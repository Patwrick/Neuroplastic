"""Independent identities and causal/cold boundary checks for delta memory."""

from dataclasses import replace

import pytest
import torch

from neuroplastic_v041.baselines.delta import DeltaModel


def model(dtype=torch.float64):
    return DeltaModel({"task": {"cue_dim": 16, "context_dim": 8, "target_dim": 8},
                       "learning": {"wmax": 1.0, "pmax": 1.0}}, dtype=dtype)


def observation(batch=1):
    cue, context = torch.zeros(batch, 16, dtype=torch.float64), torch.zeros(batch, 8, dtype=torch.float64)
    cue[:, 0], context[:, 0] = 1, 1
    return cue, context


def capture(m, state, cue=None, context=None, **kwargs):
    if cue is None:
        cue, context = observation(state.S.shape[0])
    return m.predict(state, cue, context, "opaque", 1, 3, **kwargs)


def test_delta_exact_normalized_correction_and_no_input_mutation():
    m = model()
    state = m.initial_state(1)
    cue, context = observation()
    cue[:, 1], context[:, 1] = 0.5, 0.2
    _, _, snap = capture(m, state, cue, context)
    target = torch.linspace(0.01, 0.08, 8, dtype=torch.float64)[None]
    updated, metrics = m.write(state, snap, target, beta=0.7, eps=1e-8)
    pred, _, _ = capture(m, updated, cue, context)
    factor = 1 - 0.7 * snap.key.square().sum() / (snap.key.square().sum() + 1e-8)
    torch.testing.assert_close(target - pred, factor * target, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(state.Pa) == 0
    assert torch.all(metrics["loss_after"] < metrics["loss_before"])
    assert metrics["projection_fraction"].item() == 0


def test_delta_batch_isolation_and_snapshot_independence():
    m = model()
    state = m.initial_state(2)
    cue, context = observation(2)
    _, _, snap = capture(m, state, cue, context)
    cue.zero_()
    assert snap.key[:, 0].tolist() == [1.0, 1.0]
    target = torch.zeros(2, 8, dtype=torch.float64)
    target[0, 0] = 0.1
    updated, _ = m.write(state, snap, target)
    assert torch.count_nonzero(updated.Pa[0]) > 0
    assert torch.count_nonzero(updated.Pa[1]) == 0
    copied = updated.clone()
    copied.Pa[0, 0, 0] = 0.8
    assert updated.Pa[0, 0, 0] != 0.8


def test_delta_inward_correction_and_independent_decay():
    m = model()
    state = m.initial_state(1)
    state.S[:, 0, 0] = 1
    state = replace(state, Pu=torch.full_like(state.Pu, -0.1))
    _, _, snap = capture(m, state)
    target = torch.zeros(1, 8, dtype=torch.float64)
    assert state.version == snap.version
    assert state.storage_bytes() == 3 * 8 * 128 * 8 + 24
    assert snap.storage_bytes() >= 128 * 8 + 41
    target[:, 0] = 0.5
    updated, _ = m.write(state, snap, target)
    assert updated.Pa[0, 0, 0] < 0
    for a, u in [(0, 0), (0, 1), (1, 0), (1, 1), (0.2, 0.7)]:
        decayed = m.decay(updated, a, u)
        assert torch.max((decayed.S + decayed.Pa + decayed.Pu).abs()) <= 1


def test_delta_cold_disables_both_tiers_and_cannot_rewrite():
    m = model()
    state = m.initial_state(1)
    state.S[:, 0, 0], state.Pa[:, 0, 0], state.Pu[:, 0, 0] = 0.1, 0.2, 0.3
    before = state.clone()
    for _ in range(3):
        pred, state, snap = capture(m, state, cold=True)
        assert pred[0, 0].item() == 0.1
        with pytest.raises(ValueError, match="cold"):
            m.write(state, snap, torch.zeros(1, 8, dtype=torch.float64))
    for name in ("S", "Pa", "Pu"):
        assert torch.equal(getattr(before, name), getattr(state, name))


def test_delta_feedback_timing_version_shape_and_numeric_guards():
    m = model()
    state = m.initial_state(1)
    _, _, snap = capture(m, state)
    target = torch.zeros(1, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="not available"):
        m.write(state, snap, target, feedback_time=2)
    for field in ("model_version", "feature_version", "topology_version"):
        with pytest.raises(ValueError, match="stale"):
            m.write(replace(state, **{field: 1}), snap, target)
    with pytest.raises(ValueError, match="nonfinite"):
        m.write(state, snap, torch.full_like(target, float("nan")))
    with pytest.raises(ValueError, match="externally frozen"):
        m.write(state, snap, target.clone().requires_grad_(True))
    with pytest.raises(ValueError, match="endpoint"):
        m.write(replace(state, S=state.S + 1.1), snap, target)
    with pytest.raises(ValueError, match="batch axis"):
        m.predict(state, torch.zeros(16), torch.zeros(8), "id", 0, 0)


@pytest.mark.parametrize("zero_key,beta", [(True, 0.7), (False, 0.0)])
def test_delta_no_ops(zero_key, beta):
    m = model()
    state = m.initial_state(1)
    cue, context = observation()
    if zero_key:
        cue.zero_()
    _, _, snap = capture(m, state, cue, context)
    updated, metrics = m.write(state, snap, torch.ones(1, 8, dtype=torch.float64), beta=beta)
    assert torch.equal(updated.Pa, state.Pa)
    assert metrics["no_progress"].item() == 1


def test_delta_full_outer_path_matches_finite_difference():
    m = model()
    def objective(theta, detached=False):
        state = m.initial_state(1)
        cue, context = observation()
        cue = cue + theta * torch.nn.functional.one_hot(torch.tensor([1]), 16)
        _, _, snap = capture(m, state, cue, context)
        if detached:
            snap = replace(snap, key=snap.key.detach())
        target = torch.full((1, 8), 0.1, dtype=torch.float64)
        updated, _ = m.write(state, snap, target)
        query, query_context = observation()
        query[:, 1] = 0.3
        pred, _, _ = capture(m, updated, query, query_context)
        return (pred - target).square().sum()
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    loss = objective(theta)
    derivative = torch.autograd.grad(loss, theta)[0]
    delta = 1e-6
    finite_difference = (objective(theta.detach()+delta) - objective(theta.detach()-delta)) / (2*delta)
    torch.testing.assert_close(derivative, finite_difference, rtol=1e-7, atol=1e-10)
    assert abs(derivative.item()) > 1e-6
    assert not objective(theta, detached=True).requires_grad


def test_delta_slow_leaf_can_fit_and_cold_uses_only_slow():
    m = model()
    state = m.initial_state(1)
    slow = state.S.detach().clone().requires_grad_(True)
    state = replace(state, S=slow, Pa=state.Pa + 0.2)
    pred, _, _ = capture(m, state, cold=True)
    target = torch.zeros_like(pred)
    target[0, 0] = 0.1
    loss = (pred - target).square().sum()
    grad = torch.autograd.grad(loss, slow)[0]
    assert grad[0, 0, 0] == -0.2
    assert torch.count_nonzero(grad) == 1
