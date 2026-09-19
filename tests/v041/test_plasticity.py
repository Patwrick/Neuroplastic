"""Independent NumPy release parity and regression tests for v0.4.1 math."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from neuroplastic_v041.plasticity import (
    decay, effective_intervals, feasible, projected_write, reachability,
    storage_intervals, transfer, validate_state, vertices,
)


def _reference():
    # Import a read-only oracle without writing __pycache__ in the release tree.
    path = Path(__file__).resolve().parents[2] / "Docs/research/csgn_v0_4_1/reference/math_reference.py"
    spec = importlib.util.spec_from_file_location("v041_numpy_math_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


REF = _reference()
DTYPE = torch.float64


def tensor(value, *, dtype=DTYPE):
    return torch.tensor(value, dtype=dtype)


def fixture(seed=3, batch=3, edges=7, width=4):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-0.8, 0.8, (batch, edges))
    other = rng.uniform(-0.15, 0.15, s.shape)
    intervals = [REF.storage_intervals(si, oi, 1.0, 0.7) for si, oi in zip(s, other)]
    chosen = np.stack([rng.uniform(lo * 0.8, hi * 0.8) for lo, hi in intervals])
    A = rng.normal(size=(batch, width, edges))
    b = rng.normal(size=(batch, width))
    allowances = rng.uniform(0, 0.15, s.shape)
    return tuple(tensor(x) for x in (A, b, s, chosen, other, allowances))


def write(args, **kwargs):
    A, b, s, chosen, other, allowances = args
    return projected_write(A, b, s, chosen, other, wmax=1.0, pmax=0.7,
                           allowances=allowances, write_budget=allowances.sum(-1), **kwargs)


@pytest.mark.parametrize("seed", range(30))
def test_batched_numpy_reference_parity_and_descent(seed):
    args = fixture(seed)
    A, b, s, p, o, d = args
    before = [x.clone() for x in args]
    result = write(args, beta=0.83)
    lo, hi = effective_intervals(s, p, o, wmax=1.0, pmax=0.7, allowances=d)
    for i in range(3):
        expected = REF.projected_write(A[i].numpy(), b[i].numpy(), s[i].numpy(), p[i].numpy(), o[i].numpy(),
                                       allowances=d[i].numpy(), beta=0.83, wmax=1.0, pmax=0.7)
        bounds = REF.effective_intervals(s[i].numpy(), p[i].numpy(), o[i].numpy(), 1.0, 0.7, d[i].numpy())
        np.testing.assert_allclose(lo[i].numpy(), bounds[0], atol=2e-15, rtol=2e-15)
        np.testing.assert_allclose(hi[i].numpy(), bounds[1], atol=2e-15, rtol=2e-15)
        for name in ("z", "chosen", "eta", "loss_before", "loss_after", "theoretical_upper", "l1_write", "max_projection_change"):
            np.testing.assert_allclose(getattr(result, name)[i].detach().numpy(), getattr(expected, name), atol=2e-13, rtol=2e-13)
    assert feasible(s, result.chosen, o, 1.0, 0.7)
    assert torch.all(result.loss_after <= result.theoretical_upper + 1e-12)
    assert torch.all(result.loss_after <= result.loss_before + 1e-12)
    assert torch.all(result.l1_write <= d.sum(-1) + 1e-12)
    assert torch.equal(result.roundoff_repairs, torch.zeros(3, dtype=torch.long))
    for actual, original in zip(args, before):
        assert torch.equal(actual, original)


def test_exact_intervals_match_four_endpoints_including_opposite_residuals():
    s = tensor([[1.0, -1.0, 0.85, -0.5]])
    other = tensor([[-0.2, 0.3, -0.5, 0.1]])
    lo, hi = storage_intervals(s, other, wmax=1, pmax=0.8)
    np.testing.assert_allclose(lo.numpy()[0], REF.storage_intervals(s.numpy()[0], other.numpy()[0], 1, 0.8)[0])
    np.testing.assert_allclose(hi.numpy()[0], REF.storage_intervals(s.numpy()[0], other.numpy()[0], 1, 0.8)[1])
    for ratio in torch.linspace(0, 1, 23, dtype=DTYPE):
        assert feasible(s, lo + ratio * (hi - lo), other, 1, 0.8)
    assert not feasible(s, lo - 1e-5, other, 1, 0.8)
    assert not feasible(s, hi + 1e-5, other, 1, 0.8)


def test_saturated_slow_weight_accepts_inward_correction():
    s, zero = tensor([[1.0]]), tensor([[0.0]])
    result = projected_write(tensor([[[1.0]]]), tensor([[0.7]]), s, zero, zero,
                             wmax=1, pmax=0.5, allowances=tensor([[0.5]]), write_budget=0.5)
    assert result.chosen[0, 0] < -0.29
    assert float(result.loss_after) < 1e-22
    assert feasible(s, result.chosen, zero, 1, 0.5)
    # Regression: the superseded absolute-sum reserve would reject this state.
    assert float((s.abs() + result.chosen.abs()).sum()) > 1.0


def test_batch_state_and_gradients_are_independent():
    A, b, s, p, o, d = fixture()
    p = p.clone().requires_grad_()
    result = write((A, b, s, p, o, d))
    first_gradient = torch.autograd.grad(result.z[0].sum(), p)[0]
    assert torch.equal(first_gradient[1:], torch.zeros_like(first_gradient[1:]))
    b_changed = b.clone()
    b_changed[0] += 8
    other = write((A, b_changed, s, p, o, d))
    assert torch.equal(other.z[1:], result.z[1:])
    result.chosen[0, 0] = 0  # Outputs have independent storage, not aliased inputs.
    assert torch.equal(p[1:], fixture()[3][1:])


def test_two_tier_decay_is_independent_and_preserves_convex_endpoints():
    _, _, s, pa, pu, _ = fixture()
    la = tensor([[0.0], [0.4], [1.0]])
    lu = tensor([[1.0], [0.2], [0.0]])
    s1, a1, u1 = decay(s, pa, pu, la, lu, wmax=1, pmax=0.7)
    vv = vertices(s, pa, pu)
    combined = ((1-la)*(1-lu)*vv[:, 0] + la*(1-lu)*vv[:, 1] +
                (1-la)*lu*vv[:, 2] + la*lu*vv[:, 3])
    torch.testing.assert_close(s1+a1+u1, combined, atol=2e-16, rtol=2e-15)
    assert feasible(s1, a1, u1, 1, 0.7)
    assert torch.equal(a1[0], torch.zeros_like(a1[0]))
    assert torch.equal(u1[0], pu[0])
    for _ in range(5):
        s1, a1, u1 = decay(s1, a1, u1, 0.8, 0.4, wmax=1, pmax=0.7)
        assert feasible(s1, a1, u1, 1, 0.7)


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_exact_transfer_preserves_sum_endpoints_and_gradient(fraction):
    _, _, s, pa, pu, _ = fixture()
    pa = pa.requires_grad_()
    out = transfer(s, pa, pu, fraction, wmax=1, pmax=0.7)
    torch.testing.assert_close(sum(out), s+pa+pu, atol=2e-16, rtol=2e-15)
    assert feasible(*out, 1, 0.7)
    torch.testing.assert_close(torch.autograd.grad(sum(out).sum(), pa)[0], torch.ones_like(pa))
    for i in range(3):
        ref = REF.transfer(s[i].numpy(), pa[i].detach().numpy(), pu[i].numpy(), fraction, 1, 0.7)
        for x, expected in zip(out, ref):
            np.testing.assert_allclose(x[i].detach().numpy(), expected, atol=2e-16, rtol=2e-15)
    assert out[2].data_ptr() != pu.data_ptr()


@pytest.mark.parametrize("mode", ["zero_design", "zero_beta", "zero_allowance", "empty_edges"])
def test_no_op_paths_are_exact_and_logged(mode):
    A, b, s, p, o, d = fixture()
    beta = 1.0
    if mode == "zero_design":
        A = torch.zeros_like(A)
    elif mode == "zero_beta":
        beta = 0.0
    elif mode == "zero_allowance":
        d = torch.zeros_like(d)
    else:
        A, s, p, o, d = A[..., :0], s[..., :0], p[..., :0], o[..., :0], d[..., :0]
    result = write((A, b, s, p, o, d), beta=beta)
    assert result.no_op.all()
    assert torch.equal(result.chosen, p)
    assert torch.equal(result.loss_before, result.loss_after)
    assert torch.equal(result.l1_write, torch.zeros_like(result.l1_write))
    if mode in ("zero_design", "zero_beta", "empty_edges"):
        assert torch.equal(result.chosen, p)
        assert torch.equal(result.eta, torch.zeros_like(result.eta))


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_rejected_before_clipping(index, bad):
    args = list(fixture())
    args[index] = args[index].clone()
    args[index].flatten()[0] = bad
    with pytest.raises(ValueError, match="finite"):
        write(args)


@pytest.mark.parametrize("name,value", [("wmax", 0), ("pmax", -1), ("eps", 0), ("eps", float("inf")),
                                        ("beta", -0.1), ("beta", 1.1), ("beta", float("nan")),
                                        ("write_budget", -0.1), ("write_budget", float("nan"))])
def test_invalid_scalar_parameters_rejected(name, value):
    A, b, s, p, o, d = fixture()
    kw = dict(wmax=1., pmax=0.7, eps=1e-12, beta=1., write_budget=d.sum(-1))
    kw[name] = value
    with pytest.raises(ValueError):
        projected_write(A, b, s, p, o, allowances=d, **kw)


@pytest.mark.parametrize("kind", ["slow", "other", "chosen", "allowance", "budget", "shape", "dtype"])
def test_invalid_state_and_budget_boundaries(kind):
    A, b, s, p, o, d = fixture()
    if kind == "slow": s[0, 0] = 1.2
    if kind == "other": o[0, 0] = 1.2
    if kind == "chosen": p[0, 0] = 1.2
    if kind == "allowance": d[0, 0] = -0.1
    if kind == "shape": p = p[:, :2]
    if kind == "dtype": A = A.float()
    with pytest.raises(ValueError):
        projected_write(A, b, s, p, o, wmax=1, pmax=0.7, allowances=d,
                        write_budget=d.sum(-1) - (0.01 if kind == "budget" else 0))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_decay_and_transfer_fractions(value):
    _, _, s, pa, pu, _ = fixture()
    with pytest.raises(ValueError):
        decay(s, pa, pu, value, 1, wmax=1, pmax=0.7)
    with pytest.raises(ValueError):
        transfer(s, pa, pu, value, wmax=1, pmax=0.7)


def test_finite_overflow_is_rejected_before_it_can_be_clipped():
    A, b, s, p, o, d = fixture()
    A = torch.full_like(A, 1e250)
    with pytest.raises(ValueError, match="finite"):
        write((A, b, s, p, o, d))


def test_tiny_roundoff_repair_is_explicit_and_gross_failure_rejected():
    s, chosen = tensor([[0.0]]), tensor([[0.0]])
    other = tensor([[0.5 + 1e-13]])
    result = projected_write(tensor([[[0.2]]]), tensor([[0.0]]), s, chosen, other,
                             wmax=1, pmax=0.5, allowances=tensor([[0.1]]), write_budget=0.1)
    assert int(result.roundoff_repairs) == 1
    assert int(result.boundary_tolerance_uses) == 1
    with pytest.raises(ValueError, match="infeasible"):
        projected_write(tensor([[[0.2]]]), tensor([[0.0]]), s, chosen, other + 1e-5,
                        wmax=1, pmax=0.5, allowances=tensor([[0.1]]), write_budget=0.1)


def outer_objective(theta, beta, *, detached=False):
    base = tensor([[0.5, -0.2, 0.3], [0.1, 0.4, -0.25]])
    direction = tensor([[0.12, 0.05, -0.08], [0.07, -0.06, 0.11]])
    A = base + theta * direction
    s = tensor([0.03, -0.02, 0.01]) + theta * tensor([0.01, 0.015, -0.02])
    p, o = torch.zeros_like(s), torch.zeros_like(s)
    result = projected_write(A, tensor([0.09, -0.035]), s, p, o, wmax=1, pmax=0.7,
                             allowances=torch.full_like(s, 0.3), write_budget=0.9, beta=beta,
                             detached_feature_write_path=detached)
    # Direct query feature path remains connected in both ablation conditions.
    query = tensor([[0.3, 0.1, -0.2], [-0.25, 0.2, 0.4]]) + theta * direction
    return ((query @ result.z - tensor([0.015, 0.02])) ** 2).sum()


@pytest.mark.parametrize("theta_value", [-0.4, -0.1, 0.2, 0.5])
def test_outer_gradient_including_features_initial_s_and_rate_matches_finite_difference(theta_value):
    theta = tensor(theta_value).requires_grad_()
    beta = tensor(0.63).requires_grad_()
    analytic = torch.autograd.grad(outer_objective(theta, beta), (theta, beta))
    step = 1e-6
    fd_theta = (outer_objective(theta.detach()+step, beta.detach()) - outer_objective(theta.detach()-step, beta.detach())) / (2*step)
    fd_beta = (outer_objective(theta.detach(), beta.detach()+step) - outer_objective(theta.detach(), beta.detach()-step)) / (2*step)
    torch.testing.assert_close(analytic[0], fd_theta, atol=1e-11, rtol=1e-6)
    torch.testing.assert_close(analytic[1], fd_beta, atol=1e-11, rtol=1e-6)


def test_detached_feature_path_is_distinct_ablation_retaining_direct_query_gradient():
    theta = tensor(0.2).requires_grad_()
    full = outer_objective(theta, tensor(0.63))
    detached = outer_objective(theta, tensor(0.63), detached=True)
    full_grad = torch.autograd.grad(full, theta)[0]
    detached_grad = torch.autograd.grad(detached, theta)[0]
    assert torch.equal(full, detached)
    assert abs(float(full_grad-detached_grad)) > 1e-6
    assert abs(float(detached_grad)) > 1e-6


def test_active_face_feature_derivative_is_zero_away_from_projection_kink():
    feature = tensor(1.0).requires_grad_()
    result = projected_write(feature.reshape(1, 1), tensor([5.0]), tensor([0.0]), tensor([0.0]), tensor([0.0]),
                             wmax=1, pmax=0.5, allowances=tensor([0.1]), write_budget=0.1)
    assert float(result.z.detach()) == pytest.approx(0.1)
    assert float(torch.autograd.grad(result.z.sum(), feature)[0]) == 0
    for value in (0.999999, 1.000001):
        other = projected_write(tensor([[value]]), tensor([5.0]), tensor([0.0]), tensor([0.0]), tensor([0.0]),
                                wmax=1, pmax=0.5, allowances=tensor([0.1]), write_budget=0.1)
        assert torch.equal(other.z, result.z)


def test_active_face_preserves_continuous_allowance_gradient_and_masked_state():
    allowance = tensor([0.1, 0.0]).requires_grad_()
    s, p, o = tensor([0.03, 0.13]), tensor([0.01, 0.07]), tensor([0.02, -0.04])
    result = projected_write(tensor([[1.0, 0.2]]), tensor([5.0]), s, p, o,
                             wmax=1, pmax=0.5, allowances=allowance, write_budget=0.1)
    derivative = torch.autograd.grad(result.z[0], allowance)[0]
    torch.testing.assert_close(derivative, tensor([1.0, 0.0]))
    assert torch.equal(result.chosen[1], p[1])


def test_float32_matches_float64_at_declared_precision():
    args = fixture(7)
    double = write(args)
    single = write(tuple(x.float() for x in args))
    torch.testing.assert_close(single.z.double(), double.z, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(single.loss_after.double(), double.loss_after, atol=5e-7, rtol=2e-6)


def test_storage_reachability_known_floor_and_fixed_coordinates():
    A = tensor([[[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]]])
    b = tensor([[2., 0.3], [0.1, 0.4]])
    lo = tensor([[-1., 0.], [0., 0.]])
    hi = tensor([[1., 0.], [0., 0.]])
    reports = reachability(A, b, lo, hi)
    assert reports[0]["loss_lower"] == pytest.approx(0.545)
    assert reports[0]["loss_upper"] == pytest.approx(0.545)
    assert reports[0]["free_rank"] == 1
    assert reports[0]["singular_values"] == [1.0]
    assert reports[1]["free_rank"] == 0
    for i, report in enumerate(reports):
        expected = REF.reachability(A[i].numpy(), b[i].numpy(), lo[i].numpy(), hi[i].numpy())
        for key in ("loss_upper", "loss_lower", "primal_dual_gap", "free_rank", "free_variables"):
            assert report[key] == pytest.approx(expected[key], abs=1e-12)
        assert report["tight_at_declared_tolerance"]


def test_storage_diagnostic_does_not_confuse_zero_allowance_with_no_capacity():
    s = tensor([0.0])
    storage = effective_intervals(s, s, s, wmax=1, pmax=1)
    step = effective_intervals(s, s, s, wmax=1, pmax=1, allowances=s)
    assert reachability(tensor([[1.0]]), tensor([0.5]), *storage)[0]["loss_upper"] < 1e-20
    assert reachability(tensor([[1.0]]), tensor([0.5]), *step)[0]["loss_lower"] == 0.125
