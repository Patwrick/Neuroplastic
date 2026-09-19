"""Post-decay captured objectives, causal boundaries and diagnostic integration."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neuroplastic_v041.diagnostics import reachability
from neuroplastic_v041.feedback import graph_decay, graph_write
from neuroplastic_v041.state import AgentState, ModelVersion


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / 'Docs/research/csgn_v0_4_1/configs/cpu_smoke.json').read_text())
    config['learning'].update(admissible_decay=0.65, ephemeral_time_constant=5.0)
    return config


def fixture(batch=1):
    s = tensor([[0.2, -0.1, 0.05]]).repeat(batch, 1)
    pa = tensor([[0.08, -0.04, 0.1]]).repeat(batch, 1)
    pu = tensor([[0.1, -0.05, 0.03]]).repeat(batch, 1)
    state = AgentState(s, pa, pu, tensor([[[0.0]]]).repeat(batch, 1, 1), torch.zeros(batch, 0, 1))
    full = tensor([[[0.6, -0.3, 0.2], [0.15, 0.25, -0.5]]]).repeat(batch, 1, 1)
    mask = torch.tensor([[True, False, True]]).repeat(batch, 1)
    snap = snapshot(state, full, mask)
    return state, snap, tensor([[0.1, 0.04]]).repeat(batch, 1)


def snapshot(state, full, mask):
    design = full * mask.unsqueeze(1)
    result = SimpleNamespace(A=design, full_design=full, writable_mask=mask,
                             prediction=torch.einsum('bve,be->bv', full, state.effective_weights),
                             version=state.version)
    result.fixed_contribution = lambda weights: torch.einsum('bve,be->bv', full-design, weights)
    return result


def test_captured_objective_uses_post_decay_other_tier_and_fixed_columns(config):
    state, snap, target = fixture(batch=2)
    post = graph_decay(state, config, elapsed=2)
    updated, result = graph_write(state, snap, target, config, elapsed=2)
    predicted_before = torch.einsum('bve,be->bv', snap.full_design, post.effective_weights)
    predicted_after = torch.einsum('bve,be->bv', snap.full_design, updated.effective_weights)
    torch.testing.assert_close(result.loss_before, 0.5*(predicted_before-target).square().sum(-1), atol=1e-16, rtol=1e-14)
    torch.testing.assert_close(result.loss_after, 0.5*(predicted_after-target).square().sum(-1), atol=1e-16, rtol=1e-14)
    assert torch.all(result.loss_after < result.loss_before)
    torch.testing.assert_close(updated.Pu, state.Pu*math.exp(-2/5), atol=0, rtol=0)
    torch.testing.assert_close(updated.Pa[:, 1], state.Pa[:, 1]*0.65, atol=0, rtol=0)
    assert not torch.equal(snap.fixed_contribution(post.effective_weights), snap.fixed_contribution(state.effective_weights))
    for field in ('S', 'Pa', 'Pu'):
        torch.testing.assert_close(getattr(state, field), getattr(fixture(batch=2)[0], field), atol=0, rtol=0)


def test_preapplied_decay_matches_default_and_no_write_branch_has_identical_baseline(config):
    state, snap, target = fixture()
    baseline = graph_decay(state, config, elapsed=3)
    implicit, implicit_result = graph_write(state, snap, target, config, elapsed=3)
    explicit, explicit_result = graph_write(baseline, snap, target, config, apply_decay=False)
    for field in ('S', 'Pa', 'Pu'):
        assert torch.equal(getattr(implicit, field), getattr(explicit, field))
    assert torch.equal(implicit_result.loss_before, explicit_result.loss_before)
    assert torch.equal(baseline.Pu, explicit.Pu)
    config['learning']['beta'] = 0.0
    no_write, result = graph_write(baseline, snap, target, config, apply_decay=False)
    for field in ('S', 'Pa', 'Pu'):
        assert torch.equal(getattr(no_write, field), getattr(baseline, field))
    assert result.l1_write.item() == 0.0
    assert result.no_op.item()
    assert (state.Pa-baseline.Pa).abs().sum().item() > 0
    assert (state.Pu-baseline.Pu).abs().sum().item() > 0


def test_detach_ablation_removes_all_captured_features_but_retains_state_gradient(config):
    config['learning']['admissible_decay'] = 1.0
    theta = tensor(0.7).requires_grad_()
    s = tensor([[0.1, 0.2]]).requires_grad_()
    state = AgentState(s, torch.zeros_like(s), torch.zeros_like(s), tensor([[[0.0]]]), torch.zeros(1, 0, 1))
    full = theta * tensor([[[1.0, 1.0]]])
    snap = snapshot(state, full, torch.tensor([[True, False]]))
    target = tensor([[0.19]])
    complete, _ = graph_write(state, snap, target, config, apply_decay=False)
    ablated, _ = graph_write(state, snap, target, config, apply_decay=False, detached_feature_write_path=True)
    assert torch.equal(complete.Pa, ablated.Pa)
    complete_feature_grad = torch.autograd.grad(complete.Pa.sum(), theta, retain_graph=True)[0]
    missing_feature_grad, state_grad = torch.autograd.grad(ablated.Pa.sum(), (theta, s), allow_unused=True)
    assert complete_feature_grad.abs() > 1e-3
    assert missing_feature_grad is None or missing_feature_grad.item() == 0
    # The held-fixed nonwritable synapse remains an outer state dependency.
    assert state_grad[0, 1].abs() > 0.1


def test_graph_write_rate_tensor_retains_outer_gradient(config):
    state, snap, target = fixture()
    beta = tensor(0.5).requires_grad_()
    config['learning']['beta'] = beta
    updated, _ = graph_write(state, snap, target, config)
    gradient = torch.autograd.grad(updated.Pa.sum(), beta)[0]
    step = 1e-6
    config['learning']['beta'] = beta.detach()+step
    plus = graph_write(state, snap, target, config)[0].Pa.sum()
    config['learning']['beta'] = beta.detach()-step
    minus = graph_write(state, snap, target, config)[0].Pa.sum()
    torch.testing.assert_close(gradient, (plus-minus)/(2*step), atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize('bad', [-1.0, float('inf'), float('nan')])
def test_elapsed_boundary_rejects_nonfinite_or_negative(config, bad):
    state, snap, target = fixture()
    with pytest.raises(ValueError, match='elapsed'):
        graph_decay(state, config, bad)
    with pytest.raises(ValueError, match='elapsed'):
        graph_write(state, snap, target, config, elapsed=bad, apply_decay=False)


@pytest.mark.parametrize('bad', [0.0, -1.0, float('inf'), float('nan')])
def test_ephemeral_time_constant_must_be_finite_positive(config, bad):
    config['learning']['ephemeral_time_constant'] = bad
    with pytest.raises(ValueError, match='time_constant'):
        graph_decay(fixture()[0], config)


def test_stale_version_and_invalid_mask_rejected_before_write(config):
    state, snap, target = fixture()
    stale = replace(state, version=ModelVersion(feature=1))
    with pytest.raises(ValueError, match='stale'):
        graph_write(stale, snap, target, config)
    with pytest.raises(ValueError, match='stale'):
        reachability(snap, stale, target, config)
    snap.writable_mask = snap.writable_mask.float()
    with pytest.raises(ValueError, match='boolean'):
        graph_write(state, snap, target, config)


def test_reachability_fixes_nonwritable_edges_and_reports_same_post_decay_objective(config):
    state, snap, target = fixture()
    post = graph_decay(state, config, 2)
    report = reachability(snap, post, target, config)
    solution = tensor([report['x']])
    assert solution[0, 1] == post.effective_weights[0, 1]
    actual = torch.einsum('bve,be->bv', snap.full_design, solution)
    assert report['loss_upper'] == pytest.approx(float(0.5*(actual-target).square().sum()), abs=1e-14)
    assert report['free_variables'] == 2
    assert report['free_rank'] == 2
    assert report['solver_success'] and report['tight_at_declared_tolerance']
    assert report['projected_gradient_residual'] < 1e-10


def test_reachability_rejects_multiple_lifetimes_instead_of_silently_dropping_them(config):
    state, snap, target = fixture(batch=2)
    with pytest.raises(ValueError, match='exactly one lifetime'):
        reachability(snap, state, target, config)


def test_empty_writable_mask_is_a_logged_noop_and_fixed_box_diagnostic(config):
    state, snap, target = fixture()
    snap = snapshot(state, snap.full_design, torch.zeros_like(snap.writable_mask))
    post = graph_decay(state, config)
    updated, result = graph_write(post, snap, target, config, apply_decay=False)
    assert torch.equal(updated.Pa, post.Pa)
    assert result.zero_design.item() and result.no_op.item()
    report = reachability(snap, post, target, config)
    assert report['free_variables'] == report['free_rank'] == 0
    assert report['loss_upper'] == pytest.approx(result.loss_before.item(), abs=1e-14)
