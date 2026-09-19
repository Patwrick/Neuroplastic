"""Executable graph contracts, independent of learned-performance claims."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from neuroplastic_v041.graph import GraphModel
from neuroplastic_v041.plasticity import projected_write
from neuroplastic_v041.routing import Topology, build_topology, exact_route, validate_topology
from neuroplastic_v041.state import ModelVersion


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "Docs/research/csgn_v0_4_1/configs/cpu_smoke.json").read_text())
    config["graph"].update(columns=16, regions=4, out_slots=4, state_dim=12,
                            message_dim=8, descriptor_dim=4, active_regions=2,
                            active_senders=4, incoming_capacity=8, updated_node_cap=16)
    return config


@pytest.fixture
def model(config):
    torch.set_num_threads(1)
    return GraphModel(config, seed=73, dtype=torch.float64)


def observations(batch=1):
    generator = torch.Generator().manual_seed(41)
    cue = torch.randn(batch, 16, generator=generator, dtype=torch.float64)
    context = torch.randn(batch, 8, generator=generator, dtype=torch.float64)
    return cue / cue.norm(dim=-1, keepdim=True), context / context.norm(dim=-1, keepdim=True)


def test_topology_protected_connectivity_unique_targets_generations(config):
    topology = build_topology(64, 8, 8, 4, 2, seed=0)
    validate_topology(topology, 64, 8)
    assert topology.edge_count == 512
    assert torch.count_nonzero(topology.generation) == 0
    assert torch.all(topology.target[topology.output_delivery] < 4)
    assert torch.all(topology.protected[topology.output_delivery])
    malformed = replace(topology, target=topology.target.clone())
    malformed.target[1] = malformed.target[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_topology(malformed, 64, 8)


def test_exact_snapshot_with_nonwritable_contribution(model):
    state = model.initial_state(2)
    state = replace(state, Pa=torch.full_like(state.Pa, 0.12), Pu=torch.full_like(state.Pu, -0.02))
    cue, context = observations(2)
    mask = torch.arange(model.topology.edge_count) % 2 == 0
    prediction, after, snapshot = model.predict(state, cue, context, "captured", 2, 4, writable_mask=mask)
    actual = torch.einsum("bve,be->bv", snapshot.A, state.effective_weights)
    actual += snapshot.fixed_contribution(state.effective_weights)
    torch.testing.assert_close(actual, prediction, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(snapshot.reconstruct(), prediction, atol=1e-14, rtol=1e-12)
    changed = state.effective_weights + 0.1
    expected = torch.einsum("bve,be->bv", snapshot.full_design - snapshot.A, changed)
    torch.testing.assert_close(snapshot.fixed_contribution(changed), expected)
    assert snapshot.A.requires_grad and snapshot.A.grad_fn is not None
    assert after.S is state.S and torch.equal(after.Pa, state.Pa)
    assert snapshot.prediction_time == 2 and snapshot.feedback_available_time == 4


def test_snapshot_values_do_not_alias_live_state_or_observation(model):
    state = model.initial_state()
    cue, context = observations()
    _, _, snapshot = model.predict(state, cue, context, "id", 0, 0)
    before = snapshot.reconstruct().detach().clone()
    state.S.add_(0.2)
    cue.mul_(0)
    torch.testing.assert_close(snapshot.reconstruct(), before)
    assert torch.count_nonzero(snapshot.input_cue) > 0


def test_independent_state_rows_and_clones_keep_gradient(model):
    original = model.initial_state(2)
    original.Pa[0, 0] = 0.2
    assert original.Pa[1, 0] == 0
    meta_initial = model.initial_slow.clone().requires_grad_()
    state = replace(original, S=meta_initial.unsqueeze(0).expand(2, -1).clone())
    branch = state.clone()
    assert branch.S.data_ptr() != state.S.data_ptr()
    branch.S.sum().backward()
    torch.testing.assert_close(meta_initial.grad, torch.full_like(meta_initial, 2))
    branch.Pa[0, 0] = 0.3
    assert original.Pa[0, 0] == 0.2
    assert state.clone(detach=True).S.grad_fn is None
    assert state.reset_transient().demand_queues == ((), ())


def simple_route(availability=None, logits=None, capacity=2):
    # Two sources with two routes each; node 2 is protected delivery.
    topology = Topology(torch.tensor([0, 0, 1, 1]), torch.tensor([2, 3, 2, 3]),
                        torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
                        torch.tensor([True, False, True, False]), torch.tensor([True, False, True, False]),
                        torch.zeros(4, dtype=torch.long),
                        torch.ones(4, dtype=torch.float64) if availability is None else availability)
    keys = torch.tensor([[[1.], [0.9], [-1.], [-1.]]], dtype=torch.float64)
    logits = torch.zeros(1, 4, dtype=torch.float64) if logits is None else logits
    return exact_route(topology, torch.ones(1, 1, dtype=torch.float64), keys, logits,
                       torch.zeros(1, 4, dtype=torch.float64), active_regions=1, active_senders=2,
                       real_routes=2, incoming_capacity=capacity, updated_node_cap=4,
                       output_receivers=(2, 3), event_time=2)


def test_null_availability_before_topk_and_no_double_availability():
    availability = torch.tensor([1., 0., 0.5, 1.], dtype=torch.float64)
    logits = torch.tensor([[0., 10000., 0., 0.]], dtype=torch.float64)
    route = simple_route(availability, logits)
    assert not route.support_mask[0, 1]
    torch.testing.assert_close(route.gates[0, 0], torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(route.gates[0, 2], torch.tensor(0.2, dtype=torch.float64))
    torch.testing.assert_close(route.null_gates[0, 1], torch.tensor(0.4, dtype=torch.float64))
    logits[0, 1] = -10000.
    torch.testing.assert_close(simple_route(availability, logits).gates, route.gates)


def test_receiver_rejection_does_not_redistribute_and_ties_are_stable():
    route = simple_route(capacity=1)
    assert route.executed_mask.tolist() == [[True, True, False, False]]
    assert route.metrics[0]["admission_rejections"] == 2
    torch.testing.assert_close(route.admitted_gates[0, :2], torch.full((2,), 1 / 3, dtype=torch.float64))
    assert route.admitted_gates[0, 2:].sum() == 0
    assert route.null_gates[0, 1] == route.null_gates[0, 0]


def test_selected_logits_retain_gradients_through_null_and_admission():
    logits = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    route = simple_route(logits=logits)
    route.admitted_gates[0, 0].backward()
    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] < 0


def test_actual_bounded_messages_envelope_and_unupdated_boundary(model):
    state = model.initial_state()
    state = replace(state, h=torch.full_like(state.h, 0.5), S=torch.ones_like(state.S))
    cue, context = observations()
    cycle = model.prepare_cycle(state, cue, context)
    next_h, aggregates, messages = model.inner_round(state.h, cycle)
    assert float(messages.detach().abs().max()) <= 1
    assert float(aggregates.detach().norm(dim=-1).max()) <= model.message_dim ** 0.5 + 1e-12
    assert float(next_h.detach().abs().max()) <= 1
    inactive = ~cycle.routing.updated_mask
    torch.testing.assert_close(next_h[inactive], state.h[inactive], atol=0, rtol=0)
    # Envelope is unchanged if writable weights change.
    altered = model.prepare_cycle(replace(state, S=state.S * 0.2), cue, context)
    torch.testing.assert_close(cycle.receiver_budgets, altered.receiver_budgets, atol=0, rtol=0)


def test_exact_norm_enforcement_and_frozen_cycle_contraction(model):
    with torch.no_grad():
        model.Ah_raw.mul_(100)
        model.Br_raw.mul_(100)
        model.U_raw.mul_(100)
    metrics = model.operator_metrics()
    assert metrics["Ah_operator_norm"] <= 0.25 + 1e-12
    assert metrics["Br_operator_norm"] * metrics["Lm"] <= 0.5 + 1e-12
    assert metrics["q"] <= 0.875 + 1e-12
    state = model.initial_state()
    cue, context = observations()
    cycle = model.prepare_cycle(state, cue, context)
    second = state.h + cycle.routing.updated_mask.unsqueeze(-1) * 0.01
    first_next = model.inner_round(state.h, cycle)[0]
    second_next = model.inner_round(second, cycle)[0]
    before = (second - state.h).norm(dim=-1).max()
    after = (second_next - first_next).norm(dim=-1).max()
    assert after <= metrics["q"] * before + 1e-12
    # The frozen cycle remains numerically independent of later live weight edits.
    stored = cycle.weights.clone()
    state.S.add_(0.1)
    torch.testing.assert_close(cycle.weights, stored, atol=0, rtol=0)


def test_protected_output_delivery_and_actual_queue_service(model):
    state = model.initial_state()
    cue, context = observations()
    _, advanced, first = model.predict(state, cue, context, "first", 0, 0)
    assert first.routing.metrics[0]["executed_output_paths"] == 4
    assert first.routing.metrics[0]["queue_occupancy"] > 0
    _, _, second = model.predict(advanced, cue, context, "second", 1, 1)
    assert second.routing.metrics[0]["queue_served"] > 0
    assert second.routing.metrics[0]["queue_service_max_event_latency"] == 1
    assert len(advanced.demand_queues[0]) <= model.columns


def test_queue_overflow_is_observable(config):
    model = GraphModel(config, dtype=torch.float64)
    state = model.initial_state()
    cue, context = observations()
    cycle = model.prepare_cycle(state, cue, context)
    topology = model.topology
    route = exact_route(topology, torch.ones(1, 4, dtype=torch.float64),
                        torch.zeros(1, 16, 4, dtype=torch.float64),
                        torch.zeros(1, topology.edge_count, dtype=torch.float64),
                        torch.zeros(1, 16, dtype=torch.float64), active_regions=2, active_senders=4,
                        real_routes=4, incoming_capacity=8, updated_node_cap=16, queue_capacity=0)
    assert route.metrics[0]["queue_overflow"] > 0
    assert route.next_queues == ((),)


def test_cold_requires_permanent_fast_off_and_version_includes_scale(model):
    state = model.initial_state()
    cue, context = observations()
    with pytest.raises(ValueError, match="both residual"):
        model.predict(replace(state, Pu=state.Pu + 0.01), cue, context, "cold", 0, 0, cold=True)
    before = state.S.clone()
    for t in range(3):
        _, state, _ = model.predict(state, cue, context, str(t), t, t, cold=True)
        assert torch.count_nonzero(state.Pa) + torch.count_nonzero(state.Pu) == 0
    torch.testing.assert_close(state.S, before, atol=0, rtol=0)
    assert ModelVersion(quantization_scale=(0.1,)) != ModelVersion(quantization_scale=(0.2,))
    with pytest.raises(ValueError, match="version"):
        model.predict(replace(state, version=ModelVersion(feature=1)), cue, context, "bad", 4, 4)


def test_recurrent_write_query_outer_gradient_matches_central_difference(model):
    # A declared all-node derivative fixture ensures the query reads written
    # edges. The production smoke retains its sparse active sender budgets.
    model.graph_config.update(active_regions=4, active_senders=16, incoming_capacity=16)
    cue, context = observations()
    target = torch.tensor([[0., 0.1, 0., 0., 0., 0., 0., 0.]], dtype=torch.float64)

    def unroll(detached=False):
        state = model.initial_state()
        _, state, snapshot = model.predict(state, cue, context, "support", 0, 0)
        allowances = snapshot.writable_mask.to(torch.float64) * 0.05
        result = projected_write(snapshot.A, target - snapshot.fixed_contribution(state.effective_weights),
                                 state.S, state.Pa, state.Pu, wmax=1., pmax=1., allowances=allowances,
                                 write_budget=10., beta=0.7, eps=1e-8, detached_feature_write_path=detached)
        state = replace(state, Pa=result.chosen)
        prediction, _, query_snapshot = model.predict(state, cue, context, "query", 1, 1)
        return 1000 * (prediction - target).square().sum(), snapshot, query_snapshot

    objective, snapshot, query_snapshot = unroll()
    gradient = torch.autograd.grad(objective, model.encoder)[0]
    index = tuple(torch.unravel_index(gradient.abs().argmax(), gradient.shape))
    step = 1e-5
    original = model.encoder[index].detach().clone()
    with torch.no_grad():
        model.encoder[index] = original + step
    plus, plus_snapshot, plus_query = unroll()
    with torch.no_grad():
        model.encoder[index] = original - step
    minus, minus_snapshot, minus_query = unroll()
    with torch.no_grad():
        model.encoder[index] = original
    for perturbed in (plus_snapshot, minus_snapshot):
        assert torch.equal(perturbed.routing.support_mask, snapshot.routing.support_mask)
    for perturbed in (plus_query, minus_query):
        assert torch.equal(perturbed.routing.support_mask, query_snapshot.routing.support_mask)
    numeric = (plus - minus).detach() / (2 * step)
    torch.testing.assert_close(gradient[index], numeric, atol=2e-8, rtol=2e-5)
    detached_objective, _, _ = unroll(detached=True)
    detached_gradient = torch.autograd.grad(detached_objective, model.encoder)[0]
    assert float((gradient - detached_gradient).abs().max()) > 1e-6
    assert float(detached_gradient.abs().max()) > 0


def test_context_changes_query_key_order_not_shared_additive_offset(model):
    cue, context = observations()
    state = model.initial_state()
    first = model.prepare_cycle(state, cue, context).routing
    second = model.prepare_cycle(state, -cue, -context).routing
    assert not torch.equal(first.active_senders, second.active_senders)


def test_nonfinite_and_future_time_inputs_are_rejected(model):
    state = model.initial_state()
    cue, context = observations()
    with pytest.raises(ValueError, match="precede"):
        model.predict(state, cue, context, "bad", 2, 1)
    _, advanced, _ = model.predict(state, cue, context, "future-queue", 4, 4)
    with pytest.raises(ValueError, match="queue request"):
        model.predict(advanced, cue, context, "backward-clock", 0, 0)
    cue[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model.predict(state, cue, context, "bad", 0, 0)
