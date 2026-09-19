"""Small dense-storage CSGN reference, with gather/scatter recurrent execution.

There is no sparse-speed claim: local node keys and all B-by-E route logits are
computed exactly and charged. Top-k branch choices are discrete. Selected gate,
message, state, and snapshot tensors retain their continuous outer gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn

from .routing import Topology, RoutingDecision, bounded_norm, build_topology, exact_route
from .state import AgentState, ModelVersion, WriteSnapshot


def constrain_operator(matrix: Tensor, maximum: float) -> Tensor:
    """Exact SVD norm at this tiny scale; differentiable on smooth branches."""
    norm = torch.linalg.matrix_norm(matrix, ord=2)
    return matrix / (norm / maximum).clamp_min(1.0)


@dataclass(frozen=True)
class FrozenCycle:
    routing: RoutingDecision
    context: Tensor
    weights: Tensor
    drive: Tensor
    receiver_budgets: Tensor
    U: Tensor
    V: Tensor
    Ah: Tensor
    Br: Tensor


class GraphModel(nn.Module):
    def __init__(self, config: dict, *, seed: int = 0, dtype=torch.float32):
        super().__init__()
        self.config = config
        graph, workspace = config["graph"], config["workspace"]
        self.graph_config = graph
        self.columns, self.state_dim = graph["columns"], graph["state_dim"]
        self.message_dim, self.heads = graph["message_dim"], graph["heads"]
        self.cue_dim, self.context_dim = config["task"]["cue_dim"], config["task"]["context_dim"]
        self.feature_dim, self.route_dim = 32, graph["descriptor_dim"]
        self.output_receivers = tuple(graph["output_receivers"])
        if self.output_receivers != (0, 1, 2, 3) or self.message_dim < 8:
            raise ValueError("reference decoder requires receivers 0..3 and at least 8 message coordinates")
        if graph["address_mode"] != "state_dependent":
            raise ValueError("only declared state-dependent address profile implemented")
        self.kappa = workspace["kappa"]
        self.wmax = float(config["learning"]["wmax"])
        self.pmax = float(config["learning"]["pmax"])
        self.ah_max = float(workspace["Ah_spectral_norm_max"])
        self.u_max = 0.5
        self.br_max = float(workspace["Br_times_Lm_max"]) / self.u_max
        q_bound = 1 - self.kappa + self.kappa * (self.ah_max + self.br_max * self.u_max)
        if not 0 < self.kappa <= 1 or q_bound >= 1 or q_bound > workspace["required_q_max"]:
            raise ValueError("workspace bounds do not satisfy configured contraction")
        self.version = ModelVersion()
        self.topology = build_topology(self.columns, graph["regions"], graph["out_slots"],
                                       graph["protected_slots"], self.heads, seed=seed + 1, dtype=dtype)
        generator = torch.Generator().manual_seed(seed)

        def matrix(*shape, scale=1.0):
            return torch.randn(shape, generator=generator, dtype=dtype) * (scale / math.sqrt(shape[-1]))

        self.encoder = nn.Parameter(matrix(self.feature_dim, self.cue_dim + self.context_dim))
        self.query_matrix = nn.Parameter(matrix(self.route_dim, self.feature_dim))
        self.key_state = nn.Parameter(matrix(self.route_dim, self.state_dim))
        self.key_descriptor = nn.Parameter(matrix(self.route_dim, graph["descriptor_dim"]))
        self.head_bias = nn.Parameter(torch.zeros(self.heads, dtype=dtype))
        self.null_logit = nn.Parameter(torch.zeros((), dtype=dtype))
        self.U_raw = nn.Parameter(matrix(self.heads, self.message_dim, self.state_dim))
        self.V = nn.Parameter(matrix(self.heads, self.message_dim, self.feature_dim))
        self.Ah_raw = nn.Parameter(matrix(self.state_dim, self.state_dim))
        self.Br_raw = nn.Parameter(matrix(self.state_dim, self.message_dim))
        self.context_drive = nn.Parameter(matrix(self.state_dim, self.feature_dim))
        self.descriptor_drive = nn.Parameter(matrix(self.state_dim, graph["descriptor_dim"], scale=0.3))
        self.register_buffer("descriptors", bounded_norm(matrix(self.columns, graph["descriptor_dim"])))
        self.register_buffer("initial_slow", matrix(self.topology.edge_count, scale=0.02))

    def initial_state(self, batch_size: int = 1) -> AgentState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        slow = self.initial_slow.unsqueeze(0).expand(batch_size, -1).clone()
        return AgentState(slow, torch.zeros_like(slow), torch.zeros_like(slow),
                          slow.new_zeros(batch_size, self.columns, self.state_dim),
                          slow.new_zeros(batch_size, 0, self.context_dim), self.version,
                          tuple(() for _ in range(batch_size)), {"wm_policy": "zero_capacity_no_write"})

    def encode(self, cue: Tensor, context: Tensor) -> Tensor:
        if cue.ndim != 2 or context.ndim != 2 or cue.shape[-1] != self.cue_dim or context.shape[-1] != self.context_dim:
            raise ValueError("cue/context feature dimensions must match the public task interface")
        if not bool(torch.isfinite(cue).all() & torch.isfinite(context).all()):
            raise ValueError("nonfinite observation")
        if bool((cue.abs() > 1 + 1e-6).any() | (context.abs() > 1 + 1e-6).any()):
            raise ValueError("observation coordinates must be bounded by one")
        return torch.tanh(torch.cat((cue, context), -1) @ self.encoder.T)

    def operators(self) -> tuple[Tensor, Tensor, Tensor]:
        U = torch.stack([constrain_operator(head, self.u_max) for head in self.U_raw])
        return U, constrain_operator(self.Ah_raw, self.ah_max), constrain_operator(self.Br_raw, self.br_max)

    def operator_metrics(self) -> dict:
        U, Ah, Br = self.operators()
        a = float(torch.linalg.matrix_norm(Ah, ord=2).detach())
        b = float(torch.linalg.matrix_norm(Br, ord=2).detach())
        lm = max(float(torch.linalg.matrix_norm(head, ord=2).detach()) for head in U)
        return {"Ah_operator_norm": a, "Br_operator_norm": b, "Lm": lm,
                "q": 1 - self.kappa + self.kappa * (a + b * lm),
                "norm_method": "exact_singular_values", "Mmax": 1.0}

    def prepare_cycle(self, state: AgentState, cue: Tensor, context: Tensor,
                      *, event_time: int = 0, cold: bool = False) -> FrozenCycle:
        if state.version != self.version:
            raise ValueError("state/model version mismatch")
        if state.S.shape != (cue.shape[0], self.topology.edge_count):
            raise ValueError("synaptic shape must be [independent lifetimes, reserved edges]")
        if state.h.shape != (cue.shape[0], self.columns, self.state_dim):
            raise ValueError("workspace shape mismatch")
        if cold and bool(torch.count_nonzero(state.Pa) + torch.count_nonzero(state.Pu)):
            raise ValueError("cold evaluation requires both residual tiers permanently zero")
        from .plasticity import validate_state
        validate_state(state.S, state.Pa, state.Pu, wmax=self.wmax, pmax=self.pmax)
        if not bool(torch.isfinite(state.h).all()) or bool((state.h.abs() > 1 + 1e-6).any()):
            raise ValueError("workspace must be finite in [-1,1]")
        features = self.encode(cue, context)
        query = bounded_norm(torch.tanh(features @ self.query_matrix.T))
        keys = bounded_norm(torch.tanh(state.h @ self.key_state.T + self.descriptors @ self.key_descriptor.T))
        edge_keys = (keys[:, self.topology.source] + keys[:, self.topology.target]) * 0.5
        logits = torch.einsum("bd,bed->be", query, edge_keys) / math.sqrt(self.route_dim)
        logits = logits + self.head_bias[self.topology.head]
        graph = self.graph_config
        routing = exact_route(self.topology, query, keys, logits,
                              self.null_logit.expand(cue.shape[0], self.columns),
                              active_regions=graph["active_regions"], active_senders=graph["active_senders"],
                              real_routes=graph["real_routes_per_sender"],
                              incoming_capacity=graph["incoming_capacity"],
                              updated_node_cap=graph["updated_node_cap"], queues=state.demand_queues,
                              event_time=event_time, queue_capacity=self.columns,
                              output_receivers=self.output_receivers)
        receiver_mass = features.new_zeros(cue.shape[0], self.columns).scatter_add(
            1, self.topology.target.expand(cue.shape[0], -1), routing.admitted_gates)
        budgets = (self.wmax * receiver_mass).clamp_min(1.0)
        drive = torch.tanh((features @ self.context_drive.T).unsqueeze(1) + self.descriptors @ self.descriptor_drive.T)
        U, Ah, Br = self.operators()
        return FrozenCycle(routing, features.clone(), state.effective_weights.clone(), drive,
                           budgets, U, self.V.clone(), Ah, Br)

    def inner_round(self, h: Tensor, cycle: FrozenCycle) -> tuple[Tensor, Tensor, Tensor]:
        # Dense tiny typed source-head computation; no E-by-N-by-d temporary.
        messages = torch.tanh(torch.einsum("bnd,hmd->bnhm", h, cycle.U)
                              + torch.einsum("bf,hmf->bhm", cycle.context, cycle.V).unsqueeze(1))
        edge_messages = messages[:, self.topology.source, self.topology.head]
        weighted = edge_messages * (cycle.routing.admitted_gates * cycle.weights).unsqueeze(-1)
        aggregates = h.new_zeros(h.shape[0], self.columns, self.message_dim).scatter_add(
            1, self.topology.target.view(1, -1, 1).expand(h.shape[0], -1, self.message_dim), weighted)
        aggregates = aggregates / cycle.receiver_budgets.unsqueeze(-1)
        candidate = (1 - self.kappa) * h + self.kappa * torch.tanh(h @ cycle.Ah.T + aggregates @ cycle.Br.T + cycle.drive)
        next_h = torch.where(cycle.routing.updated_mask.unsqueeze(-1), candidate, h)
        return next_h, aggregates, edge_messages

    def predict(self, state: AgentState, cue: Tensor, context: Tensor, decision_id: str,
                prediction_time: int, feedback_available_time: int, *, cold: bool = False,
                writable_mask: Tensor | None = None) -> tuple[Tensor, AgentState, WriteSnapshot]:
        if feedback_available_time < prediction_time:
            raise ValueError("feedback cannot precede the captured prediction")
        cycle = self.prepare_cycle(state, cue, context, event_time=prediction_time, cold=cold)
        h = state.h
        for _ in range(self.graph_config["inner_rounds"]):
            h, aggregates, edge_messages = self.inner_round(h, cycle)
        prediction = aggregates[:, self.output_receivers, :8].mean(1)
        output_edges = torch.isin(self.topology.target, torch.tensor(self.output_receivers))
        factors = cycle.routing.admitted_gates / cycle.receiver_budgets[:, self.topology.target]
        factors = factors * output_edges.to(factors.dtype).unsqueeze(0) / len(self.output_receivers)
        full_design = (factors.unsqueeze(-1) * edge_messages[:, :, :8]).transpose(1, 2)
        writable = cycle.routing.executed_mask & output_edges.unsqueeze(0)
        if writable_mask is not None:
            writable = writable & writable_mask
        design = full_design * writable.unsqueeze(1).to(full_design.dtype)
        snapshot = WriteSnapshot(design.clone(), full_design.clone(), writable.clone(),
                                 cycle.weights.clone(), prediction.clone(), edge_messages.clone(),
                                 cycle.routing.admitted_gates.clone(), cycle.receiver_budgets.clone(),
                                 torch.arange(self.topology.edge_count), self.topology.generation.clone(),
                                 state.version, decision_id, prediction_time, feedback_available_time,
                                 cycle.routing, cycle.context.clone(), cue.clone())
        metadata = dict(state.metadata or {})
        metadata["last_prediction_time"] = prediction_time
        return prediction, replace(state, h=h, demand_queues=cycle.routing.next_queues,
                                   metadata=metadata), snapshot

    def storage_bytes(self) -> dict:
        return {"theta_bytes": sum(p.numel() * p.element_size() for p in self.parameters()),
                "buffer_bytes": sum(p.numel() * p.element_size() for p in self.buffers()),
                "topology_bytes": self.topology.storage_bytes(),
                "layout": "dense_BxE_correctness_reference",
                "computed_source_head_messages_per_round": self.columns * self.heads,
                "computed_edge_logits_per_decision": self.topology.edge_count,
                "region_summary_member_reads_per_decision": self.columns,
                "region_summary_policy": "exact_full_recomputation_dense_reference"}
