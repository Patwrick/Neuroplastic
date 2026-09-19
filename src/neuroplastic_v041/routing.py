"""Exact small-graph routing; hard choices use stable ascending identity ties."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class Topology:
    source: Tensor
    target: Tensor
    head: Tensor
    generation: Tensor
    protected: Tensor
    output_delivery: Tensor
    region: Tensor
    availability: Tensor

    @property
    def edge_count(self) -> int:
        return self.source.numel()

    def storage_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in vars(self).values())


def build_topology(columns: int, regions: int, out_slots: int, protected_slots: int,
                   heads: int, *, seed: int = 0, dtype=torch.float32) -> Topology:
    if columns % regions or protected_slots != 4 or not (4 <= out_slots < columns):
        raise ValueError("profile requires equal regions, four protected slots, unique nonself targets")
    generator = torch.Generator().manual_seed(seed)
    width = columns // regions
    targets, protection, delivery = [], [], []
    for source in range(columns):
        chosen: list[int] = []
        for target in ((source + 1) % columns, (source - 1) % columns,
                       (source + width) % columns):
            while target == source or target in chosen:
                target = (target + 1) % columns
            chosen.append(target)
        output_options = [(source + offset) % 4 for offset in range(4)]
        candidates = [v for v in output_options if v != source and v not in chosen]
        if not candidates:
            raise ValueError("no distinct protected output delivery slot")
        chosen.append(candidates[0])
        remaining = [v for v in range(columns) if v != source and v not in chosen]
        order = torch.randperm(len(remaining), generator=generator).tolist()
        chosen.extend(remaining[i] for i in order[:out_slots - 4])
        targets.extend(chosen)
        protection.extend([True] * 4 + [False] * (out_slots - 4))
        delivery.extend([False, False, False, True] + [False] * (out_slots - 4))
    count = columns * out_slots
    topology = Topology(torch.arange(columns).repeat_interleave(out_slots),
                        torch.tensor(targets), torch.arange(count) % heads,
                        torch.zeros(count, dtype=torch.long), torch.tensor(protection),
                        torch.tensor(delivery), torch.arange(columns) // width,
                        torch.ones(count, dtype=dtype))
    validate_topology(topology, columns, out_slots)
    return topology


def validate_topology(topology: Topology, columns: int, out_slots: int) -> None:
    for node in range(columns):
        edges = topology.source == node
        targets = topology.target[edges].tolist()
        if len(targets) != out_slots or len(set(targets)) != out_slots or node in targets:
            raise ValueError("duplicate/self target or incorrect reserved slot count")
        if int(topology.output_delivery[edges].sum()) != 1:
            raise ValueError("each source requires one protected output route")
    # Actual directed reachability of the protected graph, independently of service.
    for origin in range(columns):
        reached, frontier = {origin}, [origin]
        while frontier:
            node = frontier.pop()
            for target in topology.target[(topology.source == node) & topology.protected].tolist():
                if target not in reached:
                    reached.add(target)
                    frontier.append(target)
        if len(reached) != columns:
            raise ValueError("protected graph is not strongly connected")


def stable_topk(scores: Tensor, eligible: Tensor, k: int) -> Tensor:
    """Identity order breaks equal-score ties; zero availability is excluded first."""
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    order = torch.argsort(scores[indices], descending=True, stable=True)
    return indices[order[:k]]


def bounded_norm(value: Tensor) -> Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0)


@dataclass(frozen=True)
class RoutingDecision:
    active_senders: Tensor
    selected_regions: Tensor
    scored_mask: Tensor
    support_mask: Tensor
    executed_mask: Tensor
    gates: Tensor
    admitted_gates: Tensor
    null_gates: Tensor
    updated_mask: Tensor
    next_queues: tuple[tuple[tuple[int, int], ...], ...]
    metrics: tuple[dict, ...]

    def storage_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in vars(self).values()
                   if isinstance(v, Tensor))


def exact_route(topology: Topology, query: Tensor, keys: Tensor, edge_logits: Tensor,
                null_logits: Tensor, *, active_regions: int, active_senders: int,
                real_routes: int, incoming_capacity: int, updated_node_cap: int,
                queues=(), event_time: int = 0, queue_capacity: int | None = None,
                output_receivers=(0, 1, 2, 3)) -> RoutingDecision:
    """Select support once, retain null, admit deterministically without renormalizing.

    Demands prioritize regions then senders in FIFO order within hard budgets.
    Full queues reject the newest request, recording overflow. Queue requests are
    service hints, never stored answers. Direct output delivery is counted as an
    executed one-round path only when its positive gate is actually admitted.
    """
    batch, columns = keys.shape[:2]
    edges = topology.edge_count
    regions = int(topology.region.max()) + 1
    if not (1 <= active_regions <= regions and 1 <= active_senders <= columns and real_routes >= 1):
        raise ValueError("invalid region, sender, or route budget")
    if incoming_capacity < 1 or updated_node_cap < len(output_receivers):
        raise ValueError("invalid admission budget")
    for value in (query, keys, edge_logits, null_logits, topology.availability):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("routing inputs must be finite")
    if bool(((topology.availability < 0) | (topology.availability > 1)).any()):
        raise ValueError("availability must be in [0,1]")
    queue_capacity = columns if queue_capacity is None else queue_capacity
    queues = queues or tuple(() for _ in range(batch))
    region_keys = torch.stack([keys[:, topology.region == r].mean(1) for r in range(regions)], 1)
    region_scores = torch.einsum("bd,brd->br", query, bounded_norm(region_keys)) / math.sqrt(query.shape[-1])
    local_scores = torch.einsum("bd,bnd->bn", query, keys) / math.sqrt(query.shape[-1])
    active_rows, region_rows, scored_rows, support_rows = [], [], [], []
    executed_rows, gate_rows, admitted_rows, null_rows, updated_rows = [], [], [], [], []
    next_queues, all_metrics = [], []
    for b in range(batch):
        queued = list(queues[b])
        if any(queued_at > event_time for _, queued_at in queued):
            raise ValueError("event time precedes a pending queue request")
        demanded_regions = list(dict.fromkeys(int(topology.region[node]) for node, _ in queued))
        ranked_regions = stable_topk(region_scores[b], torch.ones(regions, dtype=torch.bool), regions).tolist()
        chosen_regions = list(dict.fromkeys(demanded_regions + ranked_regions))[:active_regions]
        eligible_nodes = torch.isin(topology.region, torch.tensor(chosen_regions))
        demanded_nodes = [node for node, _ in queued if bool(eligible_nodes[node])]
        ranked_nodes = stable_topk(local_scores[b], eligible_nodes, columns).tolist()
        senders = list(dict.fromkeys(demanded_nodes + ranked_nodes))[:active_senders]
        active = torch.zeros(columns, dtype=torch.bool)
        active[senders] = True
        scored = active[topology.source]
        support = torch.zeros(edges, dtype=torch.bool)
        gates = torch.zeros_like(edge_logits[b])
        null = torch.ones_like(null_logits[b])
        adjusted = edge_logits[b] + torch.log(topology.availability.clamp_min(torch.finfo(query.dtype).tiny))
        for node in senders:
            eligible = (topology.source == node) & (topology.availability > 0)
            reserved = torch.nonzero(eligible & topology.output_delivery).flatten()
            remaining = eligible & ~topology.output_delivery
            selected = torch.cat((reserved, stable_topk(adjusted, remaining, real_routes - len(reserved))))
            support[selected] = True
            distribution = torch.softmax(torch.cat((null_logits[b, node:node + 1], adjusted[selected])), 0)
            gates = gates.index_copy(0, selected, distribution[1:])
            null = null.index_copy(0, torch.tensor([node]), distribution[:1])
        execution = torch.zeros(edges, dtype=torch.bool)
        updated = set(senders) | set(output_receivers)
        if len(updated) > updated_node_cap:
            raise ValueError("sender and interface set exceeds updated-node cap")
        incoming = [0] * columns
        # Priority is protected delivery, adjusted logit, ascending edge identity.
        candidates = stable_topk(adjusted, support, edges).tolist()
        candidates.sort(key=lambda edge: not bool(topology.output_delivery[edge]))
        for edge in candidates:
            target = int(topology.target[edge])
            if incoming[target] >= incoming_capacity:
                continue
            if target not in updated and len(updated) >= updated_node_cap:
                continue
            execution[edge] = True
            incoming[target] += 1
            updated.add(target)
        admitted = gates * execution.to(gates.dtype)
        updated_mask = torch.zeros(columns, dtype=torch.bool)
        updated_mask[list(updated)] = True
        served = [(node, time) for node, time in queued if node in senders]
        future = [(node, time) for node, time in queued if node not in senders]
        already = {node for node, _ in future}
        overflow = 0
        for target in topology.target[execution].tolist():
            if target in output_receivers or target in already:
                continue
            if len(future) >= queue_capacity:
                overflow += 1
            else:
                future.append((target, event_time))
                already.add(target)
        all_metrics.append({"regions_scored": regions, "local_scores_computed": columns,
                            "local_candidates_in_selected_regions": int(eligible_nodes.sum()),
                            "edges_scored": int(scored.sum()), "edges_selected": int(support.sum()),
                            "edges_executed": int(execution.sum()),
                            "admission_rejections": int((support & ~execution).sum()),
                            "updated_nodes": len(updated), "queue_served": len(served),
                            "queue_service_max_event_latency": max((event_time - t for _, t in served), default=0),
                            "queue_occupancy": len(future), "queue_overflow": overflow,
                            "executed_output_paths": int((execution & topology.output_delivery & (gates > 0)).sum()),
                            "output_path_internal_round_latency": 1,
                            "hard_selection_gradient": "conditional_on_fixed_support"})
        selected_region = torch.zeros(regions, dtype=torch.bool)
        selected_region[chosen_regions] = True
        for dest, value in ((active_rows, active), (region_rows, selected_region), (scored_rows, scored),
                            (support_rows, support), (executed_rows, execution), (gate_rows, gates),
                            (admitted_rows, admitted), (null_rows, null), (updated_rows, updated_mask)):
            dest.append(value)
        next_queues.append(tuple(future))
    return RoutingDecision(*(torch.stack(rows) for rows in
                             (active_rows, region_rows, scored_rows, support_rows, executed_rows,
                              gate_rows, admitted_rows, null_rows, updated_rows)),
                           tuple(next_queues), tuple(all_metrics))
