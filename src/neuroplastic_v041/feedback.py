"""Bounded causal feedback records, with explicit lost-credit accounting."""
from dataclasses import dataclass, replace
import math
import torch
from .plasticity import decay, projected_write


@dataclass(frozen=True)
class PendingRecord:
    snapshot: object
    deadline: int
    charged_bytes: int


class PendingFeedback:
    def __init__(self, capacity=32, byte_capacity=16_000_000, max_delay=16):
        if min(capacity, byte_capacity) < 1 or max_delay < 0:
            raise ValueError('invalid pending budgets')
        self.capacity, self.byte_capacity, self.max_delay = capacity, byte_capacity, max_delay
        self.records = {}
        self.bytes = self.peak_bytes = 0
        self.counts = {name: 0 for name in ('overflow', 'expired', 'version_rejected', 'applied', 'invalidated_at_sleep')}

    def add(self, snapshot):
        if snapshot.decision_id in self.records:
            raise ValueError('duplicate decision record')
        charge = snapshot.storage_bytes() if hasattr(snapshot, 'storage_bytes') else snapshot.key.numel()*snapshot.key.element_size()+128
        if len(self.records) >= self.capacity or self.bytes+charge > self.byte_capacity:
            self.counts['overflow'] += 1
            return False
        self.records[snapshot.decision_id] = PendingRecord(snapshot, snapshot.prediction_time+self.max_delay, charge)
        self.bytes += charge
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return True

    def _remove(self, identifier):
        record = self.records.pop(identifier)
        self.bytes -= record.charged_bytes
        return record

    def take(self, identifier, now, version, generations=None):
        if identifier not in self.records:
            raise ValueError('unknown or already consumed feedback')
        record = self.records[identifier]
        snapshot = record.snapshot
        if now < snapshot.feedback_available_time or now < snapshot.prediction_time:
            raise ValueError('feedback arrived before available time')
        record = self._remove(identifier)
        if now > record.deadline:
            self.counts['expired'] += 1
            return None
        if snapshot.version != version or (generations is not None and
                not torch.equal(snapshot.edge_generations, generations)):
            self.counts['version_rejected'] += 1
            return None
        return record.snapshot

    def expire(self, now):
        for identifier in tuple(self.records):
            if now > self.records[identifier].deadline:
                self._remove(identifier)
                self.counts['expired'] += 1

    def reconcile_commit(self):
        # Reset-boundary policy: unresolved pre-cutoff records are explicitly lost.
        self.counts['invalidated_at_sleep'] += len(self.records)
        self.records.clear()
        self.bytes = 0


def graph_decay(state, config, elapsed=1):
    """Post-decay baseline shared by written/unwritten diagnostic branches.

    Pa retention is per event; Pu uses the elapsed interval. Intentional writes
    are separate. Callers log passive movement against the incoming state.
    """
    learning = config['learning']
    elapsed = float(elapsed)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('elapsed must be finite and nonnegative')
    time_constant = float(learning['ephemeral_time_constant'])
    if not math.isfinite(time_constant) or time_constant <= 0:
        raise ValueError('ephemeral_time_constant must be finite and positive')
    s, pa, pu = decay(state.S, state.Pa, state.Pu,
                     retention_a=learning['admissible_decay'],
                     retention_u=math.exp(-elapsed/time_constant),
                     wmax=learning['wmax'], pmax=learning['pmax'])
    return replace(state, S=s, Pa=pa, Pu=pu)


def graph_write(state, snapshot, target, config, *, elapsed=1,
                detached_feature_write_path=False, apply_decay=True):
    """Apply one authorized snapshot; PendingFeedback handles timing/generations.

    ``apply_decay=False`` takes an already decayed state, useful when cloning the
    same baseline for no-write utility and storage-only reachability diagnostics.
    """
    learning = config['learning']
    if state.version != snapshot.version:
        raise ValueError('stale snapshot version')
    if target.shape != snapshot.prediction.shape or not torch.isfinite(target).all():
        raise ValueError('invalid external target')
    if target.requires_grad:
        raise ValueError('reference targets must be externally frozen')
    if snapshot.writable_mask.shape != state.S.shape or snapshot.writable_mask.dtype != torch.bool:
        raise ValueError('writable_mask must be a boolean tensor matching synaptic state')
    if not math.isfinite(float(elapsed)) or float(elapsed) < 0:
        raise ValueError('elapsed must be finite and nonnegative')
    post_decay = graph_decay(state, config, elapsed) if apply_decay else state
    s, pa, pu = post_decay.S, post_decay.Pa, post_decay.Pu
    effective = s + pa + pu
    fixed_features = snapshot.full_design - snapshot.A
    if detached_feature_write_path:
        # Both writable and nonwritable captured factors are feature-to-write
        # paths. Detach factors only: inherited synaptic state remains connected.
        fixed_features = fixed_features.detach()
    b = target - torch.einsum('bve,be->bv', fixed_features, effective)
    # Allocation is fixed before the write and its sum is bounded per lifetime.
    mask = snapshot.writable_mask.to(pa.dtype)
    per_edge = torch.minimum(torch.full_like(mask.sum(-1, keepdim=True), learning['per_edge_allowance']),
                             learning['total_write_budget']/mask.sum(-1, keepdim=True).clamp_min(1))
    result = projected_write(snapshot.A, b, s, pa, pu,
                             wmax=learning['wmax'], pmax=learning['pmax'],
                             allowances=mask*per_edge, beta=learning['beta'],
                             write_budget=learning['total_write_budget'], eps=learning['rate_epsilon'],
                             detached_feature_write_path=detached_feature_write_path)
    return replace(state, S=s, Pa=result.chosen, Pu=pu), result
