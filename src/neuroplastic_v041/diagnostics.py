"""Offline diagnostics. Solver results never flow into model state or targets."""
import torch
from .plasticity import storage_intervals, reachability as solve_storage_box


def reachability(snapshot, state, target, config):
    """Single-lifetime smoke diagnostic on an explicitly post-decay state.

    Nonwritable coordinates are fixed. The shared solver accepts general batches,
    but this smoke adapter returns one dictionary and therefore rejects B != 1
    instead of silently discarding independent lifetimes.
    """
    if state.S.ndim != 2 or state.S.shape[0] != 1:
        raise ValueError('smoke reachability requires exactly one lifetime (B=1)')
    if state.version != snapshot.version:
        raise ValueError('stale snapshot version')
    if snapshot.writable_mask.shape != state.S.shape or snapshot.writable_mask.dtype != torch.bool:
        raise ValueError('writable_mask must be a boolean tensor matching synaptic state')
    learning = config['learning']
    lower, upper = storage_intervals(state.S, state.Pu,
                                     wmax=learning['wmax'], pmax=learning['pmax'])
    offset = state.S + state.Pu
    effective = state.S + state.Pa + state.Pu
    lo = torch.where(snapshot.writable_mask, offset+lower, effective)
    hi = torch.where(snapshot.writable_mask, offset+upper, effective)
    b = target - snapshot.fixed_contribution(effective)
    report = solve_storage_box(snapshot.A, b, lo, hi)[0]
    report['scope'] = 'storage-only fixed-snapshot half-squared loss; no downstream floor claim'
    return report


def address_drift(write_snapshot, read_snapshot):
    old, new = write_snapshot.A.flatten(), read_snapshot.A.flatten()
    cosine = torch.nn.functional.cosine_similarity(old[None],new[None]).item()
    a, b = write_snapshot.writable_mask, read_snapshot.writable_mask
    union = (a|b).sum().item()
    return dict(design_cosine=cosine, support_jaccard=(a&b).sum().item()/union if union else 1.)
