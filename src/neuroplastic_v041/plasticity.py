"""Batched functional v0.4.1 endpoint geometry and fixed-snapshot writes.

Leading dimensions identify independent lifetimes. No operation averages memory
between them. Validation reads detached values only to reject bad inputs; every
continuous value used in a write keeps its autograd lineage. The one deliberately
detached feature path is an explicitly named ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


def _finite(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(x.detach()).all()):
        raise ValueError(f"{name} must be finite")


def _state(*xs: Tensor) -> None:
    for i, x in enumerate(xs):
        _finite(x, f"state[{i}]")
    if not xs or xs[0].ndim < 1:
        raise ValueError("State requires a final edge dimension")
    if any(x.shape != xs[0].shape or x.dtype != xs[0].dtype or x.device != xs[0].device for x in xs):
        raise ValueError("State tensors must have identical shape, dtype and device")


def _scalar(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _tolerance(x: Tensor, atol: float | None) -> float:
    # These are numerical boundary tolerances, never a reserve relaxation.
    return _scalar(atol, "atol", zero=True) if atol is not None else (1e-12 if x.dtype == torch.float64 else 1e-6)


def _parameter(value: Tensor | float, reference: Tensor, name: str, shape: tuple[int, ...]) -> Tensor:
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    _finite(value, name)
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc


def vertices(s: Tensor, pa: Tensor, pu: Tensor) -> Tensor:
    """Return [..., 4, E] endpoints in S, S+Pa, S+Pu, S+Pa+Pu order."""
    _state(s, pa, pu)
    result = torch.stack((s, s + pa, s + pu, s + pa + pu), dim=-2)
    _finite(result, "endpoint arithmetic")
    return result


def feasible(s: Tensor, pa: Tensor, pu: Tensor, wmax: float, pmax: float,
             atol: float | None = None) -> bool:
    """Check every lifetime; nonfinite inputs raise instead of passing comparisons."""
    vv = vertices(s, pa, pu)
    wmax, pmax = _scalar(wmax, "wmax"), _scalar(pmax, "pmax")
    tol = _tolerance(s, atol)
    return bool((vv.detach().abs() <= wmax + tol).all() and
                (pa.detach().abs() + pu.detach().abs() <= pmax + tol).all())


def validate_state(s: Tensor, pa: Tensor, pu: Tensor, *, wmax: float,
                   pmax: float, atol: float | None = None) -> None:
    if not feasible(s, pa, pu, wmax, pmax, atol):
        raise ValueError("Infeasible four-endpoint or residual-budget state")


def _storage(s: Tensor, other: Tensor, wmax: float, pmax: float,
             atol: float | None) -> tuple[Tensor, Tensor, Tensor]:
    _state(s, other)
    wmax, pmax = _scalar(wmax, "wmax"), _scalar(pmax, "pmax")
    tol = _tolerance(s, atol)
    fixed = s + other
    _finite(fixed, "unchanged endpoints")
    if bool(((s.detach().abs() > wmax + tol) | (fixed.detach().abs() > wmax + tol) |
             (other.detach().abs() > pmax + tol)).any()):
        raise ValueError("Unchanged slow/other-tier endpoints are infeasible")
    radius = pmax - other.abs()
    repaired = radius < 0
    radius = radius.clamp_min(0)
    lo = torch.maximum(torch.maximum(-radius, -wmax - s), -wmax - fixed)
    hi = torch.minimum(torch.minimum(radius, wmax - s), wmax - fixed)
    _finite(lo, "storage lower bound")
    _finite(hi, "storage upper bound")
    if bool((lo.detach() > hi.detach() + tol).any()):
        raise ValueError("Empty storage interval")
    # Only a <=tol inversion can reach here. Collapse it explicitly and count it.
    inverted = lo > hi
    midpoint = (lo + hi) / 2
    lo, hi = torch.where(inverted, midpoint, lo), torch.where(inverted, midpoint, hi)
    return lo, hi, (repaired | inverted).sum(dim=-1)


def storage_intervals(s: Tensor, other: Tensor, *, wmax: float, pmax: float,
                      atol: float | None = None) -> tuple[Tensor, Tensor]:
    """Exact storage-only bounds for the chosen residual, not effective weight."""
    lo, hi, _ = _storage(s, other, wmax, pmax, atol)
    return lo, hi


def _effective(s: Tensor, chosen: Tensor, other: Tensor, *, wmax: float,
               pmax: float, allowances: Tensor | None,
               atol: float | None) -> tuple[Tensor, Tensor, Tensor]:
    _state(s, chosen, other)
    lo_p, hi_p, repairs = _storage(s, other, wmax, pmax, atol)
    tol = _tolerance(s, atol)
    if bool(((chosen.detach() < lo_p.detach() - tol) |
             (chosen.detach() > hi_p.detach() + tol)).any()):
        raise ValueError("Starting chosen residual is infeasible")
    offset, z0 = s + other, s + other + chosen
    _finite(z0, "effective starting weight")
    lo, hi = offset + lo_p, offset + hi_p
    if allowances is not None:
        _state(s, allowances)
        if bool((allowances.detach() < 0).any()):
            raise ValueError("Negative write allowance")
        lo, hi = torch.maximum(lo, z0 - allowances), torch.minimum(hi, z0 + allowances)
    _finite(lo, "effective lower bound")
    _finite(hi, "effective upper bound")
    if bool((lo.detach() > hi.detach() + tol).any()):
        raise ValueError("Empty effective interval")
    inverted = lo > hi
    midpoint = (lo + hi) / 2
    return (torch.where(inverted, midpoint, lo), torch.where(inverted, midpoint, hi),
            repairs + inverted.sum(dim=-1))


def effective_intervals(s: Tensor, chosen: Tensor, other: Tensor, *, wmax: float,
                        pmax: float, allowances: Tensor | None = None,
                        atol: float | None = None) -> tuple[Tensor, Tensor]:
    """Bounds for effective z; omit allowances only for storage diagnostics."""
    lo, hi, _ = _effective(s, chosen, other, wmax=wmax, pmax=pmax,
                          allowances=allowances, atol=atol)
    return lo, hi


@dataclass(frozen=True)
class WriteResult:
    """Per-lifetime diagnostics; bound fractions use positive-allowance edges."""
    z: Tensor
    chosen: Tensor
    eta: Tensor
    loss_before: Tensor
    loss_after: Tensor
    theoretical_upper: Tensor
    l1_write: Tensor
    max_projection_change: Tensor
    no_op: Tensor
    zero_design: Tensor
    zero_rate: Tensor
    roundoff_repairs: Tensor
    boundary_tolerance_uses: Tensor
    write_budget: Tensor
    allowance_sum: Tensor
    storage_bound_fraction: Tensor
    step_bound_fraction: Tensor


def projected_write(A: Tensor, b: Tensor, s: Tensor, chosen: Tensor, other: Tensor,
                    *, wmax: float, pmax: float, allowances: Tensor,
                    write_budget: Tensor | float, beta: Tensor | float = 1.0,
                    eps: float = 1e-12, detached_feature_write_path: bool = False,
                    atol: float | None = None) -> WriteResult:
    """One joint projected step, with budget and diagnostics per lifetime.

    A has shape [..., V, E], b [..., V], and every state/allowance [..., E].
    Nonselected coordinates must have zero allowance. Targets are externally
    supplied; this operator does not authorize their availability or provenance.
    """
    _state(s, chosen, other, allowances)
    for name, x in (("A", A), ("b", b)):
        _finite(x, name)
        if x.dtype != s.dtype or x.device != s.device:
            raise ValueError(f"{name} must share state dtype and device")
    if A.ndim != s.ndim + 1 or A.shape[:-2] != s.shape[:-1] or A.shape[-1] != s.shape[-1] or b.shape != A.shape[:-1]:
        raise ValueError("Expected A[..., V, E], b[..., V], state[..., E]")
    eps = _scalar(eps, "eps")
    tol = _tolerance(s, atol)
    rate = _parameter(beta, s, "beta", s.shape[:-1])
    budget = _parameter(write_budget, s, "write_budget", s.shape[:-1])
    if bool(((rate.detach() < 0) | (rate.detach() > 1)).any()):
        raise ValueError("beta must be in [0, 1]")
    if bool((budget.detach() < 0).any()):
        raise ValueError("Negative write budget")
    allowance_sum = allowances.sum(dim=-1)
    _finite(allowance_sum, "allowance sum")
    if bool((allowance_sum.detach() > budget.detach() + tol).any()):
        raise ValueError("Allowance sum exceeds the global intentional-write budget")
    lo, hi, repairs = _effective(s, chosen, other, wmax=wmax, pmax=pmax,
                                 allowances=allowances, atol=atol)
    Af = A.detach() if detached_feature_write_path else A
    z0 = s + other + chosen
    r0 = torch.matmul(Af, z0.unsqueeze(-1)).squeeze(-1) - b
    grad = torch.matmul(Af.transpose(-1, -2), r0.unsqueeze(-1)).squeeze(-1)
    energy = Af.square().sum(dim=(-2, -1))
    for name, x in (("residual", r0), ("gradient", grad), ("design energy", energy)):
        _finite(x, name)
    zero_design, zero_rate = energy == 0, rate == 0
    active = ~zero_design & ~zero_rate
    eta = torch.where(active, rate / (energy + eps), torch.zeros_like(rate))
    proposal = z0 - eta.unsqueeze(-1) * grad
    _finite(proposal, "write proposal")
    projected = torch.minimum(torch.maximum(proposal, lo), hi)
    # These no-op branches must preserve the exact input, including roundoff.
    writable = active.unsqueeze(-1) & allowances.ne(0)
    z1 = torch.where(writable, projected, z0)
    p1 = torch.where(writable, z1 - s - other, chosen)
    validate_state(s, p1, other, wmax=wmax, pmax=pmax, atol=tol)
    r1 = torch.matmul(Af, z1.unsqueeze(-1)).squeeze(-1) - b
    dz = z1 - z0
    f0, f1 = 0.5 * r0.square().sum(dim=-1), 0.5 * r1.square().sum(dim=-1)
    if A.shape[-1] and A.shape[-2]:
        lipschitz = torch.linalg.matrix_norm(Af, ord=2).square()
    else:
        lipschitz = torch.zeros_like(energy)
    safe_eta = torch.where(active, eta, torch.ones_like(eta))
    upper = torch.where(active, f0 - (safe_eta.reciprocal() - lipschitz / 2) * dz.square().sum(dim=-1), f0)
    l1 = dz.abs().sum(dim=-1)
    for name, x in (("loss before", f0), ("loss after", f1), ("descent bound", upper), ("write size", l1)):
        _finite(x, name)
    if bool((l1.detach() > budget.detach() + tol * max(1, s.shape[-1])).any()):
        raise ArithmeticError("Write violated its intentional budget")
    if bool((dz.detach().abs() > allowances.detach() + tol).any()):
        raise ArithmeticError("Write violated a per-edge allowance")
    # Numerical checks are scale-aware only for loss arithmetic, not storage.
    loss_tol = tol * (1 + f0.detach())
    if bool((f1.detach() > upper.detach() + loss_tol).any()):
        raise ArithmeticError("Captured descent bound violated")
    storage_lo, storage_hi = effective_intervals(s, chosen, other, wmax=wmax, pmax=pmax, atol=tol)
    boundary_uses = ((vertices(s, chosen, other).detach().abs() > wmax).any(dim=-2) |
                     (chosen.detach().abs() + other.detach().abs() > pmax)).sum(dim=-1)
    permitted = allowances > 0
    count = permitted.sum(dim=-1).clamp_min(1)
    maximum = (z1 - proposal).abs().amax(dim=-1) if s.shape[-1] else torch.zeros_like(energy)
    return WriteResult(z1, p1, eta, f0, f1, upper, l1, maximum, dz.eq(0).all(dim=-1),
                       zero_design, zero_rate, repairs, boundary_uses, budget, allowance_sum,
                       (((z1 - storage_lo).abs().le(tol) | (z1 - storage_hi).abs().le(tol)) & permitted).sum(dim=-1) / count,
                       ((dz.abs() >= allowances - tol) & permitted).sum(dim=-1) / count)


def decay(s: Tensor, pa: Tensor, pu: Tensor, retention_a: Tensor | float,
          retention_u: Tensor | float, *, wmax: float, pmax: float,
          atol: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Independent passive tier decay; intentional writes are accounted separately."""
    validate_state(s, pa, pu, wmax=wmax, pmax=pmax, atol=atol)
    la = _parameter(retention_a, s, "retention_a", s.shape)
    lu = _parameter(retention_u, s, "retention_u", s.shape)
    if bool(((la.detach() < 0) | (la.detach() > 1) | (lu.detach() < 0) | (lu.detach() > 1)).any()):
        raise ValueError("Retention fractions must be in [0, 1]")
    out = s.clone(), pa * la, pu * lu
    validate_state(*out, wmax=wmax, pmax=pmax, atol=atol)
    return out


def transfer(s: Tensor, pa: Tensor, pu: Tensor, fraction: Tensor | float,
             *, wmax: float, pmax: float,
             atol: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Exact optional Pa-to-S transfer; it is not slow-only behavioral fitting."""
    validate_state(s, pa, pu, wmax=wmax, pmax=pmax, atol=atol)
    q = _parameter(fraction, s, "fraction", s.shape)
    if bool(((q.detach() < 0) | (q.detach() > 1)).any()):
        raise ValueError("Transfer fractions must be in [0, 1]")
    out = s + q * pa, (1 - q) * pa, pu.clone()
    validate_state(*out, wmax=wmax, pmax=pmax, atol=atol)
    return out


def reachability(A: Tensor, b: Tensor, lower: Tensor, upper: Tensor,
                 *, tolerance: float = 1e-10) -> list[dict]:
    """Offline storage-box diagnostic, with independent numerical primal/dual bounds.

    This intentionally detached SciPy solve is never called by the write path.
    One result is returned per flattened lifetime; callers supply storage-only
    intervals, never the per-event allowance box.
    """
    import numpy as np
    from scipy.optimize import lsq_linear

    _state(lower, upper)
    for name, x in (("A", A), ("b", b)):
        _finite(x, name)
    tolerance = _scalar(tolerance, "tolerance")
    if b.ndim < 1 or A.ndim < 2 or A.shape != (*lower.shape[:-1], b.shape[-1], lower.shape[-1]) or b.shape[:-1] != lower.shape[:-1]:
        raise ValueError("Reachability shapes disagree")
    if bool((lower.detach() > upper.detach()).any()):
        raise ValueError("Inverted storage bounds")
    mats = A.detach().cpu().double().reshape(-1, A.shape[-2], A.shape[-1]).numpy()
    rhs = b.detach().cpu().double().reshape(len(mats), b.shape[-1]).numpy()
    lows = lower.detach().cpu().double().reshape(len(mats), lower.shape[-1]).numpy()
    highs = upper.detach().cpu().double().reshape(len(mats), upper.shape[-1]).numpy()
    result = []
    for matrix, target, lo, hi in zip(mats, rhs, lows, highs, strict=True):
        fixed = lo == hi
        free = ~fixed
        x = lo.copy()
        if free.any():
            sol = lsq_linear(matrix[:, free], target - matrix[:, fixed] @ x[fixed],
                             bounds=(lo[free], hi[free]), method="bvls", tol=tolerance, max_iter=1000)
            x[free] = np.clip(sol.x, lo[free], hi[free])
            success, status, optimality = bool(sol.success), int(sol.status), float(sol.optimality)
        else:
            success, status, optimality = True, 3, 0.0
        residual = matrix @ x - target
        c = matrix.T @ residual
        primal = float(0.5 * residual @ residual)
        dual = max(0.0, float(-0.5 * residual @ residual - target @ residual + np.minimum(c * lo, c * hi).sum()))
        gap = primal - dual
        singular = np.linalg.svd(matrix[:, free], compute_uv=False) if free.any() else np.zeros(0)
        rank_tol = max(matrix.shape) * np.finfo(float).eps * singular[0] if singular.size else 0.0
        nonzero = singular[singular > rank_tol]
        numerical = bool(gap >= -100 * tolerance * (1 + primal))
        result.append({"x": x.tolist(), "loss_upper": primal, "loss_lower": dual,
                       "primal_dual_gap": gap, "numerical_bound_check": numerical,
                       "solver_success": success, "solver_status": status, "solver_optimality": optimality,
                       "free_rank": int(nonzero.size), "free_variables": int(free.sum()),
                       "singular_values": singular.tolist(),
                       "condition_nonzero_subspace": float(nonzero[0] / nonzero[-1]) if nonzero.size else None,
                       "projected_gradient_residual": float(np.max(np.abs(x - np.clip(x - c, lo, hi)), initial=0)),
                       "active_bounds": int(((np.abs(x - lo) <= tolerance) | (np.abs(x - hi) <= tolerance)).sum()),
                       "tight_at_declared_tolerance": bool(success and numerical and gap <= tolerance * (1 + primal)),
                       "scope": "fixed-snapshot storage box; not downstream policy capacity"})
    return result
