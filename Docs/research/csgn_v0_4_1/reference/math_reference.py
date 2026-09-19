"""CSGN v0.4.1 numerical reference, not a trained or complete CSGN.

NumPy float64 makes the local contracts easy to inspect. PyTorch implementations
should be independently compared with this file, not wrap it during meta-training.
All losses here are half squared Euclidean norms unless documented otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]

def _vector(x: ArrayLike, name: str) -> F64:
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return v

def _positive(x: float, name: str, allow_zero: bool = False) -> float:
    x = float(x)
    if not np.isfinite(x) or (x < 0 if allow_zero else x <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return x

def vertices(s: ArrayLike, pa: ArrayLike, pu: ArrayLike) -> F64:
    """Rows are S, S+Pa, S+Pu, S+Pa+Pu, in that order."""
    s, pa, pu = (_vector(v, n) for v, n in zip((s, pa, pu), ("s", "pa", "pu")))
    if s.shape != pa.shape or s.shape != pu.shape:
        raise ValueError("State vectors must have identical shapes")
    return np.stack((s, s + pa, s + pu, s + pa + pu))

def feasible(s: ArrayLike, pa: ArrayLike, pu: ArrayLike, wmax: float,
             pmax: float, atol: float = 1e-12) -> bool:
    wmax, pmax = _positive(wmax, "wmax"), _positive(pmax, "pmax")
    _positive(atol, "atol", True)
    vv = vertices(s, pa, pu)
    pa, pu = np.asarray(pa), np.asarray(pu)
    return bool(np.all(np.abs(vv) <= wmax + atol)
                and np.all(np.abs(pa) + np.abs(pu) <= pmax + atol))

def storage_intervals(s: ArrayLike, other: ArrayLike, wmax: float,
                      pmax: float, atol: float = 1e-12) -> tuple[F64, F64]:
    """Bounds for the chosen *residual*, not for effective weight.

    Check the two unchanged vertices first. The returned interval enforces the
    other two vertices and |chosen|+|other|<=pmax. No step allowance is included.
    """
    s, o = _vector(s, "s"), _vector(other, "other")
    if s.shape != o.shape:
        raise ValueError("Shape mismatch")
    wmax, pmax = _positive(wmax, "wmax"), _positive(pmax, "pmax")
    if (np.any(np.abs(s) > wmax + atol) or np.any(np.abs(s + o) > wmax + atol)
            or np.any(np.abs(o) > pmax + atol)):
        raise ValueError("Unchanged tier/slow vertices are infeasible")
    radius = pmax - np.abs(o)
    # Clamping only compensates the declared input roundoff tolerance.
    radius = np.maximum(radius, 0.0)
    lo = np.maximum.reduce((-radius, -wmax - s, -wmax - s - o))
    hi = np.minimum.reduce((radius, wmax - s, wmax - s - o))
    if np.any(lo > hi + atol):
        raise ValueError("Empty storage interval")
    return lo, hi

def effective_intervals(s: ArrayLike, chosen: ArrayLike, other: ArrayLike,
                        wmax: float, pmax: float,
                        allowances: ArrayLike | None = None,
                        atol: float = 1e-12) -> tuple[F64, F64]:
    """Storage-only or storage-and-step box for effective weights z=S+o+p."""
    s, p, o = (_vector(v, n) for v, n in zip((s, chosen, other), ("s", "chosen", "other")))
    if s.shape != p.shape or s.shape != o.shape:
        raise ValueError("Shape mismatch")
    lo_p, hi_p = storage_intervals(s, o, wmax, pmax, atol)
    if np.any(p < lo_p - atol) or np.any(p > hi_p + atol):
        raise ValueError("Starting chosen residual is infeasible")
    offset, z = s + o, s + o + p
    lo, hi = offset + lo_p, offset + hi_p
    if allowances is not None:
        d = _vector(allowances, "allowances")
        if d.shape != z.shape or np.any(d < 0):
            raise ValueError("Allowances must match the state and be nonnegative")
        lo, hi = np.maximum(lo, z - d), np.minimum(hi, z + d)
    if np.any(lo > hi + atol):
        raise ValueError("Empty effective interval")
    return lo, hi

@dataclass(frozen=True)
class WriteResult:
    z: F64
    chosen: F64
    eta: float
    loss_before: float
    loss_after: float
    theoretical_upper: float
    l1_write: float
    max_projection_change: float


def projected_write(A: ArrayLike, b: ArrayLike, s: ArrayLike,
                    chosen: ArrayLike, other: ArrayLike, *, wmax: float,
                    pmax: float, allowances: ArrayLike, beta: float = 1.0,
                    eps: float = 1e-12) -> WriteResult:
    """One fixed-snapshot projected step; no whole-policy descent claim."""
    A, b = np.asarray(A, dtype=np.float64), _vector(b, "b")
    s, p, o = (_vector(v, n) for v, n in zip((s, chosen, other), ("s", "chosen", "other")))
    lo, hi = effective_intervals(s, p, o, wmax, pmax, allowances)
    if A.ndim != 2 or A.shape != (b.size, s.size) or not np.isfinite(A).all():
        raise ValueError("A must be finite with shape (len(b), len(s))")
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    _positive(eps, "eps")
    z0 = s + o + p
    r0 = A @ z0 - b
    f0 = float(0.5 * r0 @ r0)
    eta = float(beta / (np.sum(A * A) + eps)) if np.any(A) and beta else 0.0
    grad = A.T @ r0
    proposed = z0 - eta * grad
    z1 = np.clip(proposed, lo, hi) if eta else z0.copy()
    r1, dz = A @ z1 - b, z1 - z0
    L = float(np.linalg.norm(A, 2) ** 2) if A.size and np.any(A) else 0.0
    upper = f0 - (1 / eta - L / 2) * float(dz @ dz) if eta else f0
    return WriteResult(z1, z1 - s - o, eta, f0, float(0.5 * r1 @ r1), upper,
                       float(np.abs(dz).sum()), float(np.max(np.abs(z1-proposed), initial=0)))


def delta_write(W: ArrayLike, key: ArrayLike, target: ArrayLike,
                beta: float = 1.0, eps: float = 1e-12) -> F64:
    W, k, v = np.asarray(W, dtype=np.float64), _vector(key, "key"), _vector(target, "target")
    if W.shape != (v.size, k.size) or not np.isfinite(W).all():
        raise ValueError("Incompatible finite matrix/key/target")
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    _positive(eps, "eps", True)
    if not np.any(k) or beta == 0:
        return W.copy()
    return W + beta * np.outer(v - W @ k, k) / (eps + k @ k)


def transfer(s: ArrayLike, pa: ArrayLike, pu: ArrayLike, fraction: ArrayLike,
             wmax: float, pmax: float) -> tuple[F64, F64, F64]:
    s, pa, pu = (_vector(v, n) for v, n in zip((s, pa, pu), ("s", "pa", "pu")))
    q = np.broadcast_to(np.asarray(fraction, dtype=np.float64), s.shape)
    if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1):
        raise ValueError("Transfer fractions must be in [0, 1]")
    if not feasible(s, pa, pu, wmax, pmax):
        raise ValueError("Cannot transfer infeasible state")
    out = (s + q * pa, (1 - q) * pa, pu.copy())
    if not feasible(*out, wmax, pmax):
        raise ArithmeticError("Transfer violated the endpoint invariant")
    return out


def reachability(A: ArrayLike, b: ArrayLike, lower: ArrayLike, upper: ArrayLike,
                 tol: float = 1e-10) -> dict[str, Any]:
    """Box-constrained snapshot reference with primal/dual objective bounds.

    SciPy is a diagnostic dependency. The result is not evidence of downstream
    task reachability. Bounds use 0.5*||Az-b||^2; multiply by 2 for E_floor SSE.
    Zero-width coordinates are removed before calling lsq_linear.
    """
    from scipy.optimize import lsq_linear
    A = np.asarray(A, dtype=np.float64)
    b, lo, hi = _vector(b, "b"), _vector(lower, "lower"), _vector(upper, "upper")
    if A.ndim != 2 or A.shape != (b.size, lo.size) or lo.shape != hi.shape:
        raise ValueError("Reachability shape mismatch")
    if not np.isfinite(A).all() or np.any(lo > hi):
        raise ValueError("Nonfinite design matrix or infeasible bounds")
    _positive(tol, "tol")
    fixed = lo == hi
    x = lo.copy()
    free = ~fixed
    if free.any():
        rhs = b - A[:, fixed] @ x[fixed]
        sol = lsq_linear(A[:, free], rhs, bounds=(lo[free], hi[free]),
                         method="bvls", tol=tol, max_iter=1000)
        x[free] = np.clip(sol.x, lo[free], hi[free])
        success, status, optimality = bool(sol.success), int(sol.status), float(sol.optimality)
    else:
        success, status, optimality = True, 3, 0.0
    residual = A @ x - b
    primal = float(0.5 * residual @ residual)
    c = A.T @ residual
    # Fenchel dual feasible for any residual vector. Every interval is finite.
    dual = float(-0.5 * residual @ residual - b @ residual
                 + np.sum(np.where(c >= 0, c * lo, c * hi)))
    lower_bound = max(0.0, dual)
    gap = primal - lower_bound
    singular = np.linalg.svd(A[:, free], compute_uv=False) if free.any() else np.zeros(0)
    rank_tol = (max(A.shape) * np.finfo(float).eps * singular[0]) if singular.size else 0.0
    rank = int(np.sum(singular > rank_tol))
    nonzero = singular[singular > rank_tol]
    condition = float(nonzero[0] / nonzero[-1]) if nonzero.size else None
    grad = A.T @ residual
    kkt_residual = float(np.max(np.abs(x - np.clip(x - grad, lo, hi)), initial=0.0))
    numerical_valid = gap >= -100 * tol * (1 + primal)
    return {"x": x.tolist(), "loss_upper": primal, "loss_lower": lower_bound,
            "primal_dual_gap": gap, "numerical_bound_check": numerical_valid,
            "solver_success": success, "solver_status": status,
            "solver_optimality": optimality, "projected_gradient_residual": kkt_residual,
            "free_rank": rank, "free_variables": int(free.sum()),
            "condition_nonzero_subspace": condition,
            "scope": "fixed-snapshot bounded least squares; not policy capacity"}
