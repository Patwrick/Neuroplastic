"""Differentiable reference only; not a complete CSGN or training runner.

Features remain constant in the partial derivative with respect to z, while the
outer autograd graph through their construction is retained. No in-place writes.
Hard support is selected outside this function and fixed for the write.
"""
from __future__ import annotations
import torch
from torch import Tensor


def projected_write(A: Tensor, b: Tensor, s: Tensor, chosen: Tensor, other: Tensor,
                    allowances: Tensor, beta: Tensor, *, wmax: float = 1.0,
                    pmax: float = 1.0, eps: float = 1e-12,
                    detach_feature_write_path: bool = False) -> tuple[Tensor, Tensor]:
    """Single unbatched functional step. Detach option is a named ablation.

    At max/min/clip ties the derivative is not unique. Finite-difference checks
    must avoid branch boundaries; do not call a straight-through estimator exact.
    The reference assumes feasible, finite inputs; the NumPy wrapper validates
    storage inputs, and the implementation must add equivalent boundary checks.
    """
    if A.ndim != 2 or b.shape != (A.shape[0],) or s.shape != (A.shape[1],):
        raise ValueError("Expected A[m,n], b[m], state[n]")
    if any(x.shape != s.shape for x in (chosen, other, allowances)):
        raise ValueError("State/allowance shape mismatch")
    if beta.numel() != 1 or eps <= 0 or wmax <= 0 or pmax <= 0:
        raise ValueError("Scalar beta and positive budgets/epsilon required")
    if not 0 <= float(beta.detach()) <= 1:
        raise ValueError("beta must be in [0,1]")
    if bool((allowances.detach() < 0).any()):
        raise ValueError("Negative write allowance")
    for x in (A, b, s, chosen, other, allowances, beta):
        if not bool(torch.isfinite(x.detach()).all()):
            raise ValueError("Nonfinite input")
    vv = torch.stack((s, s+chosen, s+other, s+chosen+other))
    # Deliberately check outside differentiation, never modify the valid input.
    tol = 1e-9 if s.dtype == torch.float64 else 1e-5
    if bool((vv.detach().abs() > wmax+tol).any()) or bool(
            (chosen.detach().abs()+other.detach().abs() > pmax+tol).any()):
        raise ValueError("Infeasible input state")
    rad = pmax - other.abs()
    lp = torch.maximum(torch.maximum(-rad, -wmax-s), -wmax-s-other)
    up = torch.minimum(torch.minimum(rad, wmax-s), wmax-s-other)
    z0 = s+other+chosen
    lo = torch.maximum(s+other+lp, z0-allowances)
    hi = torch.minimum(s+other+up, z0+allowances)
    Af = A.detach() if detach_feature_write_path else A
    # Key distinction: this is the partial z-gradient, but its formula remains
    # differentiable with respect to theta through Af when full mode is used.
    grad_z = Af.T @ (Af @ z0 - b)
    eta = beta / (Af.square().sum() + eps)
    proposal = z0 - eta*grad_z
    z1 = torch.minimum(torch.maximum(proposal, lo), hi)
    return z1, z1-s-other
