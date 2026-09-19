"""Bounded context-keyed delta memory (paper equations 16--18).

The key is the tensor product of the permitted cue and context.  This is a
separate, explicitly engineered baseline: it does not share graph features or
claim a matched-memory or matched-latency comparison.  The normalized delta
step has no graph per-event allowance; only the declared storage bounds apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _copy(value: Tensor, detach: bool) -> Tensor:
    return value.detach().clone() if detach else value.clone()


@dataclass(frozen=True)
class DeltaState:
    S: Tensor
    Pa: Tensor
    Pu: Tensor
    h: Tensor
    model_version: int = 0
    feature_version: int = 0
    topology_version: int = 0

    @property
    def version(self) -> tuple[int, int, int]:
        return self.model_version, self.feature_version, self.topology_version

    def storage_bytes(self) -> int:
        return sum(value.numel() * value.element_size()
                   for value in (self.S, self.Pa, self.Pu, self.h)) + 24

    def clone(self, detach: bool = False) -> "DeltaState":
        return DeltaState(
            *(_copy(x, detach) for x in (self.S, self.Pa, self.Pu, self.h)),
            self.model_version, self.feature_version, self.topology_version,
        )

    def reset_transient(self) -> "DeltaState":
        return DeltaState(
            self.S.clone(), self.Pa.clone(), self.Pu.clone(),
            torch.zeros_like(self.h), self.model_version,
            self.feature_version, self.topology_version,
        )


@dataclass(frozen=True)
class DeltaSnapshot:
    key: Tensor
    decision_id: Any
    prediction_time: int
    feedback_available_time: int
    model_version: int
    feature_version: int
    topology_version: int
    cold: bool = False

    @property
    def version(self) -> tuple[int, int, int]:
        return self.model_version, self.feature_version, self.topology_version

    def storage_bytes(self) -> int:
        # Key, three int64 versions, two int64 times, cold flag and opaque ID.
        return self.key.numel() * self.key.element_size() + 41 + len(str(self.decision_id).encode("utf-8"))


class DeltaModel:
    """Functional per-lifetime memory with a fixed public output interface."""

    def __init__(
        self, config: dict[str, Any], seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.seed = seed  # Identity only; there is no randomized key encoder.
        self.dtype = dtype
        task = config["task"]
        self.cue_dim = int(task["cue_dim"])
        self.context_dim = int(task["context_dim"])
        self.output_dim = int(task["target_dim"])
        self.key_dim = self.cue_dim * self.context_dim
        self.wmax = float(config["learning"]["wmax"])
        self.pmax = float(config["learning"]["pmax"])
        if min(self.cue_dim, self.context_dim, self.output_dim) <= 0:
            raise ValueError("positive feature/output dimensions required")
        if not (0 < self.wmax < float("inf") and 0 < self.pmax < float("inf")):
            raise ValueError("finite positive storage bounds required")

    def initial_state(self, batch_size: int = 1) -> DeltaState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.output_dim, self.key_dim)
        return DeltaState(
            torch.zeros(shape, dtype=self.dtype),
            torch.zeros(shape, dtype=self.dtype),
            torch.zeros(shape, dtype=self.dtype),
            torch.zeros((batch_size, 0), dtype=self.dtype),
        )

    def _validate(self, state: DeltaState) -> None:
        expected = (state.S.shape[0], self.output_dim, self.key_dim)
        if any(t.shape != expected for t in (state.S, state.Pa, state.Pu)):
            raise ValueError("invalid delta state shape")
        if not all(torch.isfinite(t).all() for t in (state.S, state.Pa, state.Pu)):
            raise ValueError("nonfinite delta state")
        tolerance = 8 * torch.finfo(state.S.dtype).eps
        endpoints = torch.stack((state.S, state.S + state.Pa,
                                 state.S + state.Pu, state.S + state.Pa + state.Pu))
        if torch.any(endpoints.abs() > self.wmax + tolerance):
            raise ValueError("delta endpoint bound violated")
        if torch.any(state.Pa.abs() + state.Pu.abs() > self.pmax + tolerance):
            raise ValueError("delta residual budget violated")

    def predict(
        self, state: DeltaState, cue: Tensor, context: Tensor,
        decision_id: Any, prediction_time: int, feedback_available_time: int,
        cold: bool = False,
    ) -> tuple[Tensor, DeltaState, DeltaSnapshot]:
        self._validate(state)
        batch = state.S.shape[0]
        if cue.shape != (batch, self.cue_dim) or context.shape != (batch, self.context_dim):
            raise ValueError("cue/context must have explicit independent-lifetime batch axis")
        if not torch.isfinite(cue).all() or not torch.isfinite(context).all():
            raise ValueError("nonfinite observation")
        if torch.any(cue.abs() > 1 + 1e-6) or torch.any(context.abs() > 1 + 1e-6):
            raise ValueError("observation coordinates must be bounded by one")
        if feedback_available_time < prediction_time:
            raise ValueError("feedback cannot predate prediction")
        key = (cue.unsqueeze(2) * context.unsqueeze(1)).flatten(1)
        weights = state.S if cold else state.S + state.Pa + state.Pu
        prediction = torch.einsum("bvk,bk->bv", weights, key)
        captured = DeltaSnapshot(
            key.clone(), decision_id, prediction_time, feedback_available_time,
            state.model_version, state.feature_version, state.topology_version, cold,
        )
        # No recurrent workspace, episodic store, or answer retrieval exists here.
        return prediction, state.clone(), captured

    def write(
        self, state: DeltaState, snapshot: DeltaSnapshot, target: Tensor,
        beta: float | Tensor = 0.7, eps: float = 1e-8,
        feedback_time: int | None = None,
    ) -> tuple[DeltaState, dict[str, Tensor]]:
        self._validate(state)
        if snapshot.cold:
            raise ValueError("cold snapshots never permit writes")
        if feedback_time is None:
            feedback_time = snapshot.feedback_available_time
        if feedback_time < snapshot.feedback_available_time:
            raise ValueError("feedback is not available yet")
        if (state.model_version, state.feature_version, state.topology_version) != (
            snapshot.model_version, snapshot.feature_version, snapshot.topology_version
        ):
            raise ValueError("stale delta snapshot version")
        if target.shape != (state.S.shape[0], self.output_dim):
            raise ValueError("invalid target shape")
        if target.requires_grad:
            raise ValueError("reference targets must be externally frozen")
        if snapshot.key.shape != (state.S.shape[0], self.key_dim):
            raise ValueError("snapshot/state lifetime mismatch")
        if not torch.isfinite(target).all() or not torch.isfinite(snapshot.key).all():
            raise ValueError("nonfinite target or key")
        rate = torch.as_tensor(beta, dtype=state.S.dtype, device=state.S.device)
        if rate.ndim != 0 or not torch.isfinite(rate) or not (0 <= rate <= 1):
            raise ValueError("beta must be a finite scalar in [0, 1]")
        if not (0 < eps < float("inf")):
            raise ValueError("eps must be finite and positive")
        key = snapshot.key
        z = state.S + state.Pa + state.Pu
        error = target.detach() - torch.einsum("bvk,bk->bv", z, key)
        key_norm_sq = key.square().sum(dim=1)
        eta = rate / (key_norm_sq + eps)
        # Zero keys produce an exact no-op without a NumPy or no_grad boundary.
        proposal = z + eta[:, None, None] * error[:, :, None] * key[:, None, :]
        reserve = self.pmax - state.Pu.abs()
        lower = torch.maximum(-reserve, torch.maximum(-self.wmax-state.S,
                                                       -self.wmax-state.S-state.Pu))
        upper = torch.minimum(reserve, torch.minimum(self.wmax-state.S,
                                                      self.wmax-state.S-state.Pu))
        offset = state.S + state.Pu
        projected = torch.maximum(offset + lower, torch.minimum(offset + upper, proposal))
        changed = projected - z
        result = DeltaState(
            state.S.clone(), projected - offset, state.Pu.clone(), state.h.clone(),
            state.model_version, state.feature_version, state.topology_version,
        )
        self._validate(result)
        after = target.detach() - torch.einsum("bvk,bk->bv", projected, key)
        metrics = {
            "loss_before": error.square().sum(dim=1) / 2,
            "loss_after": after.square().sum(dim=1) / 2,
            "eta": eta,
            "intentional_write_l1": changed.abs().sum(dim=(1, 2)),
            "projection_fraction": (projected != proposal).to(state.S.dtype).mean(dim=(1, 2)),
            "zero_key": (key_norm_sq == 0).to(state.S.dtype),
            "no_progress": (changed == 0).all(dim=2).all(dim=1).to(state.S.dtype),
        }
        return result, metrics

    def decay(self, state: DeltaState, admissible: float = 1.0,
              ephemeral: float = 1.0) -> DeltaState:
        self._validate(state)
        if not (0 <= admissible <= 1 and 0 <= ephemeral <= 1):
            raise ValueError("decay factors must belong to [0, 1]")
        result = DeltaState(
            state.S.clone(), state.Pa * admissible, state.Pu * ephemeral,
            state.h.clone(), state.model_version, state.feature_version,
            state.topology_version,
        )
        self._validate(result)
        return result
