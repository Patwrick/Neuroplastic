"""Independent lifetime state and graph-connected, captured write factors."""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModelVersion:
    theta: int = 0
    topology: int = 0
    feature: int = 0
    decoder: str = "mean_four_outputs_first8_v1"
    quantization_scale: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SlotIdentity:
    source: int
    target: int
    head: int
    generation: int


def _clone(value: Tensor, detach: bool) -> Tensor:
    return value.detach().clone() if detach else value.clone()


@dataclass(frozen=True)
class AgentState:
    """Dense B-by-E synapses; every row is a distinct lifetime."""
    S: Tensor
    Pa: Tensor
    Pu: Tensor
    h: Tensor
    wm: Tensor
    version: ModelVersion = ModelVersion()
    demand_queues: tuple[tuple[tuple[int, int], ...], ...] = ()
    metadata: dict[str, Any] | None = None

    @property
    def effective_weights(self) -> Tensor:
        return self.S + self.Pa + self.Pu

    def clone(self, detach: bool = False) -> AgentState:
        return replace(self, **{name: _clone(getattr(self, name), detach)
                               for name in ("S", "Pa", "Pu", "h", "wm")},
                       demand_queues=copy.deepcopy(self.demand_queues),
                       metadata=copy.deepcopy(self.metadata))

    def reset_transient(self) -> AgentState:
        return replace(self, h=torch.zeros_like(self.h), wm=torch.zeros_like(self.wm),
                       demand_queues=tuple(() for _ in range(self.S.shape[0])),
                       metadata=copy.deepcopy(self.metadata))

    def storage_bytes(self) -> int:
        return sum(getattr(self, name).numel() * getattr(self, name).element_size()
                   for name in ("S", "Pa", "Pu", "h", "wm"))


@dataclass(frozen=True)
class WriteSnapshot:
    """Numerically captured tensors are cloned, never implicitly detached.

    Frozen dataclasses prevent rebinding; callers must also treat tensor contents
    as read-only. No snapshot tensor aliases a mutable live state tensor.
    """
    A: Tensor
    full_design: Tensor
    writable_mask: Tensor
    captured_weights: Tensor
    prediction: Tensor
    source_messages: Tensor
    gates: Tensor
    receiver_budgets: Tensor
    edge_ids: Tensor
    edge_generations: Tensor
    version: ModelVersion
    decision_id: str
    prediction_time: int
    feedback_available_time: int
    routing: Any
    context: Tensor
    input_cue: Tensor

    def fixed_contribution(self, effective_weights: Tensor) -> Tensor:
        """Use the current post-decay values explicitly held fixed for a write."""
        return torch.einsum("bve,be->bv", self.full_design - self.A, effective_weights)

    def reconstruct(self, effective_weights: Tensor | None = None) -> Tensor:
        weights = self.captured_weights if effective_weights is None else effective_weights
        return torch.einsum("bve,be->bv", self.full_design, weights)

    def storage_bytes(self) -> int:
        # Counts retained tensors even if an optimization could share buffers.
        return sum(value.numel() * value.element_size() for value in vars(self).values()
                   if isinstance(value, Tensor)) + self.routing.storage_bytes()
