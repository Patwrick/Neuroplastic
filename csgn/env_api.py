from __future__ import annotations

from typing import Any, Protocol, TypedDict

import numpy as np


class Obs(TypedDict, total=False):
    # Required by convention across backends.
    rgb: np.ndarray
    pov: np.ndarray
    inventory: dict
    equipped: dict
    location: dict
    status: dict
    message: str | dict | None
    raw: dict


StepReturn = tuple[Obs, float, bool, bool, dict]


class EmbodiedEnv(Protocol):
    @property
    def action_space(self) -> Any:
        ...

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict]:
        ...

    def step(self, action: Any) -> StepReturn:
        ...

    def close(self) -> None:
        ...
