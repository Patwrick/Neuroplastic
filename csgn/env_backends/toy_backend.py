from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from csgn.env_api import EmbodiedEnv, Obs, StepReturn


class ToyEnv(EmbodiedEnv):
    def __init__(self, obs_size: tuple[int, int] = (128, 128), episode_len: int = 200):
        if len(obs_size) != 2:
            raise ValueError(f"obs_size must be (H, W), got {obs_size!r}")
        if episode_len <= 0:
            raise ValueError(f"episode_len must be > 0, got {episode_len!r}")

        self._obs_size = (int(obs_size[0]), int(obs_size[1]))
        self._episode_len = int(episode_len)
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._action_space = gym.spaces.Discrete(2)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._action_space

    def _random_rgb(self) -> np.ndarray:
        h, w = self._obs_size
        return self._rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        obs: Obs = {
            "rgb": self._random_rgb(),
            "raw": {
                "backend": "toy",
                "event": "reset",
                "step": self._step_count,
            },
        }
        info = {"backend": "toy", "step": self._step_count}
        return obs, info

    def step(self, action: Any) -> StepReturn:
        self._step_count += 1
        truncated = self._step_count >= self._episode_len

        obs: Obs = {
            "rgb": self._random_rgb(),
            "raw": {
                "backend": "toy",
                "event": "step",
                "step": self._step_count,
                "action": action,
            },
        }
        info = {"backend": "toy", "step": self._step_count}
        return obs, 0.0, False, truncated, info

    def close(self) -> None:
        return None
