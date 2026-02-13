from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from csgn.data.json_safe import to_jsonable
from csgn.env_api import Obs


class TrajectoryWriter:
    _EXTRA_KEYS = ("status", "location", "inventory", "equipped", "message")

    def __init__(self, run_dir: str, max_steps: int, rgb_shape: tuple[int, int, int], meta: dict):
        self.run_dir = Path(run_dir)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps!r}")

        if len(rgb_shape) != 3:
            raise ValueError(f"rgb_shape must be (H, W, C), got {rgb_shape!r}")
        self.rgb_shape = (int(rgb_shape[0]), int(rgb_shape[1]), int(rgb_shape[2]))
        if any(dim <= 0 for dim in self.rgb_shape):
            raise ValueError(f"rgb_shape dimensions must be > 0, got {rgb_shape!r}")

        self._meta = dict(meta or {})
        self._closed = False

        self.run_dir.mkdir(parents=True, exist_ok=True)

        h, w, c = self.rgb_shape
        self._rgb_memmap = np.lib.format.open_memmap(
            str(self.run_dir / "rgb.npy"),
            mode="w+",
            dtype=np.uint8,
            shape=(self.max_steps + 1, h, w, c),
        )
        self._reward_memmap = np.lib.format.open_memmap(
            str(self.run_dir / "reward.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(self.max_steps,),
        )
        self._terminated_memmap = np.lib.format.open_memmap(
            str(self.run_dir / "terminated.npy"),
            mode="w+",
            dtype=np.bool_,
            shape=(self.max_steps,),
        )
        self._truncated_memmap = np.lib.format.open_memmap(
            str(self.run_dir / "truncated.npy"),
            mode="w+",
            dtype=np.bool_,
            shape=(self.max_steps,),
        )

        self._actions_fp = (self.run_dir / "actions.jsonl").open("w", encoding="utf-8", buffering=1)
        self._obs_extra_fp = (self.run_dir / "obs_extra.jsonl").open("w", encoding="utf-8", buffering=1)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TrajectoryWriter is already closed")

    def _check_obs_index(self, k: int) -> int:
        idx = int(k)
        if idx < 0 or idx > self.max_steps:
            raise IndexError(f"obs index out of range: {idx} (expected 0..{self.max_steps})")
        return idx

    def _check_step_index(self, t: int) -> int:
        idx = int(t)
        if idx < 0 or idx >= self.max_steps:
            raise IndexError(f"step index out of range: {idx} (expected 0..{self.max_steps - 1})")
        return idx

    def _write_jsonl(self, fp: Any, payload: dict[str, Any]) -> None:
        fp.write(json.dumps(to_jsonable(payload)) + "\n")
        fp.flush()

    def write_obs(self, k: int, obs: Obs) -> None:
        self._ensure_open()
        idx = self._check_obs_index(k)

        if "rgb" not in obs:
            raise KeyError("obs must contain 'rgb'")

        rgb = np.asarray(obs["rgb"])
        if rgb.shape != self.rgb_shape:
            raise ValueError(f"obs['rgb'] shape mismatch: expected {self.rgb_shape}, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)

        self._rgb_memmap[idx] = rgb

        extras: dict[str, Any] = {"k": idx}
        for key in self._EXTRA_KEYS:
            value = obs.get(key)
            if value is not None:
                extras[key] = value
        self._write_jsonl(self._obs_extra_fp, extras)

    def write_action(self, t: int, action: Any) -> None:
        self._ensure_open()
        idx = self._check_step_index(t)
        self._write_jsonl(self._actions_fp, {"t": idx, "action": action})

    def write_transition(self, t: int, reward: float, terminated: bool, truncated: bool) -> None:
        self._ensure_open()
        idx = self._check_step_index(t)
        self._reward_memmap[idx] = np.float32(reward)
        self._terminated_memmap[idx] = bool(terminated)
        self._truncated_memmap[idx] = bool(truncated)

    def close(self, num_steps: int) -> None:
        if self._closed:
            return

        steps = int(num_steps)
        if steps < 0 or steps > self.max_steps:
            raise ValueError(f"num_steps must be in [0, {self.max_steps}], got {steps}")

        self._rgb_memmap.flush()
        self._reward_memmap.flush()
        self._terminated_memmap.flush()
        self._truncated_memmap.flush()

        self._actions_fp.flush()
        self._actions_fp.close()
        self._obs_extra_fp.flush()
        self._obs_extra_fp.close()

        h, w, c = self.rgb_shape
        out_meta = {
            **to_jsonable(self._meta),
            "num_steps": steps,
            "num_obs": steps + 1,
            "shapes": {
                "rgb": [steps + 1, h, w, c],
                "reward": [steps],
                "terminated": [steps],
                "truncated": [steps],
            },
            "allocated_shapes": {
                "rgb": [self.max_steps + 1, h, w, c],
                "reward": [self.max_steps],
                "terminated": [self.max_steps],
                "truncated": [self.max_steps],
            },
        }
        with (self.run_dir / "meta.json").open("w", encoding="utf-8") as fp:
            json.dump(out_meta, fp, indent=2)
            fp.write("\n")

        self._closed = True
