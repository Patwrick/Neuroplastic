from __future__ import annotations

from typing import Any

import numpy as np

from csgn.env_api import EmbodiedEnv, Obs, StepReturn


class MineStudioEnv(EmbodiedEnv):
    def __init__(
        self,
        action_type: str = "env",
        obs_size: tuple[int, int] = (224, 224),
        render_size: tuple[int, int] = (640, 360),
        seed: int = 0,
        callbacks: list | None = None,
        debug: bool = False,
    ):
        try:
            from minestudio.simulator import MinecraftSim
        except ImportError as exc:
            raise ImportError(
                "MineStudio not installed. Install with: pip install -r requirements-minestudio.txt"
            ) from exc

        self._obs_size = (int(obs_size[0]), int(obs_size[1]))
        self._debug = bool(debug)
        self._sim = MinecraftSim(
            action_type=action_type,
            obs_size=obs_size,
            render_size=render_size,
            seed=seed,
            callbacks=callbacks or [],
        )

    @property
    def action_space(self) -> Any:
        return self._sim.action_space

    def _to_rgb_uint8(self, frame: Any) -> np.ndarray:
        arr = np.asarray(frame)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            h, w = self._obs_size
            arr = np.zeros((h, w, 3), dtype=np.uint8)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    def _normalize_obs(self, obs: Any, info: dict) -> Obs:
        obs_dict = obs if isinstance(obs, dict) else {}
        info_dict = info if isinstance(info, dict) else {}

        rgb_source = obs_dict.get("image")
        if rgb_source is None:
            rgb_source = info_dict.get("pov")

        if rgb_source is None:
            h, w = self._obs_size
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            rgb = self._to_rgb_uint8(rgb_source)

        out: Obs = {"rgb": rgb}

        pov = info_dict.get("pov")
        if pov is not None:
            out["pov"] = self._to_rgb_uint8(pov)

        inventory = info_dict.get("inventory")
        if inventory is not None:
            out["inventory"] = inventory

        equipped = info_dict.get("equipped_items")
        if equipped is not None:
            out["equipped"] = equipped

        location = info_dict.get("location_stats") or info_dict.get("player_pos")
        if location is not None:
            out["location"] = location

        status: dict[str, Any] = {}
        for key in ("health", "food_level", "is_gui_open", "isGuiOpen"):
            if key in info_dict:
                status[key] = info_dict[key]
        if status:
            out["status"] = status

        if "message" in info_dict:
            out["message"] = info_dict.get("message")

        if self._debug:
            out["raw"] = {"obs": obs, "info": info}

        return out

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict]:
        if seed is not None and hasattr(self._sim, "seed"):
            self._sim.seed(seed)

        result = self._sim.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        if not isinstance(info, dict):
            info = {"raw_info": info}

        return self._normalize_obs(obs, info), info

    def step(self, action: Any) -> StepReturn:
        result = self._sim.step(action)

        if not isinstance(result, tuple):
            raise ValueError(f"Unexpected step return type: {type(result)!r}")

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            # Fallback for non-gymnasium style backends.
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raise ValueError(f"Unexpected step return length: {len(result)}")

        if not isinstance(info, dict):
            info = {"raw_info": info}

        normalized_obs = self._normalize_obs(obs, info)
        return normalized_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._sim.close()
