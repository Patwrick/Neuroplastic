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
        np_version = getattr(np, "__version__", "unknown")
        try:
            np_major = int(str(np_version).split(".", 1)[0])
        except (TypeError, ValueError):
            np_major = 0

        if np_major >= 2:
            raise RuntimeError(
                f"Detected NumPy {np_version}. MineStudio pulls in Gym, which may not support NumPy 2.x. "
                'Run: pip install "numpy<2" --upgrade'
            )

        self._obs_size = (int(obs_size[0]), int(obs_size[1]))
        self._render_size = (int(render_size[0]), int(render_size[1]))
        self._action_type = action_type
        self._callbacks = list(callbacks or [])
        self._debug = bool(debug)
        self._sim: Any | None = None
        self._make_sim(seed)
        self._seed = int(seed)

    def _make_sim(self, seed: int) -> None:
        try:
            from minestudio.simulator import MinecraftSim
        except ModuleNotFoundError as e:
            if e.name == "minestudio":
                raise ImportError(
                    "MineStudio not installed. Install with: pip install -r requirements-minestudio.txt"
                ) from e
            raise RuntimeError(
                f"MineStudio installed but missing dependency: {e.name}. Fix with: pip install {e.name} "
                "(or reinstall requirements-minestudio.txt)"
            ) from e
        except ImportError as e:
            raise RuntimeError(f"Failed to import MineStudio simulator: {e}") from e
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import MineStudio simulator ({exc.__class__.__name__}: {exc})"
            ) from exc

        self._sim = MinecraftSim(
            action_type=self._action_type,
            obs_size=self._obs_size,
            render_size=self._render_size,
            seed=int(seed),
            callbacks=list(self._callbacks),
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
        if seed is not None:
            seed_int = int(seed)
            if seed_int != self._seed:
                try:
                    if self._sim is not None:
                        self._sim.close()
                except Exception:
                    pass
                self._make_sim(seed_int)
                self._seed = seed_int

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
