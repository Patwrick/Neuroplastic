from __future__ import annotations

import platform
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    numpy_exc: Exception | None = None
    np_version = "unknown"
    try:
        import numpy as np

        np_version = getattr(np, "__version__", "unknown")
    except Exception as exc:
        np_version = "<import failed>"
        numpy_exc = exc

    print("=== MineStudio Smoke Test ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np_version}")

    if numpy_exc is not None:
        traceback.print_exception(numpy_exc)
        print(
            "If this is a ModuleNotFoundError for a dependency, install it or reinstall requirements-minestudio.txt"
        )
        raise SystemExit(1)

    env = None
    try:
        from csgn.env_factory import make_env

        env = make_env(
            "minestudio",
            action_type="env",
            obs_size=(128, 128),
            render_size=(640, 360),
        )

        obs, info = env.reset()
        rgb = obs["rgb"]
        print(f"reset obs keys: {sorted(obs.keys())}")
        print(f"reset rgb: shape={rgb.shape} dtype={rgb.dtype}")
        print(f"reset has pov: {'pov' in obs}")
        print(f"reset info keys (sample): {list(info.keys())[:5]}")

        for i in range(20):
            action = env.action_space.sample() if hasattr(env.action_space, "sample") else None
            obs, reward, terminated, truncated, info = env.step(action)
            rgb = obs["rgb"]
            print(
                f"step={i + 1} reward={reward} rgb shape={rgb.shape} dtype={rgb.dtype} "
                f"terminated={terminated} truncated={truncated} info_keys={list(info.keys())[:5]}"
            )
            if terminated or truncated:
                break
    except Exception:
        traceback.print_exc()
        print(
            "If this is a ModuleNotFoundError for a dependency, install it or reinstall requirements-minestudio.txt"
        )
        raise SystemExit(1)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
