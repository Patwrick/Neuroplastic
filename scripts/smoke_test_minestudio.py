from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.env_factory import make_env


def main() -> None:
    try:
        env = make_env(
            "minestudio",
            action_type="env",
            obs_size=(128, 128),
            render_size=(640, 360),
        )
    except ImportError as exc:
        print(exc)
        raise SystemExit(0)

    try:
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
