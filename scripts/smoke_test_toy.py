from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.env_factory import make_env


def main() -> None:
    env = make_env("toy")
    try:
        obs, info = env.reset(seed=0)
        rgb = obs["rgb"]
        print(f"reset: rgb shape={rgb.shape} dtype={rgb.dtype} info_keys={list(info.keys())[:5]}")

        for i in range(20):
            action = env.action_space.sample() if hasattr(env.action_space, "sample") else None
            obs, reward, terminated, truncated, info = env.step(action)
            rgb = obs["rgb"]
            print(
                f"step={i + 1} reward={reward} rgb shape={rgb.shape} dtype={rgb.dtype} "
                f"terminated={terminated} truncated={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
