#!/usr/bin/env python3
import os
import sys


def _unwrap_reset(reset_out):
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out


def _step(env, action):
    step_out = env.step(action)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = step_out
    return obs, reward, done, info


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from envs.make_env import make_env

    env = make_env("InfantSurvive-v0")
    obs = _unwrap_reset(env.reset())
    action = env.action_space.noop()

    for _ in range(5):
        obs, reward, done, info = _step(env, action)
        if done:
            obs = _unwrap_reset(env.reset())

    print("top-level keys:", list(obs.keys()))

    inventory = obs.get("inventory")
    if isinstance(inventory, dict):
        print("inventory keys:", list(inventory.keys()))
    else:
        print("inventory keys:", None)

    life_stats = obs.get("life_stats")
    if isinstance(life_stats, dict):
        print("life_stats keys:", list(life_stats.keys()))
    else:
        print("life_stats keys:", None)

    location = obs.get("location")
    if isinstance(location, dict):
        print("location keys:", list(location.keys()))
    else:
        print("location keys:", None)

    pov = obs.get("pov")
    if pov is not None and hasattr(pov, "shape"):
        print("pov shape:", tuple(pov.shape))
        print("pov dtype:", pov.dtype)
    else:
        print("pov shape:", None)
        print("pov dtype:", None)

    env.close()


if __name__ == "__main__":
    main()
