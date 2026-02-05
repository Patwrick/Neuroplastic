#!/usr/bin/env python3
import os
import sys

import gym
import minerl  # noqa: F401 - ensure MineRL envs are registered
import numpy as np  # noqa: F401


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


def _shape(value):
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    return type(value).__name__


def _pick_key(obs, keys):
    for key in keys:
        if key in obs:
            return key
    return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from envs import infant_survive_envspec

    infant_survive_envspec.register()

    env = gym.make("InfantSurvive-v0")
    obs = _unwrap_reset(env.reset())
    action = env.action_space.noop()

    reward = 0.0
    done = False
    for _ in range(200):
        obs, reward, done, info = _step(env, action)
        if done:
            obs = _unwrap_reset(env.reset())
            done = False

    print("Observation keys:", list(obs.keys()))

    key = _pick_key(obs, ["pov"])
    print("pov shape:", _shape(obs.get(key) if key else None))

    key = _pick_key(obs, ["location_stats", "location"])
    print("location shape:", _shape(obs.get(key) if key else None))

    key = _pick_key(obs, ["life_stats", "lifestats"])
    print("lifestats shape:", _shape(obs.get(key) if key else None))

    key = _pick_key(obs, ["inventory"])
    print("inventory shape:", _shape(obs.get(key) if key else None))

    print("reward:", reward)
    print("done:", done)

    env.close()


if __name__ == "__main__":
    main()
