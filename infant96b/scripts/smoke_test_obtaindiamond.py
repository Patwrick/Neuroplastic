#!/usr/bin/env python3
import gym
import minerl  # noqa: F401 - ensures MineRL envs are registered
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


def _keys(x):
    return list(x.keys()) if hasattr(x, "keys") else type(x)


def main():
    env = gym.make("MineRLObtainDiamond-v0")
    obs = _unwrap_reset(env.reset())
    action = env.action_space.noop()

    for _ in range(100):
        obs, reward, done, info = _step(env, action)
        if done:
            obs = _unwrap_reset(env.reset())

    print("Observation keys:", _keys(obs))
    print("Action keys:", _keys(action))
    env.close()


if __name__ == "__main__":
    main()
