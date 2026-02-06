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


def _shape(value):
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    return type(value).__name__


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from envs.make_env import make_env

    env = make_env("InfantSurvive-v0")
    obs = _unwrap_reset(env.reset())
    action = env.action_space.noop()

    reward = 0.0
    done = False
    for step_idx in range(200):
        obs, reward, done, info = _step(env, action)
        milestone_reward = info.get("milestone_reward", 0) if isinstance(info, dict) else 0
        milestones_unlocked = info.get("milestones_unlocked", []) if isinstance(info, dict) else []
        print(
            f"step {step_idx}: reward={reward} done={done} "
            f"milestone_reward={milestone_reward} milestones_unlocked={milestones_unlocked}"
        )
        if done:
            obs = _unwrap_reset(env.reset())
            done = False

    print("Observation keys:", list(obs.keys()))

    print("pov shape:", _shape(obs.get("pov")))

    print("location shape:", _shape(obs.get("location")))

    print("life_stats shape:", _shape(obs.get("life_stats")))

    print("inventory shape:", _shape(obs.get("inventory")))

    print("reward:", reward)
    print("done:", done)

    env.close()


if __name__ == "__main__":
    main()
