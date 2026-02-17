from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.data.trajectory_writer import TrajectoryWriter
from csgn.env_factory import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record trajectories from CSGN backends.")
    parser.add_argument("--backend", default="minestudio", choices=["toy", "minestudio"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--out_dir", default="data/trajectories")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--action_type", default="env")
    parser.add_argument("--obs_h", type=int, default=128)
    parser.add_argument("--obs_w", type=int, default=128)
    parser.add_argument("--render_w", type=int, default=640)
    parser.add_argument("--render_h", type=int, default=360)

    return parser.parse_args()


def make_run_id(backend: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{backend}_seed{seed}"


def choose_action(env: Any) -> Any:
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "sample"):
        return action_space.sample()
    return 0


def read_git_stamp() -> tuple[str | None, str | None]:
    commit = os.getenv("CSGN_GIT_COMMIT")
    dirty = os.getenv("CSGN_GIT_DIRTY")
    commit_out = commit.strip() if commit is not None and commit.strip() else None
    dirty_out = dirty.strip() if dirty is not None and dirty.strip() else None
    return commit_out, dirty_out


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError(f"--steps must be > 0, got {args.steps}")
    if args.log_every <= 0:
        raise ValueError(f"--log_every must be > 0, got {args.log_every}")

    print(
        f"start: backend={args.backend} steps={args.steps} seed={args.seed} out_dir={args.out_dir}",
        flush=True,
    )
    git_commit, git_dirty = read_git_stamp()

    env_kwargs: dict[str, Any] = {}
    if args.backend == "minestudio":
        env_kwargs = {
            "action_type": args.action_type,
            "obs_size": (args.obs_h, args.obs_w),
            "render_size": (args.render_w, args.render_h),
            "seed": args.seed,
        }

    run_dir = Path(args.out_dir) / make_run_id(args.backend, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir created: {run_dir}", flush=True)

    env = make_env(args.backend, **env_kwargs)
    writer: TrajectoryWriter | None = None
    num_steps = 0

    try:
        print("before reset", flush=True)
        try:
            obs, info = env.reset(seed=args.seed)
        except Exception as exc:
            error_file = run_dir / "error.txt"
            error_file.write_text(
                f"{exc.__class__.__name__}: {exc}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
            print(f"reset failed; wrote error marker: {error_file}", flush=True)
            raise
        rgb = np.asarray(obs["rgb"])
        if rgb.ndim != 3:
            raise ValueError(f"obs['rgb'] must be rank-3 [H, W, C], got shape {rgb.shape}")
        print(f"after reset: rgb_shape={tuple(rgb.shape)} run_dir={run_dir}", flush=True)

        writer = TrajectoryWriter(
            run_dir=str(run_dir),
            max_steps=args.steps,
            rgb_shape=(int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])),
            meta={
                "backend": args.backend,
                "seed": args.seed,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "action_type": args.action_type if args.backend == "minestudio" else None,
                "obs_size": [args.obs_h, args.obs_w] if args.backend == "minestudio" else None,
                "render_size": [args.render_w, args.render_h] if args.backend == "minestudio" else None,
                "reset_info": info,
            },
        )
        writer.write_obs(0, obs)

        for t in range(args.steps):
            action = choose_action(env)
            writer.write_action(t, action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            writer.write_transition(t, reward, terminated, truncated)
            writer.write_obs(t + 1, next_obs)
            num_steps = t + 1

            if (num_steps % args.log_every == 0) or terminated or truncated:
                print(
                    f"progress: t={t} reward={reward} terminated={terminated} truncated={truncated}",
                    flush=True,
                )

            if terminated or truncated:
                break
    finally:
        close_errors: list[Exception] = []
        if writer is not None:
            try:
                writer.close(num_steps=num_steps)
            except Exception as exc:  # pragma: no cover - cleanup fallback.
                close_errors.append(exc)

        try:
            env.close()
        except Exception as exc:  # pragma: no cover - cleanup fallback.
            close_errors.append(exc)

        if close_errors:
            raise RuntimeError(
                "Cleanup failed: " + "; ".join(f"{type(err).__name__}: {err}" for err in close_errors)
            )

    print(f"exit: num_steps={num_steps} run_dir={run_dir}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"num_steps: {num_steps}", flush=True)
    print(f"rgb_file: {run_dir / 'rgb.npy'}", flush=True)
    print(f"actions_file: {run_dir / 'actions.jsonl'}", flush=True)
    print(f"meta_file: {run_dir / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
