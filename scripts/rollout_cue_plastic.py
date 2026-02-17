from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.agents.plastic_cue_agent import PlasticCueAgent
from csgn.agents.hash_slot_cue_agent import HashSlotCueAgent
from csgn.agents.slot_plastic_cue_agent import SlotPlasticCueAgent
from csgn.data.trajectory_dataset import ActionSpec, TrajectoryDataset, infer_action_spec
from csgn.data.trajectory_writer import TrajectoryWriter
from csgn.env_factory import make_env
from csgn.env_wrappers.cue_reward_wrapper import CueRewardWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out cue-learning plastic agent in MineStudio.")
    parser.add_argument("--backend", default="minestudio", choices=["minestudio", "toy"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="data/rollouts")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--episode_steps", type=int, default=200)
    parser.add_argument("--episode_len", type=int, default=None)
    parser.add_argument("--change_every", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--sleep_every", type=int, default=0)
    parser.add_argument("--sleep_alpha", type=float, default=0.1)
    parser.add_argument(
        "--mode",
        default="plastic",
        choices=["plastic", "static", "slot_static", "slot_plastic", "slot_hash_static", "slot_hash_plastic"],
    )
    parser.add_argument("--cue_dist", default="uniform", choices=["uniform", "zipf"])
    parser.add_argument("--zipf_alpha", type=float, default=1.2)
    parser.add_argument("--permute_ids", action="store_true")
    parser.add_argument("--permute_every", type=int, default=0)
    parser.add_argument("--permute_seed", type=int, default=-1)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def make_run_id(mode: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_cue_{mode}_seed{seed}"


def infer_spec_from_trajectories(data_dir: str = "data/trajectories") -> ActionSpec:
    run_dirs = TrajectoryDataset._discover_run_dirs(Path(data_dir))
    if not run_dirs:
        raise ValueError(f"No trajectories found in {data_dir}. Record MineStudio data first.")
    return infer_action_spec([str(run_dir) for run_dir in run_dirs])


def base_action_template(action_spec: ActionSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in action_spec.keys:
        size = int(action_spec.sizes[key])
        if key == "camera":
            out[key] = [0.0, 0.0]
        elif size <= 1:
            out[key] = 0
        else:
            out[key] = [0.0 for _ in range(size)]

    if "camera" not in out:
        out["camera"] = [0.0, 0.0]
    if "forward" not in out:
        out["forward"] = 0
    if "back" not in out:
        out["back"] = 0
    return out


def action_dict_for_id(action_id: int, base_template: dict[str, Any]) -> dict[str, Any]:
    action: dict[str, Any] = {}
    for key, value in base_template.items():
        if isinstance(value, list):
            action[key] = list(value)
        else:
            action[key] = value

    action["camera"] = [0.0, 0.0]
    if int(action_id) == 0:
        action["forward"] = 1
        action["back"] = 0
    else:
        action["forward"] = 0
        action["back"] = 1
    return action


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
    if args.K <= 0:
        raise ValueError(f"--K must be > 0, got {args.K}")
    if args.log_every <= 0:
        raise ValueError(f"--log_every must be > 0, got {args.log_every}")
    if args.slots < 0:
        raise ValueError(f"--slots must be >= 0, got {args.slots}")
    if args.permute_every < 0:
        raise ValueError(f"--permute_every must be >= 0, got {args.permute_every}")

    episode_len = int(args.episode_steps if args.episode_len is None else args.episode_len)
    if episode_len <= 0:
        episode_len = int(args.steps)
    permute_seed = None if int(args.permute_seed) < 0 else int(args.permute_seed)

    git_commit, git_dirty = read_git_stamp()

    action_spec = infer_spec_from_trajectories("data/trajectories")
    action_template = base_action_template(action_spec)

    env_kwargs: dict[str, Any] = {}
    if args.backend == "minestudio":
        env_kwargs = {
            "action_type": "env",
            "obs_size": (128, 128),
            "render_size": (640, 360),
            "seed": int(args.seed),
        }

    base_env = make_env(args.backend, **env_kwargs)
    env = CueRewardWrapper(
        base_env,
        K=int(args.K),
        episode_steps=int(episode_len),
        change_every=int(args.change_every),
        cue_dist=str(args.cue_dist),
        zipf_alpha=float(args.zipf_alpha),
        permute_ids=bool(args.permute_ids),
        permute_every=int(args.permute_every),
        permute_seed=permute_seed,
    )

    is_slot_mode = args.mode in ("slot_static", "slot_plastic")
    is_hash_mode = args.mode in ("slot_hash_static", "slot_hash_plastic")
    is_plastic = args.mode in ("plastic", "slot_plastic", "slot_hash_plastic")
    slots = int(args.slots) if int(args.slots) > 0 else int(args.K)
    run_config: dict[str, Any] = {
        "backend": str(args.backend),
        "mode": str(args.mode),
        "seed": int(args.seed),
        "steps_requested": int(args.steps),
        "K": int(args.K),
        "slots": int(slots),
        "cue_dist": str(args.cue_dist),
        "zipf_alpha": float(args.zipf_alpha),
        "permute_ids": bool(args.permute_ids),
        "permute_every": int(args.permute_every),
        "permute_seed": permute_seed,
        "change_every": int(args.change_every),
        "episode_len": int(episode_len),
        "episode_steps": int(episode_len),
        "eta": float(args.eta),
        "decay": float(args.decay),
        "epsilon": float(args.epsilon),
        "sleep_every": int(args.sleep_every),
        "sleep_alpha": float(args.sleep_alpha),
        "log_every": int(args.log_every),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }

    if is_hash_mode:
        agent: Any = HashSlotCueAgent(
            slots=slots,
            action_dim=2,
            epsilon=float(args.epsilon),
            eta=float(args.eta),
            decay=float(args.decay),
            baseline_ema=0.95,
            seed=int(args.seed),
        )
    elif is_slot_mode:
        agent: Any = SlotPlasticCueAgent(
            K=int(args.K),
            slots=slots,
            eta=float(args.eta),
            decay=float(args.decay),
            epsilon=float(args.epsilon),
            seed=int(args.seed),
        )
    else:
        agent = PlasticCueAgent(
            K=int(args.K),
            eta=float(args.eta),
            decay=float(args.decay),
            epsilon=float(args.epsilon),
            sleep_every=int(args.sleep_every),
            sleep_alpha=float(args.sleep_alpha),
            seed=int(args.seed),
        )

    run_dir = Path(args.out_dir) / make_run_id(args.mode, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"

    print(
        f"cue_rollout_start: backend={args.backend} mode={args.mode} steps={args.steps} "
        f"seed={args.seed} run_dir={run_dir}",
        flush=True,
    )

    writer: TrajectoryWriter | None = None
    num_steps = 0
    recent_rewards: list[float] = []
    rewired_count = 0
    episode_idx = 0
    episode_t = 0

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_fp:
        csv_writer = csv.writer(metrics_fp)
        csv_writer.writerow(
            [
                "backend",
                "mode",
                "seed",
                "steps_requested",
                "K",
                "slots",
                "cue_dist",
                "zipf_alpha",
                "permute_ids",
                "permute_every",
                "permute_seed",
                "change_every",
                "episode_len",
                "git_commit",
                "git_dirty",
                "episode_idx",
                "t_global",
                "t",
                "cue_id",
                "cue_target",
                "action_id",
                "reward",
                "baseline",
                "p0",
                "p1",
                "slot_idx",
                "rewired",
                "slots_used",
                "wfast_norm",
                "wslow_norm",
            ]
        )

        try:
            obs, info = env.reset(seed=args.seed)
            rgb = np.asarray(obs["rgb"])
            if rgb.ndim != 3:
                raise ValueError(f"obs['rgb'] must be rank-3 [H, W, C], got shape {rgb.shape}")

            writer = TrajectoryWriter(
                run_dir=str(run_dir),
                max_steps=args.steps,
                rgb_shape=(int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])),
                meta={
                    **run_config,
                    "run_config": dict(run_config),
                    "reset_info": info,
                },
            )
            writer.write_obs(0, obs)

            while num_steps < args.steps:
                t_global = int(num_steps)
                row_episode_idx = int(episode_idx)
                row_episode_t = int(episode_t)
                cue_id = int(obs.get("cue_id", 0))
                cue_target = int(obs.get("cue_target", 0))
                cue_onehot = np.asarray(obs.get("cue_onehot"), dtype=np.float32)
                if cue_onehot.shape != (args.K,):
                    cue_onehot = np.zeros((args.K,), dtype=np.float32)
                    cue_onehot[cue_id % args.K] = 1.0

                if is_hash_mode:
                    action_id, probs, slot_idx = agent.act(cue_onehot, cue_id=cue_id)
                    rewired = False
                elif is_slot_mode:
                    action_id, probs, slot_idx, rewired = agent.act(cue_onehot, cue_id=cue_id)
                    if rewired:
                        rewired_count += 1
                else:
                    action_id, probs = agent.act(cue_onehot)
                    slot_idx = -1
                    rewired = False
                action = action_dict_for_id(action_id, action_template)

                writer.write_action(t_global, action)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                writer.write_transition(t_global, reward, terminated, truncated)

                next_obs_for_writer = next_obs
                needs_reset = bool(terminated or truncated)
                if needs_reset and (t_global + 1) < args.steps:
                    episode_idx += 1
                    episode_t = 0
                    reset_seed = int(args.seed) + episode_idx
                    next_obs_for_writer, reset_info = env.reset(seed=reset_seed)
                writer.write_obs(t_global + 1, next_obs_for_writer)
                num_steps = t_global + 1

                if is_plastic:
                    agent.update(float(reward))
                    if is_slot_mode or is_hash_mode:
                        if args.sleep_every > 0 and num_steps % args.sleep_every == 0:
                            agent.consolidate(args.sleep_alpha)
                    else:
                        agent.maybe_sleep(num_steps)
                else:
                    agent.baseline = 0.95 * float(agent.baseline) + 0.05 * float(reward)

                if is_slot_mode or is_hash_mode:
                    slot_stats = agent.stats()
                    slots_used = int(slot_stats["slots_used"])
                    wfast_norm = float(slot_stats["wfast_norm"])
                    wslow_norm = float(slot_stats["wslow_norm"])
                else:
                    slots_used = -1
                    wfast_norm = float(torch.norm(agent.W_fast).item())
                    wslow_norm = float(torch.norm(agent.W_slow).item())

                csv_writer.writerow(
                    [
                        run_config["backend"],
                        run_config["mode"],
                        run_config["seed"],
                        run_config["steps_requested"],
                        run_config["K"],
                        run_config["slots"],
                        run_config["cue_dist"],
                        run_config["zipf_alpha"],
                        int(bool(run_config["permute_ids"])),
                        run_config["permute_every"],
                        run_config["permute_seed"],
                        run_config["change_every"],
                        run_config["episode_len"],
                        run_config["git_commit"],
                        run_config["git_dirty"],
                        row_episode_idx,
                        int(t_global),
                        row_episode_t,
                        cue_id,
                        cue_target,
                        int(action_id),
                        float(reward),
                        float(agent.baseline),
                        float(probs[0]),
                        float(probs[1]),
                        int(slot_idx),
                        int(bool(rewired)),
                        int(slots_used),
                        wfast_norm,
                        wslow_norm,
                    ]
                )
                metrics_fp.flush()

                recent_rewards.append(float(reward))
                if len(recent_rewards) > 25:
                    recent_rewards = recent_rewards[-25:]

                if (num_steps % args.log_every == 0) or terminated or truncated:
                    avg_r = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                    print(
                        f"t_global={t_global} avg_reward_last_25={avg_r:.3f} baseline={agent.baseline:.3f} "
                        f"wfast_norm={wfast_norm:.4f} slots_used={slots_used} rewired_count_so_far={rewired_count}",
                        flush=True,
                    )

                obs = next_obs_for_writer
                if needs_reset:
                    continue
                episode_t += 1
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

    print(f"run_dir: {run_dir}", flush=True)
    print(f"num_steps: {num_steps}", flush=True)
    print(f"metrics_file: {metrics_path}", flush=True)
    print(f"rgb_file: {run_dir / 'rgb.npy'}", flush=True)
    print(f"actions_file: {run_dir / 'actions.jsonl'}", flush=True)
    print(f"meta_file: {run_dir / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
