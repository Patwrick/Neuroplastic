from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.envs.cue_bandit_env import CueBanditEnv
from csgn.models.plastic_mlp import PlasticMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural plastic policy on CueBanditEnv.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", default="neural_plastic", choices=["neural_plastic", "neural_static", "neural_fast_only"])
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--cue_dist", default="uniform", choices=["uniform", "zipf"])
    parser.add_argument("--zipf_alpha", type=float, default=1.2)
    parser.add_argument("--permute_ids", action="store_true")
    parser.add_argument("--permute_every", type=int, default=0)
    parser.add_argument("--permute_seed", type=int, default=-1)
    parser.add_argument("--change_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_path", default="runs/plastic_cue.pt")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def read_git_stamp(repo_root: Path) -> tuple[str | None, str | None]:
    commit = (os.getenv("CSGN_GIT_COMMIT") or "").strip() or None
    dirty = (os.getenv("CSGN_GIT_DIRTY") or "").strip() or None

    if commit is None:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            commit = out or None
        except Exception:
            commit = None

    if dirty is None:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            dirty = "1" if out.strip() else "0"
        except Exception:
            dirty = None

    return commit, dirty


def main() -> None:
    args = parse_args()
    if args.K <= 0:
        raise ValueError(f"--K must be > 0, got {args.K}")
    if args.action_dim <= 0:
        raise ValueError(f"--action_dim must be > 0, got {args.action_dim}")
    if args.hidden_dim <= 0:
        raise ValueError(f"--hidden_dim must be > 0, got {args.hidden_dim}")
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be > 0, got {args.episodes}")
    if args.episode_len <= 0:
        raise ValueError(f"--episode_len must be > 0, got {args.episode_len}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}")
    if args.log_every <= 0:
        raise ValueError(f"--log_every must be > 0, got {args.log_every}")
    if args.permute_every < 0:
        raise ValueError(f"--permute_every must be >= 0, got {args.permute_every}")
    if args.change_every < 0:
        raise ValueError(f"--change_every must be >= 0, got {args.change_every}")

    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print(f"device: {device}", flush=True)

    permute_seed = None if int(args.permute_seed) < 0 else int(args.permute_seed)
    resolved_env_config: dict[str, Any] = {
        "K": int(args.K),
        "action_dim": int(args.action_dim),
        "episode_len": int(args.episode_len),
        "batch_size": int(args.batch_size),
        "cue_dist": str(args.cue_dist),
        "zipf_alpha": float(args.zipf_alpha),
        "permute_ids": bool(args.permute_ids),
        "permute_every": int(args.permute_every),
        "permute_seed": permute_seed,
        "change_every": int(args.change_every),
    }
    env = CueBanditEnv(
        K=resolved_env_config["K"],
        action_dim=resolved_env_config["action_dim"],
        episode_steps=resolved_env_config["episode_len"],
        cue_dist=resolved_env_config["cue_dist"],
        zipf_alpha=resolved_env_config["zipf_alpha"],
        permute_ids=resolved_env_config["permute_ids"],
        permute_every=resolved_env_config["permute_every"],
        permute_seed=resolved_env_config["permute_seed"],
        change_every=resolved_env_config["change_every"],
        batch_size=resolved_env_config["batch_size"],
    )

    torch.manual_seed(int(args.seed))
    model = PlasticMLP(
        input_dim=resolved_env_config["K"],
        hidden_dim=int(args.hidden_dim),
        action_dim=resolved_env_config["action_dim"],
        eta=float(args.eta),
        decay=float(args.decay),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    is_static_mode = args.mode == "neural_static"
    is_fast_enabled = args.mode in ("neural_plastic", "neural_fast_only")
    train_slow_params = args.mode in ("neural_plastic", "neural_static")

    print(
        f"train_start: mode={args.mode} episodes={args.episodes} episode_len={args.episode_len} "
        f"batch_size={args.batch_size} K={args.K} A={args.action_dim}",
        flush=True,
    )
    print(f"resolved_env_config: {resolved_env_config}", flush=True)

    episode_reward_means: list[float] = []
    for episode in range(args.episodes):
        obs, _info = env.reset(seed=int(args.seed) + episode * 1000)
        model.reset_fast_state(batch_size=int(args.batch_size), device=device)
        if is_static_mode:
            model.plastic.H.zero_()
        baseline = torch.zeros((int(args.batch_size),), device=device, dtype=torch.float32)

        ep_loss = 0.0
        ep_reward = 0.0
        for _t in range(args.episode_len):
            cue = torch.as_tensor(obs["cue_onehot"], dtype=torch.float32, device=device)
            target = torch.as_tensor(obs["target_action_idx"], dtype=torch.long, device=device)

            if train_slow_params:
                logits, pre = model(cue, return_pre=True)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    logits, pre = model(cue, return_pre=True)
                    loss = F.cross_entropy(logits, target)

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1)
                reward = (action == target).float()
                if is_fast_enabled:
                    post = F.one_hot(action, num_classes=int(args.action_dim)).float()
                    mod = reward - baseline
                    model.plastic_update(pre=pre.detach(), post=post.detach(), mod=mod.detach())
                baseline = 0.95 * baseline + 0.05 * reward

            obs, _reward_np, _term_np, truncated_np, _ = env.step(action.detach().cpu().numpy())
            ep_loss += float(loss.item())
            ep_reward += float(reward.mean().item())

            if bool(truncated_np.all()):
                break

        episode_reward_mean = ep_reward / max(1, int(args.episode_len))
        episode_reward_means.append(float(episode_reward_mean))

        if ((episode + 1) % int(args.log_every) == 0) or (episode == 0) or (episode + 1 == args.episodes):
            avg_loss = ep_loss / max(1, int(args.episode_len))
            avg_reward = episode_reward_mean
            fast_norm = model.stats()["wfast_norm"]
            print(
                f"episode={episode + 1}/{args.episodes} "
                f"avg_loss={avg_loss:.4f} avg_reward={avg_reward:.4f} wfast_norm={fast_norm:.4f}",
                flush=True,
            )

    repo_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = read_git_stamp(repo_root=repo_root)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "mode": str(args.mode),
        "config": {
            "mode": str(args.mode),
            "K": int(args.K),
            "action_dim": int(args.action_dim),
            "hidden_dim": int(args.hidden_dim),
            "episodes": int(args.episodes),
            "episode_len": int(args.episode_len),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "eta": float(args.eta),
            "decay": float(args.decay),
            "cue_dist": str(args.cue_dist),
            "zipf_alpha": float(args.zipf_alpha),
            "permute_ids": bool(args.permute_ids),
            "permute_every": int(args.permute_every),
            "permute_seed": permute_seed,
            "change_every": int(args.change_every),
            "seed": int(args.seed),
            "env_config": dict(resolved_env_config),
        },
        "resolved_env_config": dict(resolved_env_config),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    torch.save(payload, save_path)
    if episode_reward_means:
        reward_mean = float(sum(episode_reward_means) / len(episode_reward_means))
        tail = episode_reward_means[-min(100, len(episode_reward_means)) :]
        reward_last = float(sum(tail) / len(tail))
    else:
        reward_mean = 0.0
        reward_last = 0.0
    print(f"train_done: mode={args.mode} reward_mean={reward_mean:.4f} reward_last={reward_last:.4f}", flush=True)
    print(f"checkpoint: {save_path}", flush=True)


if __name__ == "__main__":
    main()
