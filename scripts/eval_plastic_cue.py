from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.data.json_safe import to_jsonable
from csgn.envs.cue_bandit_env import CueBanditEnv
from csgn.models.plastic_mlp import PlasticMLP


TRAIN_DEFAULTS: dict[str, Any] = {
    "mode": "neural_plastic",
    "K": 32,
    "action_dim": 8,
    "hidden_dim": 128,
    "episode_len": 200,
    "batch_size": 64,
    "eta": 0.1,
    "decay": 0.01,
    "cue_dist": "uniform",
    "zipf_alpha": 1.2,
    "permute_ids": False,
    "permute_every": 0,
    "permute_seed": None,
    "change_every": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained plastic cue model and write rollout-style metrics.")
    parser.add_argument("--checkpoint", default="runs/plastic_cue.pt")
    parser.add_argument("--tag", default="PLASTIC_CUE_EVAL")
    parser.add_argument("--rollouts_root", default="data/rollouts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--log_every", type=int, default=100)

    # Parity/override args (matching train names/defaults).
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=8)
    parser.add_argument("--cue_dist", default="uniform", choices=["uniform", "zipf"])
    parser.add_argument("--zipf_alpha", type=float, default=1.2)
    parser.add_argument("--permute_ids", action="store_true")
    parser.add_argument("--permute_every", type=int, default=0)
    parser.add_argument("--permute_seed", type=int, default=-1)
    parser.add_argument("--change_every", type=int, default=0)
    parser.add_argument("--episode_len", "--episode_steps", dest="episode_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def parse_seed_list(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        out.append(int(stripped))
    if not out:
        raise ValueError(f"No seeds parsed from --seeds={text!r}")
    return out


def make_run_id(mode: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_cue_{mode}_seed{seed}"


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def arg_provided(argv_tokens: list[str], *names: str) -> bool:
    for token in argv_tokens:
        for name in names:
            if token == name or token.startswith(name + "="):
                return True
    return False


def resolve_eval_config(args: argparse.Namespace, ckpt_config: dict[str, Any], argv_tokens: list[str]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for key, default in TRAIN_DEFAULTS.items():
        value = ckpt_config.get(key, default)
        cfg[key] = default if value is None else value

    override_spec: list[tuple[str, tuple[str, ...], Any]] = [
        ("K", ("--K",), int(args.K)),
        ("action_dim", ("--action_dim",), int(args.action_dim)),
        ("cue_dist", ("--cue_dist",), str(args.cue_dist)),
        ("zipf_alpha", ("--zipf_alpha",), float(args.zipf_alpha)),
        ("permute_ids", ("--permute_ids",), bool(args.permute_ids)),
        ("permute_every", ("--permute_every",), int(args.permute_every)),
        ("permute_seed", ("--permute_seed",), (None if int(args.permute_seed) < 0 else int(args.permute_seed))),
        ("change_every", ("--change_every",), int(args.change_every)),
        ("episode_len", ("--episode_len", "--episode_steps"), int(args.episode_len)),
        ("batch_size", ("--batch_size",), int(args.batch_size)),
    ]
    for key, flags, value in override_spec:
        if arg_provided(argv_tokens, *flags):
            cfg[key] = value

    cfg["K"] = int(cfg["K"])
    cfg["action_dim"] = int(cfg["action_dim"])
    cfg["hidden_dim"] = int(cfg.get("hidden_dim", TRAIN_DEFAULTS["hidden_dim"]))
    cfg["episode_len"] = int(cfg["episode_len"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["zipf_alpha"] = float(cfg["zipf_alpha"])
    cfg["permute_ids"] = bool(cfg["permute_ids"])
    cfg["permute_every"] = int(cfg["permute_every"])
    raw_perm_seed = cfg.get("permute_seed")
    cfg["permute_seed"] = None if raw_perm_seed is None else int(raw_perm_seed)
    cfg["change_every"] = int(cfg["change_every"])
    cfg["eta"] = float(cfg.get("eta", TRAIN_DEFAULTS["eta"]))
    cfg["decay"] = float(cfg.get("decay", TRAIN_DEFAULTS["decay"]))
    cfg["mode"] = str(cfg.get("mode", TRAIN_DEFAULTS["mode"]))

    if cfg["K"] <= 0:
        raise ValueError(f"K must be > 0, got {cfg['K']}")
    if cfg["action_dim"] <= 0:
        raise ValueError(f"action_dim must be > 0, got {cfg['action_dim']}")
    if cfg["hidden_dim"] <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {cfg['hidden_dim']}")
    if cfg["episode_len"] <= 0:
        raise ValueError(f"episode_len must be > 0, got {cfg['episode_len']}")
    if cfg["batch_size"] <= 0:
        raise ValueError(f"batch_size must be > 0, got {cfg['batch_size']}")
    if cfg["permute_every"] < 0:
        raise ValueError(f"permute_every must be >= 0, got {cfg['permute_every']}")
    if cfg["change_every"] < 0:
        raise ValueError(f"change_every must be >= 0, got {cfg['change_every']}")
    if cfg["mode"] not in ("neural_plastic", "neural_static", "neural_fast_only"):
        cfg["mode"] = "neural_plastic"

    return cfg


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError(f"--steps must be > 0, got {args.steps}")
    if args.log_every <= 0:
        raise ValueError(f"--log_every must be > 0, got {args.log_every}")

    device = resolve_device(args.device)
    seed_list = parse_seed_list(args.seeds)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict")
    ckpt_config = payload.get("config")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint missing state_dict: {ckpt_path}")
    if not isinstance(ckpt_config, dict):
        ckpt_config = {}
    ckpt_mode = payload.get("mode")
    if isinstance(ckpt_mode, str) and ckpt_mode:
        ckpt_config["mode"] = ckpt_mode

    resolved = resolve_eval_config(args, ckpt_config=ckpt_config, argv_tokens=sys.argv[1:])
    eval_mode = str(resolved["mode"])
    is_static_mode = eval_mode == "neural_static"
    is_fast_enabled = eval_mode in ("neural_plastic", "neural_fast_only")
    print(f"eval_mode: {eval_mode}", flush=True)
    print(f"resolved_config: {json.dumps(to_jsonable(resolved), sort_keys=True)}", flush=True)

    if int(resolved["batch_size"]) != 1:
        print(
            f"note: eval currently runs one seed per environment instance; --batch_size={resolved['batch_size']} is accepted for parity.",
            flush=True,
        )

    git_commit = payload.get("git_commit")
    git_dirty = payload.get("git_dirty")

    model = PlasticMLP(
        input_dim=int(resolved["K"]),
        hidden_dim=int(resolved["hidden_dim"]),
        action_dim=int(resolved["action_dim"]),
        eta=float(resolved["eta"]),
        decay=float(resolved["decay"]),
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint with resolved overrides. "
            f"Check dimension overrides (K/action_dim/hidden_dim): {exc}"
        ) from exc
    model.to(device)
    model.eval()

    num_slow_params = int(sum(p.numel() for p in model.parameters()))
    slow_param_bytes_fp16 = int(num_slow_params * 2)
    slow_param_bytes_fp32 = int(num_slow_params * 4)
    fast_state_shape = [1, int(model.plastic.out_features), int(model.plastic.in_features)]
    fast_state_numel = int(fast_state_shape[0] * fast_state_shape[1] * fast_state_shape[2])
    fast_state_bytes_fp16 = int(fast_state_numel * 2)
    fast_state_bytes_fp32 = int(fast_state_numel * 4)
    print(
        f"model_size: num_slow_params={num_slow_params} "
        f"slow_param_bytes_fp16={slow_param_bytes_fp16} slow_param_bytes_fp32={slow_param_bytes_fp32} "
        f"fast_state_shape={fast_state_shape} fast_state_bytes_fp32={fast_state_bytes_fp32}",
        flush=True,
    )

    run_root = Path(args.rollouts_root) / str(args.tag)
    run_root.mkdir(parents=True, exist_ok=True)

    print(
        f"eval_start: checkpoint={ckpt_path} device={device} seeds={seed_list} "
        f"steps={args.steps} out_root={run_root}",
        flush=True,
    )

    for seed in seed_list:
        env = CueBanditEnv(
            K=int(resolved["K"]),
            action_dim=int(resolved["action_dim"]),
            episode_steps=int(resolved["episode_len"]),
            cue_dist=str(resolved["cue_dist"]),
            zipf_alpha=float(resolved["zipf_alpha"]),
            permute_ids=bool(resolved["permute_ids"]),
            permute_every=int(resolved["permute_every"]),
            permute_seed=resolved["permute_seed"],
            change_every=int(resolved["change_every"]),
            batch_size=1,
        )

        run_dir = run_root / make_run_id(mode=eval_mode, seed=int(seed))
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.csv"

        run_config: dict[str, Any] = {
            "backend": "cue_bandit",
            "mode": eval_mode,
            "seed": int(seed),
            "steps_requested": int(args.steps),
            "K": int(resolved["K"]),
            "action_dim": int(resolved["action_dim"]),
            "hidden_dim": int(resolved["hidden_dim"]),
            "slots": 0,
            "cue_dist": str(resolved["cue_dist"]),
            "zipf_alpha": float(resolved["zipf_alpha"]),
            "permute_ids": bool(resolved["permute_ids"]),
            "permute_every": int(resolved["permute_every"]),
            "permute_seed": resolved["permute_seed"],
            "change_every": int(resolved["change_every"]),
            "episode_len": int(resolved["episode_len"]),
            "episode_steps": int(resolved["episode_len"]),
            "batch_size_requested": int(resolved["batch_size"]),
            "batch_size_used": 1,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "checkpoint": str(ckpt_path),
        }

        obs, reset_info = env.reset(seed=int(seed))
        model.reset_fast_state(batch_size=1, device=device)
        if is_static_mode:
            model.plastic.H.zero_()
        baseline = 0.0
        num_steps = 0
        episode_idx = 0
        episode_t = 0
        rewards: list[float] = []

        with metrics_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "backend",
                    "mode",
                    "seed",
                    "steps_requested",
                    "K",
                    "action_dim",
                    "hidden_dim",
                    "slots",
                    "cue_dist",
                    "zipf_alpha",
                    "permute_ids",
                    "permute_every",
                    "permute_seed",
                    "change_every",
                    "episode_len",
                    "batch_size_requested",
                    "batch_size_used",
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

            while num_steps < int(args.steps):
                t_global = int(num_steps)
                row_episode_idx = int(episode_idx)
                row_episode_t = int(episode_t)
                cue_id = int(obs["cue_id"][0])
                cue_target = int(obs["target_action_idx"][0])

                cue = torch.as_tensor(obs["cue_onehot"], dtype=torch.float32, device=device)
                logits, pre = model(cue, return_pre=True)
                probs = torch.softmax(logits, dim=1)
                action_tensor = torch.argmax(probs, dim=1)
                action_id = int(action_tensor.item())

                next_obs, reward_arr, term_arr, trunc_arr, _ = env.step(action_tensor.detach().cpu().numpy())
                reward = float(reward_arr[0])
                terminated = bool(term_arr[0])
                truncated = bool(trunc_arr[0])

                post = F.one_hot(action_tensor, num_classes=int(resolved["action_dim"])).to(dtype=torch.float32)
                if is_fast_enabled:
                    mod = torch.as_tensor([reward - baseline], dtype=torch.float32, device=device)
                    model.plastic_update(pre=pre.detach(), post=post.detach(), mod=mod.detach())
                baseline = 0.95 * baseline + 0.05 * reward
                rewards.append(reward)

                stats = model.stats()
                p0 = float(probs[0, 0].item()) if int(resolved["action_dim"]) > 0 else 0.0
                p1 = float(probs[0, 1].item()) if int(resolved["action_dim"]) > 1 else 0.0
                writer.writerow(
                    [
                        run_config["backend"],
                        run_config["mode"],
                        run_config["seed"],
                        run_config["steps_requested"],
                        run_config["K"],
                        run_config["action_dim"],
                        run_config["hidden_dim"],
                        run_config["slots"],
                        run_config["cue_dist"],
                        run_config["zipf_alpha"],
                        int(bool(run_config["permute_ids"])),
                        run_config["permute_every"],
                        run_config["permute_seed"],
                        run_config["change_every"],
                        run_config["episode_len"],
                        run_config["batch_size_requested"],
                        run_config["batch_size_used"],
                        run_config["git_commit"],
                        run_config["git_dirty"],
                        row_episode_idx,
                        t_global,
                        row_episode_t,
                        cue_id,
                        cue_target,
                        action_id,
                        reward,
                        baseline,
                        p0,
                        p1,
                        -1,
                        0,
                        0,
                        float(stats["wfast_norm"]),
                        float(stats["wslow_norm"]),
                    ]
                )
                fp.flush()

                num_steps += 1
                if (num_steps % int(args.log_every) == 0) or terminated or truncated:
                    print(
                        f"seed={seed} t_global={t_global} reward={reward:.3f} "
                        f"baseline={baseline:.3f} wfast={stats['wfast_norm']:.4f}",
                        flush=True,
                    )

                if (terminated or truncated) and num_steps < int(args.steps):
                    episode_idx += 1
                    episode_t = 0
                    next_obs, _ = env.reset(seed=int(seed) + episode_idx)
                else:
                    episode_t += 1

                obs = next_obs

        meta_payload = {
            **run_config,
            "run_config": dict(run_config),
            "resolved_config": dict(resolved),
            "reset_info": reset_info,
            "num_steps": int(num_steps),
            "num_episodes": int(episode_idx + 1),
            "num_slow_params": num_slow_params,
            "slow_param_bytes_fp16": slow_param_bytes_fp16,
            "slow_param_bytes_fp32": slow_param_bytes_fp32,
            "fast_state_shape": fast_state_shape,
            "fast_state_numel": fast_state_numel,
            "fast_state_bytes_fp16": fast_state_bytes_fp16,
            "fast_state_bytes_fp32": fast_state_bytes_fp32,
        }
        with (run_dir / "meta.json").open("w", encoding="utf-8") as fp:
            json.dump(to_jsonable(meta_payload), fp, indent=2)
            fp.write("\n")

        env.close()
        if rewards:
            reward_mean = float(sum(rewards) / len(rewards))
            tail = rewards[-min(100, len(rewards)) :]
            reward_last = float(sum(tail) / len(tail))
        else:
            reward_mean = 0.0
            reward_last = 0.0
        print(
            f"run_dir: {run_dir} num_steps={num_steps} reward_mean={reward_mean:.4f} reward_last={reward_last:.4f}",
            flush=True,
        )

    print("eval_done", flush=True)


if __name__ == "__main__":
    main()
