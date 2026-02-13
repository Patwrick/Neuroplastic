from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.data.trajectory_dataset import ActionSpec
from csgn.data.trajectory_writer import TrajectoryWriter
from csgn.env_factory import make_env
from csgn.models.tiny_bc import TinyBC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out a trained BC policy in MineStudio.")
    parser.add_argument("--checkpoint", default="runs/bc_baseline.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="data/rollouts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera_scale", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def make_run_id(seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_bc_rollout_seed{seed}"


def action_spec_from_checkpoint(payload: dict[str, Any]) -> ActionSpec:
    keys = [str(k) for k in payload["keys"]]
    sizes = {str(k): int(v) for k, v in dict(payload["sizes"]).items()}
    offsets = {
        str(k): (int(v[0]), int(v[1]))
        for k, v in dict(payload["offsets"]).items()
    }
    dim = int(payload["dim"])
    return ActionSpec(keys=keys, sizes=sizes, offsets=offsets, dim=dim)


def _remap_encoder_plain_to_net(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.") and not key.startswith("encoder.net."):
            remapped[f"encoder.net.{key[len('encoder.'):]}"] = value
        else:
            remapped[key] = value
    return remapped


def _remap_encoder_net_to_plain(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.net."):
            remapped[f"encoder.{key[len('encoder.net.'):]}"] = value
        else:
            remapped[key] = value
    return remapped


def load_state_dict_compatible(model: TinyBC, state_dict: dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        message = str(exc)
        has_missing = "Missing key(s) in state_dict" in message
        has_unexpected = "Unexpected key(s) in state_dict" in message
        missing_net = "encoder.net." in message
        missing_plain_only = ("encoder." in message) and ("encoder.net." not in message)

        if has_missing and has_unexpected and missing_net:
            remapped = _remap_encoder_plain_to_net(state_dict)
            model.load_state_dict(remapped, strict=True)
            print("state_dict_remap: encoder.* -> encoder.net.* (legacy checkpoint)", flush=True)
            return

        if has_missing and has_unexpected and missing_plain_only:
            remapped = _remap_encoder_net_to_plain(state_dict)
            model.load_state_dict(remapped, strict=True)
            print("state_dict_remap: encoder.net.* -> encoder.*", flush=True)
            return

        raise


def decode_action(vec: torch.Tensor, action_spec: ActionSpec, camera_scale: float) -> dict[str, Any]:
    vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
    if vec.numel() < action_spec.dim:
        padded = torch.zeros(action_spec.dim, dtype=torch.float32)
        padded[: vec.numel()] = vec
        vec = padded
    elif vec.numel() > action_spec.dim:
        vec = vec[: action_spec.dim]

    out: dict[str, Any] = {}
    for key in action_spec.keys:
        start, end = action_spec.offsets[key]
        sl = vec[start:end]

        if key == "camera":
            cam = torch.zeros(2, dtype=torch.float32)
            cam[: min(2, sl.numel())] = sl[:2]
            cam = torch.tanh(cam) * float(camera_scale)
            out[key] = [float(cam[0].item()), float(cam[1].item())]
            continue

        value = float(sl[0].item()) if sl.numel() > 0 else 0.0
        p = torch.sigmoid(torch.tensor(value, dtype=torch.float32))
        out[key] = int((p > 0.5).item())

    return out


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError(f"--steps must be > 0, got {args.steps}")
    if args.log_every <= 0:
        raise ValueError(f"--log_every must be > 0, got {args.log_every}")
    if args.camera_scale <= 0:
        raise ValueError(f"--camera_scale must be > 0, got {args.camera_scale}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / make_run_id(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"rollout_start: checkpoint={args.checkpoint} steps={args.steps} seed={args.seed} "
        f"device={args.device} out_dir={run_dir}",
        flush=True,
    )

    env = None
    writer: TrajectoryWriter | None = None
    num_steps = 0

    try:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        requested_device = str(args.device).strip()
        requested_lower = requested_device.lower()
        if requested_lower.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Install/verify CUDA torch and retry with --device cpu if needed."
            )
        device = torch.device(requested_device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        action_spec_payload = checkpoint.get("action_spec")
        if not isinstance(action_spec_payload, dict):
            raise ValueError(f"Checkpoint missing action_spec dict: {checkpoint_path}")
        action_spec = action_spec_from_checkpoint(action_spec_payload)

        model = TinyBC(action_dim=action_spec.dim)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint missing state_dict: {checkpoint_path}")
        load_state_dict_compatible(model, state_dict)
        model.to(device)
        model.eval()

        env = make_env(
            "minestudio",
            action_type="env",
            obs_size=(128, 128),
            render_size=(640, 360),
            seed=args.seed,
        )

        obs, info = env.reset(seed=args.seed)
        rgb = np.asarray(obs["rgb"])
        if rgb.ndim != 3:
            raise ValueError(f"obs['rgb'] must be rank-3 [H, W, C], got shape {rgb.shape}")

        writer = TrajectoryWriter(
            run_dir=str(run_dir),
            max_steps=args.steps,
            rgb_shape=(int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])),
            meta={
                "backend": "minestudio",
                "seed": int(args.seed),
                "checkpoint": str(checkpoint_path),
                "device": str(device),
                "camera_scale": float(args.camera_scale),
                "reset_info": info,
            },
        )
        writer.write_obs(0, obs)

        for t in range(args.steps):
            rgb_tensor = torch.from_numpy(np.asarray(obs["rgb"], dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
            rgb_tensor = rgb_tensor / 255.0
            rgb_tensor = rgb_tensor.to(device, non_blocking=True)

            with torch.no_grad():
                pred = model(rgb_tensor)[0].detach().cpu()

            action = decode_action(pred, action_spec, camera_scale=float(args.camera_scale))

            writer.write_action(t, action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            writer.write_transition(t, reward, terminated, truncated)
            writer.write_obs(t + 1, next_obs)
            num_steps = t + 1
            obs = next_obs

            if (num_steps % args.log_every == 0) or terminated or truncated:
                print(
                    f"progress: t={t} reward={reward} terminated={terminated} truncated={truncated}",
                    flush=True,
                )

            if terminated or truncated:
                break
    except Exception:
        error_file = run_dir / "error.txt"
        error_file.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"rollout failed; wrote error marker: {error_file}", flush=True)
        raise
    finally:
        close_errors: list[Exception] = []
        if writer is not None:
            try:
                writer.close(num_steps=num_steps)
            except Exception as exc:  # pragma: no cover - cleanup fallback.
                close_errors.append(exc)
        if env is not None:
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
    print(f"rgb_file: {run_dir / 'rgb.npy'}", flush=True)
    print(f"actions_file: {run_dir / 'actions.jsonl'}", flush=True)
    print(f"meta_file: {run_dir / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
