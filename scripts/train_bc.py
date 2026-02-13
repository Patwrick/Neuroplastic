from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.data.trajectory_dataset import TrajectoryDataset
from csgn.models.tiny_bc import TinyBC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal BC baseline on trajectory data.")
    parser.add_argument("--data_dir", default="data/trajectories")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_path", default="runs/bc_baseline.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}")
    if args.log_every <= 0:
        raise ValueError(f"--log_every must be > 0, got {args.log_every}")

    requested_device = str(args.device).strip()
    requested_lower = requested_device.lower()

    if requested_lower == "auto":
        selected_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device_auto_selected: {selected_device_str}", flush=True)
    elif requested_lower.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested --device cuda but torch.cuda.is_available() is False. "
            "Install CUDA PyTorch with .\\scripts\\install_torch_cu128.ps1 "
            "or .\\scripts\\install_torch_nightly_cu128.ps1, then run "
            "python scripts/verify_torch_cuda.py.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    else:
        selected_device_str = requested_device

    try:
        device = torch.device(selected_device_str)
    except Exception as exc:
        print(f"Invalid --device value {args.device!r}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    print(f"training_device: requested={args.device} selected={device}", flush=True)

    dataset = TrajectoryDataset(data_dir=args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyBC(action_dim=dataset.action_spec.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(
        f"training_start: steps={len(dataset)} runs={len(dataset.runs)} action_dim={dataset.action_spec.dim} "
        f"device={device}",
        flush=True,
    )

    final_loss = float("nan")
    for epoch in range(args.epochs):
        model.train()
        running_sum = 0.0
        running_count = 0

        for step_i, batch in enumerate(loader, start=1):
            rgb = batch["rgb"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)

            pred = model(rgb)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            final_loss = loss_val
            running_sum += loss_val
            running_count += 1

            if step_i % args.log_every == 0:
                avg_loss = running_sum / max(1, running_count)
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step_i}/{len(loader)} avg_loss={avg_loss:.6f}",
                    flush=True,
                )
                running_sum = 0.0
                running_count = 0

        if running_count > 0:
            avg_loss = running_sum / running_count
            print(f"epoch={epoch + 1}/{args.epochs} avg_loss={avg_loss:.6f}", flush=True)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    action_spec_payload = {
        "keys": list(dataset.action_spec.keys),
        "sizes": dict(dataset.action_spec.sizes),
        "offsets": {k: [v[0], v[1]] for k, v in dataset.action_spec.offsets.items()},
        "dim": int(dataset.action_spec.dim),
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "action_spec": action_spec_payload,
            "args": vars(args),
        },
        save_path,
    )

    print(f"saved_checkpoint: {save_path}", flush=True)
    print(f"final_loss: {final_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
