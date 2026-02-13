from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.data.trajectory_dataset import TrajectoryDataset, infer_action_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect trajectory dataset statistics.")
    parser.add_argument("--data_dir", default="data/trajectories")
    parser.add_argument("--max_lines_per_run", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dirs = TrajectoryDataset._discover_run_dirs(Path(args.data_dir))
    action_spec = infer_action_spec([str(run_dir) for run_dir in run_dirs], max_lines_per_run=args.max_lines_per_run)
    dataset = TrajectoryDataset(data_dir=args.data_dir, action_spec=action_spec)

    print(f"runs: {len(dataset.runs)}")
    print(f"total_steps: {len(dataset)}")
    print(f"rgb_shape_first_run: {tuple(dataset.runs[0].rgb.shape)}")
    print(f"action_dim: {dataset.action_spec.dim}")
    print("run_step_counts:")
    for run in dataset.runs:
        meta_steps = run.meta.get("num_steps")
        steps = int(meta_steps) if isinstance(meta_steps, (int, float)) else run.num_steps
        print(f"  {run.run_dir.name}: {steps}")

    print("action_keys (first 20):")
    for key in dataset.action_spec.keys[:20]:
        print(f"  {key}: {dataset.action_spec.sizes[key]}")

    sample = dataset[0]
    rgb = sample["rgb"]
    action = sample["action"]

    print(f"sample_rgb_min_max: {float(torch.min(rgb)):.6f}, {float(torch.max(rgb)):.6f}")
    print(f"sample_action_norm: {float(torch.norm(action)):.6f}")
    print(
        f"sample_reward_done: reward={sample['reward']} terminated={sample['terminated']} "
        f"truncated={sample['truncated']}"
    )


if __name__ == "__main__":
    main()
