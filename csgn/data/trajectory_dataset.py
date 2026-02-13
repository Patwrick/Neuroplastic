from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _scalar_to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, bool):
            return 1.0 if item else 0.0
        if isinstance(item, (int, float)):
            return float(item)
    return None


def _flatten_seq(values: list | tuple) -> list[float]:
    out: list[float] = []
    for item in values:
        as_float = _scalar_to_float(item)
        if as_float is not None:
            out.append(as_float)
            continue
        if isinstance(item, (list, tuple)):
            out.extend(_flatten_seq(item))
    return out


def _flatten_into(prefix: str, obj: Any, out: dict[str, list[float]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = str(key) if not prefix else f"{prefix}.{key}"
            _flatten_into(child, value, out)
        return

    key = prefix or "value"
    as_float = _scalar_to_float(obj)
    if as_float is not None:
        out[key] = [as_float]
        return

    if isinstance(obj, (list, tuple)):
        flat = _flatten_seq(obj)
        if flat:
            out[key] = flat


def flatten_action(action_obj: Any) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    _flatten_into("", action_obj, out)
    return out


@dataclass
class ActionSpec:
    keys: list[str]
    sizes: dict[str, int]
    offsets: dict[str, tuple[int, int]]
    dim: int

    def encode(self, action_obj: Any) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        flat = flatten_action(action_obj)
        for key in self.keys:
            start, end = self.offsets[key]
            size = end - start
            values = flat.get(key, [])
            if not values:
                continue
            clipped = values[:size]
            vec[start : start + len(clipped)] = torch.tensor(clipped, dtype=torch.float32)
        return vec

    def decode(self, vec: Any) -> dict[str, list[float]]:
        arr = torch.as_tensor(vec, dtype=torch.float32).flatten()
        if arr.numel() < self.dim:
            padded = torch.zeros(self.dim, dtype=torch.float32)
            padded[: arr.numel()] = arr
            arr = padded
        elif arr.numel() > self.dim:
            arr = arr[: self.dim]

        out: dict[str, list[float]] = {}
        for key in self.keys:
            start, end = self.offsets[key]
            out[key] = arr[start:end].tolist()
        return out


def infer_action_spec(run_dirs: list[str], max_lines_per_run: int = 2000) -> ActionSpec:
    max_sizes: dict[str, int] = {}
    line_cap = max(1, int(max_lines_per_run))

    for run_dir in run_dirs:
        actions_path = Path(run_dir) / "actions.jsonl"
        if not actions_path.is_file():
            continue

        with actions_path.open("r", encoding="utf-8") as fp:
            for line_i, line in enumerate(fp):
                if line_i >= line_cap:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue

                flat = flatten_action(obj.get("action"))
                for key, values in flat.items():
                    if not values:
                        continue
                    prev = max_sizes.get(key, 0)
                    if len(values) > prev:
                        max_sizes[key] = len(values)

    if not max_sizes:
        max_sizes = {"value": 1}

    keys = sorted(max_sizes.keys())
    sizes = {key: int(max_sizes[key]) for key in keys}
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for key in keys:
        size = sizes[key]
        offsets[key] = (cursor, cursor + size)
        cursor += size

    return ActionSpec(keys=keys, sizes=sizes, offsets=offsets, dim=cursor)


class TrajectoryRun:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

        self.meta_path = self.run_dir / "meta.json"
        self.rgb_path = self.run_dir / "rgb.npy"
        self.reward_path = self.run_dir / "reward.npy"
        self.terminated_path = self.run_dir / "terminated.npy"
        self.truncated_path = self.run_dir / "truncated.npy"
        self.actions_path = self.run_dir / "actions.jsonl"
        self.obs_extra_path = self.run_dir / "obs_extra.jsonl"

        with self.meta_path.open("r", encoding="utf-8") as fp:
            self.meta = json.load(fp)

        self.rgb = np.load(self.rgb_path, mmap_mode="r")
        self.reward = np.load(self.reward_path, mmap_mode="r")
        self.terminated = np.load(self.terminated_path, mmap_mode="r")
        self.truncated = np.load(self.truncated_path, mmap_mode="r")

        self.num_steps = int(self.reward.shape[0])

        if self.terminated.shape != (self.num_steps,):
            raise ValueError(f"Invalid terminated shape in {self.run_dir}: {self.terminated.shape}")
        if self.truncated.shape != (self.num_steps,):
            raise ValueError(f"Invalid truncated shape in {self.run_dir}: {self.truncated.shape}")
        if self.rgb.shape[0] < self.num_steps + 1:
            raise ValueError(f"Invalid rgb shape in {self.run_dir}: {self.rgb.shape}")


class TrajectoryDataset(Dataset):
    REQUIRED_FILES = ("meta.json", "rgb.npy", "reward.npy", "terminated.npy", "truncated.npy", "actions.jsonl")

    def __init__(
        self,
        data_dir: str = "data/trajectories",
        split: str = "all",
        action_spec: ActionSpec | None = None,
        normalize_rgb: bool = True,
        max_runs: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize_rgb = bool(normalize_rgb)

        if split != "all":
            raise ValueError(f"Unsupported split {split!r}. Only 'all' is supported.")

        run_dirs = self._discover_run_dirs(self.data_dir)
        if max_runs is not None:
            run_dirs = run_dirs[: max(0, int(max_runs))]
        self.runs = [TrajectoryRun(run_dir) for run_dir in run_dirs]

        if not self.runs:
            raise ValueError(f"No trajectory runs found in {self.data_dir}")

        if action_spec is None:
            action_spec = infer_action_spec([str(run.run_dir) for run in self.runs])
        self.action_spec = action_spec

        self._line_offsets: list[list[int]] = []
        self._global_index: list[tuple[int, int]] = []

        for run_i, run in enumerate(self.runs):
            offsets = self._build_line_offsets(run.actions_path)
            if len(offsets) < run.num_steps:
                raise ValueError(
                    f"actions.jsonl has fewer lines than steps in {run.run_dir}: "
                    f"lines={len(offsets)} steps={run.num_steps}"
                )
            self._line_offsets.append(offsets)
            for t in range(run.num_steps):
                self._global_index.append((run_i, t))

    @staticmethod
    def _discover_run_dirs(data_dir: Path) -> list[Path]:
        if not data_dir.exists():
            return []

        out: list[Path] = []
        for child in sorted(data_dir.iterdir()):
            if not child.is_dir():
                continue
            if all((child / name).is_file() for name in TrajectoryDataset.REQUIRED_FILES):
                out.append(child)
        return out

    @staticmethod
    def _build_line_offsets(path: Path) -> list[int]:
        offsets: list[int] = []
        with path.open("rb") as fp:
            while True:
                pos = fp.tell()
                line = fp.readline()
                if not line:
                    break
                offsets.append(pos)
        return offsets

    @staticmethod
    def _read_action_at(path: Path, offset: int) -> Any:
        with path.open("rb") as fp:
            fp.seek(int(offset))
            raw = fp.readline()

        if not raw:
            return None

        text = raw.decode("utf-8").strip()
        if not text:
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict):
            return payload.get("action")
        return None

    def __len__(self) -> int:
        return len(self._global_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        run_i, t = self._global_index[int(idx)]
        run = self.runs[run_i]

        rgb_hwc = np.asarray(run.rgb[t], dtype=np.float32)
        rgb = torch.from_numpy(rgb_hwc).permute(2, 0, 1).contiguous()
        if self.normalize_rgb:
            rgb = rgb / 255.0

        offset = self._line_offsets[run_i][t]
        action_obj = self._read_action_at(run.actions_path, offset)
        action_vec = self.action_spec.encode(action_obj)

        return {
            "rgb": rgb,
            "action": action_vec,
            "reward": float(run.reward[t]),
            "terminated": bool(run.terminated[t]),
            "truncated": bool(run.truncated[t]),
            "run_dir": str(run.run_dir),
            "t": int(t),
        }
