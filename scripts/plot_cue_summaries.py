from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


MODE_ORDER = ("static", "plastic", "slot_static", "slot_plastic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cue rollout summary CSVs as PNG charts.")
    parser.add_argument("--summary_dir", default="runs", help="Directory containing cue_group_summary.csv and cue_paired_deltas.csv")
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: Any) -> int | None:
    parsed = maybe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def sort_none_last(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    return (0, value)


def mode_sort_key(mode: str) -> tuple[int, str]:
    if mode in MODE_ORDER:
        return (MODE_ORDER.index(mode), mode)
    return (len(MODE_ORDER), mode)


def config_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row.get("backend") or None,
        maybe_int(row.get("K")),
        row.get("cue_dist") or None,
        maybe_float(row.get("zipf_alpha")),
    )


def config_label(cfg: tuple[Any, ...]) -> str:
    backend, k_val, cue_dist, zipf_alpha = cfg
    return (
        f"{backend or 'NA'}\n"
        f"K={k_val if k_val is not None else 'NA'} "
        f"{cue_dist or 'NA'} "
        f"a={zipf_alpha if zipf_alpha is not None else 'NA'}"
    )


def slot_value(row: dict[str, str]) -> int | None:
    return maybe_int(row.get("slots"))


def slot_label(slot: int | None) -> str:
    if slot is None:
        return "none"
    return str(slot)


def _build_subplot_grid(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def plot_group_reward_mean(group_rows: list[dict[str, str]], out_path: Path, plt: Any) -> None:
    valid_rows = [row for row in group_rows if maybe_float(row.get("reward_mean_mean")) is not None]
    if not valid_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No group data to plot", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    slots = sorted({slot_value(row) for row in valid_rows}, key=sort_none_last)
    nrows, ncols = _build_subplot_grid(len(slots))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    flat_axes = list(axes.flatten())

    all_modes = sorted({(row.get("mode") or "unknown") for row in valid_rows}, key=mode_sort_key)
    if not all_modes:
        all_modes = ["unknown"]
    width = 0.8 / max(len(all_modes), 1)

    for i, slot in enumerate(slots):
        ax = flat_axes[i]
        rows = [row for row in valid_rows if slot_value(row) == slot]
        cfgs = sorted({config_key(row) for row in rows}, key=lambda x: tuple(sort_none_last(v) for v in x))
        x = np.arange(len(cfgs), dtype=np.float64)
        lookup: dict[tuple[Any, ...], dict[str, dict[str, str]]] = {}
        for row in rows:
            cfg = config_key(row)
            mode = row.get("mode") or "unknown"
            lookup.setdefault(cfg, {})[mode] = row

        for mode_idx, mode in enumerate(all_modes):
            vals: list[float] = []
            errs: list[float] = []
            for cfg in cfgs:
                row = lookup.get(cfg, {}).get(mode)
                mean = maybe_float(row.get("reward_mean_mean")) if row is not None else None
                std = maybe_float(row.get("reward_mean_std")) if row is not None else None
                vals.append(float("nan") if mean is None else mean)
                errs.append(0.0 if std is None else std)

            offset = -0.4 + (mode_idx + 0.5) * width
            ax.bar(x + offset, vals, width=width, yerr=errs, capsize=3, label=mode, alpha=0.9)

        ax.set_title(f"slots={slot_label(slot)}")
        ax.set_xlabel("config")
        ax.set_ylabel("reward_mean")
        ax.set_xticks(x)
        ax.set_xticklabels([config_label(cfg) for cfg in cfgs], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    for j in range(len(slots), len(flat_axes)):
        flat_axes[j].axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("Cue Group Summary: reward_mean by mode", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def paired_label(row: dict[str, str]) -> str:
    return (
        f"{row.get('mode_a')}-{row.get('mode_b')}\n"
        f"K={row.get('K') or 'NA'} slots={row.get('slots') or 'NA'} "
        f"{row.get('cue_dist') or 'NA'} a={row.get('zipf_alpha') or 'NA'}"
    )


def plot_paired_deltas(paired_rows: list[dict[str, str]], out_path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    if not paired_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No paired deltas to plot", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    x = np.arange(len(paired_rows), dtype=np.float64)
    width = 0.36

    delta_mean = [maybe_float(row.get("delta_reward_mean_mean")) for row in paired_rows]
    delta_mean_std = [maybe_float(row.get("delta_reward_mean_std")) for row in paired_rows]
    delta_last = [maybe_float(row.get("delta_reward_last_mean")) for row in paired_rows]
    delta_last_std = [maybe_float(row.get("delta_reward_last_std")) for row in paired_rows]

    mean_vals = [float("nan") if v is None else v for v in delta_mean]
    mean_errs = [0.0 if v is None else v for v in delta_mean_std]
    last_vals = [float("nan") if v is None else v for v in delta_last]
    last_errs = [0.0 if v is None else v for v in delta_last_std]

    ax.bar(x - width / 2.0, mean_vals, width=width, yerr=mean_errs, capsize=3, label="delta_reward_mean")
    ax.bar(x + width / 2.0, last_vals, width=width, yerr=last_errs, capsize=3, label="delta_reward_last")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("paired comparison")
    ax.set_ylabel("delta")
    ax.set_title("Cue Paired Deltas")
    ax.set_xticks(x)
    ax.set_xticklabels([paired_label(row) for row in paired_rows], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    group_csv = summary_dir / "cue_group_summary.csv"
    paired_csv = summary_dir / "cue_paired_deltas.csv"
    out_dir = summary_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib import failed: {type(exc).__name__}: {exc}")
        print("Install matplotlib in this environment to generate plots.")
        return 1

    try:
        group_rows = read_csv_rows(group_csv)
        paired_rows = read_csv_rows(paired_csv)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    group_out = out_dir / "group_reward_mean.png"
    paired_out = out_dir / "paired_deltas.png"

    plot_group_reward_mean(group_rows, group_out, plt)
    plot_paired_deltas(paired_rows, paired_out, plt)

    print(f"Wrote: {group_out}")
    print(f"Wrote: {paired_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
