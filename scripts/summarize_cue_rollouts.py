from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_FIELDS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cue benchmark rollout runs.")
    parser.add_argument("--rollouts_dir", default="data/rollouts")
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--include_regex", default="cue_")
    parser.add_argument("--min_steps", type=int, default=1)
    parser.add_argument("--dedupe", choices=["none", "latest", "longest"], default="longest")
    parser.add_argument("--last_windows", default="25,100,500")
    return parser.parse_args()


def parse_windows(value: str) -> list[int]:
    windows: list[int] = []
    for chunk in value.split(","):
        token = chunk.strip()
        if not token:
            continue
        window = int(token)
        if window <= 0:
            raise ValueError(f"All --last_windows must be positive integers, got {window}")
        windows.append(window)
    if not windows:
        raise ValueError("--last_windows produced an empty window list")
    return windows


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, bool):
        return float(int(value))
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in ("true", "t", "yes"):
        return 1.0
    if lower in ("false", "f", "no"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def maybe_int(value: Any) -> int | None:
    parsed = maybe_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return int(parsed)


def maybe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def parse_mode_from_name(name: str) -> str | None:
    match = re.search(r"cue_(slot_hash_plastic|slot_hash_static|slot_plastic|slot_static|plastic|static)", name)
    if not match:
        return None
    return match.group(1)


def parse_seed_from_name(name: str) -> int | None:
    match = re.search(r"seed(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def load_config(run_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = {field: None for field in CONFIG_FIELDS}

    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.csv"
    raw_meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw_meta.update(loaded)
                nested_meta = loaded.get("meta")
                if isinstance(nested_meta, dict):
                    raw_meta.update(nested_meta)
                run_config = loaded.get("run_config")
                if isinstance(run_config, dict):
                    raw_meta.update(run_config)
        except Exception:
            pass

    config["backend"] = maybe_text(raw_meta.get("backend"))
    config["mode"] = maybe_text(raw_meta.get("mode"))
    config["seed"] = maybe_int(raw_meta.get("seed"))
    config["steps_requested"] = maybe_int(raw_meta.get("steps_requested"))
    config["K"] = maybe_int(raw_meta.get("K"))
    config["slots"] = maybe_int(raw_meta.get("slots"))
    config["cue_dist"] = maybe_text(raw_meta.get("cue_dist"))
    config["zipf_alpha"] = maybe_float(raw_meta.get("zipf_alpha"))
    config["permute_ids"] = maybe_int(raw_meta.get("permute_ids"))
    config["permute_every"] = maybe_int(raw_meta.get("permute_every"))
    config["permute_seed"] = maybe_int(raw_meta.get("permute_seed"))
    config["change_every"] = maybe_int(raw_meta.get("change_every"))
    config["episode_len"] = maybe_int(first_non_none(raw_meta.get("episode_len"), raw_meta.get("episode_steps")))
    config["git_commit"] = maybe_text(raw_meta.get("git_commit"))
    config["git_dirty"] = maybe_text(raw_meta.get("git_dirty"))

    metrics_row: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.DictReader(fp)
                first_row = next(reader, None)
                if isinstance(first_row, dict):
                    metrics_row = first_row
        except Exception:
            pass

    if config["backend"] is None:
        config["backend"] = maybe_text(metrics_row.get("backend"))
    if config["mode"] is None:
        config["mode"] = maybe_text(metrics_row.get("mode"))
    if config["seed"] is None:
        config["seed"] = maybe_int(metrics_row.get("seed"))
    if config["steps_requested"] is None:
        config["steps_requested"] = maybe_int(metrics_row.get("steps_requested"))
    if config["K"] is None:
        config["K"] = maybe_int(metrics_row.get("K"))
    if config["slots"] is None:
        config["slots"] = maybe_int(metrics_row.get("slots"))
    if config["cue_dist"] is None:
        config["cue_dist"] = maybe_text(metrics_row.get("cue_dist"))
    if config["zipf_alpha"] is None:
        config["zipf_alpha"] = maybe_float(metrics_row.get("zipf_alpha"))
    if config["permute_ids"] is None:
        config["permute_ids"] = maybe_int(metrics_row.get("permute_ids"))
    if config["permute_every"] is None:
        config["permute_every"] = maybe_int(metrics_row.get("permute_every"))
    if config["permute_seed"] is None:
        config["permute_seed"] = maybe_int(metrics_row.get("permute_seed"))
    if config["change_every"] is None:
        config["change_every"] = maybe_int(metrics_row.get("change_every"))
    if config["episode_len"] is None:
        config["episode_len"] = maybe_int(metrics_row.get("episode_len"))
    if config["git_commit"] is None:
        config["git_commit"] = maybe_text(metrics_row.get("git_commit"))
    if config["git_dirty"] is None:
        config["git_dirty"] = maybe_text(metrics_row.get("git_dirty"))

    if config["seed"] is None:
        config["seed"] = parse_seed_from_name(run_dir.name)
    if config["mode"] is None:
        config["mode"] = parse_mode_from_name(run_dir.name)

    return config


def compute_run_stats(metrics_path: Path, windows: list[int]) -> dict[str, Any]:
    rewards: list[float] = []
    rewired_total = 0.0
    episode_max: int | None = None
    baseline_last: float | None = None
    wfast_last: float | None = None
    wslow_last: float | None = None
    slots_used_last: int | None = None
    has_baseline = False
    has_wfast = False
    has_wslow = False
    has_rewired = False
    has_slots_used = False
    has_episode_idx = False

    with metrics_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            reward = maybe_float(row.get("reward"))
            if reward is None:
                continue
            rewards.append(reward)

            baseline = maybe_float(row.get("baseline"))
            if baseline is not None:
                has_baseline = True
                baseline_last = baseline

            wfast = maybe_float(row.get("wfast_norm"))
            if wfast is not None:
                has_wfast = True
                wfast_last = wfast

            wslow = maybe_float(row.get("wslow_norm"))
            if wslow is not None:
                has_wslow = True
                wslow_last = wslow

            rewired = maybe_float(row.get("rewired"))
            if rewired is not None:
                has_rewired = True
                rewired_total += rewired

            slots_used = maybe_int(row.get("slots_used"))
            if slots_used is not None:
                has_slots_used = True
                slots_used_last = slots_used

            episode_idx = maybe_int(row.get("episode_idx"))
            if episode_idx is not None:
                has_episode_idx = True
                if episode_max is None or episode_idx > episode_max:
                    episode_max = episode_idx

    num_steps = len(rewards)
    stats: dict[str, Any] = {"num_steps": num_steps}

    if num_steps == 0:
        stats["reward_mean"] = None
        for window in windows:
            stats[f"reward_last_{window}"] = None
        stats["baseline_last"] = baseline_last if has_baseline else None
        stats["wfast_last"] = wfast_last if has_wfast else None
        stats["wslow_last"] = wslow_last if has_wslow else None
        stats["rewired_total"] = rewired_total if has_rewired else None
        stats["rewired_rate"] = None
        stats["slots_used_last"] = slots_used_last if has_slots_used else None
        stats["episodes"] = (episode_max + 1) if has_episode_idx and episode_max is not None else None
        return stats

    reward_array = np.asarray(rewards, dtype=np.float64)
    stats["reward_mean"] = float(np.mean(reward_array))
    for window in windows:
        clamped = min(window, num_steps)
        stats[f"reward_last_{window}"] = float(np.mean(reward_array[-clamped:])) if clamped > 0 else None

    stats["baseline_last"] = baseline_last if has_baseline else None
    stats["wfast_last"] = wfast_last if has_wfast else None
    stats["wslow_last"] = wslow_last if has_wslow else None
    stats["rewired_total"] = rewired_total if has_rewired else None
    stats["rewired_rate"] = (rewired_total / num_steps) if has_rewired else None
    stats["slots_used_last"] = slots_used_last if has_slots_used else None
    stats["episodes"] = (episode_max + 1) if has_episode_idx and episode_max is not None else None
    return stats


def dedupe_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        run.get("mode"),
        run.get("seed"),
        run.get("K"),
        run.get("slots"),
        run.get("cue_dist"),
        run.get("zipf_alpha"),
        run.get("permute_ids"),
        run.get("permute_every"),
        run.get("backend"),
    )


def group_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        run.get("backend"),
        run.get("mode"),
        run.get("K"),
        run.get("slots"),
        run.get("cue_dist"),
        run.get("zipf_alpha"),
        run.get("permute_ids"),
        run.get("permute_every"),
    )


def base_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        run.get("backend"),
        run.get("K"),
        run.get("slots"),
        run.get("cue_dist"),
        run.get("zipf_alpha"),
        run.get("permute_ids"),
        run.get("permute_every"),
    )


def choose_best_run(runs: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if policy == "latest":
        return max(runs, key=lambda item: item["run_name"])
    return max(runs, key=lambda item: (item["num_steps"], item["run_name"]))


def apply_dedupe(runs: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    if policy == "none":
        return sorted(runs, key=lambda item: item["run_name"])

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        buckets.setdefault(dedupe_key(run), []).append(run)

    deduped = [choose_best_run(bucket, policy) for bucket in buckets.values()]
    deduped.sort(key=lambda item: item["run_name"])
    return deduped


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def sorted_repr(value: Any) -> tuple[int, Any]:
    if value is None:
        return 1, ""
    return 0, value


def build_group_summary(runs: list[dict[str, Any]], windows: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(group_key(run), []).append(run)

    rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        backend, mode, k_val, slots, cue_dist, zipf_alpha, permute_ids, permute_every = key
        seeds = sorted({item.get("seed") for item in items if item.get("seed") is not None})
        row: dict[str, Any] = {
            "backend": backend,
            "mode": mode,
            "K": k_val,
            "slots": slots,
            "cue_dist": cue_dist,
            "zipf_alpha": zipf_alpha,
            "permute_ids": permute_ids,
            "permute_every": permute_every,
            "n_runs": len(items),
            "seeds": ",".join(str(seed) for seed in seeds),
            "git_commit_set": "|".join(
                sorted({str(item.get("git_commit")) for item in items if item.get("git_commit") is not None})
            ),
            "git_dirty_set": "|".join(
                sorted({str(item.get("git_dirty")) for item in items if item.get("git_dirty") is not None})
            ),
            "permute_seed_set": "|".join(
                sorted({str(item.get("permute_seed")) for item in items if item.get("permute_seed") is not None})
            ),
        }

        reward_mean_values = [item["reward_mean"] for item in items if item.get("reward_mean") is not None]
        reward_mean_avg, reward_mean_std = mean_std(reward_mean_values)
        row["reward_mean_mean"] = reward_mean_avg
        row["reward_mean_std"] = reward_mean_std

        for window in windows:
            field = f"reward_last_{window}"
            values = [item[field] for item in items if item.get(field) is not None]
            avg, std = mean_std(values)
            row[f"{field}_mean"] = avg
            row[f"{field}_std"] = std

        rows.append(row)

    rows.sort(
        key=lambda item: (
            sorted_repr(item.get("cue_dist")),
            sorted_repr(item.get("zipf_alpha")),
            sorted_repr(item.get("K")),
            sorted_repr(item.get("slots")),
            sorted_repr(item.get("mode")),
            sorted_repr(item.get("permute_ids")),
            sorted_repr(item.get("permute_every")),
            sorted_repr(item.get("backend")),
        )
    )
    return rows


def choose_best_per_seed(runs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for run in runs:
        seed = run.get("seed")
        if seed is None:
            continue
        existing = by_seed.get(seed)
        if existing is None:
            by_seed[seed] = run
            continue
        if (run["num_steps"], run["run_name"]) > (existing["num_steps"], existing["run_name"]):
            by_seed[seed] = run
    return by_seed


def build_paired_deltas(runs: list[dict[str, Any]], max_window: int) -> list[dict[str, Any]]:
    pair_modes = [
        ("plastic", "static"),
        ("slot_plastic", "slot_static"),
        ("slot_plastic", "slot_hash_plastic"),
        ("slot_hash_plastic", "slot_hash_static"),
    ]
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        grouped.setdefault(base_key(run), {}).setdefault(run.get("mode"), []).append(run)

    rows: list[dict[str, Any]] = []
    last_field = f"reward_last_{max_window}"
    for base, mode_map in grouped.items():
        backend, k_val, slots, cue_dist, zipf_alpha, permute_ids, permute_every = base
        for mode_a, mode_b in pair_modes:
            if mode_a not in mode_map or mode_b not in mode_map:
                continue

            runs_a = choose_best_per_seed(mode_map[mode_a])
            runs_b = choose_best_per_seed(mode_map[mode_b])
            common_seeds = sorted(set(runs_a.keys()) & set(runs_b.keys()))
            if not common_seeds:
                continue

            delta_mean: list[float] = []
            delta_last: list[float] = []
            for seed in common_seeds:
                run_a = runs_a[seed]
                run_b = runs_b[seed]
                metric_a = run_a.get("reward_mean")
                metric_b = run_b.get("reward_mean")
                if metric_a is not None and metric_b is not None:
                    delta_mean.append(float(metric_a) - float(metric_b))

                last_a = run_a.get(last_field)
                last_b = run_b.get(last_field)
                if last_a is not None and last_b is not None:
                    delta_last.append(float(last_a) - float(last_b))

            if not delta_mean and not delta_last:
                continue

            mean_avg, mean_std_val = mean_std(delta_mean)
            last_avg, last_std = mean_std(delta_last)
            rows.append(
                {
                    "backend": backend,
                    "K": k_val,
                    "slots": slots,
                    "cue_dist": cue_dist,
                    "zipf_alpha": zipf_alpha,
                    "permute_ids": permute_ids,
                    "permute_every": permute_every,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "metric_last_window": max_window,
                    "n_pairs": len(common_seeds),
                    "seeds": ",".join(str(seed) for seed in common_seeds),
                    "delta_reward_mean_mean": mean_avg,
                    "delta_reward_mean_std": mean_std_val,
                    "delta_reward_last_mean": last_avg,
                    "delta_reward_last_std": last_std,
                }
            )

    rows.sort(
        key=lambda item: (
            sorted_repr(item.get("cue_dist")),
            sorted_repr(item.get("zipf_alpha")),
            sorted_repr(item.get("K")),
            sorted_repr(item.get("slots")),
            sorted_repr(item.get("mode_a")),
            sorted_repr(item.get("permute_ids")),
            sorted_repr(item.get("permute_every")),
            sorted_repr(item.get("backend")),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def print_group_table(rows: list[dict[str, Any]], max_window: int) -> None:
    print("\nGrouped summary:")
    print("cue_dist  zipf_alpha  K    slots  mode          n_runs  reward_mean  reward_last  seeds")
    print(f"(reward_last means reward_last_{max_window})")
    for row in rows:
        print(
            f"{str(row.get('cue_dist')):<8}  "
            f"{str(row.get('zipf_alpha')):<10}  "
            f"{str(row.get('K')):<3}  "
            f"{str(row.get('slots')):<5}  "
            f"{str(row.get('mode')):<12}  "
            f"{int(row.get('n_runs', 0)):<6}  "
            f"{fmt_float(row.get('reward_mean_mean')):<11}  "
            f"{fmt_float(row.get(f'reward_last_{max_window}_mean')):<11}  "
            f"{row.get('seeds')}"
        )


def print_paired_table(rows: list[dict[str, Any]], max_window: int) -> None:
    print("\nPaired deltas:")
    if not rows:
        print("No overlapping seeds found for requested mode pairs.")
        return
    print(
        "cue_dist  zipf_alpha  K    slots  pair                       n_pairs  "
        "delta_reward_mean  delta_reward_last"
    )
    print(f"(delta_reward_last means reward_last_{max_window})")
    for row in rows:
        pair_name = f"{row.get('mode_a')}-{row.get('mode_b')}"
        print(
            f"{str(row.get('cue_dist')):<8}  "
            f"{str(row.get('zipf_alpha')):<10}  "
            f"{str(row.get('K')):<3}  "
            f"{str(row.get('slots')):<5}  "
            f"{pair_name:<25}  "
            f"{int(row.get('n_pairs', 0)):<7}  "
            f"{fmt_float(row.get('delta_reward_mean_mean')):<17}  "
            f"{fmt_float(row.get('delta_reward_last_mean')):<17}"
        )


def main() -> int:
    args = parse_args()
    windows = parse_windows(args.last_windows)

    rollouts_dir = Path(args.rollouts_dir)
    out_dir = Path(args.out_dir)
    try:
        include_pattern = re.compile(args.include_regex)
    except re.error:
        include_pattern = re.compile(re.escape(args.include_regex))

    if not rollouts_dir.exists():
        print(f"rollouts_dir does not exist: {rollouts_dir}")
        return 1

    discovered: list[Path] = []
    for child in sorted(rollouts_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not (child / "metrics.csv").exists():
            continue
        discovered.append(child)

    analyzed: list[dict[str, Any]] = []
    for run_dir in discovered:
        if not include_pattern.search(run_dir.name):
            continue
        stats = compute_run_stats(run_dir / "metrics.csv", windows)
        if stats["num_steps"] < args.min_steps:
            continue

        config = load_config(run_dir)
        row = {
            "run_name": run_dir.name,
            "run_path": str(run_dir),
            **config,
            **stats,
        }
        analyzed.append(row)

    kept = apply_dedupe(analyzed, args.dedupe)

    print(f"Discovered valid runs: {len(discovered)}")
    print(f"Kept runs after filters/dedupe: {len(kept)}")

    if not kept:
        print(
            "No valid runs found after filtering. Check --rollouts_dir, --include_regex, "
            "--min_steps, and --dedupe settings."
        )
        return 1

    max_window = max(windows)
    group_rows = build_group_summary(kept, windows)
    paired_rows = build_paired_deltas(kept, max_window)

    run_fields = [
        "run_name",
        "run_path",
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
        "num_steps",
        "reward_mean",
    ]
    run_fields.extend([f"reward_last_{window}" for window in windows])
    run_fields.extend(
        [
            "baseline_last",
            "wfast_last",
            "wslow_last",
            "rewired_total",
            "rewired_rate",
            "slots_used_last",
            "episodes",
        ]
    )

    group_fields = [
        "backend",
        "mode",
        "K",
        "slots",
        "cue_dist",
        "zipf_alpha",
        "permute_ids",
        "permute_every",
        "n_runs",
        "seeds",
        "git_commit_set",
        "git_dirty_set",
        "permute_seed_set",
        "reward_mean_mean",
        "reward_mean_std",
    ]
    for window in windows:
        group_fields.extend([f"reward_last_{window}_mean", f"reward_last_{window}_std"])

    paired_fields = [
        "backend",
        "K",
        "slots",
        "cue_dist",
        "zipf_alpha",
        "permute_ids",
        "permute_every",
        "mode_a",
        "mode_b",
        "metric_last_window",
        "n_pairs",
        "seeds",
        "delta_reward_mean_mean",
        "delta_reward_mean_std",
        "delta_reward_last_mean",
        "delta_reward_last_std",
    ]

    write_csv(out_dir / "cue_run_summary.csv", kept, run_fields)
    write_csv(out_dir / "cue_group_summary.csv", group_rows, group_fields)
    write_csv(out_dir / "cue_paired_deltas.csv", paired_rows, paired_fields)

    print_group_table(group_rows, max_window)
    print_paired_table(paired_rows, max_window)
    print(f"\nWrote: {out_dir / 'cue_run_summary.csv'}")
    print(f"Wrote: {out_dir / 'cue_group_summary.csv'}")
    print(f"Wrote: {out_dir / 'cue_paired_deltas.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
