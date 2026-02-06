from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import torch
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csgn.model.plastic_graph_core import PlasticGraphCore
from csgn.model.sleep_controller import SleepController
from csgn.toy.replay_buffer import ReplayBuffer
from csgn.toy.task_switch_stream import TaskSwitchStream
from csgn.utils.csv_logger import CSVLogger
from csgn.utils.seed import set_global_seed


TRAIN_BATCH_SIZE = 32
EVAL_N = 512


def evaluate_task(model: PlasticGraphCore, stream: TaskSwitchStream, task: str, n: int = EVAL_N) -> float:
    state = model.snapshot_state()
    try:
        def model_fn(x: torch.Tensor) -> torch.Tensor:
            x_d = x.to(model.device)
            model.reset_state(batch_size=x_d.shape[0])
            logits = model.step(x_d)
            return torch.sigmoid(logits).detach().cpu()

        return stream.evaluate_accuracy(model_fn, task=task, n=n)
    finally:
        model.restore_state(state)


def maybe_run_sleep_cycle(
    *,
    condition: str,
    model: PlasticGraphCore,
    replay: ReplayBuffer,
    sleep_replay_steps: int,
    batch_size: int,
) -> None:
    if condition not in {"sleep", "sleep_no_rewire"}:
        return
    if len(replay) == 0:
        return

    replay_batch = min(batch_size, len(replay))
    model.sleep_replay(replay, steps=sleep_replay_steps, batch_size=replay_batch)
    model.sleep_consolidate()
    if condition == "sleep":
        model.sleep_rewire(prune_frac=0.10, exploration=0.20)


def run_condition(
    *,
    condition: str,
    args: argparse.Namespace,
    base_state: dict[str, torch.Tensor],
    timestamp: str,
) -> dict[str, float | str]:
    set_global_seed(args.seed)
    stream = TaskSwitchStream(d_in=args.d_in, seed=args.seed)
    replay = ReplayBuffer(capacity=args.capacity)

    model = PlasticGraphCore(
        N=args.N,
        d=args.d,
        k_out=args.k_out,
        d_in=args.d_in,
        device=args.device,
    )
    model.load_state_dict(deepcopy(base_state), strict=True)
    model.reset_state(batch_size=TRAIN_BATCH_SIZE)

    controller = SleepController(max_awake=args.sleep_every, sleep_steps=1)

    out_path = Path("runs") / f"toy_{timestamp}_{condition}.csv"
    logger = CSVLogger(
        str(out_path),
        fieldnames=["condition", "task", "step", "accA_est", "accB_est", "syn_load", "drift"],
    )

    global_step = 0
    phases = [("A", args.steps_A), ("B", args.steps_B)]
    acc_A_before = 0.0

    for task_name, num_steps in phases:
        for _ in trange(num_steps, desc=f"{condition}:{task_name}", leave=False):
            global_step += 1

            x, y = stream.sample(task_name, batch_size=TRAIN_BATCH_SIZE)
            x_d = x.to(model.device)
            y_d = y.to(model.device)

            logits = model.step(x_d)
            y_pred = torch.sigmoid(logits)

            model.wake_update(y_d, y_pred)
            if condition == "no_sleep":
                model.wake_update_slow_direct(y_d, y_pred)

            replay.add(x_d, y_d, task_name)

            syn_load = model.synaptic_load()
            pooled = model.pooled_state()
            should_sleep, sleep_cycles = controller.update_and_maybe_sleep(
                synaptic_load=syn_load,
                pooled_state=pooled,
            )

            if should_sleep:
                for _ in range(sleep_cycles):
                    maybe_run_sleep_cycle(
                        condition=condition,
                        model=model,
                        replay=replay,
                        sleep_replay_steps=args.sleep_replay_steps,
                        batch_size=TRAIN_BATCH_SIZE,
                    )

            if global_step % 100 == 0:
                accA_est = evaluate_task(model, stream, "A")
                accB_est = evaluate_task(model, stream, "B")
                logger.log_row(
                    {
                        "condition": condition,
                        "task": task_name,
                        "step": global_step,
                        "accA_est": f"{accA_est:.6f}",
                        "accB_est": f"{accB_est:.6f}",
                        "syn_load": f"{controller.ma_synaptic_load:.6f}",
                        "drift": f"{controller.ma_drift:.6f}",
                    }
                )

        if task_name == "A":
            acc_A_before = evaluate_task(model, stream, "A", n=1024)

    acc_A_after = evaluate_task(model, stream, "A", n=1024)
    acc_B_after = evaluate_task(model, stream, "B", n=1024)
    forgetting_A = acc_A_before - acc_A_after

    return {
        "condition": condition,
        "acc_A_before": acc_A_before,
        "acc_A_after": acc_A_after,
        "acc_B_after": acc_B_after,
        "forgetting_A": forgetting_A,
        "csv_path": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy continual-learning experiment for sleep/replay.")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--k_out", type=int, default=8)
    p.add_argument("--d_in", type=int, default=16)
    p.add_argument("--steps_A", type=int, default=5000)
    p.add_argument("--steps_B", type=int, default=5000)
    p.add_argument("--sleep_every", type=int, default=500)
    p.add_argument("--sleep_replay_steps", type=int, default=512)
    p.add_argument("--capacity", type=int, default=20000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    base_model = PlasticGraphCore(
        N=args.N,
        d=args.d,
        k_out=args.k_out,
        d_in=args.d_in,
        device=args.device,
    )
    base_state = deepcopy(base_model.state_dict())

    conditions = ["no_sleep", "sleep", "sleep_no_rewire"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for condition in conditions:
        result = run_condition(
            condition=condition,
            args=args,
            base_state=base_state,
            timestamp=timestamp,
        )
        results.append(result)

    print("")
    print("Final Summary")
    print(
        f"{'condition':<18} {'acc_A_before':>12} {'acc_A_after':>12} "
        f"{'acc_B_after':>12} {'forget_A':>10} {'csv':>30}"
    )
    for row in results:
        print(
            f"{str(row['condition']):<18} "
            f"{float(row['acc_A_before']):>12.4f} "
            f"{float(row['acc_A_after']):>12.4f} "
            f"{float(row['acc_B_after']):>12.4f} "
            f"{float(row['forgetting_A']):>10.4f} "
            f"{str(row['csv_path']):>30}"
        )


if __name__ == "__main__":
    main()
