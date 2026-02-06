#!/usr/bin/env python3
"""Run toy experiments to evaluate neuroplasticity behavior."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from model.cortex.plastic_graph_core import PlasticGraphCore
from model.sleep.sleep_controller import SleepController
from train.toy.replay_buffer import ReplayBuffer
from train.toy.task_switch_env import TaskSwitchEnv


@dataclass
class ExperimentConfig:
    N: int = 64
    d: int = 32
    k_out: int = 8
    steps: int = 2000
    batch_size: int = 32
    seed: int = 0
    lr: float = 1e-3
    sleep_lr: float = 0.05
    prune_frac: float = 0.1
    replay_batches: int = 3
    buffer_size: int = 500


@dataclass
class Condition:
    name: str
    sleep_enabled: bool
    rewire_enabled: bool


class ToyAgent(nn.Module):
    def __init__(self, N: int, d: int, k_out: int) -> None:
        super().__init__()
        self.core = PlasticGraphCore(N=N, d=d, k_out=k_out)
        self.classifier = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, readout = self.core(x)
        logits = self.classifier(readout).squeeze(-1)
        return logits


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _evaluate(agent: ToyAgent, env: TaskSwitchEnv, task_id: int, batches: int, batch_size: int) -> float:
    agent.eval()
    agent.core.reset_state(batch_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = env.sample_batch(task_id, batch_size)
            logits = agent(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    agent.train()
    return float(correct) / float(max(1, total))


def run_condition(config: ExperimentConfig, condition: Condition) -> Dict[str, float]:
    _set_seed(config.seed)
    env = TaskSwitchEnv(config.d, seed=config.seed)
    agent = ToyAgent(config.N, config.d, config.k_out)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    buffer = ReplayBuffer(max_episodes=config.buffer_size)
    controller = SleepController()

    drift_vals: List[float] = []
    syn_vals: List[float] = []

    steps_a = config.steps // 2
    steps_b = config.steps - steps_a

    for step in range(config.steps):
        task_id = 0 if step < steps_a else 1
        x, y = env.sample_batch(task_id, config.batch_size)
        logits = agent(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            mod = (preds == y).float().mean()
            agent.core.wake_plastic_update(mod)

        buffer.add_episode([(x.detach().cpu(), y.detach().cpu())])

        drift = agent.core.drift_risk()
        syn_load = agent.core.synaptic_load()
        drift_vals.append(drift)
        syn_vals.append(syn_load)

        if condition.sleep_enabled:
            decision = controller.update(syn_load, drift)
            if decision.should_sleep:
                for _ in range(decision.sleep_steps):
                    labeled_batches = buffer.sample_labeled_batches(
                        num_batches=config.replay_batches,
                        batch_size=config.batch_size,
                    )
                    for x_replay, y_replay in labeled_batches:
                        logits_replay = agent(x_replay)
                        loss_replay = loss_fn(logits_replay, y_replay)
                        optimizer.zero_grad()
                        loss_replay.backward()
                        optimizer.step()

                    replay_batches = [x for x, _ in labeled_batches]
                    if replay_batches:
                        agent.core.sleep_consolidate(replay_batches, lr=config.sleep_lr)
                    if condition.rewire_enabled:
                        agent.core.sleep_rewire(prune_frac=config.prune_frac)
                agent.core.reset_state(config.batch_size)
        else:
            controller.update(syn_load, drift)

    acc_task_a_post_b = _evaluate(agent, env, task_id=0, batches=50, batch_size=config.batch_size)

    window = min(100, len(drift_vals))
    drift_avg = float(sum(drift_vals[-window:]) / max(1, window))
    syn_avg = float(sum(syn_vals[-window:]) / max(1, window))

    return {
        "condition": condition.name,
        "sleep_enabled": float(condition.sleep_enabled),
        "rewire_enabled": float(condition.rewire_enabled),
        "acc_taskA_post_B": acc_task_a_post_b,
        "drift_risk": drift_avg,
        "synaptic_load": syn_avg,
        "steps": float(config.steps),
        "seed": float(config.seed),
    }


def run_conditions(
    config: ExperimentConfig,
    conditions: Iterable[Condition],
    output_csv: str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for cond in conditions:
        rows.append(run_condition(config, cond))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def default_conditions() -> List[Condition]:
    return [
        Condition(name="no_sleep", sleep_enabled=False, rewire_enabled=False),
        Condition(name="sleep", sleep_enabled=True, rewire_enabled=True),
        Condition(name="sleep_no_rewire", sleep_enabled=True, rewire_enabled=False),
    ]


def run_all(config: ExperimentConfig, output_dir: str) -> List[Dict[str, float]]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"toy_experiment_{timestamp}.csv")
    return run_conditions(config, default_conditions(), output_csv)


if __name__ == "__main__":
    cfg = ExperimentConfig()
    output = run_all(cfg, output_dir=os.path.join(os.path.dirname(__file__), "..", "..", "runs"))
    for row in output:
        print(row)
