#!/usr/bin/env python3
"""Toy environment with task switching to measure forgetting."""

from __future__ import annotations

from typing import List, Tuple

import torch


class TaskSwitchEnv:
    """Generates labeled batches with two alternating task mappings."""

    def __init__(self, input_dim: int, seed: int = 0) -> None:
        self.input_dim = int(input_dim)
        gen = torch.Generator().manual_seed(seed)
        self.w_a = torch.randn(self.input_dim, generator=gen)
        self.w_b = torch.randn(self.input_dim, generator=gen)

    def sample_batch(self, task_id: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, self.input_dim)
        if int(task_id) == 0:
            logits = x @ self.w_a
        else:
            logits = x @ self.w_b
        y = (logits > 0).float()
        return x, y

    def sample_episode(
        self, task_id: int, length: int, batch_size: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.sample_batch(task_id, batch_size) for _ in range(length)]
