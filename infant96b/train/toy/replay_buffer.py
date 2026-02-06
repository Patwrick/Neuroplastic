#!/usr/bin/env python3
"""Simple replay buffer for toy experiments."""

from __future__ import annotations

from typing import List, Tuple

import random
import torch


class ReplayBuffer:
    def __init__(self, max_episodes: int = 500) -> None:
        self.max_episodes = int(max_episodes)
        self._episodes: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

    def add_episode(self, episode: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self._episodes.append(episode)
        if len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)

    def sample_batches(self, num_batches: int, batch_size: int) -> List[torch.Tensor]:
        if not self._episodes:
            return []

        batches: List[torch.Tensor] = []
        for _ in range(num_batches):
            episode = random.choice(self._episodes)
            x, _ = random.choice(episode)
            if x.size(0) != batch_size:
                idx = torch.randint(0, x.size(0), (batch_size,))
                x = x[idx]
            batches.append(x)
        return batches

    def sample_labeled_batches(
        self, num_batches: int, batch_size: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._episodes:
            return []

        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_batches):
            episode = random.choice(self._episodes)
            x, y = random.choice(episode)
            if x.size(0) != batch_size:
                idx = torch.randint(0, x.size(0), (batch_size,))
                x = x[idx]
                y = y[idx]
            batches.append((x, y))
        return batches
