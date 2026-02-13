from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.net(rgb)
        return x.mean(dim=(-2, -1))


class TinyBC(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.encoder = TinyCNN()
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.encoder(rgb)
        return self.head(x)
