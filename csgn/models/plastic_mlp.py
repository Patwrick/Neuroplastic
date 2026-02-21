from __future__ import annotations

import torch
from torch import nn


class PlasticLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eta: float = 0.1,
        decay: float = 0.01,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.eta = float(eta)
        self.decay = float(decay)

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.alpha = nn.Parameter(torch.full((self.out_features, self.in_features), 0.1))
        nn.init.xavier_uniform_(self.W)

        self.register_buffer("H", torch.zeros(1, self.out_features, self.in_features), persistent=False)

    def reset_fast_state(self, batch_size: int, device: torch.device | None = None) -> None:
        bsz = int(batch_size)
        if bsz <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        target_device = device if device is not None else self.W.device
        self.H = torch.zeros(
            (bsz, self.out_features, self.in_features),
            device=target_device,
            dtype=self.W.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.H is None or self.H.ndim != 3 or self.H.shape[0] != x.shape[0]:
            self.reset_fast_state(batch_size=int(x.shape[0]), device=x.device)
        x = x.to(dtype=self.W.dtype)
        base = x @ self.W.t()
        fast = torch.einsum("boi,bi->bo", self.alpha.unsqueeze(0) * self.H, x)
        return base + fast

    @torch.no_grad()
    def plastic_update(self, pre: torch.Tensor, post: torch.Tensor, mod: torch.Tensor) -> None:
        if self.H is None or self.H.ndim != 3 or self.H.shape[0] != pre.shape[0]:
            self.reset_fast_state(batch_size=int(pre.shape[0]), device=pre.device)

        pre_t = pre.to(dtype=self.W.dtype)
        post_t = post.to(dtype=self.W.dtype)
        mod_t = mod.to(dtype=self.W.dtype)
        if mod_t.ndim == 1:
            mod_t = mod_t.unsqueeze(1)

        hebb = torch.einsum("bo,bi->boi", post_t, pre_t)
        self.H.mul_(1.0 - self.decay).add_(self.eta * mod_t.unsqueeze(2) * hebb)

    def fast_norm(self) -> torch.Tensor:
        return torch.norm(self.H)


class PlasticMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        eta: float = 0.1,
        decay: float = 0.01,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.plastic = PlasticLinear(self.hidden_dim, self.action_dim, eta=eta, decay=decay)

    def reset_fast_state(self, batch_size: int, device: torch.device | None = None) -> None:
        self.plastic.reset_fast_state(batch_size=batch_size, device=device)

    def forward(self, x: torch.Tensor, return_pre: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pre = self.encoder(x)
        logits = self.plastic(pre)
        if return_pre:
            return logits, pre
        return logits

    @torch.no_grad()
    def plastic_update(self, pre: torch.Tensor, post: torch.Tensor, mod: torch.Tensor) -> None:
        self.plastic.plastic_update(pre=pre, post=post, mod=mod)

    def stats(self) -> dict[str, float]:
        return {
            "wfast_norm": float(self.plastic.fast_norm().item()),
            "wslow_norm": float(torch.norm(self.plastic.W).item()),
        }
