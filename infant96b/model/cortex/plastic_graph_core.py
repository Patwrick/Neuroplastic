#!/usr/bin/env python3
"""Plastic graph core for neuroplasticity experiments."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn


class PlasticGraphCore(nn.Module):
    """Graph of N nodes with fixed out-degree and fast/slow plasticity."""

    def __init__(
        self,
        N: int,
        d: int,
        k_out: int,
        p_active: float = 0.2,
        alpha: float = 0.5,
        w_scale: float = 0.01,
        lr_fast: float = 0.05,
        util_decay: float = 0.99,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.N = int(N)
        self.d = int(d)
        self.k_out = int(k_out)
        self.p_active = float(p_active)
        self.alpha = float(alpha)
        self.w_scale = float(w_scale)
        self.lr_fast = float(lr_fast)
        self.util_decay = float(util_decay)

        self.mlp = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.gru = nn.GRUCell(self.d, self.d)
        self.readout = nn.Linear(self.d, self.d)

        targets = torch.randint(0, self.N, (self.N, self.k_out), device=device)
        w_slow = torch.zeros((self.N, self.k_out), dtype=torch.int8, device=device)
        f_fast = torch.zeros((self.N, self.k_out), dtype=torch.float16, device=device)
        u_util = torch.zeros((self.N, self.k_out), dtype=torch.float32, device=device)

        self.register_buffer("targets", targets)
        self.register_buffer("w_slow", w_slow)
        self.register_buffer("f_fast", f_fast)
        self.register_buffer("u_util", u_util)

        self.h: Optional[torch.Tensor] = None
        self._last_pre: Optional[torch.Tensor] = None
        self._last_post: Optional[torch.Tensor] = None
        self._last_activity: Optional[torch.Tensor] = None
        self._last_drift: float = 0.0

    def reset_state(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state and traces for a batch."""
        device = self.targets.device
        self.h = torch.zeros((batch_size, self.N, self.d), device=device)
        self._last_pre = None
        self._last_post = None
        self._last_activity = None
        self._last_drift = 0.0
        return self.h

    def _active_indices(self) -> torch.Tensor:
        if self.h is None:
            raise RuntimeError("Hidden state is not initialized.")
        act = self.h.abs().mean(dim=-1)  # [B, N]
        score = act.mean(dim=0)
        num_active = max(1, int(round(self.p_active * self.N)))
        num_active = min(num_active, self.N)
        _, active_idx = torch.topk(score, k=num_active)
        return active_idx

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one update and return (node_states, readout)."""
        if x.dim() == 2:
            batch_size = x.size(0)
        elif x.dim() == 3:
            batch_size = x.size(0)
        else:
            raise ValueError("x must be [B, d] or [B, N, d]")

        if self.h is None or self.h.size(0) != batch_size:
            self.reset_state(batch_size)

        if self.h is None:
            raise RuntimeError("Hidden state is not initialized.")

        if x.dim() == 2:
            x_in = x[:, None, :].expand(-1, self.N, -1)
        else:
            x_in = x
        x_in = x_in.to(self.h.device).to(self.h.dtype)

        prev_h = self.h
        active_idx = self._active_indices()
        m = self.mlp(prev_h)

        w_eff = self.w_scale * self.w_slow.float() + self.alpha * self.f_fast.float()

        edge_targets = self.targets[active_idx].reshape(-1)
        edge_weights = w_eff[active_idx].reshape(-1).to(prev_h.dtype)

        agg = torch.zeros((batch_size, self.N, self.d), device=prev_h.device, dtype=prev_h.dtype)
        if edge_targets.numel() > 0:
            for b in range(batch_size):
                m_active = m[b, active_idx, :]
                m_repeat = m_active.repeat_interleave(self.k_out, dim=0)
                contrib = m_repeat * edge_weights[:, None]
                agg_b = torch.zeros((self.N, self.d), device=prev_h.device, dtype=prev_h.dtype)
                agg_b.index_add_(0, edge_targets, contrib)
                agg[b] = agg_b

        gru_in = agg + x_in
        h_flat = prev_h.reshape(batch_size * self.N, self.d)
        in_flat = gru_in.reshape(batch_size * self.N, self.d)
        new_h = self.gru(in_flat, h_flat).view(batch_size, self.N, self.d)

        self.h = new_h
        self._last_pre = prev_h.detach()
        self._last_post = new_h.detach()
        self._last_activity = new_h.abs().mean(dim=0).detach()
        self._last_drift = float((new_h - prev_h).abs().mean().item())

        readout = self.readout(new_h.mean(dim=1))
        return new_h, readout

    def wake_plastic_update(self, modulator: torch.Tensor) -> None:
        """Update fast traces and utility with a simple 3-factor rule."""
        if self._last_pre is None or self._last_post is None:
            return

        with torch.no_grad():
            mod = modulator
            if not isinstance(mod, torch.Tensor):
                mod = torch.tensor(mod, device=self.targets.device)
            mod = mod.float().mean()

            pre = self._last_pre.abs().mean(dim=-1).mean(dim=0)  # [N]
            post = self._last_post.abs().mean(dim=-1).mean(dim=0)  # [N]
            post_target = post[self.targets]  # [N, k]

            delta = self.lr_fast * mod * pre[:, None] * post_target
            f_fast = self.f_fast.float() + delta
            f_fast = torch.clamp(f_fast, -1.0, 1.0)
            self.f_fast.copy_(f_fast.to(torch.float16))

            util = self.u_util * self.util_decay + (1.0 - self.util_decay) * delta.abs()
            self.u_util.copy_(util)

    def sleep_consolidate(self, replay_batches: List[torch.Tensor], lr: float) -> None:
        """Consolidate slow weights and decay fast traces using replay."""
        with torch.no_grad():
            for batch in replay_batches:
                self.forward(batch)

        w_slow = self.w_slow.float() + lr * torch.tanh(self.f_fast.float())
        w_slow = torch.clamp(w_slow, -127.0, 127.0).round()
        self.w_slow.copy_(w_slow.to(torch.int8))

        f_fast = (self.f_fast.float() * 0.9).clamp(-1.0, 1.0)
        self.f_fast.copy_(f_fast.to(torch.float16))

    def sleep_rewire(self, prune_frac: float, regrow_candidates: str = "coactivation") -> None:
        """Prune low-utility edges and regrow new targets while keeping degree fixed."""
        if prune_frac <= 0.0:
            return

        with torch.no_grad():
            k_prune = max(1, int(round(self.k_out * prune_frac)))
            k_prune = min(k_prune, self.k_out)

            if regrow_candidates == "coactivation" and self._last_activity is not None:
                probs = self._last_activity.clone().float()
                probs = probs + 1e-3
            else:
                probs = torch.ones(self.N, device=self.targets.device)

            probs = probs / probs.sum()

            for i in range(self.N):
                util_row = self.u_util[i]
                _, prune_idx = torch.topk(util_row, k=k_prune, largest=False)
                existing = set(self.targets[i].tolist())

                for slot in prune_idx.tolist():
                    new_target = None
                    for _ in range(10):
                        cand = int(torch.multinomial(probs, 1).item())
                        if cand not in existing:
                            new_target = cand
                            break
                    if new_target is None:
                        new_target = int(torch.randint(0, self.N, (1,)).item())

                    self.targets[i, slot] = new_target
                    existing.add(new_target)
                    self.w_slow[i, slot] = 0
                    self.f_fast[i, slot] = 0
                    self.u_util[i, slot] = 0

    def synaptic_load(self) -> float:
        return float(self.f_fast.float().abs().mean().item())

    def drift_risk(self) -> float:
        return float(self._last_drift)
