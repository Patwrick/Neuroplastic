from __future__ import annotations

import math

import torch
from torch import nn


class PlasticGraphCore(nn.Module):
    def __init__(
        self,
        N: int = 128,
        d: int = 64,
        k_out: int = 8,
        d_in: int = 16,
        p_active: float = 0.1,
        lambda_e: float = 0.95,
        lambda_f: float = 0.90,
        lambda_u: float = 0.99,
        eta_fast: float = 0.10,
        eta_wake_slow: float = 0.005,
        eta_sleep_slow: float = 0.05,
        rho_fast_decay: float = 0.2,
        use_gating: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        if N <= 0 or d <= 0 or k_out <= 0 or d_in <= 0:
            raise ValueError("N, d, k_out, and d_in must all be > 0")
        if not (0.0 < p_active <= 1.0):
            raise ValueError(f"p_active must be in (0, 1], got {p_active!r}")

        self.N = int(N)
        self.d = int(d)
        self.k_out = int(k_out)
        self.d_in = int(d_in)
        self.p_active = float(p_active)

        self.lambda_e = float(lambda_e)
        self.lambda_f = float(lambda_f)
        self.lambda_u = float(lambda_u)

        self.eta_fast = float(eta_fast)
        self.eta_wake_slow = float(eta_wake_slow)
        self.eta_sleep_slow = float(eta_sleep_slow)
        self.rho_fast_decay = float(rho_fast_decay)
        self.use_gating = bool(use_gating)

        self.input_proj = nn.Linear(self.d_in, self.d)
        self.msg_mlp = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.gru = nn.GRUCell(input_size=self.d, hidden_size=self.d)

        if self.use_gating:
            self.Wq = nn.Linear(self.d, self.d)
            self.Wk = nn.Linear(self.d, self.d)

        targets = torch.randint(0, self.N, (self.N, self.k_out), dtype=torch.long)
        w_slow = torch.zeros(self.N, self.k_out, dtype=torch.float32)
        self.register_buffer("targets", targets)
        self.register_buffer("w_slow", w_slow)

        self.register_buffer("h", torch.zeros(1, self.N, self.d, dtype=torch.float32))
        self.register_buffer("f_fast", torch.zeros(1, self.N, self.k_out, dtype=torch.float32))
        self.register_buffer("e_elig", torch.zeros(1, self.N, self.k_out, dtype=torch.float32))
        self.register_buffer("u_util", torch.zeros(1, self.N, self.k_out, dtype=torch.float32))

        self.to(device)

    @property
    def device(self) -> torch.device:
        return self.w_slow.device

    @torch.no_grad()
    def reset_state(self, batch_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size!r}")

        self.h = torch.zeros(batch_size, self.N, self.d, dtype=torch.float32, device=self.device)
        self.f_fast = torch.zeros(batch_size, self.N, self.k_out, dtype=torch.float32, device=self.device)
        self.e_elig = torch.zeros(batch_size, self.N, self.k_out, dtype=torch.float32, device=self.device)
        self.u_util = torch.zeros(batch_size, self.N, self.k_out, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def snapshot_state(self) -> dict[str, torch.Tensor]:
        return {
            "h": self.h.clone(),
            "f_fast": self.f_fast.clone(),
            "e_elig": self.e_elig.clone(),
            "u_util": self.u_util.clone(),
        }

    @torch.no_grad()
    def restore_state(self, state: dict[str, torch.Tensor]) -> None:
        self.h = state["h"].to(self.device)
        self.f_fast = state["f_fast"].to(self.device)
        self.e_elig = state["e_elig"].to(self.device)
        self.u_util = state["u_util"].to(self.device)

    @torch.no_grad()
    def synaptic_load(self) -> float:
        return float(self.f_fast.abs().mean().item())

    @torch.no_grad()
    def pooled_state(self) -> torch.Tensor:
        # Mean over nodes then batch -> [d]
        return self.h.mean(dim=1).mean(dim=0)

    @torch.no_grad()
    def _ensure_batch_size(self, batch_size: int) -> None:
        if self.h.shape[0] != batch_size:
            self.reset_state(batch_size)

    def _active_count(self) -> int:
        return max(1, min(self.N, int(math.ceil(self.p_active * self.N))))

    def _target_hidden(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, d], returns target h per outgoing edge: [B, N, k_out, d]
        bsz = h.shape[0]
        flat_targets = self.targets.reshape(-1)
        return h[:, flat_targets, :].reshape(bsz, self.N, self.k_out, self.d)

    def _edge_pre_post_dot(self, h: torch.Tensor) -> torch.Tensor:
        pre = h.unsqueeze(2)  # [B, N, 1, d]
        post = self._target_hidden(h)  # [B, N, k_out, d]
        return (pre * post).sum(dim=-1)  # [B, N, k_out]

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if x_t.ndim != 2 or x_t.shape[1] != self.d_in:
            raise ValueError(f"x must be [B, {self.d_in}], got {tuple(x_t.shape)}")

        bsz = x_t.shape[0]
        self._ensure_batch_size(bsz)

        x_embed = self.input_proj(x_t)  # [B, d]
        m = self.msg_mlp(self.h)  # [B, N, d]

        scores = self.h.norm(dim=-1)  # [B, N]
        k_active = self._active_count()
        top_idx = torch.topk(scores, k=k_active, dim=1).indices  # [B, k_active]
        active_mask = torch.zeros(bsz, self.N, dtype=torch.float32, device=self.device)
        active_mask.scatter_(1, top_idx, 1.0)

        eff = self.w_slow.unsqueeze(0) + self.f_fast  # [B, N, k_out]

        if self.use_gating:
            q = self.Wq(self.h)  # [B, N, d]
            target_h = self._target_hidden(self.h)  # [B, N, k_out, d]
            k_t = self.Wk(target_h)  # [B, N, k_out, d]
            gate_logits = (q.unsqueeze(2) * k_t).sum(dim=-1) / math.sqrt(float(self.d))
            gate = torch.softmax(gate_logits, dim=-1)  # [B, N, k_out]
        else:
            gate = torch.full(
                (bsz, self.N, self.k_out),
                1.0 / float(self.k_out),
                dtype=torch.float32,
                device=self.device,
            )

        coeff = eff * gate * active_mask.unsqueeze(-1)  # [B, N, k_out]
        edge_msgs = coeff.unsqueeze(-1) * m.unsqueeze(2)  # [B, N, k_out, d]

        offsets = torch.arange(bsz, device=self.device, dtype=torch.long).view(bsz, 1, 1) * self.N
        flat_targets = (self.targets.unsqueeze(0) + offsets).reshape(-1)  # [B*N*k_out]

        agg_flat = torch.zeros(bsz * self.N, self.d, dtype=torch.float32, device=self.device)
        agg_flat.index_add_(0, flat_targets, edge_msgs.reshape(-1, self.d))
        agg = agg_flat.reshape(bsz, self.N, self.d)  # [B, N, d]

        gru_in = agg + x_embed.unsqueeze(1)
        new_h = self.gru(gru_in.reshape(-1, self.d), self.h.reshape(-1, self.d)).reshape(bsz, self.N, self.d)
        self.h = new_h

        pool = new_h.mean(dim=1)  # [B, d]
        logit = pool[:, 0:1]  # [B, 1]
        return logit

    @torch.no_grad()
    def wake_update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        y_t = torch.as_tensor(y_true, dtype=torch.float32, device=self.device).reshape(-1, 1)
        y_p = torch.as_tensor(y_pred, dtype=torch.float32, device=self.device).reshape(-1, 1)
        if y_t.shape[0] != self.h.shape[0]:
            raise ValueError(f"Batch mismatch: y batch {y_t.shape[0]} vs state batch {self.h.shape[0]}")

        mu = (y_t - y_p).unsqueeze(-1)  # [B, 1, 1]
        dot_pre_post = self._edge_pre_post_dot(self.h)  # [B, N, k_out]

        self.e_elig.mul_(self.lambda_e).add_(dot_pre_post)
        delta = mu * self.e_elig
        self.f_fast.mul_(self.lambda_f).add_(self.eta_fast * delta)
        self.u_util.mul_(self.lambda_u).add_((1.0 - self.lambda_u) * delta.abs())

    @torch.no_grad()
    def wake_update_slow_direct(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        y_t = torch.as_tensor(y_true, dtype=torch.float32, device=self.device).reshape(-1, 1)
        y_p = torch.as_tensor(y_pred, dtype=torch.float32, device=self.device).reshape(-1, 1)
        if y_t.shape[0] != self.h.shape[0]:
            raise ValueError(f"Batch mismatch: y batch {y_t.shape[0]} vs state batch {self.h.shape[0]}")

        mu = (y_t - y_p).unsqueeze(-1)  # [B, 1, 1]
        delta = mu * self.e_elig
        self.w_slow.add_(self.eta_wake_slow * delta.mean(dim=0))
        self.w_slow.clamp_(-2.0, 2.0)

    @torch.no_grad()
    def sleep_replay(self, replay_buffer, steps: int = 256, batch_size: int = 16) -> None:
        if steps <= 0 or len(replay_buffer) == 0:
            return

        self.f_fast.zero_()
        self.e_elig.zero_()

        for _ in range(int(steps)):
            x, y, _task = replay_buffer.sample(batch_size=batch_size)
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.step(x)
            y_pred = torch.sigmoid(logits)
            self.wake_update(y, y_pred)

    @torch.no_grad()
    def sleep_consolidate(self) -> None:
        if self.f_fast.numel() == 0:
            return

        mean_fast = self.f_fast.mean(dim=0)
        self.w_slow.add_(self.eta_sleep_slow * torch.tanh(mean_fast))
        self.w_slow.clamp_(-2.0, 2.0)
        self.f_fast.mul_(self.rho_fast_decay)

    @torch.no_grad()
    def sleep_rewire(self, prune_frac: float = 0.10, exploration: float = 0.20) -> None:
        if self.u_util.numel() == 0:
            return
        if prune_frac <= 0.0:
            return

        exploration = float(min(1.0, max(0.0, exploration)))
        prune_count = int(math.floor(self.k_out * float(prune_frac)))
        prune_count = max(0, min(self.k_out, prune_count))
        if prune_count == 0:
            return

        utility = self.u_util.mean(dim=0)  # [N, k_out]
        node_activity = self.h.norm(dim=-1).mean(dim=0)  # [N]
        if float(node_activity.sum().item()) <= 1e-8:
            activity_probs = torch.full((self.N,), 1.0 / float(self.N), device=self.device)
        else:
            activity_probs = node_activity / node_activity.sum()

        for i in range(self.N):
            low_idx = torch.topk(utility[i], k=prune_count, largest=False).indices
            for slot in low_idx.tolist():
                if torch.rand((), device=self.device).item() < exploration:
                    new_j = int(torch.randint(0, self.N, (1,), device=self.device).item())
                else:
                    new_j = int(torch.multinomial(activity_probs, 1).item())

                self.targets[i, slot] = new_j
                self.f_fast[:, i, slot] = 0.0
                self.e_elig[:, i, slot] = 0.0
                self.u_util[:, i, slot] = 0.0
