from __future__ import annotations

import numpy as np
import torch


class HashSlotCueAgent:
    def __init__(
        self,
        slots: int,
        action_dim: int = 2,
        epsilon: float = 0.1,
        eta: float = 0.5,
        decay: float = 0.01,
        baseline_ema: float = 0.95,
        seed: int = 0,
    ):
        self.slots = int(slots)
        self.action_dim = int(action_dim)
        if self.slots <= 0:
            raise ValueError(f"slots must be > 0, got {slots}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if not (0.0 <= float(baseline_ema) <= 1.0):
            raise ValueError(f"baseline_ema must be in [0, 1], got {baseline_ema}")

        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.decay = float(decay)
        self.baseline_ema = float(baseline_ema)
        self.baseline = 0.0
        self._rng = np.random.default_rng(int(seed))

        self.W_slow = torch.zeros((self.slots, self.action_dim), dtype=torch.float32)
        self.W_fast = torch.zeros((self.slots, self.action_dim), dtype=torch.float32)

        self._last_slot_idx = 0
        self._last_probs = torch.full((self.action_dim,), 1.0 / self.action_dim, dtype=torch.float32)
        self._last_action_id = 0

    def _slot_idx(self, cue_id: int) -> int:
        return int(cue_id) % self.slots

    def act(self, cue_onehot: np.ndarray, cue_id: int) -> tuple[int, np.ndarray, int]:
        _ = cue_onehot  # cue_id fully determines slot selection in hash mode.
        slot_idx = self._slot_idx(int(cue_id))

        logits = self.W_slow[slot_idx] + self.W_fast[slot_idx]
        probs = torch.softmax(logits, dim=0)

        if self._rng.random() < self.epsilon:
            action_id = int(self._rng.integers(0, self.action_dim))
        else:
            action_id = int(torch.multinomial(probs, num_samples=1).item())

        self._last_slot_idx = slot_idx
        self._last_probs = probs
        self._last_action_id = action_id
        return action_id, probs.detach().cpu().numpy(), slot_idx

    def update(self, reward: float) -> None:
        advantage = float(reward) - float(self.baseline)

        action_onehot = torch.zeros((self.action_dim,), dtype=torch.float32)
        action_onehot[self._last_action_id] = 1.0
        grad_logits = action_onehot - self._last_probs

        idx = int(self._last_slot_idx)
        self.W_fast[idx] += self.eta * advantage * grad_logits
        self.W_fast[idx] *= (1.0 - self.decay)
        self.baseline = self.baseline_ema * float(self.baseline) + (1.0 - self.baseline_ema) * float(reward)

    def consolidate(self, alpha: float) -> None:
        a = float(alpha)
        idx = int(self._last_slot_idx)
        self.W_slow[idx] += a * self.W_fast[idx]
        self.W_fast[idx] *= (1.0 - a)

    def stats(self) -> dict[str, float]:
        return {
            "slots_used": int(self.slots),
            "wfast_norm": float(torch.norm(self.W_fast).item()),
            "wslow_norm": float(torch.norm(self.W_slow).item()),
        }
