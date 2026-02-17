from __future__ import annotations

import numpy as np
import torch


class PlasticCueAgent:
    def __init__(
        self,
        K: int,
        eta: float = 0.5,
        decay: float = 0.01,
        epsilon: float = 0.1,
        sleep_every: int = 0,
        sleep_alpha: float = 0.1,
        seed: int = 0,
    ):
        self.K = int(K)
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {K}")

        self.eta = float(eta)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.sleep_every = int(sleep_every)
        self.sleep_alpha = float(sleep_alpha)
        self.baseline = 0.0
        self._rng = np.random.default_rng(int(seed))

        self.W_slow = torch.zeros((self.K, 2), dtype=torch.float32)
        self.W_fast = torch.zeros((self.K, 2), dtype=torch.float32)

        self._last_cue = torch.zeros((self.K,), dtype=torch.float32)
        self._last_probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        self._last_action_id = 0

    def _cue_tensor(self, cue_onehot: np.ndarray) -> torch.Tensor:
        cue = torch.as_tensor(cue_onehot, dtype=torch.float32).flatten()
        if cue.numel() < self.K:
            padded = torch.zeros((self.K,), dtype=torch.float32)
            padded[: cue.numel()] = cue
            cue = padded
        elif cue.numel() > self.K:
            cue = cue[: self.K]
        return cue

    def act(self, cue_onehot: np.ndarray) -> tuple[int, np.ndarray]:
        cue = self._cue_tensor(cue_onehot)
        logits = cue @ (self.W_slow + self.W_fast)
        probs = torch.softmax(logits, dim=0)

        if self._rng.random() < self.epsilon:
            action_id = int(self._rng.integers(0, 2))
        else:
            action_id = int(torch.multinomial(probs, num_samples=1).item())

        self._last_cue = cue
        self._last_probs = probs
        self._last_action_id = action_id
        return action_id, probs.detach().cpu().numpy()

    def update(self, reward: float) -> None:
        advantage = float(reward) - float(self.baseline)

        action_onehot = torch.zeros((2,), dtype=torch.float32)
        action_onehot[self._last_action_id] = 1.0
        grad_logits = action_onehot - self._last_probs

        self.W_fast += self.eta * advantage * torch.outer(self._last_cue, grad_logits)
        self.W_fast *= (1.0 - self.decay)
        self.baseline = 0.95 * float(self.baseline) + 0.05 * float(reward)

    def consolidate(self, alpha: float) -> None:
        a = float(alpha)
        self.W_slow += a * self.W_fast
        self.W_fast *= (1.0 - a)

    def maybe_sleep(self, step: int) -> bool:
        if self.sleep_every > 0 and step > 0 and step % self.sleep_every == 0:
            self.consolidate(self.sleep_alpha)
            return True
        return False
