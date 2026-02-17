from __future__ import annotations

import numpy as np
import torch


class SlotPlasticCueAgent:
    def __init__(
        self,
        K: int,
        slots: int,
        eta: float,
        decay: float,
        epsilon: float,
        utility_decay: float = 0.99,
        utility_bonus: float = 1.0,
        seed: int = 0,
    ):
        self.K = int(K)
        self.slots = int(slots)
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if self.slots <= 0:
            raise ValueError(f"slots must be > 0, got {slots}")

        self.eta = float(eta)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.utility_decay = float(utility_decay)
        self.utility_bonus = float(utility_bonus)
        self.baseline = 0.0
        self._rng = np.random.default_rng(int(seed))

        self.slot_cue = np.full((self.slots,), fill_value=-1, dtype=np.int64)
        self.W_slow = torch.zeros((self.slots, 2), dtype=torch.float32)
        self.W_fast = torch.zeros((self.slots, 2), dtype=torch.float32)
        self.utility = torch.zeros((self.slots,), dtype=torch.float32)

        self._last_slot_idx = 0
        self._last_probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        self._last_action_id = 0

    def _find_or_assign_slot(self, cue_id: int) -> tuple[int, bool]:
        cue_int = int(cue_id)
        matched = np.where(self.slot_cue == cue_int)[0]
        if matched.size > 0:
            return int(matched[0]), False

        free = np.where(self.slot_cue == -1)[0]
        if free.size > 0:
            idx = int(free[0])
        else:
            idx = int(torch.argmin(self.utility).item())

        self.slot_cue[idx] = cue_int
        self.W_fast[idx].zero_()
        self.W_slow[idx].zero_()
        self.utility[idx] = 0.0
        return idx, True

    def act(self, cue_onehot: np.ndarray, cue_id: int) -> tuple[int, np.ndarray, int, bool]:
        _ = cue_onehot  # cue identity is carried explicitly by cue_id in slot mode.
        slot_idx, rewired = self._find_or_assign_slot(int(cue_id))

        logits = self.W_slow[slot_idx] + self.W_fast[slot_idx]
        probs = torch.softmax(logits, dim=0)

        if self._rng.random() < self.epsilon:
            action_id = int(self._rng.integers(0, 2))
        else:
            action_id = int(torch.multinomial(probs, num_samples=1).item())

        self._last_slot_idx = int(slot_idx)
        self._last_probs = probs
        self._last_action_id = int(action_id)
        return int(action_id), probs.detach().cpu().numpy(), int(slot_idx), bool(rewired)

    def update(self, reward: float) -> None:
        advantage = float(reward) - float(self.baseline)

        action_onehot = torch.zeros((2,), dtype=torch.float32)
        action_onehot[self._last_action_id] = 1.0
        grad_logits = action_onehot - self._last_probs

        self.W_fast[self._last_slot_idx] += self.eta * advantage * grad_logits
        self.W_fast[self._last_slot_idx] *= (1.0 - self.decay)
        self.utility[self._last_slot_idx] = (
            self.utility_decay * self.utility[self._last_slot_idx] + self.utility_bonus * abs(advantage)
        )
        self.baseline = 0.95 * float(self.baseline) + 0.05 * float(reward)

    def consolidate(self, alpha: float) -> None:
        a = float(alpha)
        idx = int(self._last_slot_idx)
        self.W_slow[idx] += a * self.W_fast[idx]
        self.W_fast[idx] *= (1.0 - a)

    def stats(self) -> dict[str, float]:
        used_mask = self.slot_cue != -1
        slots_used = int(np.sum(used_mask))

        if slots_used > 0:
            used_indices = np.where(used_mask)[0]
            wfast_norm = float(torch.norm(self.W_fast[used_indices]).item())
            wslow_norm = float(torch.norm(self.W_slow[used_indices]).item())
        else:
            wfast_norm = 0.0
            wslow_norm = 0.0

        return {
            "slots_used": slots_used,
            "wfast_norm": wfast_norm,
            "wslow_norm": wslow_norm,
        }
