from __future__ import annotations

import torch


class SleepController:
    def __init__(
        self,
        max_awake: int = 500,
        thresh_S: float = 0.20,
        thresh_D: float = 0.05,
        sleep_steps: int = 1,
        ema_beta: float = 0.95,
        fatigue_gain: float = 0.01,
        fatigue_decay_on_sleep: float = 0.5,
    ):
        if max_awake <= 0:
            raise ValueError(f"max_awake must be > 0, got {max_awake!r}")
        if sleep_steps <= 0:
            raise ValueError(f"sleep_steps must be > 0, got {sleep_steps!r}")

        self.max_awake = int(max_awake)
        self.thresh_S = float(thresh_S)
        self.thresh_D = float(thresh_D)
        self.sleep_steps = int(sleep_steps)
        self.ema_beta = float(ema_beta)
        self.fatigue_gain = float(fatigue_gain)
        self.fatigue_decay_on_sleep = float(fatigue_decay_on_sleep)

        self.awake_steps = 0
        self.fatigue = 0.0
        self.ma_synaptic_load = 0.0
        self.ma_drift = 0.0
        self.last_synaptic_load = 0.0
        self.last_drift = 0.0
        self._has_ema = False
        self._prev_pooled: torch.Tensor | None = None

    def _compute_drift(self, pooled_state: torch.Tensor | None) -> float:
        if pooled_state is None:
            return 0.0
        pooled = pooled_state.detach().to("cpu", dtype=torch.float32).reshape(-1)
        if self._prev_pooled is None or self._prev_pooled.shape != pooled.shape:
            self._prev_pooled = pooled
            return 0.0
        drift = float((pooled - self._prev_pooled).abs().mean().item())
        self._prev_pooled = pooled
        return drift

    def update_and_maybe_sleep(
        self,
        *,
        synaptic_load: float,
        pooled_state: torch.Tensor | None = None,
    ) -> tuple[bool, int]:
        self.awake_steps += 1
        drift = self._compute_drift(pooled_state)

        syn_load = float(synaptic_load)
        self.last_synaptic_load = syn_load
        self.last_drift = drift

        if not self._has_ema:
            self.ma_synaptic_load = syn_load
            self.ma_drift = drift
            self._has_ema = True
        else:
            self.ma_synaptic_load = self.ema_beta * self.ma_synaptic_load + (1.0 - self.ema_beta) * syn_load
            self.ma_drift = self.ema_beta * self.ma_drift + (1.0 - self.ema_beta) * drift

        self.fatigue = (1.0 - self.fatigue_gain) * self.fatigue + self.fatigue_gain * (syn_load + drift)

        should_sleep = (
            self.awake_steps >= self.max_awake
            or self.ma_synaptic_load > self.thresh_S
            or self.ma_drift > self.thresh_D
        )
        if should_sleep:
            self.awake_steps = 0
            self.fatigue *= self.fatigue_decay_on_sleep
            return True, self.sleep_steps
        return False, 0
