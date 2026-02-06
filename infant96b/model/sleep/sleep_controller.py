#!/usr/bin/env python3
"""Rule-based sleep controller for plastic graph experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SleepDecision:
    should_sleep: bool
    sleep_steps: int


class SleepController:
    """Tracks fatigue, synaptic load, and drift risk to decide when to sleep."""

    def __init__(
        self,
        synaptic_threshold: float = 0.2,
        drift_threshold: float = 0.05,
        max_awake: int = 500,
        base_sleep_steps: int = 5,
        fatigue_rate: float = 1.0,
    ) -> None:
        self.synaptic_threshold = float(synaptic_threshold)
        self.drift_threshold = float(drift_threshold)
        self.max_awake = int(max_awake)
        self.base_sleep_steps = int(base_sleep_steps)
        self.fatigue_rate = float(fatigue_rate)

        self.fatigue = 0.0
        self.synaptic_load = 0.0
        self.drift_risk = 0.0
        self.awake_steps = 0

    def update(self, synaptic_load: float, drift_risk: float) -> SleepDecision:
        self.synaptic_load = float(synaptic_load)
        self.drift_risk = float(drift_risk)
        self.fatigue += self.fatigue_rate
        self.awake_steps += 1

        should_sleep = (
            self.synaptic_load >= self.synaptic_threshold
            or self.drift_risk >= self.drift_threshold
            or self.awake_steps >= self.max_awake
        )
        sleep_steps = self.base_sleep_steps if should_sleep else 0

        if should_sleep:
            self.awake_steps = 0
            self.fatigue = 0.0

        return SleepDecision(should_sleep=should_sleep, sleep_steps=sleep_steps)
