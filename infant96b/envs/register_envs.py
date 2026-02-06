#!/usr/bin/env python3
"""Environment registration helpers."""

from __future__ import annotations

from envs import infant_survive_envspec


ENV_ID = "InfantSurvive-v0"


def _is_gym_registered(env_id: str) -> bool:
    try:
        from gym.envs.registration import registry
    except Exception:
        return False

    if hasattr(registry, "env_specs"):
        return env_id in registry.env_specs
    try:
        return env_id in registry
    except Exception:
        return False


def register_all_envs() -> None:
    spec = infant_survive_envspec.InfantSurvive()
    if _is_gym_registered(ENV_ID):
        return
    try:
        spec.register()
    except Exception:
        if _is_gym_registered(ENV_ID):
            return
        raise
