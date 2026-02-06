#!/usr/bin/env python3
"""Environment factory."""

from __future__ import annotations

import os

import gym

from envs.register_envs import register_all_envs
from envs.wrappers import MilestoneRewardWrapper, ObsSchemaNormalizeWrapper


def make_env(env_id: str, headless: bool = True, **kwargs):
    if env_id.startswith("InfantSurvive"):
        register_all_envs()

    if headless:
        os.environ.setdefault("MINERL_HEADLESS", "1")

    env = gym.make(env_id, **kwargs)
    env = ObsSchemaNormalizeWrapper(env)

    if env_id == "InfantSurvive-v0":
        env = MilestoneRewardWrapper(env)

    return env
