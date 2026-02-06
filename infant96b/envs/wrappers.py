#!/usr/bin/env python3
"""Custom wrappers for MineRL environments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import gym
import numpy as np


class ObsSchemaNormalizeWrapper(gym.ObservationWrapper):
    """Normalize MineRL observations into a consistent dict schema."""

    def observation(self, observation):
        if not isinstance(observation, dict):
            observation = {}

        pov = observation.get("pov")
        if pov is not None:
            pov = np.asarray(pov)
            if pov.dtype != np.uint8:
                pov = pov.astype(np.uint8)

        inventory = observation.get("inventory")
        if not isinstance(inventory, dict):
            inventory = None

        equipped = observation.get("equipped")
        if equipped is None:
            equipped = observation.get("equipped_items")
        if not isinstance(equipped, dict):
            if equipped is not None:
                equipped = None

        life_stats = observation.get("life_stats")
        if life_stats is None:
            life_stats = observation.get("lifestats")
        if not isinstance(life_stats, dict):
            if life_stats is not None:
                life_stats = None

        location = observation.get("location")
        if location is None:
            location = observation.get("location_stats")
        if not isinstance(location, dict):
            if location is not None:
                location = None

        return {
            "pov": pov,
            "inventory": inventory,
            "equipped": equipped,
            "life_stats": life_stats,
            "location": location,
        }


class MilestoneRewardWrapper(gym.Wrapper):
    """Grant milestone rewards based on inventory counts crossing from 0 to >=1."""

    MILESTONES: Dict[str, int] = {
        "log": 1,
        "cobblestone": 2,
        "crafting_table": 4,
        "wooden_pickaxe": 4,
        "stone_pickaxe": 6,
        "furnace": 6,
        "iron_ingot": 8,
        "iron_pickaxe": 12,
        "bed": 6,
    }

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._completed = set()
        self._prev_inventory = {name: 0 for name in self.MILESTONES}

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        self._completed = set()
        self._prev_inventory = self._extract_inventory(obs)

        return out

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            legacy = False
        else:
            obs, reward, done, info = out
            terminated = done
            truncated = False
            legacy = True

        milestone_reward, unlocked = self._compute_milestones(obs)
        reward = reward + milestone_reward

        if info is None:
            info = {}
        info["milestone_reward"] = milestone_reward
        info["milestones_unlocked"] = unlocked

        if legacy:
            return obs, reward, done, info
        return obs, reward, terminated, truncated, info

    def _extract_inventory(self, obs) -> Dict[str, int]:
        inventory: Dict[str, int] = {name: 0 for name in self.MILESTONES}
        if not isinstance(obs, dict):
            return inventory
        inv = obs.get("inventory")
        if not isinstance(inv, dict):
            return inventory
        for name in inventory:
            value = inv.get(name, 0)
            try:
                inventory[name] = int(value)
            except Exception:
                inventory[name] = 0
        return inventory

    def _compute_milestones(self, obs) -> Tuple[float, List[str]]:
        current = self._extract_inventory(obs)
        unlocked: List[str] = []
        reward = 0.0

        for name, bonus in self.MILESTONES.items():
            if name in self._completed:
                continue
            prev_count = self._prev_inventory.get(name, 0)
            curr_count = current.get(name, 0)
            if prev_count <= 0 and curr_count >= 1:
                self._completed.add(name)
                reward += float(bonus)
                unlocked.append(name)

        self._prev_inventory = current
        return reward, unlocked
