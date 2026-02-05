#!/usr/bin/env python3
"""InfantSurvive-v0 MineRL environment spec."""

from __future__ import annotations

import importlib
from typing import Iterable, List

from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero import handlers, mc


LOCKED_ITEM_NAMES = [
    "log",
    "planks",
    "stick",
    "cobblestone",
    "coal",
    "iron_ore",
    "iron_ingot",
    "crafting_table",
    "furnace",
    "wooden_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wooden_axe",
    "stone_axe",
    "iron_axe",
    "torch",
    "bed",
    "wool",
    "bread",
]


def _resolve_mc_item(name: str):
    if hasattr(mc, "ItemType") and hasattr(mc.ItemType, name.upper()):
        return getattr(mc.ItemType, name.upper())
    if hasattr(mc, name.upper()):
        return getattr(mc, name.upper())
    return name


def _resolve_mc_block(name: str):
    if hasattr(mc, "BlockType") and hasattr(mc.BlockType, name.upper()):
        return getattr(mc.BlockType, name.upper())
    if hasattr(mc, name.upper()):
        return getattr(mc, name.upper())
    return name


LOCKED_ITEMS = [_resolve_mc_item(name) for name in LOCKED_ITEM_NAMES]


class InfantSurvive(SimpleEmbodimentEnvSpec):
    def __init__(self, **kwargs):
        super().__init__(
            name="InfantSurvive-v0",
            max_episode_steps=36000,
            reward_threshold=999999,
            **kwargs,
        )

    def create_server_world_generators(self) -> List:
        return [handlers.DefaultWorldGenerator()]

    def create_server_decorators(self) -> List:
        air = _resolve_mc_block("air")
        stone = _resolve_mc_block("stone")
        torch = _resolve_mc_block("torch")
        return [
            handlers.DrawingDecorator(
                [
                    handlers.DrawCuboid((-3, 64, -3), (3, 84, 3), air),
                    handlers.DrawCuboid((-3, 63, -3), (3, 63, 3), stone),
                    handlers.DrawBlock((0, 64, 0), torch),
                ]
            )
        ]

    def create_server_initial_conditions(self) -> List:
        return [
            handlers.TimeInitialCondition(True, 23000),
            handlers.WeatherInitialCondition("clear"),
            handlers.SpawningInitialCondition(True),
        ]

    def create_agent_handlers(self) -> List:
        handlers_list = super().create_agent_handlers()

        handlers_list += [
            handlers.StartingHealthAgentStart(20),
            handlers.StartingFoodAgentStart(20),
            handlers.SimpleInventoryAgentStart(
                [
                    {
                        "type": "bread",
                        "quantity": 4,
                    }
                ]
            ),
            handlers.AgentStartPlacement(0, 64, 0, 0, 0),
        ]

        handlers_list += [
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats(),
            handlers.ObservationFromDamageSource(),
            handlers.FlatInventoryObservation(LOCKED_ITEMS),
        ]

        handlers_list += [
            handlers.KeybasedCommandAction("use"),
            handlers.CraftAction(LOCKED_ITEMS),
            handlers.CraftNearbyAction(LOCKED_ITEMS),
            handlers.SmeltItemNearby(LOCKED_ITEMS),
            handlers.EquipAction(LOCKED_ITEMS),
            handlers.PlaceBlock(LOCKED_ITEMS),
        ]

        return handlers_list

    def create_server_quit_producers(self) -> List:
        return [handlers.ServerQuitFromTimeUp(1800000)]

    def determine_success_from_rewards(self, rewards: Iterable[float]) -> bool:
        return False

    def is_from_folder(self, folder: str) -> bool:
        return False

    def get_docstring(self) -> str:
        return "InfantSurvive-v0: minimal survival sandbox for custom experiments."


def _resolve_register_env_spec():
    candidates = [
        "minerl.herobraine.env_spec",
        "minerl.herobraine.env_specs",
        "minerl.herobraine.envs",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr in ("register_env_spec", "register"):
            if hasattr(module, attr):
                return getattr(module, attr)
    raise ImportError("Could not find a MineRL env spec register function")


def register():
    register_env_spec = _resolve_register_env_spec()
    register_env_spec(InfantSurvive())


InfantSurviveEnvSpec = InfantSurvive
