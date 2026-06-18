from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, TypedDict


MOVE_BINS = 9
AIM_BINS = 16
FIRE_BINS = 2


class RawAction(TypedDict):
    move: int
    aim: int
    fire: int


class EngineAction(TypedDict):
    moveX: float
    moveY: float
    aimX: float
    aimY: float
    fire: int


@dataclass(frozen=True)
class ActionSpace:
    move_bins: int = MOVE_BINS
    aim_bins: int = AIM_BINS
    fire_bins: int = FIRE_BINS


MOVE_VECTORS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, -1.0),
    2: (0.0, 1.0),
    3: (-1.0, 0.0),
    4: (1.0, 0.0),
    5: (-1.0, -1.0),
    6: (1.0, -1.0),
    7: (-1.0, 1.0),
    8: (1.0, 1.0),
}


MOVE_LABELS: dict[int, str] = {
    0: "no_move",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "up_left",
    6: "up_right",
    7: "down_left",
    8: "down_right",
}


def action_space() -> ActionSpace:
    return ActionSpace()


def normalize_move(move_x: float, move_y: float) -> tuple[float, float]:
    length = math.hypot(move_x, move_y)
    if length <= 1e-6:
        return 0.0, 0.0
    return move_x / length, move_y / length


def decode_action(action: Mapping[str, int]) -> EngineAction:
    move = int(action["move"])
    aim = int(action["aim"])
    fire = int(action["fire"])

    if move not in MOVE_VECTORS:
        raise ValueError(f"move must be in [0, {MOVE_BINS - 1}], got {move}")
    if not 0 <= aim < AIM_BINS:
        raise ValueError(f"aim must be in [0, {AIM_BINS - 1}], got {aim}")
    if fire not in (0, 1):
        raise ValueError(f"fire must be 0 or 1, got {fire}")

    move_x, move_y = normalize_move(*MOVE_VECTORS[move])
    theta = (2.0 * math.pi * aim) / AIM_BINS
    return {
        "moveX": move_x,
        "moveY": move_y,
        "aimX": math.cos(theta),
        "aimY": math.sin(theta),
        "fire": fire,
    }


def random_action(rng: random.Random | None = None) -> RawAction:
    rng = rng or random
    return {
        "move": rng.randrange(MOVE_BINS),
        "aim": rng.randrange(AIM_BINS),
        "fire": rng.randrange(FIRE_BINS),
    }


def describe_action(action: Mapping[str, int]) -> dict:
    decoded = decode_action(action)
    return {
        "rawAction": dict(action),
        "moveLabel": MOVE_LABELS[int(action["move"])],
        "decoded": decoded,
    }
