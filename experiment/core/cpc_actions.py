from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, TypedDict


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


def aim_bin_to_vec(aim_bin: int, num_bins: int = AIM_BINS) -> dict[str, float]:
    theta = (2.0 * math.pi * int(aim_bin)) / int(num_bins)
    return {"x": math.cos(theta), "y": math.sin(theta)}


def vec_to_aim_bin(vec: Mapping[str, float], num_bins: int = AIM_BINS) -> int:
    x = float(vec["x"])
    y = float(vec["y"])
    length = math.hypot(x, y)
    if length <= 1e-6:
        return 0
    theta = math.atan2(y / length, x / length)
    if theta < 0.0:
        theta += 2.0 * math.pi
    return int(round((theta / (2.0 * math.pi)) * int(num_bins))) % int(num_bins)


def circular_bin_distance(a: int, b: int, num_bins: int) -> int:
    distance = abs(int(a) - int(b)) % int(num_bins)
    return min(distance, int(num_bins) - distance)


def decode_action(action: Mapping[str, Any]) -> EngineAction:
    move = int(action["move"])
    fire = int(action["fire"])

    if move not in MOVE_VECTORS:
        raise ValueError(f"move must be in [0, {MOVE_BINS - 1}], got {move}")
    if fire not in (0, 1):
        raise ValueError(f"fire must be 0 or 1, got {fire}")

    move_x, move_y = normalize_move(*MOVE_VECTORS[move])
    aim_vec = _decode_aim_vector(action)
    return {
        "moveX": move_x,
        "moveY": move_y,
        "aimX": aim_vec["x"],
        "aimY": aim_vec["y"],
        "fire": fire,
    }


def _decode_aim_vector(action: Mapping[str, Any]) -> dict[str, float]:
    if "aim_dx" in action or "aim_dy" in action:
        x = float(action.get("aim_dx", 0.0))
        y = float(action.get("aim_dy", 0.0))
        length = math.hypot(x, y)
        if length <= 1e-6:
            raise ValueError("continuous aim direction must be non-zero")
        return {"x": x / length, "y": y / length}
    if "aim_angle" in action:
        angle = float(action["aim_angle"])
        return {"x": math.cos(angle), "y": math.sin(angle)}
    if "aim_x" in action or "aim_y" in action:
        x = float(action.get("aim_x", 0.0))
        y = float(action.get("aim_y", 0.0))
        length = math.hypot(x, y)
        if length <= 1e-6:
            raise ValueError("continuous aim direction must be non-zero")
        return {"x": x / length, "y": y / length}
    if "aim" not in action:
        raise ValueError("action must include aim_dx/aim_dy, aim_x/aim_y, aim_angle, or legacy aim")
    aim = int(action["aim"])
    if not 0 <= aim < AIM_BINS:
        raise ValueError(f"aim must be in [0, {AIM_BINS - 1}], got {aim}")
    return aim_bin_to_vec(aim, AIM_BINS)


def random_action(rng: random.Random | None = None) -> RawAction:
    rng = rng or random
    return {
        "move": rng.randrange(MOVE_BINS),
        "aim": rng.randrange(AIM_BINS),
        "fire": rng.randrange(FIRE_BINS),
    }


def describe_action(action: Mapping[str, Any]) -> dict:
    decoded = decode_action(action)
    return {
        "rawAction": dict(action),
        "moveLabel": MOVE_LABELS[int(action["move"])],
        "decoded": decoded,
    }
