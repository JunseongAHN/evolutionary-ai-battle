from __future__ import annotations

import math
from typing import Iterator

try:
    from experiment.training.cpc_actions import MOVE_BINS, decode_action
except ModuleNotFoundError:
    from training.cpc_actions import MOVE_BINS, decode_action


def get_move_bin_vectors() -> dict[int, tuple[float, float]]:
    """
    Return normalized env movement vectors by move_bin.

    This matches training.cpc_actions.decode_action:
    0 is STAY, +x moves right, and +y moves down in the env/world
    coordinate frame. Diagonal bins are normalized by the env action decoder.
    """
    vectors: dict[int, tuple[float, float]] = {}
    for move_bin in range(MOVE_BINS):
        decoded = decode_action({"move": move_bin, "aim": 0, "fire": 0})
        vectors[move_bin] = (float(decoded["moveX"]), float(decoded["moveY"]))
    return vectors


def iter_move_candidates() -> Iterator[tuple[int, tuple[float, float]]]:
    """Yield move candidates in deterministic move_bin order."""
    for move_bin, vector in sorted(get_move_bin_vectors().items()):
        yield move_bin, vector


def normalize_vector(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(float(dx), float(dy))
    if length <= 1e-9:
        return 0.0, 0.0
    return float(dx) / length, float(dy) / length


def simulate_candidate_position(
    current_x: float,
    current_y: float,
    move_dx: float,
    move_dy: float,
    move_speed: float,
    dt: float,
) -> tuple[float, float]:
    """Project a candidate center position without clamping to the map."""
    return (
        float(current_x) + (float(move_dx) * float(move_speed) * float(dt)),
        float(current_y) + (float(move_dy) * float(move_speed) * float(dt)),
    )
