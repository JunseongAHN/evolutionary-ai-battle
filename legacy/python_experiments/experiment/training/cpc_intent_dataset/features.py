from __future__ import annotations

from math import isfinite
from typing import Dict, List

DATASET_SCHEMA_VERSION = "intent-dataset-v0.2"
FEATURE_SCHEMA_VERSION = "intent-feature-schema-v0.2"
LABEL_SCHEMA_VERSION = "intent-label-schema-v0.1"

DEFAULT_THRESHOLDS = {
    "low_hp": 35,
    "enemy_threat_range": 5,
    "support_range": 4,
    "isolation_range": 6,
    "fire_range": 5,
    "grid_width": 20,
    "grid_height": 20,
}

FEATURE_NAMES = [
    "selfHpNorm",
    "allyHpNorm",
    "allyDistanceNorm",
    "enemy0HpNorm",
    "enemy0DistanceNorm",
    "enemy1HpNorm",
    "enemy1DistanceNorm",
]

FEATURE_TYPES = {
    "selfHpNorm": "normalized",
    "allyHpNorm": "normalized",
    "allyDistanceNorm": "normalized",
    "enemy0HpNorm": "normalized",
    "enemy0DistanceNorm": "normalized",
    "enemy1HpNorm": "normalized",
    "enemy1DistanceNorm": "normalized",
}

LABELS = [
    "attack_nearest_enemy",
    "support_teammate_under_pressure",
    "reduce_isolation",
    "retreat_when_low_hp",
]

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def manhattan_distance(a: Dict[str, int], b: Dict[str, int]) -> int:
    return abs(int(a["x"]) - int(b["x"])) + abs(int(a["y"]) - int(b["y"]))


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def normalize_hp(hp: int) -> float:
    return round(clamp(int(hp), 0, 100) / 100.0, 6)


def normalize_distance(distance: int, grid_width: int, grid_height: int) -> float:
    max_distance = max(int(grid_width), int(grid_height), 1)
    return round(min(clamp(int(distance), 0, max_distance), max_distance) / float(max_distance), 6)


def can_fire_from_cooldown(weapon_cooldown_steps: int, alive: bool) -> int:
    return int(bool(alive) and int(weapon_cooldown_steps) <= 0)


def feature_vector_from_features(features: Dict[str, int | float]) -> List[int | float]:
    return [features[name] for name in FEATURE_NAMES]


def extract_model_features(state: Dict[str, object], thresholds: Dict[str, int] | None = None) -> Dict[str, int | float]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    players = list(state.get("players", []))
    self_player = next((player for player in players if player.get("role") == "self"), None)
    ally_player = next((player for player in players if player.get("role") == "ally"), None)
    enemy_players = [player for player in players if player.get("role") == "enemy"]

    if self_player is None:
        raise ValueError("state.players must contain a player with role='self'")
    if ally_player is None:
        raise ValueError("state.players must contain a player with role='ally'")
    if len(enemy_players) < 2:
        raise ValueError("state.players must contain two enemy players")

    enemy0 = enemy_players[0]
    enemy1 = enemy_players[1]

    grid_width = int(thresholds["grid_width"])
    grid_height = int(thresholds["grid_height"])
    scale = max(grid_width, grid_height, 1)

    self_xy = {"x": int(self_player["x"]), "y": int(self_player["y"])}
    ally_xy = {"x": int(ally_player["x"]), "y": int(ally_player["y"])}
    enemy0_xy = {"x": int(enemy0["x"]), "y": int(enemy0["y"])}
    enemy1_xy = {"x": int(enemy1["x"]), "y": int(enemy1["y"])}

    ally_distance = manhattan_distance(self_xy, ally_xy)
    enemy0_distance = manhattan_distance(self_xy, enemy0_xy)
    enemy1_distance = manhattan_distance(self_xy, enemy1_xy)

    features = {
        "selfHpNorm": normalize_hp(self_player.get("hp", 0)),
        "allyHpNorm": normalize_hp(ally_player.get("hp", 0)),
        "allyDistanceNorm": normalize_distance(ally_distance, scale, scale),
        "enemy0HpNorm": normalize_hp(enemy0.get("hp", 0)),
        "enemy0DistanceNorm": normalize_distance(enemy0_distance, scale, scale),
        "enemy1HpNorm": normalize_hp(enemy1.get("hp", 0)),
        "enemy1DistanceNorm": normalize_distance(enemy1_distance, scale, scale),
    }

    for key, value in features.items():
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"feature {key} is not finite")
    return features
