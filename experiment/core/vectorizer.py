from __future__ import annotations

import math

from .schema import TacticalObservation

DEFAULT_VECTOR_KEYS = [
    "self_hp_norm",
    "self_alive",
    "self_x_norm",
    "self_y_norm",
    "nearest_enemy_dx_norm",
    "nearest_enemy_dy_norm",
    "nearest_enemy_dist_norm",
    "nearest_enemy_hp_norm",
    "nearest_enemy_alive",
    "nearest_enemy_los",
    "nearest_ally_dx_norm",
    "nearest_ally_dy_norm",
    "nearest_ally_dist_norm",
    "nearest_ally_hp_norm",
    "nearest_ally_alive",
    "enemy_count_norm",
    "ally_under_pressure",
    "self_under_pressure",
    "recent_damage_taken",
    "recent_damage_dealt",
]


def _clean_number(value: float) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def vectorize_observation(obs: TacticalObservation, obs_dim: int = 20) -> list[float]:
    values = [_clean_number(value) for value in obs.get("vector", [])]
    if len(values) >= obs_dim:
        return values[:obs_dim]
    return values + [0.0] * (obs_dim - len(values))


def build_observation_vector(
    *,
    self_hp: float,
    max_hp: float,
    self_alive: bool,
    self_x: float,
    self_y: float,
    map_width: float,
    map_height: float,
    nearest_enemy: dict | None,
    nearest_ally: dict | None,
    visible_enemy_count: int,
    recent_damage_taken: float = 0.0,
    recent_damage_dealt: float = 0.0,
) -> list[float]:
    scale = max(map_width, map_height, 1.0)

    def entity_features(entity: dict | None) -> tuple[float, float, float, float, float, float]:
        if not entity:
            return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        rel = entity["relative_position"]
        return (
            _clean_number(rel["x"] / scale),
            _clean_number(rel["y"] / scale),
            _clean_number(entity["distance"] / scale),
            _clean_number(entity["hp"] / max(max_hp, 1.0)),
            1.0 if entity["alive"] else 0.0,
            1.0 if entity.get("has_line_of_sight", True) else 0.0,
        )

    enemy_dx, enemy_dy, enemy_dist, enemy_hp, enemy_alive, enemy_los = entity_features(nearest_enemy)
    ally_dx, ally_dy, ally_dist, ally_hp, ally_alive, _ = entity_features(nearest_ally)

    return [
        _clean_number(self_hp / max(max_hp, 1.0)),
        1.0 if self_alive else 0.0,
        _clean_number(self_x / max(map_width, 1.0)),
        _clean_number(self_y / max(map_height, 1.0)),
        enemy_dx,
        enemy_dy,
        enemy_dist,
        enemy_hp,
        enemy_alive,
        enemy_los,
        ally_dx,
        ally_dy,
        ally_dist,
        ally_hp,
        ally_alive,
        _clean_number(visible_enemy_count / 4.0),
        0.0,
        1.0 if enemy_dist < 0.20 else 0.0,
        _clean_number(recent_damage_taken / max(max_hp, 1.0)),
        _clean_number(recent_damage_dealt / max(max_hp, 1.0)),
    ]

