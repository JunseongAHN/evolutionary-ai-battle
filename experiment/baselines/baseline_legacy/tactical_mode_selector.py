from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from experiment.baselines.aim_oracle.enemy_cell_utils import find_nearest_enemy_cell
    from experiment.core.local_occupancy_grid import CHANNEL_ENEMY, CHANNEL_OBSTACLE
except ModuleNotFoundError:
    from baselines.aim_oracle.enemy_cell_utils import find_nearest_enemy_cell
    from core.local_occupancy_grid import CHANNEL_ENEMY, CHANNEL_OBSTACLE

from .local_grid_los import has_grid_line_of_sight
from .local_grid_pathfinding import build_blocked_grid


TACTICAL_MODES = ["engage", "kite", "hold_range", "reposition"]
_GRID_KEYS = ("local_occupancy_grid", "local_grid", "occupancy_grid", "grid")


class RuleBasedTacticalModeSelector:
    """Small deterministic policy for choosing which movement tactic to execute."""

    def __init__(
        self,
        weapon_range: float = 280.0,
        ideal_range_ratio: float = 0.7,
        low_hp_ratio: float = 0.35,
    ) -> None:
        self.default_weapon_range = max(1.0, float(weapon_range))
        self.ideal_range_ratio = max(0.05, float(ideal_range_ratio))
        self.low_hp_ratio = max(0.0, float(low_hp_ratio))

    def select_mode(self, obs: Any, state_snapshot: Any | None = None) -> tuple[str, dict[str, Any]]:
        observation = _mapping(obs)
        snapshot = _snapshot(state_snapshot)
        self_pos = _position(observation.get("self_pos"), _mapping(snapshot.get("state")).get("self_pos"), _agent(snapshot, "self").get("position"))
        enemy_pos = _position(observation.get("enemy_pos"), _mapping(snapshot.get("state")).get("enemy_pos"), _agent(snapshot, "enemy").get("position"))
        grid = next((observation.get(key) for key in _GRID_KEYS if observation.get(key) is not None), None)
        enemy_cell = None
        if grid is not None:
            enemy_channel = _channel_index(grid, CHANNEL_ENEMY)
            if enemy_channel is not None:
                enemy_cell = find_nearest_enemy_cell(grid, enemy_channel)
        enemy_exists = _enemy_exists(observation, snapshot, enemy_pos, enemy_cell)
        enemy_dist = _number(
            observation.get("distance_to_enemy"),
            _mapping(snapshot.get("distances")).get("self_to_enemy"),
        )
        if enemy_dist is None and self_pos is not None and enemy_pos is not None:
            enemy_dist = math.hypot(enemy_pos["x"] - self_pos["x"], enemy_pos["y"] - self_pos["y"])
        if enemy_dist is None and grid is not None and enemy_cell is not None:
            center = _center_cell(grid)
            cell_size = _number(
                grid.get("cell_size") if isinstance(grid, Mapping) else getattr(grid, "cell_size", None)
            ) or 1.0
            enemy_dist = math.hypot(enemy_cell[0] - center[0], enemy_cell[1] - center[1]) * cell_size
        weapon_range = _number(
            observation.get("weapon_range"),
            observation.get("fire_range"),
            _mapping(snapshot.get("combat")).get("fire_range"),
            _mapping(snapshot.get("combat")).get("bullet_range"),
        ) or self.default_weapon_range
        ideal_range = weapon_range * self.ideal_range_ratio
        line_of_sight, los_source = _line_of_sight(observation, snapshot, self_pos, enemy_pos)
        self_hp_low = _self_hp_low(observation, snapshot, self.low_hp_ratio)

        if not enemy_exists or enemy_dist is None:
            mode, reason = "reposition", "no_live_enemy"
        elif self_hp_low and enemy_dist < weapon_range:
            mode, reason = "kite", "low_hp_enemy_in_range"
        elif line_of_sight is False and enemy_dist <= weapon_range * 1.2:
            mode, reason = "reposition", "line_of_sight_blocked"
        elif enemy_dist > weapon_range:
            mode, reason = "engage", "enemy_out_of_weapon_range"
        elif enemy_dist < ideal_range * 0.6:
            mode, reason = "kite", "enemy_inside_minimum_spacing"
        else:
            mode, reason = "hold_range", "enemy_in_tactical_range"

        return mode, {
            "mode": mode,
            "enemy_exists": bool(enemy_exists),
            "enemy_dist": enemy_dist,
            "weapon_range": float(weapon_range),
            "ideal_range": float(ideal_range),
            "line_of_sight": line_of_sight,
            "self_hp_low": bool(self_hp_low),
            "reason": reason,
            "sources": {"line_of_sight": los_source},
        }


def _line_of_sight(
    obs: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    self_pos: dict[str, float] | None,
    enemy_pos: dict[str, float] | None,
) -> tuple[bool | None, str]:
    explicit = _bool_or_none(obs.get("line_of_sight"), obs.get("has_line_of_sight"))
    if explicit is not None:
        return explicit, "observation"
    grid = next((obs.get(key) for key in _GRID_KEYS if obs.get(key) is not None), None)
    if grid is not None:
        obstacle_channel = _channel_index(grid, CHANNEL_OBSTACLE)
        enemy_channel = _channel_index(grid, CHANNEL_ENEMY)
        if obstacle_channel is not None and enemy_channel is not None:
            enemy_cell = find_nearest_enemy_cell(grid, enemy_channel)
            if enemy_cell is not None:
                return has_grid_line_of_sight(_center_cell(grid), enemy_cell, build_blocked_grid(grid, obstacle_channel)), "local_grid"
    if self_pos is None or enemy_pos is None:
        return None, "missing_positions"
    obstacles = list(snapshot.get("obstacles", []) or [])
    obstacles.extend(_mapping(snapshot.get("map")).get("obstacles", []) or [])
    blocked = any(
        _segment_intersects_circle(self_pos, enemy_pos, obstacle)
        for obstacle in obstacles
        if isinstance(obstacle, Mapping) and str(obstacle.get("type", "circle")) == "circle"
    )
    return not blocked, "circle_obstacles"


def _self_hp_low(obs: Mapping[str, Any], snapshot: Mapping[str, Any], ratio: float) -> bool:
    explicit = _bool_or_none(obs.get("self_low_hp"), _mapping(snapshot.get("predicates")).get("self_low_hp"))
    if explicit is not None:
        return explicit
    hp = _number(obs.get("self_hp"), _agent(snapshot, "self").get("hp"), _mapping(snapshot.get("state")).get("self_hp"))
    max_hp = _number(obs.get("self_max_hp"), _agent(snapshot, "self").get("max_hp")) or 100.0
    return hp is not None and hp <= max_hp * ratio


def _enemy_exists(
    obs: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    enemy_pos: Any,
    enemy_cell: tuple[int, int] | None,
) -> bool:
    hp = _number(obs.get("enemy_hp"), _agent(snapshot, "enemy").get("hp"), _mapping(snapshot.get("state")).get("enemy_hp"))
    if hp is not None:
        return hp > 0.0
    alive = _bool_or_none(_agent(snapshot, "enemy").get("alive"))
    return alive if alive is not None else enemy_pos is not None or enemy_cell is not None


def _segment_intersects_circle(start: Mapping[str, float], end: Mapping[str, float], obstacle: Mapping[str, Any]) -> bool:
    sx, sy = start["x"], start["y"]
    dx, dy = end["x"] - sx, end["y"] - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return False
    cx, cy = float(obstacle.get("x", 0.0)), float(obstacle.get("y", 0.0))
    t = max(0.0, min(1.0, ((cx - sx) * dx + (cy - sy) * dy) / length_sq))
    radius = max(0.0, float(obstacle.get("radius", 0.0)))
    return math.hypot(sx + t * dx - cx, sy + t * dy - cy) <= radius


def _channel_index(grid: Any, name: str) -> int | None:
    if hasattr(grid, "channel_index"):
        try:
            return int(grid.channel_index(name))
        except ValueError:
            return None
    names = grid.get("channel_names") if isinstance(grid, Mapping) else None
    try:
        return list(names).index(name) if names is not None else None
    except ValueError:
        return None


def _center_cell(grid: Any) -> tuple[int, int]:
    center = grid.get("center_cell") if isinstance(grid, Mapping) else getattr(grid, "center_cell", None)
    if center is not None:
        return int(center[0]), int(center[1])
    cells = grid.get("cells") if isinstance(grid, Mapping) else getattr(grid, "cells", grid)
    return len(cells) // 2, len(cells[0]) // 2


def _snapshot(value: Any) -> dict[str, Any]:
    if hasattr(value, "get_debug_state"):
        return dict(value.get_debug_state())
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _agent(snapshot: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(_mapping(snapshot.get("agents")).get(name))


def _position(*values: Any) -> dict[str, float] | None:
    for value in values:
        if isinstance(value, Mapping) and "x" in value and "y" in value:
            return {"x": float(value["x"]), "y": float(value["y"])}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            return {"x": float(value[0]), "y": float(value[1])}
    return None


def _number(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _bool_or_none(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
    return None


__all__ = ["RuleBasedTacticalModeSelector", "TACTICAL_MODES"]
