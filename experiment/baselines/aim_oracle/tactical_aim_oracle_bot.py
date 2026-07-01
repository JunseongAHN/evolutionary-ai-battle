from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .aim_bin_utils import grid_cell_to_local_vector
from .enemy_cell_utils import find_nearest_enemy_cell


LOCAL_GRID_KEYS = (
    "local_occupancy_grid",
    "local_grid",
    "occupancy_grid",
    "grid",
)


class TacticalAimOracleBot:
    """Aim at the enemy's continuous world-space direction."""

    def __init__(
        self,
        enemy_channel_index: int,
        cell_size: float,
        stay_move_bin: int,
    ):
        self.enemy_channel_index = _non_negative_int(enemy_channel_index, "enemy_channel_index")
        self.cell_size = _positive_float(cell_size, "cell_size")
        self.stay_move_bin = _non_negative_int(stay_move_bin, "stay_move_bin")
        self.previous_aim_direction = (1.0, 0.0)

    def act(self, obs: Any) -> tuple[dict[str, int | float], dict[str, Any]]:
        position_vector = _enemy_relative_vector(obs)
        if position_vector is not None:
            dx, dy = position_vector
            aim_direction = _normalize_direction(dx, dy)
            self.previous_aim_direction = aim_direction
            action = self._action(*aim_direction)
            return action, self._debug(
                enemy_cell=None,
                local_vector=(dx, dy),
                aim_direction=aim_direction,
                aim_source="enemy_position",
                reason="enemy_position_mapped_to_direction",
                action=action,
            )

        grid = _extract_local_grid(obs)
        if grid is None:
            action = self._action(*self.previous_aim_direction)
            return action, self._debug(
                enemy_cell=None,
                local_vector=None,
                aim_direction=self.previous_aim_direction,
                aim_source="previous_direction",
                reason="missing_local_occupancy_grid",
                action=action,
            )

        enemy_cell = find_nearest_enemy_cell(grid, self.enemy_channel_index)
        if enemy_cell is None:
            action = self._action(*self.previous_aim_direction)
            return action, self._debug(
                enemy_cell=None,
                local_vector=None,
                aim_direction=self.previous_aim_direction,
                aim_source="previous_direction",
                reason="no_enemy_visible_in_local_grid",
                action=action,
            )

        grid_size = _grid_size(grid)
        cell_size = _grid_cell_size(grid, self.cell_size)
        dx, dy = grid_cell_to_local_vector(
            cell_y=enemy_cell[0],
            cell_x=enemy_cell[1],
            grid_size=grid_size,
            cell_size=cell_size,
        )
        aim_direction = _normalize_direction(dx, dy)
        self.previous_aim_direction = aim_direction
        action = self._action(*aim_direction)
        return action, self._debug(
            enemy_cell=enemy_cell,
            local_vector=(dx, dy),
            aim_direction=aim_direction,
            aim_source="local_occupancy_grid",
            reason="nearest_enemy_cell_mapped_to_direction",
            action=action,
            grid_size=grid_size,
            cell_size=cell_size,
        )

    def _action(self, aim_dx: float, aim_dy: float) -> dict[str, int | float]:
        return {
            "move": int(self.stay_move_bin),
            "aim_dx": float(aim_dx),
            "aim_dy": float(aim_dy),
            "fire": 0,
        }

    def _debug(
        self,
        *,
        enemy_cell: tuple[int, int] | None,
        local_vector: tuple[float, float] | None,
        aim_direction: tuple[float, float] | None,
        aim_source: str,
        reason: str,
        action: dict[str, int | float],
        grid_size: int | None = None,
        cell_size: float | None = None,
    ) -> dict[str, Any]:
        return {
            "enemy_cell": [int(enemy_cell[0]), int(enemy_cell[1])] if enemy_cell is not None else None,
            "local_vector": [float(local_vector[0]), float(local_vector[1])] if local_vector is not None else None,
            "aim_direction": [float(aim_direction[0]), float(aim_direction[1])] if aim_direction is not None else None,
            "aim_source": aim_source,
            "reason": reason,
            "action": dict(action),
            "action_schema": {"move_bin": "move", "aim_dx": "aim_dx", "aim_dy": "aim_dy", "fire": "fire"},
            "enemy_channel_index": int(self.enemy_channel_index),
            "grid_size": grid_size,
            "cell_size": cell_size,
            "coordinate_convention": "x/right positive, y/down positive, aim direction is a normalized continuous vector",
        }


def _enemy_relative_vector(obs: Any) -> tuple[float, float] | None:
    if not isinstance(obs, Mapping):
        return None
    enemy_hp = _number_or_none(obs.get("enemy_hp"))
    if enemy_hp is not None and enemy_hp <= 0.0:
        return None
    self_pos = _position(obs.get("self_pos"))
    enemy_pos = _position(obs.get("enemy_pos"))
    if self_pos is None or enemy_pos is None:
        return None
    dx = enemy_pos[0] - self_pos[0]
    dy = enemy_pos[1] - self_pos[1]
    return (dx, dy) if math.hypot(dx, dy) > 1e-6 else None


def _position(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_direction(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(float(dx), float(dy))
    if length <= 1e-6:
        return 1.0, 0.0
    return float(dx) / length, float(dy) / length


def _extract_local_grid(obs: Any) -> Any | None:
    if obs is None:
        return None
    if hasattr(obs, "cells"):
        return obs
    if isinstance(obs, Mapping):
        for key in LOCAL_GRID_KEYS:
            if key in obs:
                return obs[key]
        return None
    if isinstance(obs, list) or hasattr(obs, "tolist"):
        return obs
    return None


def _grid_size(grid: Any) -> int:
    height, width = _grid_dimensions(grid)
    if height != width:
        raise ValueError(f"local occupancy grid must be square, got {height}x{width}")
    return height


def _grid_dimensions(grid: Any) -> tuple[int, int]:
    shape = None
    if isinstance(grid, Mapping):
        shape = grid.get("shape")
    if shape is None and hasattr(grid, "shape"):
        shape = getattr(grid, "shape")
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    cells = _grid_cells(grid)
    height = len(cells)
    width = max((len(row) for row in cells), default=0)
    return height, width


def _grid_cell_size(grid: Any, fallback: float) -> float:
    if isinstance(grid, Mapping) and "cell_size" in grid:
        return _positive_float(grid["cell_size"], "cell_size")
    if hasattr(grid, "cell_size"):
        return _positive_float(getattr(grid, "cell_size"), "cell_size")
    return float(fallback)


def _grid_cells(grid: Any) -> Any:
    if isinstance(grid, Mapping) and "cells" in grid:
        return _to_list_if_needed(grid["cells"])
    if hasattr(grid, "cells"):
        return _to_list_if_needed(getattr(grid, "cells"))
    return _to_list_if_needed(grid)


def _to_list_if_needed(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _non_negative_int(value: int, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return number


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


__all__ = ["TacticalAimOracleBot"]
