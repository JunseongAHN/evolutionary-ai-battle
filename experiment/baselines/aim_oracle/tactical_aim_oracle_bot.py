from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .aim_bin_utils import enemy_cell_to_aim_bin, grid_cell_to_local_vector
from .enemy_cell_utils import find_nearest_enemy_cell


LOCAL_GRID_KEYS = (
    "local_occupancy_grid",
    "local_grid",
    "occupancy_grid",
    "grid",
)


class TacticalAimOracleBot:
    """Deterministic local-grid enemy-cell-to-aim-bin baseline."""

    def __init__(
        self,
        num_aim_bins: int,
        enemy_channel_index: int,
        cell_size: float,
        stay_move_bin: int,
        default_aim_bin: int = 0,
    ):
        self.num_aim_bins = _positive_int(num_aim_bins, "num_aim_bins")
        self.enemy_channel_index = _non_negative_int(enemy_channel_index, "enemy_channel_index")
        self.cell_size = _positive_float(cell_size, "cell_size")
        self.stay_move_bin = _non_negative_int(stay_move_bin, "stay_move_bin")
        self.default_aim_bin = int(default_aim_bin) % self.num_aim_bins
        self.previous_aim_bin: int | None = None

    def act(self, obs: Any) -> tuple[dict[str, int], dict[str, Any]]:
        grid = _extract_local_grid(obs)
        if grid is None:
            aim_bin = self._fallback_aim_bin()
            action = self._action(aim_bin)
            return action, self._debug(
                enemy_cell=None,
                local_vector=None,
                aim_bin=aim_bin,
                reason="missing_local_occupancy_grid",
                action=action,
            )

        enemy_cell = find_nearest_enemy_cell(grid, self.enemy_channel_index)
        if enemy_cell is None:
            aim_bin = self._fallback_aim_bin()
            action = self._action(aim_bin)
            return action, self._debug(
                enemy_cell=None,
                local_vector=None,
                aim_bin=aim_bin,
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
        aim_bin = enemy_cell_to_aim_bin(
            cell_y=enemy_cell[0],
            cell_x=enemy_cell[1],
            grid_size=grid_size,
            cell_size=cell_size,
            num_bins=self.num_aim_bins,
        )
        self.previous_aim_bin = aim_bin
        action = self._action(aim_bin)
        return action, self._debug(
            enemy_cell=enemy_cell,
            local_vector=(dx, dy),
            aim_bin=aim_bin,
            reason="nearest_enemy_cell_mapped_to_aim_bin",
            action=action,
            grid_size=grid_size,
            cell_size=cell_size,
        )

    def _fallback_aim_bin(self) -> int:
        if self.previous_aim_bin is not None:
            return int(self.previous_aim_bin)
        return int(self.default_aim_bin)

    def _action(self, aim_bin: int) -> dict[str, int]:
        return {
            "move": int(self.stay_move_bin),
            "aim": int(aim_bin) % self.num_aim_bins,
            "fire": 0,
        }

    def _debug(
        self,
        *,
        enemy_cell: tuple[int, int] | None,
        local_vector: tuple[float, float] | None,
        aim_bin: int,
        reason: str,
        action: dict[str, int],
        grid_size: int | None = None,
        cell_size: float | None = None,
    ) -> dict[str, Any]:
        return {
            "enemy_cell": [int(enemy_cell[0]), int(enemy_cell[1])] if enemy_cell is not None else None,
            "local_vector": [float(local_vector[0]), float(local_vector[1])] if local_vector is not None else None,
            "aim_bin": int(aim_bin),
            "reason": reason,
            "action": dict(action),
            "action_schema": {"move_bin": "move", "aim_bin": "aim", "fire": "fire"},
            "enemy_channel_index": int(self.enemy_channel_index),
            "grid_size": grid_size,
            "cell_size": cell_size,
            "num_aim_bins": int(self.num_aim_bins),
            "coordinate_convention": "x/right positive, y/down positive, aim bin 0 is right, bins rotate clockwise on screen",
        }


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
