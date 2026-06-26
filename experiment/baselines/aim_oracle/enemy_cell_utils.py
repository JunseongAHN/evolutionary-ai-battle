from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def find_nearest_enemy_cell(
    local_occupancy_grid: Any,
    enemy_channel_index: int,
) -> tuple[int, int] | None:
    """Return the nearest active enemy cell to the grid center.

    The expected layout is ``cells[row_y][column_x][channel]``. Active enemy
    cells are values greater than zero in ``enemy_channel_index``. Ties are
    resolved by row, then column to keep the selection deterministic.
    """

    channel = int(enemy_channel_index)
    if channel < 0:
        raise ValueError(f"enemy_channel_index must be non-negative, got {enemy_channel_index!r}")

    cells = _grid_cells(local_occupancy_grid)
    if not cells:
        return None

    height = len(cells)
    width = max((len(row) for row in cells), default=0)
    if height <= 0 or width <= 0:
        return None
    center_y, center_x = _center_cell(local_occupancy_grid, height, width)

    best: tuple[int, int, int] | None = None
    for row_index, row in enumerate(cells):
        for col_index, cell in enumerate(row):
            if _channel_value(cell, channel) <= 0.0:
                continue
            dy = row_index - center_y
            dx = col_index - center_x
            score = (dy * dy) + (dx * dx)
            candidate = (score, row_index, col_index)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return None
    return best[1], best[2]


def _grid_cells(local_occupancy_grid: Any) -> Any:
    if isinstance(local_occupancy_grid, Mapping) and "cells" in local_occupancy_grid:
        return _to_list_if_needed(local_occupancy_grid["cells"])
    if hasattr(local_occupancy_grid, "cells"):
        return _to_list_if_needed(getattr(local_occupancy_grid, "cells"))
    return _to_list_if_needed(local_occupancy_grid)


def _center_cell(local_occupancy_grid: Any, height: int, width: int) -> tuple[int, int]:
    if isinstance(local_occupancy_grid, Mapping) and "center_cell" in local_occupancy_grid:
        center = local_occupancy_grid["center_cell"]
        return int(center[0]), int(center[1])
    if hasattr(local_occupancy_grid, "center_cell"):
        center = getattr(local_occupancy_grid, "center_cell")
        return int(center[0]), int(center[1])
    return height // 2, width // 2


def _channel_value(cell: Any, channel: int) -> float:
    values = _to_list_if_needed(cell)
    try:
        return float(values[channel])
    except IndexError as exc:
        raise ValueError(f"enemy_channel_index {channel} is outside cell channel range") from exc
    except TypeError as exc:
        raise ValueError("local occupancy grid cells must be channel sequences") from exc


def _to_list_if_needed(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


__all__ = ["find_nearest_enemy_cell"]
