from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any


Cell = tuple[int, int]
BoolGrid = list[list[bool]]
ParentMap = dict[Cell, Cell | None]

_CARDINAL_STEPS: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL_STEPS: tuple[Cell, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def build_blocked_grid(local_occupancy_grid: Any, obstacle_channel_index: int) -> BoolGrid:
    """Extract an obstacle mask from ``cells[row_y][column_x][channel]``."""

    channel = int(obstacle_channel_index)
    if channel < 0:
        raise ValueError("obstacle_channel_index must be non-negative")
    cells = _grid_cells(local_occupancy_grid)
    if not cells:
        return []

    width = len(cells[0])
    if width == 0:
        return [[] for _ in cells]
    blocked: BoolGrid = []
    for row in cells:
        if len(row) != width:
            raise ValueError("local occupancy grid must be rectangular")
        blocked_row: list[bool] = []
        for cell in row:
            try:
                blocked_row.append(float(cell[channel]) > 0.0)
            except (IndexError, TypeError) as exc:
                raise ValueError(f"obstacle channel {channel} is outside the grid cell channels") from exc
        blocked.append(blocked_row)
    return blocked


def inflate_blocked_grid(blocked: Any, radius_cells: int = 1) -> BoolGrid:
    """Inflate blocked cells by a Chebyshev radius for agent clearance."""

    source = _bool_grid(blocked)
    radius = max(0, int(radius_cells))
    if radius == 0 or not source:
        return [row[:] for row in source]
    height = len(source)
    width = len(source[0])
    inflated = [[False for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for col in range(width):
            if not source[row][col]:
                continue
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    next_row = row + dy
                    next_col = col + dx
                    if _in_bounds(next_row, next_col, height, width):
                        inflated[next_row][next_col] = True
    return inflated


def bfs_reachable(
    blocked: Any,
    center: Cell,
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
) -> tuple[BoolGrid, ParentMap]:
    """Return cells reachable from center and their deterministic BFS parents.

    Grid coordinates are ``(row_y, column_x)``. Increasing row moves down and
    increasing column moves right, matching the env's positive-y convention.
    """

    mask = _bool_grid(blocked)
    if not mask:
        return [], {}
    height = len(mask)
    width = len(mask[0])
    start = (int(center[0]), int(center[1]))
    reachable = [[False for _ in range(width)] for _ in range(height)]
    if not _in_bounds(start[0], start[1], height, width) or mask[start[0]][start[1]]:
        return reachable, {}

    steps = _CARDINAL_STEPS + (_DIAGONAL_STEPS if allow_diagonal else ())
    parents: ParentMap = {start: None}
    reachable[start[0]][start[1]] = True
    queue: deque[Cell] = deque([start])
    while queue:
        row, col = queue.popleft()
        for dy, dx in steps:
            next_row = row + dy
            next_col = col + dx
            next_cell = (next_row, next_col)
            if not _in_bounds(next_row, next_col, height, width):
                continue
            if mask[next_row][next_col] or next_cell in parents:
                continue
            if dy != 0 and dx != 0 and prevent_corner_cutting:
                if mask[row + dy][col] or mask[row][col + dx]:
                    continue
            parents[next_cell] = (row, col)
            reachable[next_row][next_col] = True
            queue.append(next_cell)
    return reachable, parents


def reconstruct_path(parent_map: Mapping[Cell, Cell | None], start_cell: Cell, target_cell: Cell) -> list[Cell]:
    """Reconstruct an inclusive start-to-target path, or return an empty list."""

    start = (int(start_cell[0]), int(start_cell[1]))
    target = (int(target_cell[0]), int(target_cell[1]))
    if target not in parent_map:
        return []
    path = [target]
    seen = {target}
    while path[-1] != start:
        parent = parent_map.get(path[-1])
        if parent is None or parent in seen:
            return []
        path.append(parent)
        seen.add(parent)
    path.reverse()
    return path


def first_step_to_move_bin(center_cell: Cell, next_cell: Cell) -> int:
    """Map one grid step to the core env move-bin convention; invalid steps stay."""

    dy = int(next_cell[0]) - int(center_cell[0])
    dx = int(next_cell[1]) - int(center_cell[1])
    return {
        (0, 0): 0,
        (-1, 0): 1,
        (1, 0): 2,
        (0, -1): 3,
        (0, 1): 4,
        (-1, -1): 5,
        (-1, 1): 6,
        (1, -1): 7,
        (1, 1): 8,
    }.get((dy, dx), 0)


def _grid_cells(grid: Any) -> list[Any]:
    if isinstance(grid, Mapping) and "cells" in grid:
        return _to_list(grid["cells"])
    if hasattr(grid, "cells"):
        return _to_list(getattr(grid, "cells"))
    return _to_list(grid)


def _bool_grid(value: Any) -> BoolGrid:
    rows = _to_list(value)
    if not rows:
        return []
    width = len(rows[0])
    result: BoolGrid = []
    for row in rows:
        if len(row) != width:
            raise ValueError("blocked grid must be rectangular")
        result.append([bool(cell) for cell in row])
    return result


def _to_list(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _in_bounds(row: int, col: int, height: int, width: int) -> bool:
    return 0 <= row < height and 0 <= col < width


__all__ = [
    "bfs_reachable",
    "build_blocked_grid",
    "first_step_to_move_bin",
    "inflate_blocked_grid",
    "reconstruct_path",
]
