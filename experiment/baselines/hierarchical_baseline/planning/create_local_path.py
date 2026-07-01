from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

from ..types import AgentContext, AgentState, BaselineConfig


Cell = tuple[int, int]
_STEPS: tuple[Cell, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def create_local_path(
    ctx: AgentContext,
    state: AgentState,
    anchor: tuple[float, float],
    combat_profile: str | None,
    config: BaselineConfig,
) -> tuple[tuple[Cell, ...], Cell | None, Cell | None, int, dict]:
    del state, combat_profile
    grid = ctx.local_grid
    if grid is None:
        move_bin = _world_direction_to_move_bin(ctx.player_pos, anchor)
        return (), None, None, move_bin, {
            "reason": "missing_grid_direct_world_direction",
            "move_bin": move_bin,
            "target_cell": None,
            "next_cell": None,
            "path": [],
        }
    cells = _grid_cells(grid)
    if not cells or not cells[0]:
        return (), None, None, 0, {"reason": "empty_grid", "move_bin": 0, "path": []}
    center = _center_cell(grid, cells)
    target = _anchor_cell(ctx.player_pos, anchor, center, len(cells), len(cells[0]), _cell_size(grid, config))
    blocked = _blocked_grid(grid, cells)
    blocked = _inflate(blocked, config.obstacle_inflation_cells)
    blocked[center[0]][center[1]] = False
    parents = _bfs(blocked, center)
    reachable_target = min(
        parents,
        key=lambda cell: (
            (cell[0] - target[0]) ** 2 + (cell[1] - target[1]) ** 2,
            abs(cell[0] - center[0]) + abs(cell[1] - center[1]),
            cell,
        ),
    ) if parents else center
    path = _reconstruct(parents, center, reachable_target)
    next_cell = path[1] if len(path) > 1 else center
    move_bin = _step_to_move_bin(center, next_cell)
    return tuple(path), reachable_target, next_cell, move_bin, {
        "reason": "local_bfs_path",
        "center_cell": list(center),
        "requested_target_cell": list(target),
        "target_cell": list(reachable_target),
        "next_cell": list(next_cell),
        "path": [list(cell) for cell in path],
        "move_bin": move_bin,
        "reachable_count": len(parents),
    }


def _grid_cells(grid: Any) -> list:
    value = grid.get("cells") if isinstance(grid, Mapping) else getattr(grid, "cells", grid)
    return value.tolist() if hasattr(value, "tolist") else value


def _center_cell(grid: Any, cells: list) -> Cell:
    value = grid.get("center_cell") if isinstance(grid, Mapping) else getattr(grid, "center_cell", None)
    return (int(value[0]), int(value[1])) if value is not None else (len(cells) // 2, len(cells[0]) // 2)


def _cell_size(grid: Any, config: BaselineConfig) -> float:
    value = grid.get("cell_size") if isinstance(grid, Mapping) else getattr(grid, "cell_size", config.cell_size)
    return max(1e-6, float(value))


def _anchor_cell(
    player: tuple[float, float],
    anchor: tuple[float, float],
    center: Cell,
    height: int,
    width: int,
    cell_size: float,
) -> Cell:
    row = center[0] + int(round((anchor[1] - player[1]) / cell_size))
    col = center[1] + int(round((anchor[0] - player[0]) / cell_size))
    return max(0, min(height - 1, row)), max(0, min(width - 1, col))


def _blocked_grid(grid: Any, cells: list) -> list[list[bool]]:
    names = grid.get("channel_names") if isinstance(grid, Mapping) else getattr(grid, "channel_names", ())
    try:
        channel = list(names).index("obstacle")
    except ValueError:
        channel = 0
    return [[float(cell[channel]) > 0.0 for cell in row] for row in cells]


def _inflate(blocked: list[list[bool]], radius: int) -> list[list[bool]]:
    result = [row[:] for row in blocked]
    distance = max(0, int(radius))
    for row, values in enumerate(blocked):
        for col, value in enumerate(values):
            if not value:
                continue
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    y, x = row + dy, col + dx
                    if 0 <= y < len(result) and 0 <= x < len(result[0]):
                        result[y][x] = True
    return result


def _bfs(blocked: list[list[bool]], start: Cell) -> dict[Cell, Cell | None]:
    parents: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    while queue:
        row, col = queue.popleft()
        for dy, dx in _STEPS:
            target = row + dy, col + dx
            if not (0 <= target[0] < len(blocked) and 0 <= target[1] < len(blocked[0])):
                continue
            if blocked[target[0]][target[1]] or target in parents:
                continue
            if dy and dx and (blocked[row + dy][col] or blocked[row][col + dx]):
                continue
            parents[target] = (row, col)
            queue.append(target)
    return parents


def _reconstruct(parents: dict[Cell, Cell | None], start: Cell, target: Cell) -> list[Cell]:
    if target not in parents:
        return [start]
    path = [target]
    while path[-1] != start:
        parent = parents[path[-1]]
        if parent is None:
            return [start]
        path.append(parent)
    return list(reversed(path))


def _step_to_move_bin(start: Cell, target: Cell) -> int:
    return {
        (0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4,
        (-1, -1): 5, (-1, 1): 6, (1, -1): 7, (1, 1): 8,
    }.get((target[0] - start[0], target[1] - start[1]), 0)


def _world_direction_to_move_bin(start: tuple[float, float], target: tuple[float, float]) -> int:
    dx, dy = target[0] - start[0], target[1] - start[1]
    sx = 0 if abs(dx) < 1e-6 else 1 if dx > 0 else -1
    sy = 0 if abs(dy) < 1e-6 else 1 if dy > 0 else -1
    return {
        (0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4,
        (-1, -1): 5, (1, -1): 6, (-1, 1): 7, (1, 1): 8,
    }[(sx, sy)]


__all__ = ["create_local_path"]
