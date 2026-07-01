from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from experiment.baselines.aim_oracle.enemy_cell_utils import find_nearest_enemy_cell
except ModuleNotFoundError:
    from baselines.aim_oracle.enemy_cell_utils import find_nearest_enemy_cell

from .local_grid_los import has_grid_line_of_sight
from .local_grid_pathfinding import (
    bfs_reachable,
    build_blocked_grid,
    first_step_to_move_bin,
    inflate_blocked_grid,
    reconstruct_path,
)
from .tactical_mode_selector import TACTICAL_MODES


Cell = tuple[int, int]
_GRID_KEYS = ("local_occupancy_grid", "local_grid", "occupancy_grid", "grid")


class ModeConditionedBFSPlanner:
    """Choose a reachable local target and execute the first BFS path step."""

    def __init__(
        self,
        obstacle_channel_index: int,
        enemy_channel_index: int,
        cell_size: float,
        weapon_range: float,
        ideal_range_ratio: float = 0.7,
        *,
        obstacle_inflation_cells: int = 1,
        top_k: int = 5,
    ) -> None:
        self.obstacle_channel_index = _non_negative_int(obstacle_channel_index, "obstacle_channel_index")
        self.enemy_channel_index = _non_negative_int(enemy_channel_index, "enemy_channel_index")
        self.cell_size = _positive_float(cell_size, "cell_size")
        self.weapon_range = _positive_float(weapon_range, "weapon_range")
        self.ideal_range_ratio = _positive_float(ideal_range_ratio, "ideal_range_ratio")
        self.obstacle_inflation_cells = max(1, int(obstacle_inflation_cells))
        self.top_k = max(0, int(top_k))

    def choose_move(
        self,
        obs: Any,
        tactical_mode: str,
        state_snapshot: Any | None = None,
    ) -> tuple[int, dict[str, Any]]:
        mode = str(tactical_mode)
        if mode not in TACTICAL_MODES:
            raise ValueError(f"unsupported tactical mode: {mode!r}")
        observation = dict(obs) if isinstance(obs, Mapping) else {}
        snapshot = _snapshot(state_snapshot)
        grid = next((observation.get(key) for key in _GRID_KEYS if observation.get(key) is not None), None)
        if grid is None:
            return 0, self._fallback_debug(mode, "missing_local_occupancy_grid")

        center = _center_cell(grid)
        blocked = build_blocked_grid(grid, self.obstacle_channel_index)
        if not blocked or not blocked[0]:
            return 0, self._fallback_debug(mode, "empty_local_occupancy_grid")
        _block_world_boundaries(blocked, grid, snapshot)
        if _in_grid(center, blocked):
            blocked[center[0]][center[1]] = False
        inflated = inflate_blocked_grid(blocked, self.obstacle_inflation_cells)
        inflated[center[0]][center[1]] = False
        reachable, parent_map = bfs_reachable(
            inflated,
            center,
            allow_diagonal=True,
            prevent_corner_cutting=True,
        )
        reachable_cells = [
            (row, col)
            for row, values in enumerate(reachable)
            for col, is_reachable in enumerate(values)
            if is_reachable
        ]
        if not reachable_cells:
            return 0, self._fallback_debug(mode, "no_reachable_cells", center=center)

        enemy_cell, enemy_grid_pos = _enemy_grid_target(
            grid,
            observation,
            snapshot,
            self.enemy_channel_index,
        )
        ideal_range = self.weapon_range * self.ideal_range_ratio
        current_distance = _grid_distance(center, enemy_grid_pos, self.cell_size) if enemy_grid_pos is not None else None
        blocked_cells = [(row, col) for row, values in enumerate(inflated) for col, value in enumerate(values) if value]
        candidates: list[dict[str, Any]] = []
        for target in reachable_cells:
            path = reconstruct_path(parent_map, center, target)
            if not path:
                continue
            next_cell = path[1] if len(path) > 1 else center
            score = self._score_cell(
                mode=mode,
                center=center,
                target=target,
                next_cell=next_cell,
                enemy_cell=enemy_cell,
                enemy_grid_pos=enemy_grid_pos,
                current_distance=current_distance,
                ideal_range=ideal_range,
                path_length=len(path) - 1,
                blocked=blocked,
                inflated=inflated,
                blocked_cells=blocked_cells,
            )
            candidates.append({
                "target_cell": target,
                "next_cell": next_cell,
                "path": path,
                "path_length": len(path) - 1,
                "score": score,
            })

        if not candidates:
            return 0, self._fallback_debug(mode, "no_reconstructable_path", center=center)
        candidates.sort(key=_candidate_sort_key)
        selected = candidates[0]
        move_bin = first_step_to_move_bin(center, selected["next_cell"])
        reason = "safe_fallback_target_selected" if enemy_cell is None else f"{mode}_target_selected"
        debug = {
            "tactical_mode": mode,
            "enemy_cell": _cell_list(enemy_cell),
            "target_cell": _cell_list(selected["target_cell"]),
            "next_cell": _cell_list(selected["next_cell"]),
            "move_bin": int(move_bin),
            "selected_move_bin": int(move_bin),
            "path": [_cell_list(cell) for cell in selected["path"]],
            "reachable_count": len(reachable_cells),
            "selected_score": dict(selected["score"]),
            "top_candidates": [
                {
                    "target_cell": _cell_list(candidate["target_cell"]),
                    "next_cell": _cell_list(candidate["next_cell"]),
                    "path_length": int(candidate["path_length"]),
                    "score": dict(candidate["score"]),
                }
                for candidate in candidates[: self.top_k]
            ],
            "blocked_cells": [_cell_list(cell) for cell in blocked_cells],
            "reachable_cells": [_cell_list(cell) for cell in reachable_cells],
            "center_cell": _cell_list(center),
            "ideal_range": float(ideal_range),
            "current_enemy_distance": current_distance,
            "reason": reason,
        }
        return int(move_bin), debug

    def _score_cell(
        self,
        *,
        mode: str,
        center: Cell,
        target: Cell,
        next_cell: Cell,
        enemy_cell: Cell | None,
        enemy_grid_pos: tuple[float, float] | None,
        current_distance: float | None,
        ideal_range: float,
        path_length: int,
        blocked: list[list[bool]],
        inflated: list[list[bool]],
        blocked_cells: list[Cell],
    ) -> dict[str, float]:
        candidate_distance = _grid_distance(target, enemy_grid_pos, self.cell_size) if enemy_grid_pos is not None else None
        distance_cells = None if candidate_distance is None else candidate_distance / self.cell_size
        ideal_error_cells = None if candidate_distance is None else abs(candidate_distance - ideal_range) / self.cell_size
        line_of_sight = enemy_cell is not None and has_grid_line_of_sight(target, enemy_cell, blocked)
        next_step_los = enemy_cell is not None and has_grid_line_of_sight(next_cell, enemy_cell, blocked)
        open_space = _open_space(target, inflated)
        clearance = _obstacle_clearance(target, blocked_cells, len(inflated), len(inflated[0]))
        boundary_safety = _boundary_safety(target, len(inflated), len(inflated[0]))
        strafe = _strafe_preference(center, next_cell, enemy_grid_pos)
        stayed = target == center

        terms = {
            "distance": 0.0,
            "ideal_range": 0.0,
            "los": 0.0,
            "strafe": 0.0,
            "open_space": 0.0,
            "obstacle_clearance": 0.0,
            "boundary_safety": 0.0,
            "path_length": -0.08 * float(path_length),
            "stay_penalty": 0.0,
        }
        if mode == "engage":
            if candidate_distance is not None and current_distance is not None:
                terms["distance"] = ((current_distance - candidate_distance) / self.cell_size) * 1.5
                terms["ideal_range"] = -float(ideal_error_cells) * 2.0
            terms["los"] = 2.0 if line_of_sight else -1.0
            terms["open_space"] = open_space * 0.25
            terms["obstacle_clearance"] = clearance * 0.25
            terms["boundary_safety"] = boundary_safety * 0.2
            terms["stay_penalty"] = -0.8 if stayed else 0.0
        elif mode == "kite":
            if candidate_distance is not None and current_distance is not None:
                terms["distance"] = ((candidate_distance - current_distance) / self.cell_size) * 2.2
            terms["los"] = 0.2 if line_of_sight else 0.0
            terms["open_space"] = open_space * 0.8
            terms["obstacle_clearance"] = clearance * 0.5
            terms["boundary_safety"] = boundary_safety * 1.2
            terms["path_length"] = -0.06 * float(path_length)
            terms["stay_penalty"] = -1.0 if stayed else 0.0
        elif mode == "hold_range":
            if ideal_error_cells is not None:
                terms["ideal_range"] = -float(ideal_error_cells) * 2.4
            if candidate_distance is not None:
                terms["distance"] = -max(0.0, ideal_range - candidate_distance) / self.cell_size * 0.5
            if line_of_sight and next_step_los:
                terms["los"] = 2.5
            elif line_of_sight:
                terms["los"] = -0.5
            else:
                terms["los"] = -2.5
            terms["strafe"] = strafe * 2.0
            terms["open_space"] = open_space * 0.3
            terms["obstacle_clearance"] = clearance * 0.4
            terms["boundary_safety"] = boundary_safety * 0.3
            terms["path_length"] = -0.12 * float(path_length)
            terms["stay_penalty"] = -1.0 if stayed else 0.0
        else:
            if candidate_distance is not None and current_distance is not None:
                current_error = abs(current_distance - ideal_range)
                candidate_error = abs(candidate_distance - ideal_range)
                terms["distance"] = ((current_error - candidate_error) / self.cell_size) * 0.2
            terms["los"] = 5.0 if line_of_sight else -1.0
            terms["open_space"] = open_space * 1.0
            terms["obstacle_clearance"] = clearance * 0.8
            terms["boundary_safety"] = boundary_safety * 0.6
            terms["stay_penalty"] = -1.5 if stayed else 0.0

        terms["total"] = sum(terms.values())
        terms["enemy_distance"] = float(distance_cells) if distance_cells is not None else -1.0
        return terms

    @staticmethod
    def _fallback_debug(mode: str, reason: str, center: Cell | None = None) -> dict[str, Any]:
        selected_score = {
            "total": 0.0,
            "distance": 0.0,
            "ideal_range": 0.0,
            "los": 0.0,
            "strafe": 0.0,
            "open_space": 0.0,
            "obstacle_clearance": 0.0,
            "boundary_safety": 0.0,
            "path_length": 0.0,
            "stay_penalty": 0.0,
        }
        return {
            "tactical_mode": mode,
            "enemy_cell": None,
            "target_cell": _cell_list(center),
            "next_cell": _cell_list(center),
            "move_bin": 0,
            "selected_move_bin": 0,
            "path": [] if center is None else [_cell_list(center)],
            "reachable_count": 0,
            "selected_score": selected_score,
            "top_candidates": [],
            "reason": reason,
        }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, int, int, int]:
    row, col = candidate["target_cell"]
    return (-float(candidate["score"]["total"]), int(candidate["path_length"]), int(row), int(col))


def _enemy_grid_target(
    grid: Any,
    obs: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    enemy_channel_index: int,
) -> tuple[Cell | None, tuple[float, float] | None]:
    enemy_cell = find_nearest_enemy_cell(grid, enemy_channel_index)
    cells = _cells(grid)
    height, width = len(cells), len(cells[0])
    enemy_pos = _position(obs.get("enemy_pos"), _mapping(snapshot.get("state")).get("enemy_pos"), _agent(snapshot, "enemy").get("position"))
    enemy_alive = _enemy_alive(obs, snapshot, enemy_pos, enemy_cell)
    exact_grid_pos = None
    origin = _origin(grid)
    cell_size = _grid_cell_size(grid)
    if enemy_alive and enemy_pos is not None and origin is not None and cell_size is not None:
        exact_grid_pos = (
            (enemy_pos["y"] - origin["y"]) / cell_size,
            (enemy_pos["x"] - origin["x"]) / cell_size,
        )
        if enemy_cell is None:
            enemy_cell = (
                max(0, min(height - 1, int(round(exact_grid_pos[0])))),
                max(0, min(width - 1, int(round(exact_grid_pos[1])))),
            )
    if enemy_cell is None or not enemy_alive:
        return None, None
    return enemy_cell, exact_grid_pos or (float(enemy_cell[0]), float(enemy_cell[1]))


def _block_world_boundaries(blocked: list[list[bool]], grid: Any, snapshot: Mapping[str, Any]) -> None:
    origin = _origin(grid)
    cell_size = _grid_cell_size(grid)
    map_info = _mapping(snapshot.get("map"))
    width = _number(map_info.get("width"))
    height = _number(map_info.get("height"))
    if origin is None or cell_size is None or width is None or height is None:
        return
    radius = _number(_agent(snapshot, "self").get("radius")) or 0.0
    for row in range(len(blocked)):
        for col in range(len(blocked[0])):
            x = origin["x"] + col * cell_size
            y = origin["y"] + row * cell_size
            if x < radius or x > width - radius or y < radius or y > height - radius:
                blocked[row][col] = True


def _open_space(cell: Cell, blocked: list[list[bool]]) -> float:
    row, col = cell
    free = 0
    total = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            next_row, next_col = row + dy, col + dx
            if _in_grid((next_row, next_col), blocked):
                total += 1
                free += int(not blocked[next_row][next_col])
    return float(free) / max(1, total)


def _obstacle_clearance(cell: Cell, blocked_cells: list[Cell], height: int, width: int) -> float:
    if not blocked_cells:
        return 1.0
    distance = min(math.hypot(cell[0] - row, cell[1] - col) for row, col in blocked_cells)
    return min(1.0, distance / max(1.0, min(height, width) * 0.25))


def _boundary_safety(cell: Cell, height: int, width: int) -> float:
    edge_distance = min(cell[0], cell[1], height - 1 - cell[0], width - 1 - cell[1])
    return min(1.0, float(edge_distance) / max(1.0, min(height, width) * 0.25))


def _strafe_preference(center: Cell, next_cell: Cell, enemy: tuple[float, float] | None) -> float:
    if enemy is None or next_cell == center:
        return 0.0
    move_y, move_x = next_cell[0] - center[0], next_cell[1] - center[1]
    enemy_y, enemy_x = enemy[0] - center[0], enemy[1] - center[1]
    move_length = math.hypot(move_x, move_y)
    enemy_length = math.hypot(enemy_x, enemy_y)
    if move_length <= 1e-9 or enemy_length <= 1e-9:
        return 0.0
    return abs(move_x * enemy_y - move_y * enemy_x) / (move_length * enemy_length)


def _grid_distance(cell: Cell, target: tuple[float, float] | None, cell_size: float) -> float:
    if target is None:
        return 0.0
    return math.hypot(cell[0] - target[0], cell[1] - target[1]) * cell_size


def _center_cell(grid: Any) -> Cell:
    center = grid.get("center_cell") if isinstance(grid, Mapping) else getattr(grid, "center_cell", None)
    if center is not None:
        return int(center[0]), int(center[1])
    cells = _cells(grid)
    return len(cells) // 2, len(cells[0]) // 2


def _cells(grid: Any) -> Any:
    values = grid.get("cells") if isinstance(grid, Mapping) else getattr(grid, "cells", grid)
    return values.tolist() if hasattr(values, "tolist") else values


def _origin(grid: Any) -> dict[str, float] | None:
    value = grid.get("origin") if isinstance(grid, Mapping) else getattr(grid, "origin", None)
    return _position(value)


def _grid_cell_size(grid: Any) -> float | None:
    value = grid.get("cell_size") if isinstance(grid, Mapping) else getattr(grid, "cell_size", None)
    return _number(value)


def _enemy_alive(
    obs: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    enemy_pos: Any,
    enemy_cell: Cell | None,
) -> bool:
    hp = _number(obs.get("enemy_hp"), _agent(snapshot, "enemy").get("hp"), _mapping(snapshot.get("state")).get("enemy_hp"))
    if hp is not None:
        return hp > 0.0
    alive = _agent(snapshot, "enemy").get("alive")
    return bool(alive) if alive is not None else enemy_pos is not None or enemy_cell is not None


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
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _cell_list(cell: Cell | None) -> list[int] | None:
    return None if cell is None else [int(cell[0]), int(cell[1])]


def _in_grid(cell: Cell, grid: list[list[Any]]) -> bool:
    return bool(grid) and 0 <= cell[0] < len(grid) and 0 <= cell[1] < len(grid[0])


def _non_negative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = ["ModeConditionedBFSPlanner"]
