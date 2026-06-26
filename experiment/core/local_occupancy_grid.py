from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CHANNEL_OBSTACLE = "obstacle"
CHANNEL_ENEMY = "enemy"
CHANNEL_HAZARD = "hazard"
CHANNEL_AGENT = "agent"
DEFAULT_CHANNELS = (CHANNEL_OBSTACLE, CHANNEL_ENEMY, CHANNEL_HAZARD, CHANNEL_AGENT)


@dataclass(frozen=True)
class LocalGridConfig:
    radius_cells: int = 10
    cell_size: float = 40.0
    channels: tuple[str, ...] = DEFAULT_CHANNELS

    @property
    def size(self) -> int:
        return (int(self.radius_cells) * 2) + 1


@dataclass(frozen=True)
class LocalOccupancyGrid:
    cells: list[list[list[float]]]
    channel_names: tuple[str, ...]
    center_cell: tuple[int, int]
    origin: dict[str, float]
    cell_size: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            len(self.cells),
            len(self.cells[0]) if self.cells else 0,
            len(self.channel_names),
        )

    def channel_index(self, name: str) -> int:
        return self.channel_names.index(name)


def build_local_occupancy_grid(
    env_state: Mapping[str, Any],
    *,
    agent_id: str = "self",
    config: LocalGridConfig | None = None,
) -> LocalOccupancyGrid:
    grid_config = config or LocalGridConfig()
    size = grid_config.size
    center_index = int(grid_config.radius_cells)
    cells = [
        [[0.0 for _ in grid_config.channels] for _ in range(size)]
        for _ in range(size)
    ]
    agent = _agent(env_state, agent_id)
    agent_pos = dict(agent.get("position", {"x": 0.0, "y": 0.0}))
    origin = {
        "x": float(agent_pos["x"]) - (center_index * float(grid_config.cell_size)),
        "y": float(agent_pos["y"]) - (center_index * float(grid_config.cell_size)),
    }
    grid = LocalOccupancyGrid(
        cells=cells,
        channel_names=tuple(grid_config.channels),
        center_cell=(center_index, center_index),
        origin=origin,
        cell_size=float(grid_config.cell_size),
    )

    _mark_agent(grid)
    _mark_obstacles(grid, env_state)
    _mark_enemies(grid, env_state, agent_id)
    _mark_hazards(grid, env_state)
    return grid


def render_grid_to_png(
    grid: LocalOccupancyGrid,
    path: str | Path,
    *,
    cell_pixels: int = 16,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = max(1, int(cell_pixels))
    width = grid.shape[1] * scale
    height = grid.shape[0] * scale
    pixels: list[list[tuple[int, int, int, int]]] = []
    for y in range(height):
        row_index = y // scale
        row: list[tuple[int, int, int, int]] = []
        for x in range(width):
            col_index = x // scale
            color = _cell_color(grid, row_index, col_index)
            if x % scale == 0 or y % scale == 0:
                color = (8, 10, 14, 255)
            row.append(color)
        pixels.append(row)
    _write_png_rgba(path, width, height, pixels)


def _mark_agent(grid: LocalOccupancyGrid) -> None:
    channel = _optional_channel(grid, CHANNEL_AGENT)
    if channel is None:
        return
    row, col = grid.center_cell
    grid.cells[row][col][channel] = 1.0


def _mark_obstacles(grid: LocalOccupancyGrid, env_state: Mapping[str, Any]) -> None:
    channel = _optional_channel(grid, CHANNEL_OBSTACLE)
    if channel is None:
        return
    obstacles = list(env_state.get("obstacles", []))
    obstacles.extend(env_state.get("map", {}).get("obstacles", []))
    for obstacle in obstacles:
        if str(obstacle.get("type", "circle")) != "circle":
            continue
        position = {"x": float(obstacle.get("x", 0.0)), "y": float(obstacle.get("y", 0.0))}
        _mark_circle(grid, channel, position, float(obstacle.get("radius", 0.0)))


def _mark_enemies(grid: LocalOccupancyGrid, env_state: Mapping[str, Any], agent_id: str) -> None:
    channel = _optional_channel(grid, CHANNEL_ENEMY)
    if channel is None:
        return
    agents = env_state.get("agents", {})
    reference = _agent(env_state, agent_id)
    reference_team = reference.get("team_id")
    for name, candidate in agents.items():
        if name == agent_id:
            continue
        if not bool(candidate.get("alive", float(candidate.get("hp", 0.0)) > 0.0)):
            continue
        role = str(candidate.get("role", name))
        team_id = candidate.get("team_id")
        is_enemy = role == "enemy" or name == "enemy" or (
            reference_team is not None and team_id is not None and team_id != reference_team
        )
        if not is_enemy:
            continue
        position = candidate.get("position")
        if position:
            _mark_circle(grid, channel, position, float(candidate.get("radius", 12.0)))


def _mark_hazards(grid: LocalOccupancyGrid, env_state: Mapping[str, Any]) -> None:
    channel = _optional_channel(grid, CHANNEL_HAZARD)
    if channel is None:
        return
    for projectile in env_state.get("projectiles", env_state.get("bullets", [])):
        position = projectile.get("pos") or projectile.get("position")
        if position:
            _mark_circle(grid, channel, position, float(projectile.get("radius", 8.0)))

    map_info = env_state.get("map", {})
    if not (map_info.get("use_zone_reward") or map_info.get("shrink_safe_zone")):
        return
    safe_zone = env_state.get("safe_zone", {})
    center = safe_zone.get("center")
    radius = safe_zone.get("radius")
    if center is None or radius is None:
        return
    for row_index, row in enumerate(grid.cells):
        for col_index, cell in enumerate(row):
            center_pos = _cell_center(grid, row_index, col_index)
            if _distance(center_pos, center) > float(radius):
                cell[channel] = max(cell[channel], 0.35)


def _mark_circle(
    grid: LocalOccupancyGrid,
    channel: int,
    position: Mapping[str, Any],
    radius: float,
) -> None:
    touch_radius = max(0.0, float(radius)) + (grid.cell_size * 0.5)
    for row_index, row in enumerate(grid.cells):
        for col_index, cell in enumerate(row):
            if _distance(_cell_center(grid, row_index, col_index), position) <= touch_radius:
                cell[channel] = 1.0


def _cell_center(grid: LocalOccupancyGrid, row_index: int, col_index: int) -> dict[str, float]:
    return {
        "x": grid.origin["x"] + (col_index * grid.cell_size),
        "y": grid.origin["y"] + (row_index * grid.cell_size),
    }


def _agent(env_state: Mapping[str, Any], agent_id: str) -> Mapping[str, Any]:
    agents = env_state.get("agents", {})
    if agent_id in agents:
        return agents[agent_id]
    if agent_id == "self" and "self" in agents:
        return agents["self"]
    state = env_state.get("state", {})
    return {
        "position": state.get("self_pos", {"x": 0.0, "y": 0.0}),
        "hp": state.get("self_hp", 0.0),
        "alive": state.get("self_hp", 0.0) > 0.0,
    }


def _optional_channel(grid: LocalOccupancyGrid, name: str) -> int | None:
    try:
        return grid.channel_index(name)
    except ValueError:
        return None


def _distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _cell_color(grid: LocalOccupancyGrid, row_index: int, col_index: int) -> tuple[int, int, int, int]:
    cell = grid.cells[row_index][col_index]
    values = {name: cell[index] for index, name in enumerate(grid.channel_names)}
    if values.get(CHANNEL_AGENT, 0.0) > 0.0:
        return (72, 210, 128, 255)
    if values.get(CHANNEL_ENEMY, 0.0) > 0.0:
        return (235, 82, 82, 255)
    if values.get(CHANNEL_HAZARD, 0.0) > 0.0:
        return (245, 190, 74, 255)
    if values.get(CHANNEL_OBSTACLE, 0.0) > 0.0:
        return (125, 130, 140, 255)
    return (24, 28, 36, 255)


def _write_png_rgba(
    path: Path,
    width: int,
    height: int,
    pixels: list[list[tuple[int, int, int, int]]],
) -> None:
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b, a in row:
            raw.extend((r, g, b, a))
    data = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw))),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(data)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


__all__ = [
    "CHANNEL_AGENT",
    "CHANNEL_ENEMY",
    "CHANNEL_HAZARD",
    "CHANNEL_OBSTACLE",
    "LocalGridConfig",
    "LocalOccupancyGrid",
    "build_local_occupancy_grid",
    "render_grid_to_png",
]
