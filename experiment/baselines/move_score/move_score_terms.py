from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

VecLike = Mapping[str, float] | Sequence[float]


def obstacle_collision_penalty(
    candidate_pos: VecLike,
    obstacles: Sequence[Mapping[str, Any]],
    self_radius: float,
    penalty: float = 1.0,
) -> float:
    """Return a negative penalty when the candidate endpoint overlaps a circle obstacle."""
    candidate = _xy(candidate_pos)
    for obstacle in _circle_obstacles(obstacles):
        center = (float(obstacle["x"]), float(obstacle["y"]))
        radius = float(obstacle["radius"]) + float(self_radius)
        if _distance(candidate, center) <= radius:
            return -float(penalty)
    return 0.0


def obstacle_path_collision_penalty(
    start_pos: VecLike,
    candidate_pos: VecLike,
    obstacles: Sequence[Mapping[str, Any]],
    self_radius: float,
    penalty: float = 1.0,
) -> float:
    """Return a negative penalty when the step segment would intersect an obstacle."""
    start = _xy(start_pos)
    candidate = _xy(candidate_pos)
    for obstacle in _circle_obstacles(obstacles):
        center = (float(obstacle["x"]), float(obstacle["y"]))
        radius = float(obstacle["radius"]) + float(self_radius)
        if _segment_intersects_circle(start, candidate, center, radius):
            return -float(penalty)
    return 0.0


def map_boundary_penalty(
    candidate_pos: VecLike,
    map_width: float,
    map_height: float,
    self_radius: float,
    penalty: float = 1.0,
) -> float:
    """Return a negative penalty when the agent body would leave the map."""
    x, y = _xy(candidate_pos)
    radius = float(self_radius)
    if x - radius < 0.0 or x + radius > float(map_width) or y - radius < 0.0 or y + radius > float(map_height):
        return -float(penalty)
    return 0.0


def enemy_spacing_score(
    candidate_pos: VecLike,
    enemy_pos: VecLike,
    ideal_range: float,
    weight: float,
) -> float:
    """Prefer distances close to the configured ideal range."""
    distance = _distance(_xy(candidate_pos), _xy(enemy_pos))
    return -float(weight) * abs(distance - float(ideal_range))


def enemy_threat_penalty(
    candidate_pos: VecLike,
    enemy_pos: VecLike,
    enemy_threat_range: float,
    weight: float,
) -> float:
    """Penalize being inside the close-threat range, scaled by closeness."""
    threat_range = max(1e-6, float(enemy_threat_range))
    distance = _distance(_xy(candidate_pos), _xy(enemy_pos))
    if distance >= threat_range:
        return 0.0
    return -float(weight) * ((threat_range - distance) / threat_range)


def strafe_score(
    move_vector: VecLike,
    self_pos: VecLike,
    enemy_pos: VecLike,
    weight: float,
) -> float:
    """Reward perpendicular movement relative to the enemy direction."""
    move_dx, move_dy = _normalize(*_xy(move_vector))
    if abs(move_dx) <= 1e-9 and abs(move_dy) <= 1e-9:
        return 0.0
    sx, sy = _xy(self_pos)
    ex, ey = _xy(enemy_pos)
    to_enemy_x, to_enemy_y = _normalize(ex - sx, ey - sy)
    if abs(to_enemy_x) <= 1e-9 and abs(to_enemy_y) <= 1e-9:
        return 0.0
    perpendicular = abs((move_dx * to_enemy_y) - (move_dy * to_enemy_x))
    return float(weight) * perpendicular


def line_of_sight_score(
    candidate_pos: VecLike,
    enemy_pos: VecLike,
    obstacles: Sequence[Mapping[str, Any]],
    weight: float,
) -> float:
    """Simple circle-obstacle line-of-sight score: clear is positive, blocked is negative."""
    candidate = _xy(candidate_pos)
    enemy = _xy(enemy_pos)
    for obstacle in _circle_obstacles(obstacles):
        center = (float(obstacle["x"]), float(obstacle["y"]))
        if _segment_intersects_circle(candidate, enemy, center, float(obstacle["radius"])):
            return -float(weight)
    return float(weight)


def _circle_obstacles(obstacles: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        obstacle
        for obstacle in obstacles
        if obstacle.get("type", "circle") == "circle" and float(obstacle.get("radius", 0.0)) > 0.0
    ]


def _xy(value: VecLike) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return float(value["x"]), float(value["y"])
    return float(value[0]), float(value[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(float(dx), float(dy))
    if length <= 1e-9:
        return 0.0, 0.0
    return float(dx) / length, float(dy) / length


def _segment_intersects_circle(
    start: tuple[float, float],
    target: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    sx, sy = start
    tx, ty = target
    cx, cy = center
    dx = tx - sx
    dy = ty - sy
    length_sq = (dx * dx) + (dy * dy)
    if length_sq <= 1e-12:
        return _distance(start, center) <= float(radius)
    t = max(0.0, min(1.0, (((cx - sx) * dx) + ((cy - sy) * dy)) / length_sq))
    closest = (sx + (t * dx), sy + (t * dy))
    return _distance(closest, center) <= float(radius)
