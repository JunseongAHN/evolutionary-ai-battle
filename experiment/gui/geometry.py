from __future__ import annotations

from typing import Any, Mapping


def world_to_screen(
    pos: Mapping[str, Any],
    map_bounds: Mapping[str, Any] | None,
    screen_size: tuple[int, int],
    padding: int = 40,
) -> tuple[int, int]:
    width, height = screen_size
    map_width = _positive_float((map_bounds or {}).get("width"), 1000.0)
    map_height = _positive_float((map_bounds or {}).get("height"), 1000.0)
    usable_width = max(1, width - (padding * 2))
    usable_height = max(1, height - (padding * 2))
    scale = min(usable_width / map_width, usable_height / map_height)
    drawn_width = map_width * scale
    drawn_height = map_height * scale
    origin_x = padding + ((usable_width - drawn_width) / 2.0)
    origin_y = padding + ((usable_height - drawn_height) / 2.0)
    x = origin_x + (float(pos.get("x", 0.0)) * scale)
    y = origin_y + (float(pos.get("y", 0.0)) * scale)
    return int(round(x)), int(round(y))


def screen_to_world(
    point: tuple[int, int],
    map_bounds: Mapping[str, Any] | None,
    screen_size: tuple[int, int],
    padding: int = 40,
) -> dict[str, float]:
    left, top, width, height = map_rect(map_bounds, screen_size, padding)
    map_width = _positive_float((map_bounds or {}).get("width"), 1000.0)
    map_height = _positive_float((map_bounds or {}).get("height"), 1000.0)
    x = (float(point[0]) - left) / max(width, 1) * map_width
    y = (float(point[1]) - top) / max(height, 1) * map_height
    return {
        "x": max(0.0, min(map_width, x)),
        "y": max(0.0, min(map_height, y)),
    }


def world_radius_to_screen(
    radius: float,
    map_bounds: Mapping[str, Any] | None,
    screen_size: tuple[int, int],
    padding: int = 40,
) -> int:
    width, height = screen_size
    map_width = _positive_float((map_bounds or {}).get("width"), 1000.0)
    map_height = _positive_float((map_bounds or {}).get("height"), 1000.0)
    usable_width = max(1, width - (padding * 2))
    usable_height = max(1, height - (padding * 2))
    return int(round(float(radius) * min(usable_width / map_width, usable_height / map_height)))


def map_rect(
    map_bounds: Mapping[str, Any] | None,
    screen_size: tuple[int, int],
    padding: int = 40,
) -> tuple[int, int, int, int]:
    width, height = screen_size
    map_width = _positive_float((map_bounds or {}).get("width"), 1000.0)
    map_height = _positive_float((map_bounds or {}).get("height"), 1000.0)
    usable_width = max(1, width - (padding * 2))
    usable_height = max(1, height - (padding * 2))
    scale = min(usable_width / map_width, usable_height / map_height)
    drawn_width = int(round(map_width * scale))
    drawn_height = int(round(map_height * scale))
    left = int(round(padding + ((usable_width - drawn_width) / 2.0)))
    top = int(round(padding + ((usable_height - drawn_height) / 2.0)))
    return left, top, drawn_width, drawn_height


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default
