from __future__ import annotations

import math

try:
    from experiment.core.cpc_actions import vec_to_aim_bin as _core_vec_to_aim_bin
except ModuleNotFoundError:
    try:
        from core.cpc_actions import vec_to_aim_bin as _core_vec_to_aim_bin
    except ModuleNotFoundError:
        _core_vec_to_aim_bin = None


"""
Coordinate conventions for this baseline:

* local occupancy grids are indexed as cells[row_y][column_x][channel]
* the grid center cell corresponds to the controlled agent position
* x/column increases to the right
* row_y increases in the same direction as env/world y
* the CPC toy env uses screen-style world coordinates, so +y is down
* local vectors are therefore dx right, dy down
* aim bin 0 points along +x/right
* aim bins follow the existing CPC action convention
* because +y is down, increasing aim bins rotate clockwise on screen

For 16 bins this means:
right=0, down=4, left=8, up=12.
"""


def angle_to_aim_bin(angle_rad: float, num_bins: int) -> int:
    """Convert an angle from +x/right into the nearest CPC aim bin.

    The angle is expected to be produced by ``atan2(dy, dx)`` in the env/local
    coordinate frame where positive y points down. Positive angles therefore
    rotate clockwise on screen.
    """

    bins = _validate_num_bins(num_bins)
    angle = float(angle_rad) % (2.0 * math.pi)
    return int(round((angle / (2.0 * math.pi)) * bins)) % bins


def vector_to_aim_bin(dx: float, dy: float, num_bins: int) -> int:
    """Convert a local vector to an aim bin using existing CPC conventions."""

    bins = _validate_num_bins(num_bins)
    x = float(dx)
    y = float(dy)
    if math.hypot(x, y) <= 1e-6:
        return 0
    if _core_vec_to_aim_bin is not None:
        return int(_core_vec_to_aim_bin({"x": x, "y": y}, bins))
    return angle_to_aim_bin(math.atan2(y, x), bins)


def grid_cell_to_local_vector(
    cell_y: int,
    cell_x: int,
    grid_size: int,
    cell_size: float,
) -> tuple[float, float]:
    """Map a grid cell to a local env vector from the agent center.

    ``cell_y`` is the row index and ``cell_x`` is the column index. The center
    cell ``(grid_size // 2, grid_size // 2)`` maps to ``(0.0, 0.0)``. Columns to
    the right produce positive dx. Rows below the center produce positive dy.
    """

    size = int(grid_size)
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"grid_size must be a positive odd integer, got {grid_size!r}")
    if not 0 <= int(cell_y) < size:
        raise ValueError(f"cell_y must be in [0, {size - 1}], got {cell_y!r}")
    if not 0 <= int(cell_x) < size:
        raise ValueError(f"cell_x must be in [0, {size - 1}], got {cell_x!r}")
    scale = float(cell_size)
    if scale <= 0.0:
        raise ValueError(f"cell_size must be positive, got {cell_size!r}")
    center = size // 2
    return (float(int(cell_x) - center) * scale, float(int(cell_y) - center) * scale)


def enemy_cell_to_aim_bin(
    cell_y: int,
    cell_x: int,
    grid_size: int,
    cell_size: float,
    num_bins: int,
) -> int:
    """Convert an enemy grid cell directly to the corresponding aim bin."""

    dx, dy = grid_cell_to_local_vector(
        cell_y=cell_y,
        cell_x=cell_x,
        grid_size=grid_size,
        cell_size=cell_size,
    )
    return vector_to_aim_bin(dx, dy, num_bins)


def _validate_num_bins(num_bins: int) -> int:
    bins = int(num_bins)
    if bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins!r}")
    return bins


__all__ = [
    "angle_to_aim_bin",
    "enemy_cell_to_aim_bin",
    "grid_cell_to_local_vector",
    "vector_to_aim_bin",
]
