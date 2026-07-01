from __future__ import annotations

from typing import Any


def has_grid_line_of_sight(
    from_cell: tuple[int, int],
    to_cell: tuple[int, int],
    blocked: Any,
) -> bool:
    """Check Bresenham LOS on a ``[row_y][column_x]`` blocked grid."""

    mask = blocked.tolist() if hasattr(blocked, "tolist") else blocked
    if not mask or not mask[0]:
        return False
    height = len(mask)
    width = len(mask[0])
    start_row, start_col = int(from_cell[0]), int(from_cell[1])
    end_row, end_col = int(to_cell[0]), int(to_cell[1])
    if not (_in_bounds(start_row, start_col, height, width) and _in_bounds(end_row, end_col, height, width)):
        return False

    for index, (row, col) in enumerate(_bresenham_cells(start_row, start_col, end_row, end_col)):
        if index == 0:
            continue
        if bool(mask[row][col]):
            return False
    return True


def _bresenham_cells(start_row: int, start_col: int, end_row: int, end_col: int):
    x0, y0 = start_col, start_row
    x1, y1 = end_col, end_row
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield y0, x0
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _in_bounds(row: int, col: int, height: int, width: int) -> bool:
    return 0 <= row < height and 0 <= col < width


__all__ = ["has_grid_line_of_sight"]
