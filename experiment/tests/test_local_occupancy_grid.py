from __future__ import annotations

import pathlib
import sys
import json

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.env_config import load_env_config
from core.local_occupancy_grid import (
    CHANNEL_AGENT,
    CHANNEL_ENEMY,
    CHANNEL_OBSTACLE,
    build_local_occupancy_grid,
    render_grid_to_png,
)
from core.cpc_env import CPCEnv
from scripts.manual_env_debug import export_debug_snapshot


def test_local_occupancy_grid_shape_and_channels():
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    env.reset()

    grid = build_local_occupancy_grid(env.get_debug_state())

    assert grid.shape == (21, 21, 4)
    assert grid.channel_names == ("obstacle", "enemy", "hazard", "agent")
    assert grid.cells[10][10][grid.channel_index(CHANNEL_AGENT)] == 1.0


def test_enemy_left_and_right_land_on_expected_side():
    right_env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    left_env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_left.yaml"))
    right_env.reset()
    left_env.reset()

    right_grid = build_local_occupancy_grid(right_env.get_debug_state())
    left_grid = build_local_occupancy_grid(left_env.get_debug_state())
    enemy_channel = right_grid.channel_index(CHANNEL_ENEMY)

    assert right_grid.cells[10][15][enemy_channel] == 1.0
    assert left_grid.cells[10][5][enemy_channel] == 1.0


def test_obstacle_channel_and_png_generation(tmp_path):
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    env.reset()
    grid = build_local_occupancy_grid(env.get_debug_state())
    obstacle_channel = grid.channel_index(CHANNEL_OBSTACLE)
    path = tmp_path / "grid.png"

    assert grid.cells[10][12][obstacle_channel] == 1.0

    render_grid_to_png(grid, path)

    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_manual_env_debug_snapshot_exports_grid_and_status(tmp_path):
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    env.reset()

    saved = export_debug_snapshot(env, "configs/env/manual_enemy_right.yaml", tmp_path, "step_000_reset")

    assert saved["grid_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    status = json.loads(saved["status_json"].read_text(encoding="utf-8"))
    assert status["step"] == 0
    assert status["grid"]["shape"] == [21, 21, 4]
    assert status["env_state"]["agents"]["self"]["position"]["x"] == 400.0
