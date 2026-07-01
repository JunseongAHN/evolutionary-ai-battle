from __future__ import annotations

import pathlib
import math
import subprocess
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.aim_oracle.aim_bin_utils import grid_cell_to_local_vector
from baselines.aim_oracle.enemy_cell_utils import find_nearest_enemy_cell
from baselines.aim_oracle.tactical_aim_oracle_bot import TacticalAimOracleBot
from core.cpc_actions import decode_action
from core.cpc_env import CPCEnv
from core.env_config import load_env_config
from core.local_occupancy_grid import CHANNEL_ENEMY, build_local_occupancy_grid


ENEMY_CHANNEL = 1
CELL_SIZE = 40.0
GRID_SIZE = 21


def test_enemy_right_maps_to_right_continuous_aim():
    action, debug = _oracle_action_for_cell(10, 15)

    assert debug["local_vector"] == [200.0, 0.0]
    assert action["aim_dx"] == 1.0
    assert action["aim_dy"] == 0.0


def test_enemy_left_maps_to_left_continuous_aim():
    action, debug = _oracle_action_for_cell(10, 5)

    assert debug["local_vector"] == [-200.0, 0.0]
    assert action["aim_dx"] == -1.0
    assert action["aim_dy"] == 0.0


def test_enemy_up_maps_to_up_continuous_aim():
    action, debug = _oracle_action_for_cell(5, 10)

    assert debug["local_vector"] == [0.0, -200.0]
    assert action["aim_dx"] == 0.0
    assert action["aim_dy"] == -1.0


def test_enemy_down_maps_to_down_continuous_aim():
    action, debug = _oracle_action_for_cell(15, 10)

    assert debug["local_vector"] == [0.0, 200.0]
    assert action["aim_dx"] == 0.0
    assert action["aim_dy"] == 1.0


def test_no_enemy_returns_valid_default_action():
    bot = TacticalAimOracleBot(
        enemy_channel_index=ENEMY_CHANNEL,
        cell_size=CELL_SIZE,
        stay_move_bin=0,
    )

    action, debug = bot.act({"local_occupancy_grid": _empty_grid()})

    assert action == {"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 0}
    assert debug["enemy_cell"] is None
    assert debug["local_vector"] is None
    assert debug["reason"] == "no_enemy_visible_in_local_grid"


def test_enemy_world_position_takes_priority_over_quantized_grid_cell():
    bot = TacticalAimOracleBot(
        enemy_channel_index=ENEMY_CHANNEL,
        cell_size=CELL_SIZE,
        stay_move_bin=0,
    )
    grid = _empty_grid()
    grid[10][15][ENEMY_CHANNEL] = 1.0

    action, debug = bot.act(
        {
            "self_pos": {"x": 100.0, "y": 100.0},
            "enemy_pos": {"x": 50.0, "y": 50.0},
            "enemy_hp": 100.0,
            "local_occupancy_grid": grid,
        }
    )

    expected = -1.0 / math.sqrt(2.0)
    assert math.isclose(action["aim_dx"], expected)
    assert math.isclose(action["aim_dy"], expected)
    assert debug["local_vector"] == [-50.0, -50.0]
    assert debug["aim_source"] == "enemy_position"


def test_env_step_preserves_continuous_enemy_direction():
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_left.yaml"))
    obs = env.reset()
    grid = build_local_occupancy_grid(env.get_debug_state(), agent_id="self")
    bot = TacticalAimOracleBot(
        enemy_channel_index=grid.channel_index(CHANNEL_ENEMY),
        cell_size=grid.cell_size,
        stay_move_bin=0,
    )
    action, _ = bot.act({**obs, "local_occupancy_grid": grid})

    _, _, _, info = env.step(action)

    assert info["decoded_action"]["aimX"] == -1.0
    assert info["decoded_action"]["aimY"] == 0.0
    assert info["raw_action"]["aim_dx"] == -1.0


def test_decode_action_does_not_silently_default_missing_aim_to_right():
    try:
        decode_action({"move": 0, "fire": 0})
    except ValueError as exc:
        assert "must include" in str(exc)
    else:
        raise AssertionError("missing aim unexpectedly decoded as right")


def test_find_nearest_enemy_cell_returns_none_when_empty():
    assert find_nearest_enemy_cell(_empty_grid(), ENEMY_CHANNEL) is None


def test_find_nearest_enemy_cell_picks_nearest_active_cell():
    grid = _empty_grid(size=21)
    grid[10][15][ENEMY_CHANNEL] = 1.0
    grid[10][11][ENEMY_CHANNEL] = 1.0
    grid[0][0][ENEMY_CHANNEL] = 1.0

    assert find_nearest_enemy_cell(grid, ENEMY_CHANNEL) == (10, 11)


def test_grid_cell_to_local_vector_uses_center_cell_without_offset():
    assert grid_cell_to_local_vector(10, 10, GRID_SIZE, CELL_SIZE) == (0.0, 0.0)
    assert grid_cell_to_local_vector(9, 10, GRID_SIZE, CELL_SIZE) == (0.0, -40.0)
    assert grid_cell_to_local_vector(10, 11, GRID_SIZE, CELL_SIZE) == (40.0, 0.0)


def test_manual_up_down_configs_place_enemy_on_expected_grid_rows():
    up_grid = _grid_from_config("configs/env/manual_enemy_up.yaml")
    down_grid = _grid_from_config("configs/env/manual_enemy_down.yaml")
    enemy_channel = up_grid.channel_index(CHANNEL_ENEMY)

    assert find_nearest_enemy_cell(up_grid, enemy_channel) == (5, 10)
    assert find_nearest_enemy_cell(down_grid, enemy_channel) == (15, 10)


def test_aim_oracle_debug_runner_runs():
    result = subprocess.run(
        [
            sys.executable,
            "experiment/baselines/aim_oracle/run_aim_oracle_debug.py",
            "--config",
            "configs/env/manual_enemy_right.yaml",
            "--steps",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "aim_direction=[1.0, 0.0]" in result.stdout
    assert "aim_source=enemy_position" in result.stdout


def _oracle_action_for_cell(cell_y: int, cell_x: int) -> tuple[dict[str, int], dict]:
    bot = TacticalAimOracleBot(
        enemy_channel_index=ENEMY_CHANNEL,
        cell_size=CELL_SIZE,
        stay_move_bin=0,
    )
    grid = _empty_grid()
    grid[cell_y][cell_x][ENEMY_CHANNEL] = 1.0
    return bot.act({"local_occupancy_grid": grid})


def _empty_grid(size: int = GRID_SIZE, channels: int = 4) -> list[list[list[float]]]:
    return [[[0.0 for _ in range(channels)] for _ in range(size)] for _ in range(size)]


def _grid_from_config(config_path: str):
    env = CPCEnv.from_config(load_env_config(config_path))
    env.reset()
    return build_local_occupancy_grid(env.get_debug_state(), agent_id="self")
