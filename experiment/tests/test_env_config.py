from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.env_config import load_env_config
from core.cpc_env import CPCEnv


def test_load_env_config_manual_enemy_right():
    config = load_env_config("configs/env/manual_enemy_right.yaml")

    assert config.player.spawn.x == 400
    assert config.player.move_speed == pytest.approx(8.0)
    assert config.enemies[0].spawn.x > config.player.spawn.x
    assert config.enemies[0].move_speed == pytest.approx(2.0)


def test_env_from_config_uses_spawn_positions():
    config = load_env_config("configs/env/manual_enemy_left.yaml")
    env = CPCEnv.from_config(config)
    env.reset()
    state = env.get_debug_state()

    assert state["agents"]["self"]["position"]["x"] == pytest.approx(config.player.spawn.x)
    assert state["agents"]["self"]["position"]["y"] == pytest.approx(config.player.spawn.y)
    assert state["agents"]["enemy"]["position"]["x"] == pytest.approx(config.enemies[0].spawn.x)
    assert state["agents"]["enemy"]["position"]["y"] == pytest.approx(config.enemies[0].spawn.y)


def test_env_from_config_uses_movement_speeds():
    config = load_env_config("configs/env/manual_enemy_right.yaml")
    env = CPCEnv.from_config(config)
    env.reset()
    before_player_x = env.state["self_pos"]["x"]
    before_enemy_x = env.state["enemy_pos"]["x"]

    env.step({"move": 4, "aim": 0, "fire": 0})

    assert env.move_speed == pytest.approx(config.player.move_speed)
    assert env.enemy_move_speed == pytest.approx(config.enemies[0].move_speed)
    assert env.state["self_pos"]["x"] == pytest.approx(before_player_x + config.player.move_speed)
    assert env.state["enemy_pos"]["x"] == pytest.approx(before_enemy_x)


def test_existing_env_creation_path_still_works():
    env = CPCEnv(seed=3, max_steps=4)
    obs = env.reset(seed=3)

    assert obs["step_count"] == 0
    assert env.max_steps == 4
    assert env.state["self_pos"] == {"x": 430.0, "y": 500.0}


def test_manual_env_debug_script_runs():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/manual_env_debug.py",
            "--config",
            "configs/env/manual_enemy_right.yaml",
            "--steps",
            "2",
            "--actions",
            "stay,right",
            "--no-gui",
            "--no-grid-png",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "reset" in result.stdout
    assert "step=1 action=stay" in result.stdout
    assert "step=2 action=right" in result.stdout
