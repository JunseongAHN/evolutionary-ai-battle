from __future__ import annotations

import math
import pathlib
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


NOOP = {"move": 0, "aim": 0, "fire": 0}
MOVE_RIGHT = {"move": 4, "aim": 0, "fire": 0}
FIRE_RIGHT = {"move": 0, "aim": 0, "fire": 1}


def test_player_movement_stops_at_circle_obstacle():
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    env.reset()
    obstacle = env.obstacles[0]

    for _ in range(20):
        env.step(MOVE_RIGHT)

    player = env.state["self_pos"]
    min_distance = env.player_radius + obstacle["radius"]

    assert math.dist((player["x"], player["y"]), (obstacle["x"], obstacle["y"])) >= min_distance - 1e-4
    assert player["x"] <= obstacle["x"] - min_distance + 1e-4


def test_bullet_hits_obstacle_before_enemy():
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    env.reset()
    before_enemy_hp = env.state["enemy_hp"]

    _, _, _, fire_info = env.step(FIRE_RIGHT)
    _, _, _, obstacle_info = env.step(NOOP)

    assert fire_info["bullet_spawned"] is True
    assert env.state["enemy_hp"] == pytest.approx(before_enemy_hp)
    assert env.projectiles == []
    assert any(event["type"] == "bullet_hit_obstacle" for event in obstacle_info["bullet_events"])
    assert not any(event["type"] == "bullet_hit" for event in obstacle_info["bullet_events"])


def test_enemy_bullet_spawns_and_hits_obstacle_before_player():
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    env.reset()
    before_player_hp = env.state["self_hp"]

    _, _, _, fire_info = env.step(NOOP)
    _, _, _, obstacle_info = env.step(NOOP)

    assert env.state["self_hp"] == pytest.approx(before_player_hp)
    assert any(
        event["type"] == "bullet_spawned" and event["owner_id"] == "enemy"
        for event in fire_info["bullet_events"]
    )
    assert any(
        event["type"] == "bullet_hit_obstacle" and event["owner_id"] == "enemy"
        for event in obstacle_info["bullet_events"]
    )
