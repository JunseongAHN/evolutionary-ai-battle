from __future__ import annotations

import math
import pathlib
import subprocess
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.cpc_env import CPCEnv
from core.env_config import env_config_from_dict, load_env_config
from gui.pygame_viewer import _panel_lines


STAY = {"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 0}


def test_goal_disabled_preserves_existing_behavior():
    env = CPCEnv.from_config(load_env_config("configs/env/manual_enemy_right.yaml"))
    before = env.reset(seed=11)

    after, _, done, info = env.step(STAY)

    assert before["goal_enabled"] is False
    assert after["goal_enabled"] is False
    assert env.goal_position is None
    assert not any(event["type"].startswith("goal_") for event in info["events"])
    assert done is False


def test_goal_reached_event_emitted_when_player_enters_radius():
    env = _goal_env(respawn_on_reach=False)
    env.state["self_pos"] = _goal_position_dict(env)

    _, _, done, info = env.step(STAY)

    assert [event["type"] for event in info["events"]] == ["goal_reached"]
    assert info["events"][0]["goal_reached_count"] == 1
    assert env.goal_reached_count == 1
    assert done is False


def test_goal_respawns_after_reach_when_enabled():
    env = _goal_env(respawn_on_reach=True)
    previous_goal = env.goal_position
    env.state["self_pos"] = _goal_position_dict(env)

    _, _, _, info = env.step(STAY)

    assert env.goal_position is not None
    assert env.goal_position != previous_goal
    assert [event["type"] for event in info["events"]] == ["goal_reached", "goal_respawned"]


def test_goal_respawn_is_seed_deterministic():
    first = _goal_env(seed=29, respawn_on_reach=True)
    second = _goal_env(seed=29, respawn_on_reach=True)
    first.state["self_pos"] = _goal_position_dict(first)
    second.state["self_pos"] = _goal_position_dict(second)

    first.step(STAY)
    second.step(STAY)

    assert first.goal_position == second.goal_position


def test_goal_max_respawns_stops_goal_without_ending_episode():
    env = _goal_env(respawn_on_reach=True, max_respawns=0)
    env.state["self_pos"] = _goal_position_dict(env)

    _, _, done, info = env.step(STAY)

    assert env.goal_position is None
    assert [event["type"] for event in info["events"]] == ["goal_reached"]
    assert done is False


def test_enemy_spawns_on_goal_reached_when_enabled():
    env = _goal_env(respawn_on_reach=True, spawn_enemy_on_reach=True)
    old_enemy_id = env.enemy_id
    env.state["self_pos"] = _goal_position_dict(env)

    _, _, _, info = env.step(STAY)

    event = next(event for event in info["events"] if event["type"] == "enemy_spawned")
    assert env.enemy_id != old_enemy_id
    assert event["enemy_id"] == env.enemy_id
    assert event["reason"] == "goal_reached"
    assert event["position"] != [env.state["self_pos"]["x"], env.state["self_pos"]["y"]]
    assert env.state["enemy_hp"] == env.enemy_max_hp


def test_goal_and_enemy_do_not_spawn_inside_obstacles():
    env = _goal_env(
        position={"x": 400.0, "y": 400.0},
        respawn_on_reach=True,
        spawn_enemy_on_reach=True,
        obstacles=[{"id": "center", "type": "circle", "x": 400.0, "y": 400.0, "radius": 110.0}],
    )
    _assert_clear_of_obstacles(env.goal_position, env.goal_radius, env.obstacles)
    env.state["self_pos"] = _goal_position_dict(env)

    env.step(STAY)

    _assert_clear_of_obstacles(env.goal_position, env.goal_radius, env.obstacles)
    enemy_position = (env.state["enemy_pos"]["x"], env.state["enemy_pos"]["y"])
    _assert_clear_of_obstacles(enemy_position, env.enemy_radius, env.obstacles)


def test_snapshot_contains_goal_enemy_bullet_event_fields():
    env = _goal_env(respawn_on_reach=True, spawn_enemy_on_reach=True)
    env.state["self_pos"] = _goal_position_dict(env)
    env.step({**STAY, "fire": 1})

    snapshot = env.get_snapshot()

    assert set(snapshot) == {"step", "map", "player", "enemies", "bullets", "obstacles", "goal", "events"}
    assert {"position", "hp", "alive"}.issubset(snapshot["player"])
    assert {"id", "position", "hp", "alive"}.issubset(snapshot["enemies"][0])
    assert {"enabled", "position", "radius", "reached_count"}.issubset(snapshot["goal"])
    assert snapshot["events"]
    assert snapshot["bullets"]
    assert {"position", "velocity", "owner_id", "team", "ttl"}.issubset(snapshot["bullets"][0])

    snapshot["player"]["position"][0] = -999.0
    snapshot["events"].clear()
    fresh = env.get_snapshot()
    assert fresh["player"]["position"][0] != -999.0
    assert fresh["events"]


def test_debug_panel_contains_goal_count_distance_and_events():
    env = _goal_env(respawn_on_reach=True, spawn_enemy_on_reach=True)
    env.state["self_pos"] = _goal_position_dict(env)
    _, reward, done, info = env.step(STAY)

    lines = _panel_lines(
        env.get_debug_state(),
        {"env": {"info": info, "rewards": {"agent": reward}, "done": done}},
    )

    assert any(line.startswith("goal: (") for line in lines)
    assert any(line.startswith("goal dist: ") for line in lines)
    assert "goal count: 1" in lines
    assert any("goal_reached" in line and "enemy_spawned" in line for line in lines)


def test_autoplay_goal_loop_config_loads():
    config = load_env_config("configs/env/autoplay_goal_loop.yaml")
    env = CPCEnv.from_config(config)

    assert config.goal.enabled is True
    assert config.enemies[0].move_speed == 5.0
    assert config.enemies[0].behavior == "pursue"
    assert env.goal_position == (680.0, 680.0)
    assert env.get_snapshot()["goal"]["enabled"] is True


def test_pursuing_enemy_moves_around_obstacle():
    obstacle = {"id": "center", "type": "circle", "x": 400.0, "y": 400.0, "radius": 80.0}
    env = _goal_env(obstacles=[obstacle])
    env.max_steps = 80
    env.enemy_move = True
    env.enemy_behavior = "pursue"
    env.enemy_move_speed = 10.0
    env.state["self_pos"] = {"x": 150.0, "y": 400.0}
    env.state["enemy_pos"] = {"x": 650.0, "y": 400.0}
    initial_distance = env._distance(env.state["self_pos"], env.state["enemy_pos"])

    moved_off_direct_line = False
    for _ in range(50):
        env.step(STAY)
        enemy = env.state["enemy_pos"]
        obstacle_distance = math.hypot(enemy["x"] - obstacle["x"], enemy["y"] - obstacle["y"])
        assert obstacle_distance >= obstacle["radius"] + env.enemy_radius - 1e-3
        moved_off_direct_line = moved_off_direct_line or abs(enemy["y"] - 400.0) > 1.0

    assert moved_off_direct_line is True
    assert env._distance(env.state["self_pos"], env.state["enemy_pos"]) < initial_distance


def test_manual_debug_defaults_to_visible_goal_scenario():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/manual_env_debug.py",
            "--no-gui",
            "--steps",
            "0",
            "--no-grid-png",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "config=configs/env/autoplay_goal_loop.yaml" in result.stdout
    assert "goal=(680.0,680.0)" in result.stdout


def _goal_env(
    *,
    seed: int = 7,
    position: dict[str, float] | None = None,
    respawn_on_reach: bool = True,
    spawn_enemy_on_reach: bool = False,
    obstacles: list[dict] | None = None,
    max_respawns: int | None = None,
) -> CPCEnv:
    config = env_config_from_dict(
        {
            "env": {"seed": seed, "max_steps": 30, "dt": 1.0},
            "map": {"width": 800, "height": 800},
            "player": {
                "spawn": {"x": 100, "y": 100},
                "radius": 12,
                "hp": 100,
                "move_speed": 20,
                "aim_turn_speed": 1,
                "weapon_range": 260,
                "fire_cooldown_steps": 5,
                "bullet_speed": 140,
            },
            "enemies": [
                {
                    "id": "enemy-test",
                    "spawn": {"x": 650, "y": 100},
                    "radius": 12,
                    "hp": 100,
                    "move_speed": 0,
                    "behavior": "stationary",
                }
            ],
            "obstacles": obstacles or [],
            "goal": {
                "enabled": True,
                "position": position or {"x": 260, "y": 260},
                "radius": 24,
                "respawn_on_reach": respawn_on_reach,
                "spawn_enemy_on_reach": spawn_enemy_on_reach,
                "respawn_margin": 80,
                "max_respawns": max_respawns,
            },
            "zone": {"enabled": False},
        }
    )
    env = CPCEnv.from_config(config)
    env.enemy_fire = False
    env.reset(seed=seed)
    return env


def _goal_position_dict(env: CPCEnv) -> dict[str, float]:
    assert env.goal_position is not None
    return {"x": env.goal_position[0], "y": env.goal_position[1]}


def _assert_clear_of_obstacles(
    position: tuple[float, float] | None,
    radius: float,
    obstacles: list[dict],
) -> None:
    assert position is not None
    for obstacle in obstacles:
        distance = math.hypot(position[0] - obstacle["x"], position[1] - obstacle["y"])
        assert distance > radius + obstacle["radius"]
