from __future__ import annotations

import math

from experiment.core.cpc_env import CPCEnv
from experiment.core.env_config import load_env_config


def _enemy_spawn_event(env: CPCEnv) -> dict:
    env.state["enemy_pos"] = {
        "x": env.state["self_pos"]["x"] + 100.0,
        "y": env.state["self_pos"]["y"],
    }
    env.enemy_weapon["cooldown_remaining_steps"] = 0
    _, _, _, info = env.step({"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 0})
    return next(
        event
        for event in info["events"]
        if event.get("type") == "bullet_spawned"
        and event.get("owner_id") == "enemy"
    )


def test_enemy_aim_noise_zero_preserves_exact_aim():
    env = CPCEnv(seed=23, enemy_move=False, enemy_aim_noise_deg=0.0)

    event = _enemy_spawn_event(env)
    bullet = env.projectiles[-1]
    dx = env.state["self_pos"]["x"] - env.state["enemy_pos"]["x"]
    dy = env.state["self_pos"]["y"] - env.state["enemy_pos"]["y"]
    length = math.hypot(dx, dy)

    assert event["enemy_aim_noise_deg"] == 0.0
    assert event["applied_enemy_aim_noise_rad"] == 0.0
    assert math.isclose(bullet["direction"]["x"], dx / length)
    assert math.isclose(bullet["direction"]["y"], dy / length)


def test_enemy_aim_noise_is_seed_reproducible():
    first = CPCEnv(seed=29, enemy_move=False, enemy_aim_noise_deg=3.0)
    second = CPCEnv(seed=29, enemy_move=False, enemy_aim_noise_deg=3.0)

    first_event = _enemy_spawn_event(first)
    second_event = _enemy_spawn_event(second)

    assert first_event["applied_enemy_aim_noise_rad"] == second_event[
        "applied_enemy_aim_noise_rad"
    ]
    assert abs(first_event["applied_enemy_aim_noise_rad"]) <= math.radians(3.0)
    assert first.projectiles[-1]["direction"] == second.projectiles[-1]["direction"]


def test_noisy_goal_loop_config_enables_three_degree_noise():
    config = load_env_config("configs/env/autoplay_goal_loop_noisy.yaml")

    assert config.enemies[0].enemy_aim_noise_deg == 3.0
