from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from training.cpc_env import CPCEnv


NOOP = {"move": 0, "aim": 0, "fire": 0}
MOVE_RIGHT = {"move": 4, "aim": 0, "fire": 0}
MOVE_LEFT = {"move": 3, "aim": 0, "fire": 0}
AIM_RIGHT = {"move": 0, "aim": 0, "fire": 0}
FIRE_RIGHT = {"move": 0, "aim": 0, "fire": 1}
AIM_LEFT = {"move": 0, "aim": 8, "fire": 0}
FIRE_LEFT = {"move": 0, "aim": 8, "fire": 1}
MOVE_DOWN_RIGHT = {"move": 8, "aim": 0, "fire": 0}


def put_enemy_to_right(env: CPCEnv, *, distance: float = 120.0) -> None:
    env.state["self_pos"] = {"x": 500.0, "y": 500.0}
    env.state["ally_pos"] = {"x": 450.0, "y": 540.0}
    env.state["enemy_pos"] = {"x": 500.0 + distance, "y": 500.0}


def test_noop_full_episode_no_longer_gets_high_survival_reward():
    env = CPCEnv(seed=1, max_steps=100)
    total = 0.0
    done = False

    while not done:
        _, reward, done, _ = env.step(NOOP)
        total += reward

    assert total < 0.5


def test_moving_toward_enemy_rewards_more_than_moving_away_in_direct_contact():
    toward = CPCEnv(seed=1, max_steps=4)
    away = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(toward)
    put_enemy_to_right(away)

    _, toward_reward, _, toward_info = toward.step(MOVE_RIGHT)
    _, away_reward, _, away_info = away.step(MOVE_LEFT)

    assert toward_info["reward_components"]["approach_enemy"] > away_info["reward_components"]["approach_enemy"]
    assert toward_reward > away_reward


def test_aim_aligned_with_enemy_gives_positive_alignment_reward():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(AIM_RIGHT)

    assert info["reward_components"]["aim_alignment"] > 0.0


def test_aim_debug_points_to_enemy():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(AIM_RIGHT)

    aim_debug = info["aim_debug"]
    assert aim_debug["target_enemy_id"] == "enemy"
    assert aim_debug["ideal_aim_bin"] == 0
    assert aim_debug["aim_alignment"] == pytest.approx(1.0)
    assert aim_debug["is_aim_aligned"] is True


def test_wrong_aim_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(AIM_LEFT)

    assert info["aim_debug"]["aim_alignment"] < 0.0
    assert info["reward_components"]["bad_aim"] < 0.0


def test_fire_in_range_and_aligned_gives_attack_intent_reward():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert info["reward_components"]["attack_intent"] > 0.0


def test_aligned_shot_bonus():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert info["fire"]["shot_fired"] is True
    assert info["reward_components"]["aligned_shot"] > 0.0


def test_off_target_shot_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(FIRE_LEFT)

    assert info["fire"]["shot_fired"] is True
    assert info["reward_components"]["off_target_shot"] < 0.0


def test_outside_safe_zone_gives_zone_pressure_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    env.state["self_pos"] = {"x": 0.0, "y": 0.0}
    env.state["enemy_pos"] = {"x": 100.0, "y": 0.0}

    _, _, _, info = env.step(NOOP)

    assert info["zone_debug"]["outside_safe_zone"] is True
    assert info["reward_components"]["zone_pressure"] <= -0.20


def test_return_to_zone_reward():
    env = CPCEnv(seed=1, max_steps=4)
    env.state["self_pos"] = {"x": 0.0, "y": 0.0}
    env.state["enemy_pos"] = {"x": 100.0, "y": 0.0}

    _, _, _, info = env.step(MOVE_DOWN_RIGHT)

    assert info["zone_debug"]["outside_safe_zone"] is True
    assert info["zone_debug"]["move_toward_center"] > 0.5
    assert info["reward_components"]["return_to_zone"] > 0.0


def test_near_edge_outward_penalty():
    env = CPCEnv(seed=1, max_steps=100)
    env.state["self_pos"] = {"x": 880.0, "y": 500.0}
    env.state["enemy_pos"] = {"x": 700.0, "y": 500.0}

    _, _, _, info = env.step(MOVE_RIGHT)

    assert info["zone_debug"]["outside_safe_zone"] is False
    assert info["reward_components"]["near_edge_outward"] < 0.0


def test_damage_dealt_reward_is_positive():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    env.step(FIRE_RIGHT)
    _, _, _, info = env.step(NOOP)

    assert info["damage_delta"]["enemy_hp"] > 0.0
    assert info["reward_components"]["damage_dealt"] > 0.0


def test_bullet_hit_reward_only_on_hit():
    hit = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(hit, distance=120.0)
    hit.step(FIRE_RIGHT)
    _, _, _, hit_info = hit.step(NOOP)

    miss = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(miss, distance=400.0)
    miss.step(FIRE_RIGHT)
    _, _, _, miss_info = miss.step(NOOP)

    assert hit_info["reward_components"]["bullet_hit"] > 0.0
    assert miss_info["reward_components"]["bullet_hit"] == 0.0


def test_bullet_spawns_at_shooter_position():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=240.0)
    shooter_pos = dict(env.state["self_pos"])

    _, _, _, info = env.step(FIRE_RIGHT)

    bullet = env.projectiles[0]
    assert bullet["pos"] == shooter_pos
    assert bullet["spawn_pos"] == shooter_pos
    assert bullet["traveled_distance"] == 0.0
    assert info["bullet_spawned"] is True
    assert info["bullet_events"][-1]["type"] == "bullet_spawned"


def test_bullet_does_not_spawn_at_max_range():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=240.0)
    shooter_pos = dict(env.state["self_pos"])

    env.step(FIRE_RIGHT)

    assert env._distance(shooter_pos, env.projectiles[0]["pos"]) < env.fire_range


def test_bullet_moves_over_time():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=400.0)

    env.step(FIRE_RIGHT)
    spawned_x = env.projectiles[0]["pos"]["x"]
    _, _, _, info = env.step(NOOP)

    assert env.projectiles[0]["pos"]["x"] > spawned_x
    assert env.projectiles[0]["traveled_distance"] == env.projectile_speed
    assert any(event["type"] == "bullet_moved" for event in info["bullet_events"])


def test_bullet_expires_at_max_range():
    env = CPCEnv(seed=1, max_steps=4)
    env.fire_range = 50.0
    put_enemy_to_right(env, distance=400.0)

    env.step(FIRE_RIGHT)
    _, _, _, info = env.step(NOOP)

    assert env.projectiles == []
    assert any(event["type"] == "bullet_expired" for event in info["bullet_events"])
    assert info["reward_components"]["missed_shot"] < 0.0


def test_bullet_hit_applies_damage():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)
    before_hp = env.state["enemy_hp"]

    env.step(FIRE_RIGHT)
    _, _, _, info = env.step(NOOP)

    assert env.state["enemy_hp"] == before_hp - env.damage
    assert any(event["type"] == "bullet_hit" for event in info["bullet_events"])


def test_damage_not_applied_before_bullet_hit():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=240.0)
    before_hp = env.state["enemy_hp"]

    _, _, _, info = env.step(FIRE_RIGHT)

    assert env.state["enemy_hp"] == before_hp
    assert info["damage_delta"]["enemy_hp"] == 0.0


def test_fire_requested_does_not_always_spawn_bullet():
    env = CPCEnv(seed=1, max_steps=8)
    env.fire_interval_steps = 5
    env.reset(seed=1)
    put_enemy_to_right(env, distance=400.0)

    spawned_count = 0
    for _ in range(5):
        _, _, _, info = env.step(FIRE_RIGHT)
        spawned_count += int(info["fire"]["shot_fired"])

    assert spawned_count == 1


def test_cooldown_blocks_repeated_fire():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=400.0)

    _, _, _, first_info = env.step(FIRE_RIGHT)
    _, _, _, second_info = env.step(FIRE_RIGHT)

    assert first_info["fire"]["shot_fired"] is True
    assert second_info["fire"]["fire_requested"] is True
    assert second_info["fire"]["shot_fired"] is False
    assert second_info["fire"]["fire_blocked_reason"] == "cooldown"


def test_cooldown_allows_fire_after_interval():
    env = CPCEnv(seed=1, max_steps=8)
    env.fire_interval_steps = 3
    env.reset(seed=1)
    put_enemy_to_right(env, distance=400.0)

    _, _, _, first_info = env.step(FIRE_RIGHT)
    env.step(NOOP)
    env.step(NOOP)
    _, _, _, second_info = env.step(FIRE_RIGHT)

    assert first_info["fire"]["shot_fired"] is True
    assert second_info["fire"]["shot_fired"] is True


def test_damage_only_on_bullet_hit():
    env = CPCEnv(seed=1, max_steps=6)
    put_enemy_to_right(env, distance=400.0)
    before_hp = env.state["enemy_hp"]

    env.step(FIRE_RIGHT)
    _, _, _, cooldown_info = env.step(FIRE_RIGHT)

    assert cooldown_info["fire"]["fire_blocked_reason"] == "cooldown"
    assert env.state["enemy_hp"] == before_hp


def test_info_contains_fire_debug():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=400.0)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert set(info["fire"]) >= {
        "fire_requested",
        "shot_fired",
        "fire_blocked_reason",
        "cooldown_remaining_steps_before",
        "cooldown_remaining_steps_after",
        "fire_interval_steps",
    }


def test_observation_contains_can_fire():
    env = CPCEnv(seed=1, max_steps=4)
    obs = env.reset(seed=1)

    assert obs["can_fire"] is True
    assert obs["weapon_cooldown_fraction"] == 0.0


def test_observation_contains_aim_and_zone_fields():
    env = CPCEnv(seed=1, max_steps=4)
    obs = env.reset(seed=1)

    assert {"target_dir_x", "target_dir_y", "aim_alignment", "distance_to_center", "safe_margin_fraction", "outside_safe_zone"} <= set(obs)


def test_randomized_spawn_requires_non_right_aim():
    ideal_bins = set()
    for seed in range(12):
        env = CPCEnv(seed=seed, max_steps=4, randomize_enemy_spawn_direction=True)
        ideal_bins.add(env._aim_debug(0, {"aimX": 1.0, "aimY": 0.0})["ideal_aim_bin"])

    assert any(aim_bin != 0 for aim_bin in ideal_bins)


def test_projectile_moves_forward_and_damages_on_collision():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)
    before_hp = env.state["enemy_hp"]

    _, _, _, first_info = env.step(FIRE_RIGHT)

    assert first_info["damage_delta"]["enemy_hp"] == 0.0
    assert len(env.projectiles) == 1
    assert env.projectiles[0]["pos"]["x"] == env.state["self_pos"]["x"]

    _, _, _, second_info = env.step(NOOP)

    assert second_info["damage_delta"]["enemy_hp"] > 0.0
    assert env.state["enemy_hp"] == before_hp - env.damage
    assert env.projectiles == []


def test_damage_taken_penalty_is_negative():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(NOOP)

    assert info["damage_delta"]["self_hp"] > 0.0
    assert info["reward_components"]["damage_taken"] < 0.0
