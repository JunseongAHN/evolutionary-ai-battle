from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from training.cpc_env import AIM_BINS, CPCEnv, aim_bin_to_vec, circular_bin_distance, vec_to_aim_bin


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


def test_noop_does_not_get_survival_or_fire_request_reward():
    env = CPCEnv(seed=1, max_steps=100)

    _, _, _, info = env.step(NOOP)

    assert "survival" not in info["reward_components"]
    assert "attack_intent" not in info["reward_components"]
    assert "zone_pressure" not in info["reward_components"]


def test_stage1_safe_zone_does_not_shrink():
    env = CPCEnv(seed=1, max_steps=10)
    initial_radius = env._safe_radius()

    for _ in range(5):
        env.step(NOOP)

    assert env._safe_radius() == initial_radius


def test_no_aim_reward_without_shot():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(AIM_RIGHT)

    assert info["aim_debug"]["aim_bin_error"] == 0
    assert info["fire"]["shot_fired"] is False
    assert info["reward_components"]["aim_bin_exact"] == 0.0
    assert info["reward_components"]["aim_alignment"] == 0.0


def test_circular_bin_distance_wraparound():
    assert circular_bin_distance(0, 0, 16) == 0
    assert circular_bin_distance(0, 1, 16) == 1
    assert circular_bin_distance(0, 15, 16) == 1
    assert circular_bin_distance(0, 8, 16) == 8


def test_vec_to_aim_bin_matches_aim_bin_to_vec_convention():
    assert vec_to_aim_bin({"x": 1.0, "y": 0.0}) == 0
    assert vec_to_aim_bin({"x": -1.0, "y": 0.0}) == 8
    assert vec_to_aim_bin({"x": 0.0, "y": 1.0}) == 4
    assert vec_to_aim_bin({"x": 0.0, "y": -1.0}) == 12
    for aim_bin in range(AIM_BINS):
        assert vec_to_aim_bin(aim_bin_to_vec(aim_bin)) == aim_bin


def test_aim_debug_points_to_enemy():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(AIM_RIGHT)

    aim_debug = info["aim_debug"]
    assert aim_debug["target_enemy_id"] == "enemy"
    assert aim_debug["ideal_aim_bin"] == 0
    assert aim_debug["aim_bin_error"] == 0
    assert aim_debug["aim_alignment"] == pytest.approx(1.0)
    assert aim_debug["is_exact_aim"] is True
    assert aim_debug["is_near_aim"] is True
    assert aim_debug["is_aim_aligned"] is True


def test_aim_reward_with_exact_shot():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(FIRE_RIGHT)

    components = info["reward_components"]
    assert info["aim_debug"]["aim_bin_error"] == 0
    assert components["aim_bin_exact"] == pytest.approx(0.04)
    assert components["aim_alignment"] > 0.0
    assert components["aim_bin_wrong"] == 0.0


def test_neighbor_aim_reward():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step({"move": 0, "aim": 1, "fire": 0})

    components = info["reward_components"]
    assert info["aim_debug"]["aim_bin_error"] == 1
    assert components["aim_bin_exact"] == 0.0
    assert components["aim_bin_wrong"] == 0.0


def test_wrong_aim_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(AIM_LEFT)

    assert info["aim_debug"]["aim_alignment"] < 0.0
    assert info["aim_debug"]["aim_bin_error"] >= 3
    assert info["reward_components"]["aim_bin_wrong"] == 0.0


def test_wrong_aim_penalty_only_on_shot():
    no_shot = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(no_shot, distance=120.0)
    _, _, _, no_shot_info = no_shot.step(AIM_LEFT)

    fired = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(fired, distance=120.0)
    _, _, _, fired_info = fired.step(FIRE_LEFT)

    assert no_shot_info["aim_debug"]["aim_bin_error"] >= 2
    assert no_shot_info["reward_components"]["aim_bin_wrong"] == 0.0
    assert fired_info["fire"]["shot_fired"] is True
    assert fired_info["reward_components"]["aim_bin_wrong"] < 0.0


def test_fire_requested_gets_no_direct_reward():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert "attack_intent" not in info["reward_components"]


def test_aim_shaping_is_small():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert info["fire"]["shot_fired"] is True
    assert info["aim_debug"]["aim_bin_error"] == 0
    assert info["reward_components"]["aim_bin_exact"] == pytest.approx(0.04)
    assert info["reward_components"]["aim_bin_exact"] < 0.10


def test_aim_bin_wrong_only_penalizes_fired_bad_shot():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step({"move": 0, "aim": 1, "fire": 1})

    assert info["fire"]["shot_fired"] is True
    assert info["aim_debug"]["aim_bin_error"] == 1
    assert info["reward_components"]["aim_bin_wrong"] == 0.0


def test_bad_fired_shot_gets_small_aim_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, info = env.step(FIRE_LEFT)

    assert info["fire"]["shot_fired"] is True
    assert info["aim_debug"]["aim_bin_error"] >= 2
    assert info["reward_components"]["aim_bin_wrong"] == pytest.approx(-0.04)


def test_fire_during_cooldown_does_not_get_shot_reward():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    env.step(FIRE_RIGHT)
    _, _, _, info = env.step(FIRE_RIGHT)

    assert info["fire"]["fire_requested"] is True
    assert info["fire"]["shot_fired"] is False
    assert info["fire"]["fire_blocked_reason"] == "cooldown"
    assert info["reward_components"]["aim_bin_exact"] == 0.0
    assert info["reward_components"]["aim_alignment"] == 0.0
    assert info["reward_components"]["aim_bin_wrong"] == 0.0
    assert "attack_intent" not in info["reward_components"]


def test_stage1_reward_has_no_zone_components():
    env = CPCEnv(seed=1, max_steps=4)
    env.state["self_pos"] = {"x": 0.0, "y": 0.0}
    env.state["enemy_pos"] = {"x": 100.0, "y": 0.0}

    _, _, _, info = env.step(NOOP)

    assert "zone_pressure" not in info["reward_components"]
    assert "return_to_zone" not in info["reward_components"]
    assert "near_edge_outward" not in info["reward_components"]


def test_damage_dealt_reward_is_positive():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    env.step(FIRE_RIGHT)
    _, _, _, info = env.step(NOOP)

    assert info["damage_delta"]["enemy_hp"] > 0.0
    assert info["reward_components"]["damage_dealt_ratio"] == pytest.approx(0.10)


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


def test_stage1b_uses_stage_specific_fire_shaping():
    env = CPCEnv(seed=1, max_steps=4, stationary_target_mode=True, enemy_move=False, enemy_fire=False)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, no_fire_info = env.step(NOOP)
    _, _, _, shot_info = env.step(FIRE_RIGHT)

    assert no_fire_info["reward_components"]["no_fire_ready_penalty"] < 0.0
    assert "bullet_hit" not in no_fire_info["reward_components"]
    assert shot_info["reward_components"]["shot_fired_reward"] > 0.0
    assert shot_info["reward_components"]["bullet_hit_reward"] == 0.0
    assert "fire_requested" not in shot_info["reward_components"]
    assert shot_info["fire"]["fire_valid"] is True


def test_stage1b_reward_prefers_actual_shot_over_fire_spam():
    env = CPCEnv(seed=1, max_steps=4, stationary_target_mode=True, enemy_move=False, enemy_fire=False)
    put_enemy_to_right(env, distance=120.0)

    _, _, _, first_info = env.step(FIRE_RIGHT)
    _, _, _, second_info = env.step(FIRE_RIGHT)

    assert first_info["reward_components"]["shot_fired_reward"] > 0.0
    assert first_info["reward_components"]["no_fire_ready_penalty"] == 0.0
    assert second_info["fire"]["fire_blocked_reason"] == "cooldown"
    assert second_info["reward_components"]["shot_fired_reward"] == 0.0
    assert second_info["reward_components"]["no_fire_ready_penalty"] == 0.0
    assert second_info["reward_components"]["cooldown_blocked_fire_penalty"] < 0.0


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
    assert info["reward_components"]["missed_shot"] == pytest.approx(-0.03)


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


def test_stationary_target_mode_penalizes_no_engagement_at_terminal():
    env = CPCEnv(seed=1, max_steps=1, stationary_target_mode=True, enemy_move=False, enemy_fire=False)
    put_enemy_to_right(env, distance=120.0)

    _, reward, done, info = env.step(NOOP)

    assert done is True
    assert reward < 0.0
    assert info["reward_components"]["damage_dealt_ratio"] == 0.0
    assert info["reward_components"]["bullet_hit_reward"] == 0.0
    assert info["reward_components"]["missed_shot_penalty"] == 0.0
    assert info["reward_components"]["bad_aim_shot_penalty"] == 0.0
    assert info["reward_components"]["no_fire_ready_penalty"] < 0.0
    assert info["fire"]["fire_valid"] is True


def test_stage1b_blocks_and_penalizes_invalid_fire():
    env = CPCEnv(seed=1, max_steps=4, stationary_target_mode=True, enemy_move=False, enemy_fire=False)
    put_enemy_to_right(env, distance=120.0)

    _, reward, _, info = env.step(FIRE_LEFT)

    assert reward < 0.0
    assert info["fire"]["fire_valid"] is False
    assert info["fire"]["fire_blocked_reason"] == "invalid_fire"
    assert info["fire"]["shot_fired"] is False
    assert info["reward_components"]["cooldown_blocked_fire_penalty"] == 0.0
    assert info["reward_components"]["invalid_fire_penalty"] < 0.0


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

    assert {
        "target_dir_x",
        "target_dir_y",
        "aim_alignment",
        "distance_to_center",
        "safe_margin_fraction",
        "outside_safe_zone",
        "current_aim_bin",
        "ideal_aim_bin",
        "gt_ideal_aim_bin",
        "aim_error",
        "aim_aligned",
        "target_in_range",
        "cooldown_ready",
        "fire_valid",
    } <= set(obs)


def test_randomized_spawn_requires_non_right_aim():
    ideal_bins = set()
    for seed in range(12):
        env = CPCEnv(seed=seed, max_steps=4, randomize_enemy_spawn_direction=True)
        ideal_bins.add(env._aim_debug(0, {"aimX": 1.0, "aimY": 0.0})["ideal_aim_bin"])

    assert any(aim_bin != 0 for aim_bin in ideal_bins)


def test_fixed_enemy_spawn_direction_is_deterministic():
    env = CPCEnv(seed=1, max_steps=4, enemy_spawn_direction="left")

    aim_debug = env._aim_debug(0, {"aimX": 1.0, "aimY": 0.0})

    assert aim_debug["ideal_aim_bin"] == 8


def test_randomized_enemy_spawn_uses_configured_direction_list():
    ideal_bins = set()
    for seed in range(8):
        env = CPCEnv(
            seed=seed,
            max_steps=4,
            randomize_enemy_spawn_direction=True,
            enemy_spawn_directions=("left",),
        )
        ideal_bins.add(env._aim_debug(0, {"aimX": 1.0, "aimY": 0.0})["ideal_aim_bin"])

    assert ideal_bins == {8}


def test_metrics_include_aim_collapse_indicators():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    env.step(FIRE_RIGHT)
    summary = env.metrics.summary()

    for key in (
        "aim_bin_0_rate",
        "aim_bin_entropy",
        "current_aim_bin_distribution",
        "ideal_aim_bin_distribution",
        "exact_aim_match_rate",
        "within_1_bin_aim_rate",
        "bad_aim_rate",
        "aim_error",
        "aim_aligned_rate",
        "target_in_range_rate",
        "cooldown_ready_rate",
        "fire_valid_rate",
        "shot_exact_aim_rate",
        "shot_near_aim_rate",
        "shot_off_target_rate",
        "bullet_hit_per_shot",
        "bullet_hit_per_valid_shot",
        "shot_fired_count",
        "shot_fired_when_valid_count",
        "valid_fire_requested_count",
        "invalid_fire_requested_count",
        "blocked_invalid_fire_count",
        "no_fire_when_valid_count",
    ):
        assert key in summary


def test_timeout_hp_lead():
    env = CPCEnv(seed=1, max_steps=1)
    put_enemy_to_right(env, distance=400.0)
    env.metrics.damage_dealt = 20.0
    env.metrics.damage_taken = 5.0

    _, _, done, info = env.step(NOOP)

    assert done is True
    assert info["reward_components"]["timeout_hp_lead"] > 0.0


def test_accuracy_bonus_requires_damage():
    env = CPCEnv(seed=1, max_steps=1)
    put_enemy_to_right(env, distance=400.0)
    env.metrics.shot_fired_count = 3
    env.metrics.bullet_hit_count = 3

    _, _, done, info = env.step(NOOP)

    assert done is True
    assert info["reward_components"]["accuracy_bonus"] == 0.0


def test_range_debug_and_reward_components():
    env = CPCEnv(seed=1, max_steps=4)
    env.enemy_damage = 0.0
    put_enemy_to_right(env, distance=env.fire_range * 0.5)

    _, _, _, info = env.step(NOOP)

    assert info["range_debug"]["in_good_range"] is True
    assert info["reward_components"]["good_range"] == 0.0
    assert info["reward_components"]["too_close"] == 0.0
    assert info["reward_components"]["too_far"] == 0.0


def test_good_range_reward_requires_combat_engagement():
    env = CPCEnv(seed=1, max_steps=4)
    env.enemy_damage = 0.0
    put_enemy_to_right(env, distance=env.fire_range * 0.5)

    _, _, _, info = env.step(NOOP)

    assert info["range_debug"]["in_good_range"] is True
    assert info["damage_delta"]["enemy_hp"] == 0.0
    assert info["damage_delta"]["self_hp"] == 0.0
    assert info["reward_components"]["good_range"] == 0.0


def test_good_range_reward_with_combat_engagement():
    env = CPCEnv(seed=1, max_steps=4)
    env.enemy_damage = 0.0
    put_enemy_to_right(env, distance=env.fire_range * 0.5)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert info["fire"]["shot_fired"] is True
    assert info["range_debug"]["in_good_range"] is True
    assert info["reward_components"]["good_range"] > 0.0


def test_no_shot_episode_penalty():
    env = CPCEnv(seed=1, max_steps=1)
    env.enemy_damage = 0.0
    put_enemy_to_right(env, distance=400.0)

    _, _, done, info = env.step(NOOP)

    assert done is True
    assert info["reward_components"]["no_shot_episode"] < 0.0


def test_death_without_shooting_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    env.enemy_damage = 100.0
    put_enemy_to_right(env, distance=120.0)
    env.state["ally_hp"] = 0.0

    _, _, done, info = env.step(NOOP)

    assert done is True
    assert info["metrics"]["self_dead"] == 1.0
    assert info["reward_components"]["death_without_shooting"] < 0.0


def test_death_without_damage_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    env.enemy_damage = 100.0
    put_enemy_to_right(env, distance=120.0)
    env.state["ally_hp"] = 0.0

    _, _, done, info = env.step(NOOP)

    assert done is True
    assert info["damage_delta"]["enemy_hp"] == 0.0
    assert info["reward_components"]["death_without_damage"] < 0.0


def test_no_combat_death_total_reward_negative():
    env = CPCEnv(seed=1, max_steps=4)
    env.enemy_damage = 100.0
    put_enemy_to_right(env, distance=120.0)
    env.state["ally_hp"] = 0.0

    _, reward, done, info = env.step(AIM_RIGHT)

    assert done is True
    assert info["fire"]["shot_fired"] is False
    assert info["damage_delta"]["enemy_hp"] == 0.0
    assert info["metrics"]["self_dead"] == 1.0
    assert reward < 0.0


def test_stage1_metrics_exist():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env, distance=120.0)

    env.step(FIRE_RIGHT)
    summary = env.metrics.summary()

    for key in (
        "damage_trade_ratio",
        "hit_ratio",
        "bullet_hit_per_shot",
        "missed_shot_rate",
        "exact_aim_match_rate",
        "avg_distance_to_enemy",
        "good_range_rate",
        "total_return",
        "mean_step_reward",
    ):
        assert key in summary


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
    assert info["reward_components"]["damage_taken_ratio"] < 0.0
