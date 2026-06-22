from __future__ import annotations

import pathlib
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from training.cpc_env import CPCEnv


NOOP = {"move": 0, "aim": 0, "fire": 0}
MOVE_RIGHT = {"move": 4, "aim": 0, "fire": 0}
MOVE_LEFT = {"move": 3, "aim": 0, "fire": 0}
AIM_RIGHT = {"move": 0, "aim": 0, "fire": 0}
FIRE_RIGHT = {"move": 0, "aim": 0, "fire": 1}


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


def test_fire_in_range_and_aligned_gives_attack_intent_reward():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert info["reward_components"]["attack_intent"] > 0.0


def test_outside_safe_zone_gives_zone_pressure_penalty():
    env = CPCEnv(seed=1, max_steps=4)
    env.state["self_pos"] = {"x": 0.0, "y": 0.0}
    env.state["enemy_pos"] = {"x": 100.0, "y": 0.0}

    _, _, _, info = env.step(NOOP)

    assert info["reward_components"]["zone_pressure"] < 0.0


def test_damage_dealt_reward_is_positive():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(FIRE_RIGHT)

    assert info["damage_delta"]["enemy_hp"] > 0.0
    assert info["reward_components"]["damage_dealt"] > 0.0


def test_damage_taken_penalty_is_negative():
    env = CPCEnv(seed=1, max_steps=4)
    put_enemy_to_right(env)

    _, _, _, info = env.step(NOOP)

    assert info["damage_delta"]["self_hp"] > 0.0
    assert info["reward_components"]["damage_taken"] < 0.0
