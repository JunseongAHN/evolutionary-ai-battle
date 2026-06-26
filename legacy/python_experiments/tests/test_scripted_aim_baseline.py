from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from baselines.scripted_combat_policies import ScriptedAimAtEnemyPolicy
from eval_scripted_aim_baseline import run_scripted_baseline, summarize, sweep_directions
from training.cpc_actions import vec_to_aim_bin
from training.cpc_env import CPCEnv


def _put_enemy(env: CPCEnv, x: float, y: float) -> None:
    env.state["self_pos"] = {"x": 500.0, "y": 500.0}
    env.state["ally_pos"] = {"x": 450.0, "y": 540.0}
    env.state["enemy_pos"] = {"x": x, "y": y}


def test_scripted_aim_policy_selects_ideal_bin_right():
    env = CPCEnv(seed=1, max_steps=4)
    observation = env.reset(seed=1)
    _put_enemy(env, 620.0, 500.0)

    scripted = ScriptedAimAtEnemyPolicy().act_with_diagnostics(observation, env)

    assert scripted.action["aim"] == scripted.diagnostics["ideal_aim_bin"]
    assert scripted.diagnostics["aim_bin_error"] == 0


def test_scripted_aim_policy_selects_ideal_bin_left():
    env = CPCEnv(seed=1, max_steps=4)
    observation = env.reset(seed=1)
    _put_enemy(env, 380.0, 500.0)

    scripted = ScriptedAimAtEnemyPolicy().act_with_diagnostics(observation, env)

    assert scripted.action["aim"] == scripted.diagnostics["ideal_aim_bin"]
    assert scripted.diagnostics["aim_bin_error"] == 0


def test_scripted_aim_policy_selects_ideal_bin_diagonal():
    env = CPCEnv(seed=1, max_steps=4)
    observation = env.reset(seed=1)
    _put_enemy(env, 620.0, 620.0)

    scripted = ScriptedAimAtEnemyPolicy().act_with_diagnostics(observation, env)

    assert scripted.action["aim"] == vec_to_aim_bin({"x": 120.0, "y": 120.0})


def test_scripted_aim_fires_only_when_can_fire():
    env = CPCEnv(seed=1, max_steps=4)
    observation = env.reset(seed=1)
    _put_enemy(env, 620.0, 500.0)
    policy = ScriptedAimAtEnemyPolicy()

    assert policy.act_with_diagnostics(observation, env).action["fire"] == 1

    env.weapon["cooldown_remaining_steps"] = 3
    observation["can_fire"] = False
    assert policy.act_with_diagnostics(observation, env).action["fire"] == 0


def test_scripted_aim_baseline_hits_stationary_enemy():
    result = run_scripted_baseline(episodes=1, max_steps=20, seed=0, mode="stand_still", fixed_enemy_direction="right")
    summary = summarize(result)

    assert summary["mean_self_bullet_hit_count"] > 0
    assert summary["mean_damage_dealt_ratio"] > 0


def test_scripted_aim_projectile_damage_matches_hits():
    result = run_scripted_baseline(episodes=1, max_steps=20, seed=0, mode="stand_still", fixed_enemy_direction="right")
    summary = summarize(result)
    hits = summary["mean_self_bullet_hit_count"]
    damage = summary["mean_damage_dealt_ratio"] * 100.0

    assert damage == pytest.approx(hits * 10.0)


def test_scripted_aim_sweep_directions_hits_at_least_some_directions():
    rows = sweep_directions(episodes=1, max_steps=50, seed=0, mode="stand_still")
    right = next(row for row in rows if row["direction"] == "right")

    assert right["bullet_hit_count"] > 0
    assert any(row["bullet_hit_count"] > 0 for row in rows)


def test_scripted_baseline_does_not_require_ppo_checkpoint():
    result = run_scripted_baseline(episodes=1, max_steps=5, seed=0, mode="stand_still", fixed_enemy_direction="right")

    assert result["source"] == "eval_scripted_aim_baseline"


def test_scripted_baseline_result_contains_projectile_lifecycle():
    result = run_scripted_baseline(episodes=1, max_steps=5, seed=0, mode="stand_still", fixed_enemy_direction="right")
    step = result["episodes"][0]["steps"][0]

    assert "fire" in step
    assert "bullets" in step
    assert "events" in step
    assert "aim" in step
    assert "metrics_delta" in step
    assert "scripted" in step
