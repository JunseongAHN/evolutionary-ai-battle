from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from eval_scripted_ideal_aim_same_reset import run_scripted_ideal_aim_same_reset, summarize


def test_scripted_ideal_aim_same_reset_hits_stationary_in_range_target():
    result = run_scripted_ideal_aim_same_reset(
        config_path="experiment/configs/local_combat_in_range.yaml",
        episodes=1,
        max_steps=20,
        seed=0,
        enemy_spawn_direction="right",
    )
    summary = summarize(result)

    assert summary["mean_shot_fired_count"] > 0
    assert summary["mean_self_bullet_hit_count"] > 0
    assert summary["mean_bullet_hit_per_shot"] > 0
    assert summary["episodes"]
    assert summary["episodes"][0]["bullet_lifecycle"]
    assert summary["episodes"][0]["distance_to_enemy"] < summary["bullet_range"]
