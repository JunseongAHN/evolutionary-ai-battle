from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.cpc_env import CPCEnv


FIRE_RIGHT = {"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 1}
NOOP = {"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 0}


def test_projectile_collision_uses_bullet_and_target_radii():
    env = CPCEnv(
        enemy_move=False,
        enemy_fire=False,
        bullet_speed=100.0,
        bullet_range=300.0,
    )
    env.reset(seed=7)
    env.state["self_pos"] = {"x": 100.0, "y": 100.0}
    env.state["enemy_pos"] = {"x": 200.0, "y": 118.0}
    enemy_hp_before = float(env.state["enemy_hp"])

    env.step(FIRE_RIGHT)
    _, _, _, hit_info = env.step(NOOP)

    assert env.projectile_radius < 18.0 < env.projectile_radius + env.enemy_radius
    assert env.state["enemy_hp"] < enemy_hp_before
    hit = next(event for event in hit_info["bullet_events"] if event["type"] == "bullet_hit")
    assert hit["owner_id"] == "self"
    assert hit["target_id"] == "enemy"
    assert hit["damage"] == pytest.approx(enemy_hp_before - env.state["enemy_hp"])
