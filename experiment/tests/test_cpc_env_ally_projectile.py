from __future__ import annotations

import pathlib
import sys


EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.cpc_env import CPCEnv
from core.env_config import load_env_config


NOOP = {"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 0}
ALLY_FIRE_RIGHT = {"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 1}


def test_ally_fire_spawns_projectile_and_damages_enemy():
    env = _combat_env()
    enemy_hp_before = float(env.state["enemy_hp"])

    _, _, _, fire_info = env.step(NOOP, ally_action=ALLY_FIRE_RIGHT)

    assert fire_info["ally_fire"]["shot_fired"] is True
    assert fire_info["ally_fire"]["fire_blocked_reason"] is None
    assert any(projectile["owner_id"] == "ally" for projectile in env.projectiles)
    assert any(
        event["type"] == "bullet_spawned" and event["owner_id"] == "ally"
        for event in fire_info["events"]
    )

    _, _, _, hit_info = env.step(NOOP)

    assert env.state["enemy_hp"] == enemy_hp_before - env.damage
    assert any(
        event["type"] == "bullet_hit"
        and event["owner_id"] == "ally"
        and event["target_id"] == "enemy"
        for event in hit_info["events"]
    )


def test_ally_weapon_cooldown_blocks_immediate_repeat_shot():
    env = _combat_env()
    env.step(NOOP, ally_action=ALLY_FIRE_RIGHT)

    _, _, _, info = env.step(NOOP, ally_action=ALLY_FIRE_RIGHT)

    assert info["ally_fire"]["fire_requested"] is True
    assert info["ally_fire"]["shot_fired"] is False
    assert info["ally_fire"]["fire_blocked_reason"] == "cooldown"
    assert any(
        event["type"] == "bullet_not_spawned"
        and event["owner_id"] == "ally"
        and event["reason"] == "cooldown"
        for event in info["events"]
    )


def test_ally_projectile_snapshot_uses_player_team():
    env = _combat_env()
    env.step(NOOP, ally_action=ALLY_FIRE_RIGHT)

    bullet = next(item for item in env.get_snapshot()["bullets"] if item["owner_id"] == "ally")

    assert bullet["team"] == "player"


def _combat_env() -> CPCEnv:
    env = CPCEnv.from_config(load_env_config("configs/env/autoplay_goal_loop.yaml"))
    env.reset(seed=17)
    env.enemy_move = False
    env.enemy_fire = False
    env.state["self_pos"] = {"x": 100.0, "y": 250.0}
    env.state["ally_pos"] = {"x": 100.0, "y": 100.0}
    env.state["enemy_pos"] = {"x": 200.0, "y": 100.0}
    return env
