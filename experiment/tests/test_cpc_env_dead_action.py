from __future__ import annotations

import pathlib
import sys
from copy import deepcopy

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.cpc_env import CPCEnv
from core.env_config import load_env_config


def test_dead_agent_action_is_no_op_and_emits_debug_event():
    env = CPCEnv.from_config(load_env_config("configs/env/autoplay_enemy_in_range.yaml"))
    env.reset()
    env.enemy_fire = False
    env.state["self_hp"] = 0.0
    env.current_aim_vector = {"x": 0.0, "y": 1.0}
    before_position = deepcopy(env.state["self_pos"])
    before_hp = env.state["self_hp"]
    before_aim = deepcopy(env.current_aim_vector)
    before_cooldown = deepcopy(env.weapon)

    _, _, _, info = env.step({"move": 4, "aim_dx": 0.0, "aim_dy": 1.0, "fire": 1})

    assert env.state["self_pos"] == before_position
    assert env.state["self_hp"] == before_hp
    assert env.current_aim_vector == before_aim
    assert env.weapon == before_cooldown
    assert not any(projectile.get("owner_id") == "self" for projectile in env.projectiles)
    assert info["decoded_action"]["moveX"] == 0.0
    assert info["decoded_action"]["moveY"] == 0.0
    assert info["decoded_action"]["fire"] == 0
    assert info["fire"]["fire_requested"] is False
    assert info["fire"]["shot_fired"] is False
    assert info["fire"]["fire_blocked_reason"] == "agent_dead"
    assert info["action_debug"] == {
        "accepted": False,
        "no_op": True,
        "reason": "agent_dead",
    }
    assert any(
        event.get("type") == "action_ignored" and event.get("reason") == "agent_dead"
        for event in info["bullet_events"]
    )
