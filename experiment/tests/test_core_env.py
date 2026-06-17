from __future__ import annotations

import pathlib
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.env_core import PythonBattleCoreEnv
from core.reward import compute_agent_reward
from core.schema import SCHEMA_VERSION


def no_op_actions(env: PythonBattleCoreEnv):
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": env.episode_id,
        "step": env.step_index,
        "actions": {
            agent_id: {
                "schema_version": SCHEMA_VERSION,
                "episode_id": env.episode_id,
                "step": env.step_index,
                "agent_id": agent_id,
                "action": {"move_x": 0.0, "move_y": 0.0, "aim_x": 1.0, "aim_y": 0.0, "fire": 0.0},
            }
            for agent_id in env.agent_ids
        },
    }


def test_reset_returns_four_stable_agent_observations():
    env = PythonBattleCoreEnv()
    observations = env.reset(seed=123)

    assert list(observations.keys()) == ["team-a-0", "team-a-1", "team-b-0", "team-b-1"]
    assert env.agent_ids == ["team-a-0", "team-a-1", "team-b-0", "team-b-1"]


def test_observation_shape_contains_required_fields():
    env = PythonBattleCoreEnv()
    obs = env.reset(seed=0)["team-a-0"]

    for key in [
        "schema_version",
        "agent_id",
        "team_id",
        "self",
        "vector",
        "vector_keys",
        "visible_enemies",
        "visible_enemies_mask",
        "visible_allies",
        "visible_allies_mask",
    ]:
        assert key in obs


def test_no_op_step_increments_step_and_has_snapshot():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    step = env.step(no_op_actions(env))

    assert env.step_index == 1
    assert step["step"] == 1
    assert "snapshot" in step["info"]


def test_movement_changes_position_and_clamps_to_bounds():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    before = dict(env.agents["team-a-0"]["position"])
    actions = no_op_actions(env)
    actions["actions"]["team-a-0"]["action"]["move_x"] = -100.0
    actions["actions"]["team-a-0"]["action"]["move_y"] = -100.0

    env.step(actions)
    after = env.agents["team-a-0"]["position"]

    assert after != before
    assert 0.0 <= after["x"] <= env.width
    assert 0.0 <= after["y"] <= env.height


def test_fire_action_creates_fire_damage_and_death_events():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    env.agents["team-a-0"]["position"] = {"x": 500.0, "y": 500.0}
    env.agents["team-b-0"]["position"] = {"x": 510.0, "y": 500.0}
    env.agents["team-b-0"]["hp"] = env.damage
    actions = no_op_actions(env)
    actions["actions"]["team-a-0"]["action"]["fire"] = 1.0
    actions["actions"]["team-a-0"]["action"]["aim_x"] = 1.0

    step = env.step(actions)
    event_types = [event["event_type"] for event in step["info"]["events"]]

    assert "fire" in event_types
    assert "damage" in event_types
    assert "death" in event_types


def test_reward_returns_float_and_can_be_negative():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    env.agents["team-b-0"]["position"] = {"x": 500.0, "y": 500.0}
    env.agents["team-a-0"]["position"] = {"x": 510.0, "y": 500.0}
    actions = no_op_actions(env)
    actions["actions"]["team-b-0"]["action"]["fire"] = 1.0

    step = env.step(actions)
    reward = compute_agent_reward(step, "team-a-0")

    assert isinstance(reward, float)
    assert reward < 0

