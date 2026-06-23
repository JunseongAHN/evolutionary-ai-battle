from __future__ import annotations

import math
import pathlib
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.env_core import PythonBattleCoreEnv
from core.schema import SCHEMA_VERSION


EXPECTED_AGENT_IDS = ["team-a-0", "team-a-1", "team-b-0", "team-b-1"]


def actions_for(env: PythonBattleCoreEnv, overrides: dict | None = None):
    overrides = overrides or {}
    actions = {
        agent_id: {
            "schema_version": SCHEMA_VERSION,
            "episode_id": env.episode_id,
            "step": env.step_index,
            "agent_id": agent_id,
            "action": {"move_x": 0.0, "move_y": 0.0, "aim_x": 0.0, "aim_y": 0.0, "fire": 0.0},
        }
        for agent_id in env.agent_ids
    }
    for agent_id, action_override in overrides.items():
        actions[agent_id]["action"].update(action_override)
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": env.episode_id,
        "step": env.step_index,
        "actions": actions,
    }


def position(env: PythonBattleCoreEnv, agent_id: str) -> dict[str, float]:
    return dict(env.agents[agent_id]["position"])


def put_in_range(env: PythonBattleCoreEnv, actor_id: str = "team-a-0", target_id: str = "team-b-0") -> None:
    env.agents[actor_id]["position"] = {"x": 500.0, "y": 500.0}
    env.agents[target_id]["position"] = {"x": 520.0, "y": 500.0}


def test_reset_returns_four_agents():
    env = PythonBattleCoreEnv()
    observations = env.reset(seed=123)

    assert list(observations.keys()) == EXPECTED_AGENT_IDS
    assert env.agent_ids == EXPECTED_AGENT_IDS
    assert observations["team-a-0"]["team_id"] == "team-a"
    assert observations["team-a-1"]["team_id"] == "team-a"
    assert observations["team-b-0"]["team_id"] == "team-b"
    assert observations["team-b-1"]["team_id"] == "team-b"
    assert env.step_index == 0
    assert all(obs["step"] == 0 for obs in observations.values())


def test_observation_shape_contains_required_fields_and_stable_masks():
    env = PythonBattleCoreEnv()
    obs = env.reset(seed=0)["team-a-0"]

    for key in [
        "schema_version",
        "episode_id",
        "step",
        "agent_id",
        "team_id",
        "mode",
        "self",
        "vector",
        "vector_keys",
        "visible_enemies",
        "visible_enemies_mask",
        "visible_allies",
        "visible_allies_mask",
        "visible_obstacles",
        "visible_obstacles_mask",
        "recent_events",
        "recent_events_mask",
    ]:
        assert key in obs
    assert len(obs["vector"]) == len(obs["vector_keys"])
    assert len(obs["visible_enemies_mask"]) == 3
    assert len(obs["visible_allies_mask"]) == 1
    assert len(obs["visible_obstacles_mask"]) == 0
    assert len(obs["recent_events_mask"]) == 8


def test_step_increments_step_index():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    step = env.step(actions_for(env))

    assert env.step_index == 1
    assert step["step"] == env.step_index
    assert step["info"]["snapshot"]["step"] == env.step_index


def test_noop_action_does_not_move_agent():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    before = position(env, "team-a-0")

    env.step(actions_for(env))

    assert position(env, "team-a-0") == before


def test_movement_still_works_after_pr3():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    before = position(env, "team-a-0")

    env.step(actions_for(env, {"team-a-0": {"move_x": 1.0, "move_y": 0.0}}))
    after = position(env, "team-a-0")

    assert after["x"] == before["x"] + env.move_speed
    assert after["y"] == before["y"]


def test_movement_is_clamped_to_world_bounds():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)

    for _ in range(100):
        env.step(actions_for(env, {"team-a-0": {"move_x": -1.0, "move_y": -1.0}}))

    after = position(env, "team-a-0")
    assert 0.0 <= after["x"] <= env.width
    assert 0.0 <= after["y"] <= env.height


def test_move_vector_is_normalized():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    before = position(env, "team-a-0")

    env.step(actions_for(env, {"team-a-0": {"move_x": 1.0, "move_y": 1.0}}))
    after = position(env, "team-a-0")
    distance = math.hypot(after["x"] - before["x"], after["y"] - before["y"])

    assert math.isclose(distance, env.move_speed)


def test_aim_updates_without_movement():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    before = position(env, "team-a-0")

    env.step(actions_for(env, {"team-a-0": {"aim_x": 0.0, "aim_y": -1.0}}))

    assert position(env, "team-a-0") == before
    assert env.agents["team-a-0"]["aim"] == {"x": 0.0, "y": -1.0}
    assert env.agents["team-a-0"]["facing"] == {"x": 0.0, "y": -1.0}


def test_max_step_truncation_still_works():
    env = PythonBattleCoreEnv()
    env.max_steps = 2
    env.reset(seed=0)

    first = env.step(actions_for(env))
    second = env.step(actions_for(env))

    assert first["truncated"] is False
    assert first["terminated"] is False
    assert second["truncated"] is True
    assert second["terminated"] is False


def test_fire_event_emitted_when_agent_fires():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    step = env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))

    assert "fire" in [event["event_type"] for event in step["info"]["events"]]


def test_fire_can_damage_enemy_in_range():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    put_in_range(env)
    before_hp = env.agents["team-b-0"]["hp"]

    step = env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))
    event_types = [event["event_type"] for event in step["info"]["events"]]

    assert env.agents["team-b-0"]["hp"] == before_hp - env.damage
    assert "damage" in event_types


def test_fire_does_not_damage_teammate():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    env.agents["team-a-0"]["position"] = {"x": 500.0, "y": 500.0}
    env.agents["team-a-1"]["position"] = {"x": 510.0, "y": 500.0}
    env.agents["team-b-0"]["position"] = {"x": 520.0, "y": 500.0}
    ally_hp = env.agents["team-a-1"]["hp"]

    env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))

    assert env.agents["team-a-1"]["hp"] == ally_hp


def test_fire_out_of_range_does_not_damage():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    env.agents["team-a-0"]["position"] = {"x": 0.0, "y": 0.0}
    env.agents["team-b-0"]["position"] = {"x": env.width, "y": env.height}
    env.agents["team-b-1"]["position"] = {"x": env.width, "y": env.height}
    before_hp = env.agents["team-b-0"]["hp"]

    step = env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))
    event_types = [event["event_type"] for event in step["info"]["events"]]

    assert "fire" in event_types
    assert "damage" not in event_types
    assert env.agents["team-b-0"]["hp"] == before_hp


def test_damage_can_kill_agent():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    put_in_range(env)
    env.damage = env.max_hp

    step = env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))
    event_types = [event["event_type"] for event in step["info"]["events"]]

    assert env.agents["team-b-0"]["hp"] == 0.0
    assert env.agents["team-b-0"]["alive"] is False
    assert "death" in event_types


def test_dead_agent_cannot_move_or_fire():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    put_in_range(env, actor_id="team-a-0", target_id="team-b-0")
    env.agents["team-a-0"]["alive"] = False
    before_position = position(env, "team-a-0")
    before_hp = env.agents["team-b-0"]["hp"]

    step = env.step(actions_for(env, {"team-a-0": {"move_x": 1.0, "fire": 1.0}}))
    actor_events = [event for event in step["info"]["events"] if event.get("actor_id") == "team-a-0"]

    assert position(env, "team-a-0") == before_position
    assert env.agents["team-b-0"]["hp"] == before_hp
    assert actor_events == []


def test_team_elimination_terminates_episode():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    put_in_range(env)
    env.damage = env.max_hp
    env.agents["team-b-1"]["alive"] = False

    step = env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))

    assert step["terminated"] is True
    assert step["truncated"] is False


def test_fire_cooldown_blocks_repeated_fire():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    put_in_range(env)

    first = env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))
    hp_after_first = env.agents["team-b-0"]["hp"]
    second = env.step(actions_for(env, {"team-a-0": {"fire": 1.0}}))
    second_fire_events = [
        event for event in second["info"]["events"]
        if event["event_type"] == "fire" and event.get("actor_id") == "team-a-0"
    ]

    assert "damage" in [event["event_type"] for event in first["info"]["events"]]
    assert env.agents["team-b-0"]["hp"] == hp_after_first
    assert second_fire_events[0]["metadata"]["cooldown_blocked"] is True


def test_safe_zone_damages_agents_outside_radius():
    env = PythonBattleCoreEnv()
    env.safe_radius_start = 100.0
    env.safe_radius_end = 100.0
    env.zone_damage_per_step = 3.0
    env.reset(seed=0)
    env.agents["team-a-0"]["position"] = {"x": 0.0, "y": 0.0}
    before_hp = env.agents["team-a-0"]["hp"]

    step = env.step(actions_for(env))
    zone_events = [
        event for event in step["info"]["events"]
        if event["event_type"] == "damage"
        and event.get("target_id") == "team-a-0"
        and event.get("metadata", {}).get("source") == "safe_zone"
    ]

    assert env.agents["team-a-0"]["hp"] == before_hp - env.zone_damage_per_step
    assert zone_events
    assert step["info"]["snapshot"]["safe_zone"]["radius"] == 100.0
