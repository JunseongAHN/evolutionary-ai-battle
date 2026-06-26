from __future__ import annotations

import math
import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.env_core import PythonBattleCoreEnv
from debug.manual_control import (
    build_manual_multi_agent_action,
    build_user_action,
    keyboard_to_move,
    mouse_to_aim,
    spawn_debug_bullet,
    update_debug_bullets,
)
from debug.state_inspector import summarize_events, summarize_snapshot


def test_keyboard_state_to_action():
    assert keyboard_to_move({"w"}) == (0.0, -1.0)
    assert keyboard_to_move({"a", "s"}) == (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    assert keyboard_to_move({"a", "d"}) == (0.0, 0.0)


def test_mouse_position_to_aim():
    aim_x, aim_y = mouse_to_aim({"x": 10.0, "y": 10.0}, {"x": 13.0, "y": 14.0})

    assert math.isclose(aim_x, 0.6)
    assert math.isclose(aim_y, 0.8)


def test_build_user_action_sets_fire():
    action = build_user_action("episode", 3, "team-a-0", 0.0, 0.0, 1.0, 0.0, 1.0)

    assert action["action"]["fire"] == 1.0


def test_debug_bullet_spawns_from_agent_aim():
    agent = {
        "position": {"x": 10.0, "y": 20.0},
        "aim": {"x": 0.0, "y": 2.0},
    }

    bullet = spawn_debug_bullet(agent, max_range=100.0)

    assert bullet is not None
    assert bullet["position"] == {"x": 10.0, "y": 20.0}
    assert bullet["velocity"] == {"x": 0.0, "y": 1.0}
    assert bullet["max_range"] == 100.0


def test_debug_bullet_disappears_after_max_range():
    bullet = {
        "origin": {"x": 0.0, "y": 0.0},
        "position": {"x": 0.0, "y": 0.0},
        "velocity": {"x": 1.0, "y": 0.0},
        "distance": 90.0,
        "max_range": 100.0,
    }

    assert update_debug_bullets([bullet], speed=10.0)[0]["position"] == {"x": 100.0, "y": 0.0}
    assert update_debug_bullets([bullet], speed=11.0) == []


def test_build_multi_agent_action_contains_all_agents():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    user_action = build_user_action(env.episode_id, env.step_index, "team-a-0", 1.0, 0.0, 0.0, 1.0, 1.0)

    multi_action = build_manual_multi_agent_action(
        episode_id=env.episode_id,
        step=env.step_index,
        agent_ids=env.agent_ids,
        controlled_agent_id="team-a-0",
        user_action=user_action,
    )

    assert set(multi_action["actions"]) == set(env.agent_ids)
    assert multi_action["actions"]["team-a-0"]["source"]["policy_type"] == "user_controlled"
    assert multi_action["actions"]["team-b-0"]["action"]["fire"] == 0.0


def test_state_inspector_summarizes_snapshot():
    env = PythonBattleCoreEnv()
    env.reset(seed=0)
    snapshot = env._snapshot([])

    summary = summarize_snapshot(snapshot)

    assert "team-a-0" in summary
    assert "hp=" in summary
    assert "alive=" in summary
    assert "pos=" in summary


def test_state_inspector_summarizes_events():
    events = [
        {
            "event_id": "event-1",
            "step": 1,
            "event_type": "damage",
            "actor_id": "team-a-0",
            "target_id": "team-b-0",
            "value": 10.0,
        }
    ]

    summary = summarize_events(events)

    assert "damage" in summary
    assert "actor=team-a-0" in summary
    assert "target=team-b-0" in summary


def test_schema_is_framework_independent():
    source = (EXPERIMENT_ROOT / "core" / "schema.py").read_text(encoding="utf-8").lower()

    for token in ["pygame", "torch", "torchrl", "tensordict", "gym"]:
        assert token not in source


def test_model_action_decodes_policy_bins():
    pytest.importorskip("torch")
    from debug.model_gameplay import build_model_action

    action = build_model_action(
        episode_id="episode",
        step=1,
        agent_id="team-a-0",
        raw_action={"move": 4, "aim": 0, "fire": 1},
        policy_id="checkpoint.pt",
    )

    assert action["action"]["move_x"] == 1.0
    assert action["action"]["aim_x"] == 1.0
    assert action["action"]["fire"] == 1.0
    assert action["source"]["raw_action"] == {"move": 4, "aim": 0, "fire": 1}


def test_core_observation_maps_to_policy_feature_shape():
    torch = pytest.importorskip("torch")
    from debug.model_gameplay import _policy_features_from_core_observation

    env = PythonBattleCoreEnv()
    observations = env.reset(seed=0)
    snapshot = env._snapshot([])

    features = _policy_features_from_core_observation(
        observations["team-a-0"],
        snapshot=snapshot,
        agent_id="team-a-0",
        device=torch.device("cpu"),
    )

    assert tuple(features.shape) == (1, 13)
    assert float(features[0, 0]) == 1.0


def test_resolve_checkpoint_path_picks_selected_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from debug.model_gameplay import resolve_checkpoint_path

    selected = tmp_path / "checkpoint_selected.pt"
    latest = tmp_path / "checkpoint_latest.pt"
    latest.write_bytes(b"latest")
    selected.write_bytes(b"selected")

    assert resolve_checkpoint_path(run_dir=tmp_path) == selected
