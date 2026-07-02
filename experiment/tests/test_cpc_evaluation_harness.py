from __future__ import annotations

import inspect
import json
import pathlib
import sys
from dataclasses import fields

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.hierarchical_baseline import ExecutionDirective
from core.env_config import load_env_config
from evaluation import (
    CPC_INTENTS,
    POLICY_IDS,
    CombatAction,
    CpcIntentArbiter,
    CpcIntentInputs,
    CpcTargetResolver,
    EvaluationEpisode,
    Layer1Output,
    QuestionnaireAnswers,
    derive_decision_trace,
    format_decision_trace,
)
from gui.pygame_viewer import _panel_lines
from scripts.run_cpc_evaluation import format_step_debug


STAY = {"move": 0, "aim": 0, "fire": 0}
NEUTRAL_ANSWERS = QuestionnaireAnswers(3, 3, 3, 3, 3, "automated smoke")


def test_required_policy_and_intent_ids_are_available():
    assert POLICY_IDS == ("baseline_selfish", "cpc_support", "current_best")
    assert CPC_INTENTS == ("SOLO_OBJECTIVE", "SUPPORT_TEAMMATE", "REGROUP", "FOCUS_FIRE")


def test_questionnaire_rejects_out_of_range_rating():
    with pytest.raises(ValueError, match="helpful"):
        QuestionnaireAnswers(0, 3, 3, 3, 3)


def test_episode_outputs_share_session_id(tmp_path):
    episode = _episode(tmp_path, "current_best")
    episode.step(STAY)
    summary = episode.finish(NEUTRAL_ANSWERS, end_reason="test_complete")

    session_dir = tmp_path / episode.metadata.session_id
    trajectory = _jsonl(session_dir / "trajectory.jsonl")
    questionnaire = _json(session_dir / "questionnaire.json")
    saved_summary = _json(session_dir / "summary.json")

    assert len(trajectory) == 1
    assert trajectory[0]["session_id"] == episode.metadata.session_id
    assert questionnaire["session_id"] == episode.metadata.session_id
    assert saved_summary["session_id"] == episode.metadata.session_id
    assert summary == saved_summary


def test_trajectory_contains_minimal_layers_and_decision_trace(tmp_path):
    episode = _episode(tmp_path, "cpc_support")
    episode.step(STAY)
    episode.finish(NEUTRAL_ANSWERS, end_reason="test_complete")
    row = _jsonl(tmp_path / episode.metadata.session_id / "trajectory.jsonl")[0]

    assert {
        "session_id",
        "timestamp",
        "policy_id",
        "player_hp",
        "bot_hp",
        "enemy_hp",
        "layers",
        "decision_trace",
        "decision_trace_line",
        "teammate_distance",
        "events",
    }.issubset(row)
    assert row["layers"]["layer1"] == {"cpc_intent": "REGROUP"}
    assert row["layers"]["layer2"]["target_ref"] == {"kind": "teammate"}
    assert set(row["layers"]["layer3"]["combat_action"]) == {
        "move_bin",
        "aim",
        "fire_requested",
    }
    assert set(row["layers"]["layer4"]["applied_action"]) >= {
        "move_bin_applied",
        "fire_applied",
    }
    assert row["decision_trace"]["why"] == "regroup"


def test_print_debug_groups_layers_and_trace(tmp_path):
    episode = _episode(tmp_path, "current_best")
    _, reward, done, info = episode.step(STAY)
    line = format_step_debug(episode, STAY, reward, done, info)

    assert "layer1.cpc_intent=" in line
    assert "layer2.target_ref=" in line
    assert "layer2.anchor_position=" in line
    assert "layer3.combat_action=" in line
    assert "layer4.applied_action=" in line
    assert "decision_trace=" in line
    assert "decision_trace_line=WHEN " in line

    episode.finish(NEUTRAL_ANSWERS, end_reason="test_complete")


def test_selfish_level_changes_only_layer1_intent_thresholds():
    common = dict(
        teammate_distance=448.0,
        human_recent_damage=0.0,
        human_hp=100.0,
        human_isolated=True,
        enemy_threatening_human=False,
        bot_goal_distance=360.0,
        bot_enemy_distance=400.0,
    )

    team_first = CpcIntentArbiter().select(CpcIntentInputs(selfish_level=0.0, **common))
    solo_first = CpcIntentArbiter().select(CpcIntentInputs(selfish_level=1.0, **common))

    assert team_first == Layer1Output("REGROUP")
    assert solo_first == Layer1Output("SOLO_OBJECTIVE")
    assert "selfish" not in inspect.getsource(CpcTargetResolver)


def test_regroup_intent_uses_existing_release_hysteresis():
    arbiter = CpcIntentArbiter()
    common = dict(
        selfish_level=0.5,
        human_recent_damage=0.0,
        human_hp=100.0,
        human_isolated=True,
        enemy_threatening_human=False,
        bot_goal_distance=360.0,
        bot_enemy_distance=400.0,
    )

    assert arbiter.select(CpcIntentInputs(teammate_distance=450.0, **common)).cpc_intent == "REGROUP"
    assert arbiter.select(CpcIntentInputs(teammate_distance=430.0, **common)).cpc_intent == "REGROUP"
    assert arbiter.select(CpcIntentInputs(teammate_distance=340.0, **common)).cpc_intent == "SOLO_OBJECTIVE"


def test_combat_micro_contract_has_only_target_and_anchor():
    assert [field.name for field in fields(ExecutionDirective)] == ["target_ref", "anchor_position"]
    assert "selfish" not in inspect.getsource(ExecutionDirective)


def test_low_selfishness_regroups_while_high_selfishness_chases_goal(tmp_path):
    low = _episode(tmp_path, "current_best", selfish_level=0.0)
    high = _episode(tmp_path, "current_best", selfish_level=1.0)

    low_before = _distance(low.env.state["ally_pos"], low.env.state["self_pos"])
    high_before = _distance(high.env.state["ally_pos"], high.env.state["self_pos"])
    _, _, _, low_info = low.step(STAY)
    _, _, _, high_info = high.step(STAY)

    assert low_info["evaluation"]["bot"]["layers"]["layer1"]["cpc_intent"] == "REGROUP"
    assert high_info["evaluation"]["bot"]["layers"]["layer1"]["cpc_intent"] == "SOLO_OBJECTIVE"
    assert _distance(low.env.state["ally_pos"], low.env.state["self_pos"]) < low_before
    assert _distance(high.env.state["ally_pos"], high.env.state["self_pos"]) > high_before

    low.finish(NEUTRAL_ANSWERS, end_reason="test_complete")
    high.finish(NEUTRAL_ANSWERS, end_reason="test_complete")


def test_focus_fire_keeps_cpc_and_poke_logs_separate(tmp_path):
    episode = _episode(tmp_path, "current_best", selfish_level=1.0)
    episode.env.enemy_move = False
    episode.env.enemy_fire = False
    episode.env.state["ally_pos"] = {"x": 100.0, "y": 100.0}
    episode.env.state["self_pos"] = {"x": 100.0, "y": 250.0}
    episode.env.state["enemy_pos"] = {"x": 200.0, "y": 100.0}

    _, _, _, info = episode.step(STAY)
    bot = info["evaluation"]["bot"]

    assert bot["layers"]["layer1"]["cpc_intent"] == "FOCUS_FIRE"
    assert bot["layers"]["layer2"]["target_ref"] == {
        "kind": "enemy",
        "id": episode.env.enemy_id,
    }
    assert bot["debug"]["poke_state"] is not None
    assert bot["debug"]["movement_policy_reason"] is not None
    assert bot["layers"]["layer3"]["combat_action"]["fire_requested"] == 1
    assert bot["layers"]["layer4"]["applied_action"]["fire_applied"] == 1
    assert bot["fire_effect_enabled"] is True
    assert any(
        event["type"] == "bullet_spawned" and event["owner_id"] == "ally"
        for event in info["events"]
    )

    episode.finish(NEUTRAL_ANSWERS, end_reason="test_complete")


def test_decision_trace_is_derived_from_layer_outputs():
    layer1 = Layer1Output("SUPPORT_TEAMMATE")
    layer2 = CpcTargetResolver().resolve(
        layer1,
        bot_position={"x": 100.0, "y": 100.0},
        human_position={"x": 300.0, "y": 480.0},
        enemy_position={"x": 360.0, "y": 480.0},
        goal_position=(700.0, 700.0),
        enemy_id="enemy_0",
        weapon_range=260.0,
        map_width=800.0,
        map_height=800.0,
    )
    layer3 = CombatAction(4, (1.0, 0.0), 1)

    trace = derive_decision_trace(layer1, layer2, layer3)

    assert trace.when == "human_under_pressure"
    assert trace.why == "support_teammate"
    assert trace.who == "enemy_0"
    assert trace.what == "engage_enemy"
    assert trace.how == "poke_out"
    assert format_decision_trace(trace) == (
        "WHEN human_under_pressure -> WHY support_teammate -> WHO enemy_0 -> "
        "WHERE anchor -> WHAT engage_enemy -> HOW poke_out"
    )


def test_viewer_panel_shows_intent_anchor_and_trace():
    trace_line = (
        "WHEN human_under_pressure -> WHY support_teammate -> WHO enemy_0 -> "
        "WHERE anchor -> WHAT engage_enemy -> HOW poke_out"
    )
    lines = _panel_lines(
        {
            "cpc_debug": {
                "cpc_intent": "SUPPORT_TEAMMATE",
                "anchor_position": [320.0, 480.0],
                "decision_trace_line": trace_line,
            }
        },
        None,
    )

    assert "intent: SUPPORT_TEAMMATE" in lines
    assert "anchor: (320.0,480.0)" in lines
    assert trace_line in lines


def _episode(
    output_root: pathlib.Path,
    policy_id: str,
    selfish_level: float | None = None,
) -> EvaluationEpisode:
    config = load_env_config("configs/env/autoplay_goal_loop.yaml")
    return EvaluationEpisode(
        config,
        policy_id=policy_id,
        scenario_id="harness-test",
        seed=17,
        output_root=output_root,
        selfish_level=selfish_level,
    )


def _distance(a: dict, b: dict) -> float:
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5


def _json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
