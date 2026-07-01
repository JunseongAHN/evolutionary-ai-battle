from __future__ import annotations

import ast
import copy
import pathlib
import subprocess
import sys
from dataclasses import replace

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.hierarchical_baseline import HierarchicalBaselineAgent
from baselines.hierarchical_baseline import action, context, control, debug, planning, selection
from baselines.hierarchical_baseline.action import build_action
from baselines.hierarchical_baseline.context import build_context
from baselines.hierarchical_baseline.control import (
    build_fire_status,
    control_movement,
    controller,
)
from baselines.hierarchical_baseline.planning import (
    create_combat_anchor,
    create_global_plan_if_needed,
    create_local_plan,
)
from baselines.hierarchical_baseline.selection import select_intent, select_tactical_mode
from baselines.hierarchical_baseline.types import (
    AgentContext,
    AgentState,
    BaselineConfig,
    EnemyInfo,
    GlobalPlan,
    LocalPlan,
    default_agent_state,
)
from core.cpc_env import CPCEnv
from core.env_config import load_env_config


def test_existing_baseline_moved_to_legacy():
    legacy = REPO_ROOT / "experiment" / "baselines" / "baseline_legacy"

    assert (legacy / "tactical_baseline_bot.py").is_file()
    assert (legacy / "run_tactical_autoplay.py").is_file()
    assert "old tactical baseline" in (legacy / "README.md").read_text(encoding="utf-8").lower()


def test_agent_act_is_only_state_mutation_boundary():
    config = BaselineConfig()
    state = default_agent_state()
    original = copy.deepcopy(state)
    ctx = _context(goal=(500.0, 500.0), enemy=None)

    plan, _ = create_global_plan_if_needed(ctx, state, config)
    intent, _ = select_intent(ctx, state, plan, config)
    local_plan, _ = create_local_plan(ctx, state, intent, plan, config)
    control_value, _ = controller(ctx, state, local_plan, config)
    build_action(control_value, config)

    assert state == original

    agent = HierarchicalBaselineAgent(config)
    agent.act(_goal_obs(), _goal_snapshot(enemy_alive=False))
    assert agent.state != default_agent_state()
    assert agent.state.agent_mode == "GLOBAL_NAV"


def test_agent_and_helpers_do_not_mutate_obs_or_snapshot():
    obs = _goal_obs(enemy_alive=True)
    snapshot = _goal_snapshot(enemy_alive=True)
    obs_before = copy.deepcopy(obs)
    snapshot_before = copy.deepcopy(snapshot)
    agent = HierarchicalBaselineAgent()

    build_context(obs, snapshot, agent.state, agent.config)
    agent.act(obs, snapshot)

    assert obs == obs_before
    assert snapshot == snapshot_before


def test_create_global_plan_returns_new_plan_only_when_needed():
    config = BaselineConfig()
    state = default_agent_state()
    ctx = _context(goal=(500.0, 500.0), enemy=None)

    created, created_debug = create_global_plan_if_needed(ctx, state, config)
    state.global_plan = created
    reused, reused_debug = create_global_plan_if_needed(ctx, state, config)
    changed, changed_debug = create_global_plan_if_needed(
        _context(goal=(520.0, 500.0), enemy=None, goal_count=1), state, config
    )

    assert created is not None
    assert created_debug["reason"] == "missing_global_plan"
    assert reused is None
    assert reused_debug["reason"] == "reuse_existing_plan"
    assert changed is not None
    assert changed_debug["reason"] == "goal_position_changed"


def test_create_global_plan_does_not_mutate_state():
    state = AgentState(global_plan=GlobalPlan((100.0, 100.0), 0, ((100.0, 100.0),)))
    before = copy.deepcopy(state)

    create_global_plan_if_needed(_context(goal=(200.0, 200.0), enemy=None), state, BaselineConfig())

    assert state == before


def test_select_intent_returns_combat_when_enemy_visible():
    ctx = _context(goal=(500.0, 500.0), enemy=(180.0, 100.0))
    plan = GlobalPlan((500.0, 500.0), 0, ((500.0, 500.0),))

    intent, debug_data = select_intent(ctx, default_agent_state(), plan, BaselineConfig())

    assert intent == "COMBAT"
    assert debug_data["reason"] == "enemy_alive_visible"
    assert debug_data["combat_exit_blocked_reason"] == "enemy_alive_visible"


def test_far_alive_enemy_blocks_combat_exit():
    ctx = _context(goal=(500.0, 500.0), enemy=(700.0, 100.0))
    plan = GlobalPlan((500.0, 500.0), 0, ((500.0, 500.0),))

    intent, debug_data = select_intent(ctx, AgentState(previous_intent="COMBAT"), plan, BaselineConfig())

    assert ctx.enemy_in_detection_range is False
    assert intent == "COMBAT"
    assert debug_data["combat_exit_blocked_reason"] == "enemy_alive_tracked"


def test_select_intent_returns_global_nav_when_goal_exists_and_no_combat():
    ctx = _context(goal=(500.0, 500.0), enemy=None)
    plan = GlobalPlan((500.0, 500.0), 0, ((500.0, 500.0),))

    intent, _ = select_intent(ctx, default_agent_state(), plan, BaselineConfig())

    assert intent == "GLOBAL_NAV"


def test_create_local_plan_uses_global_plan_for_global_nav():
    ctx = _context(goal=(500.0, 500.0), enemy=None)
    plan = GlobalPlan((500.0, 500.0), 0, ((500.0, 500.0),))

    local_plan, debug_data = create_local_plan(ctx, default_agent_state(), "GLOBAL_NAV", plan, BaselineConfig())

    assert local_plan.intent == "GLOBAL_NAV"
    assert local_plan.anchor == plan.goal_pos
    assert local_plan.move_bin == 8
    assert debug_data["tactical_mode"] is None


def test_create_local_plan_uses_tactical_stack_for_combat():
    ctx = _context(goal=(500.0, 500.0), enemy=(345.0, 100.0))

    local_plan, debug_data = create_local_plan(ctx, default_agent_state(), "COMBAT", None, BaselineConfig())

    assert local_plan.intent == "COMBAT"
    assert local_plan.tactical_mode == "outer_band"
    assert local_plan.combat_profile == "strafe_outer_band"
    assert local_plan.anchor is not None
    assert debug_data["reason"] == "combat_tactical_stack"


def test_outer_range_band_thresholds():
    config = BaselineConfig()

    approach, approach_debug = select_tactical_mode(
        _context(goal=None, enemy=(500.0, 100.0)), AgentState(), config
    )
    outer_band, outer_debug = select_tactical_mode(
        _context(goal=None, enemy=(345.0, 100.0)), AgentState(), config
    )
    backoff, backoff_debug = select_tactical_mode(
        _context(goal=None, enemy=(250.0, 100.0)), AgentState(), config
    )
    lower_strafe, _ = select_tactical_mode(
        _context(goal=None, enemy=(332.0, 100.0)), AgentState(), config
    )

    assert approach == "approach"
    assert approach_debug["dist_ratio"] > 0.98
    assert outer_band == "outer_band"
    assert outer_debug["target_range_band"] == [0.9, 0.98]
    assert backoff == "backoff"
    assert backoff_debug["dist_ratio"] < 0.88
    assert lower_strafe == "outer_band"


def test_outer_backoff_transition_has_five_step_hysteresis():
    config = BaselineConfig(range_hysteresis_steps=5)
    close = _context(goal=None, enemy=(320.0, 100.0))
    in_band = _context(goal=None, enemy=(345.0, 100.0))

    held_outer, outer_debug = select_tactical_mode(
        close, AgentState(previous_tactical_mode="outer_band", tactical_mode_age=2), config
    )
    switched_backoff, switch_debug = select_tactical_mode(
        close, AgentState(previous_tactical_mode="outer_band", tactical_mode_age=5), config
    )
    held_backoff, backoff_debug = select_tactical_mode(
        in_band, AgentState(previous_tactical_mode="backoff", tactical_mode_age=2), config
    )

    assert held_outer == "outer_band"
    assert outer_debug["range_hysteresis_locked"] is True
    assert switched_backoff == "backoff"
    assert switch_debug["range_hysteresis_locked"] is False
    assert held_backoff == "backoff"
    assert backoff_debug["range_hysteresis_locked"] is True


def test_outer_band_strafe_direction_is_persistent():
    state = AgentState(
        previous_tactical_mode="outer_band",
        strafe_direction=1,
        strafe_age=5,
    )

    selected, debug_data = create_combat_anchor(
        _context(goal=None, enemy=(330.0, 100.0)),
        state,
        "strafe_outer_band",
        BaselineConfig(strafe_lock_steps=14),
    )

    assert selected != state.previous_anchor
    assert debug_data["anchor_reused"] is True
    assert debug_data["anchor_age"] == 6
    assert debug_data["strafe_direction"] == "right"
    assert debug_data["strafe_direction_sign"] == 1


def test_outer_band_strafe_direction_rotates_after_lock():
    _, debug_data = create_combat_anchor(
        _context(goal=None, enemy=(330.0, 100.0)),
        AgentState(previous_tactical_mode="outer_band", strafe_direction=1, strafe_age=14),
        "strafe_outer_band",
        BaselineConfig(strafe_lock_steps=14),
    )

    assert debug_data["strafe_direction"] == "left"
    assert debug_data["strafe_direction_sign"] == -1
    assert debug_data["strafe_age"] == 1


def test_local_plan_falls_back_to_previous_valid_plan():
    previous = LocalPlan(
        intent="COMBAT",
        tactical_mode="outer_band",
        combat_profile="strafe_outer_band",
        anchor=(100.0, 180.0),
        target_cell=(12, 10),
        next_cell=(11, 10),
        path=((10, 10), (11, 10), (12, 10)),
        move_bin=2,
    )
    state = AgentState(
        previous_tactical_mode="outer_band",
        tactical_mode_age=4,
        previous_anchor=previous.anchor,
        anchor_age=4,
        previous_local_plan=previous,
    )

    selected, debug_data = create_local_plan(
        _context(goal=None, enemy=(330.0, 100.0)),
        state,
        "COMBAT",
        None,
        BaselineConfig(),
    )

    assert selected.target_cell == previous.target_cell
    assert selected.next_cell == previous.next_cell
    assert selected.move_bin == previous.move_bin
    assert debug_data["fallback_previous_plan"] is True


def test_fire_ready_safely_in_range_strafes_while_firing():
    ctx = _context(goal=None, enemy=(345.0, 100.0))
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    control_value, control_debug = controller(ctx, AgentState(strafe_direction=1), plan, BaselineConfig())
    debug_data = control_debug["movement"]

    assert control_value.move_bin == 2
    assert control_value.fire == 1
    assert debug_data["fire_window_state"] == "HOLD"
    assert debug_data["can_fire_now"] is True
    assert debug_data["hold_movement_policy"] == "safe_perpendicular_strafe"
    assert debug_data["hold_predicted_in_range"] is True
    assert debug_data["hold_stop_used"] is False
    assert debug_data["predicted_next_dist_ratio"] <= 0.98


def test_hold_stops_at_range_edge_when_strafe_would_leave_margin():
    ctx = _context(goal=None, enemy=(358.0, 100.0))
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    control_value, control_debug = controller(ctx, AgentState(), plan, BaselineConfig())
    debug_data = control_debug["movement"]

    assert control_value.fire == 1
    assert control_value.move_bin == 0
    assert debug_data["fire_window_state"] == "HOLD"
    assert debug_data["hold_stop_used"] is True
    assert debug_data["hold_predicted_in_range"] is False
    assert debug_data["hold_movement_policy"] == "hold_no_safe_in_range_move"


def test_incoming_bullet_prevents_hold_stop_at_range_edge():
    ctx = replace(
        _context(goal=None, enemy=(358.0, 100.0)),
        incoming_bullet=True,
        incoming_bullet_position=(100.0, 0.0),
        incoming_bullet_velocity=(0.0, 100.0),
    )
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    control_value, control_debug = controller(ctx, AgentState(), plan, BaselineConfig())
    debug_data = control_debug["movement"]

    assert control_value.fire == 1
    assert control_value.move_bin != 0
    assert debug_data["incoming_bullet_stop_blocked"] is True
    assert debug_data["hold_stop_used"] is False
    assert debug_data["hold_predicted_in_range"] is True
    assert debug_data["predicted_next_dist_ratio"] <= 0.98


def test_outer_band_flips_strafe_direction_when_blocked():
    cells = [[[0.0] for _ in range(3)] for _ in range(3)]
    cells[2][1][0] = 1.0
    ctx = replace(
        _context(goal=None, enemy=(345.0, 100.0)),
        line_of_sight=False,
        local_grid={"cells": cells, "center_cell": [1, 1], "channel_names": ["obstacle"]},
    )
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    move_bin, debug_data = control_movement(ctx, AgentState(strafe_direction=1), plan, BaselineConfig())

    assert move_bin == 1
    assert debug_data["strafe_direction_sign"] == -1
    assert debug_data["strafe_blocked"] is True
    assert debug_data["strafe_flip_reason"] == "map_or_obstacle_blocked"


def test_outer_band_flips_only_when_strafe_interval_expires():
    ctx = _context(goal=None, enemy=(345.0, 100.0))
    ctx = replace(ctx, line_of_sight=False)
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)
    state = AgentState(strafe_direction=1, strafe_age=14)

    move_bin, debug_data = control_movement(ctx, state, plan, BaselineConfig())

    assert move_bin == 1
    assert debug_data["strafe_direction_sign"] == -1
    assert debug_data["strafe_flip_reason"] == "interval_expired"
    assert debug_data["strafe_lock_steps_remaining"] == 13


def test_cooldown_out_of_range_holds_instead_of_drifting_farther():
    ctx = replace(
        _context(goal=None, enemy=(365.0, 100.0)),
        cooldown_ready=False,
    )
    plan = LocalPlan("COMBAT", "approach", "approach_outer_band", ctx.player_pos, None, None, (), 4)

    move_bin, debug_data = control_movement(ctx, AgentState(strafe_direction=1), plan, BaselineConfig())

    assert move_bin == 0
    assert debug_data["outer_band_strafe_active"] is False
    assert debug_data["fire_window_state"] == "RESET"
    assert debug_data["movement_policy_reason"] == "fire_window_reset_hold_out_of_range"
    assert debug_data["retreat_diagonal_allowed"] is False


def test_fire_hold_priority_keeps_shot_stable_when_bullet_is_present():
    ctx = replace(
        _context(goal=None, enemy=(345.0, 100.0)),
        incoming_bullet=True,
        incoming_bullet_position=(100.0, -100.0),
        incoming_bullet_velocity=(0.0, 100.0),
        incoming_bullet_radius=12.0,
    )
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    state = AgentState(strafe_direction=1)
    control_value, control_debug = controller(ctx, state, plan, BaselineConfig())
    debug_data = control_debug["movement"]

    assert control_value.move_bin == 2
    assert control_value.fire == 1
    assert debug_data["fire_window_state"] == "HOLD"
    assert debug_data["bullet_strafe_lock_active"] is True
    assert debug_data["incoming_bullet_stop_blocked"] is True
    assert debug_data["hold_movement_policy"] == "incoming_bullet_perpendicular"


def test_hold_bullet_strafe_lock_keeps_perpendicular_move():
    ctx = _context(goal=None, enemy=(345.0, 100.0))
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)
    state = AgentState(
        strafe_direction=1,
        strafe_age=2,
        dodge_lock_steps_remaining=2,
        dodge_lock_move_bin=2,
    )

    move_bin, debug_data = control_movement(ctx, state, plan, BaselineConfig())

    assert move_bin == 2
    assert debug_data["bullet_strafe_lock_active"] is True
    assert debug_data["dodge_lock_steps_remaining"] == 1
    assert debug_data["dodge_lock_move_bin"] == 2
    assert debug_data["strafe_flip_reason"] is None
    assert debug_data["fire_window_state"] == "HOLD"
    assert debug_data["hold_movement_policy"] == "incoming_bullet_locked_strafe"


def test_outer_band_rejects_old_retreat_diagonal_dodge_lock():
    ctx = _context(goal=None, enemy=(345.0, 100.0))
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)
    state = AgentState(dodge_lock_steps_remaining=2, dodge_lock_move_bin=5)

    move_bin, debug_data = control_movement(ctx, state, plan, BaselineConfig())

    assert move_bin == 2
    assert debug_data["bullet_strafe_lock_active"] is False
    assert debug_data["dodge_lock_steps_remaining"] == 0
    assert debug_data["retreat_diagonal_allowed"] is False


def test_outer_band_flips_strafe_near_boundary():
    base = _context(goal=None, enemy=(345.0, 750.0))
    ctx = replace(
        base,
        player_pos=(100.0, 750.0),
        enemy_dist=245.0,
        enemy_in_range=True,
        line_of_sight=False,
        map_width=800.0,
        map_height=800.0,
        player_radius=12.0,
    )
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    move_bin, debug_data = control_movement(ctx, AgentState(strafe_direction=1), plan, BaselineConfig())

    assert move_bin == 1
    assert debug_data["strafe_direction_sign"] == -1
    assert debug_data["strafe_flip_reason"] == "near_boundary"


def test_fire_window_too_close_retreat_has_highest_priority():
    ctx = replace(
        _context(goal=None, enemy=(280.0, 100.0)),
        incoming_bullet=True,
        incoming_bullet_position=(100.0, 0.0),
        incoming_bullet_velocity=(0.0, 100.0),
    )
    plan = LocalPlan("COMBAT", "backoff", "backoff_to_outer_band", ctx.player_pos, None, None, (), 0)

    move_bin, debug_data = control_movement(ctx, AgentState(), plan, BaselineConfig())

    assert move_bin == 3
    assert debug_data["retreat_diagonal_allowed"] is True
    assert debug_data["selected_dodge_move"]["name"] == "pure_retreat"
    assert debug_data["bullet_strafe_lock_active"] is False
    assert debug_data["fire_window_state"] == "TOO_CLOSE"
    assert debug_data["movement_policy_reason"] == "fire_window_too_close_retreat"


def test_fire_ready_out_of_range_enters_instead_of_strafing():
    ctx = _context(goal=None, enemy=(400.0, 100.0))
    plan = LocalPlan("COMBAT", "approach", "approach_outer_band", ctx.player_pos, None, None, (), 0)

    move_bin, debug_data = control_movement(ctx, AgentState(), plan, BaselineConfig())

    assert move_bin == 4
    assert debug_data["outer_band_strafe_active"] is False
    assert debug_data["retreat_diagonal_allowed"] is False
    assert debug_data["fire_window_state"] == "ENTER"
    assert debug_data["fire_ready"] is True
    assert debug_data["target_in_range"] is False
    assert debug_data["movement_policy_reason"] == "fire_window_enter_approach"


def test_cooldown_out_of_range_resets_without_approaching():
    ctx = replace(
        _context(goal=None, enemy=(400.0, 100.0)),
        cooldown_ready=False,
    )
    plan = LocalPlan("COMBAT", "approach", "approach_outer_band", ctx.player_pos, None, None, (), 4)

    move_bin, debug_data = control_movement(ctx, AgentState(strafe_direction=1), plan, BaselineConfig())

    assert move_bin == 0
    assert move_bin != 4
    assert debug_data["fire_window_state"] == "RESET"
    assert debug_data["fire_ready"] is False
    assert debug_data["target_in_range"] is False
    assert debug_data["movement_policy_reason"] == "fire_window_reset_hold_out_of_range"


def test_incoming_bullet_uses_open_emergency_move_when_tangents_are_blocked():
    cells = [[[0.0] for _ in range(3)] for _ in range(3)]
    cells[0][1][0] = 1.0
    cells[2][1][0] = 1.0
    ctx = replace(
        _context(goal=None, enemy=(400.0, 100.0)),
        cooldown_ready=False,
        incoming_bullet=True,
        incoming_bullet_position=(100.0, 0.0),
        incoming_bullet_velocity=(0.0, 100.0),
        local_grid={"cells": cells, "center_cell": [1, 1], "channel_names": ["obstacle"]},
    )
    plan = LocalPlan("COMBAT", "approach", "approach_outer_band", ctx.player_pos, None, None, (), 0)

    move_bin, debug_data = control_movement(ctx, AgentState(), plan, BaselineConfig())

    assert move_bin == 4
    assert debug_data["incoming_bullet_stop_blocked"] is True
    assert debug_data["movement_policy_reason"].endswith("incoming_emergency")


def test_incoming_bullet_can_stop_only_when_every_move_is_blocked():
    cells = [[[0.0] for _ in range(3)] for _ in range(3)]
    for row in range(3):
        for col in range(3):
            if (row, col) != (1, 1):
                cells[row][col][0] = 1.0
    ctx = replace(
        _context(goal=None, enemy=(400.0, 100.0)),
        cooldown_ready=False,
        incoming_bullet=True,
        incoming_bullet_position=(100.0, 0.0),
        incoming_bullet_velocity=(0.0, 100.0),
        local_grid={"cells": cells, "center_cell": [1, 1], "channel_names": ["obstacle"]},
    )
    plan = LocalPlan("COMBAT", "approach", "approach_outer_band", ctx.player_pos, None, None, (), 0)

    move_bin, debug_data = control_movement(ctx, AgentState(), plan, BaselineConfig())

    assert move_bin == 0
    assert debug_data["incoming_bullet_stop_blocked"] is False


def test_cooldown_inside_range_backs_off_slightly():
    ctx = replace(
        _context(goal=None, enemy=(308.0, 100.0)),
        cooldown_ready=False,
    )
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    move_bin, debug_data = control_movement(ctx, AgentState(), plan, BaselineConfig())

    assert move_bin == 3
    assert debug_data["fire_window_state"] == "RESET"
    assert debug_data["movement_policy_reason"] == "fire_window_reset_backoff"
    assert debug_data["reset_soft_backoff_active"] is True


def test_agent_act_applies_dodge_lock_state_delta():
    agent = HierarchicalBaselineAgent()
    obs = _goal_obs(enemy_alive=True)
    obs["enemy_pos"] = {"x": 345.0, "y": 100.0}
    obs["can_fire"] = False
    snapshot = _goal_snapshot(enemy_alive=True)
    snapshot["agents"]["enemy"]["position"] = {"x": 345.0, "y": 100.0}
    snapshot["weapon"]["cooldown_remaining_steps"] = 3
    snapshot["bullets"] = [
        {
            "bullet_id": "incoming",
            "owner_id": "enemy",
            "position": [100.0, -100.0],
            "velocity": [0.0, 100.0],
            "radius": 12.0,
            "alive": True,
        }
    ]

    _, debug_data = agent.act(obs, snapshot)

    assert debug_data["incoming_bullet_stop_blocked"] is True
    assert debug_data["fire_window_state"] == "RESET"
    assert debug_data["reset_soft_backoff_active"] is True
    assert debug_data["move_bin"] != 0


def test_context_detects_only_hostile_bullets_approaching_player_path():
    snapshot = _goal_snapshot(enemy_alive=True)
    snapshot["bullets"] = [
        {
            "bullet_id": "incoming",
            "owner_id": "enemy",
            "position": [100.0, 0.0],
            "velocity": [0.0, 100.0],
            "alive": True,
        },
        {
            "bullet_id": "passing",
            "owner_id": "enemy",
            "position": [200.0, 0.0],
            "velocity": [0.0, 100.0],
            "alive": True,
        },
        {
            "bullet_id": "friendly",
            "owner_id": "self",
            "position": [100.0, 0.0],
            "velocity": [0.0, 100.0],
            "alive": True,
        },
    ]

    ctx, debug_data = build_context(_goal_obs(enemy_alive=True), snapshot, AgentState(), BaselineConfig())

    assert ctx.incoming_bullet is True
    assert ctx.incoming_bullet_position == (100.0, 0.0)
    assert ctx.incoming_bullet_velocity == (0.0, 100.0)
    assert ctx.incoming_bullet_radius == 12.0
    assert debug_data["incoming_bullet"] is True


def test_controller_builds_continuous_aim_action():
    ctx = _context(goal=None, enemy=(50.0, 100.0))
    local_plan, _ = create_local_plan(ctx, default_agent_state(), "COMBAT", None, BaselineConfig())

    control_value, _ = controller(ctx, default_agent_state(), local_plan, BaselineConfig())
    env_action, _ = build_action(control_value, BaselineConfig())

    assert env_action["aim_dx"] == -1.0
    assert env_action["aim_dy"] == 0.0
    assert "aim" not in env_action
    assert "aim_bin" not in env_action


def test_fire_status_is_exposed_before_movement_and_controls_fire():
    ctx = _context(goal=None, enemy=(345.0, 100.0))
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)
    state = AgentState()

    status = build_fire_status(ctx, state, plan, BaselineConfig(), (1.0, 0.0))
    control_value, control_debug = controller(ctx, state, plan, BaselineConfig())

    assert {
        "fire_ready",
        "target_in_range",
        "aim_ok",
        "los_ok",
        "fire_reason",
        "can_fire_now",
    }.issubset(status)
    assert status["can_fire_now"] is True
    assert control_debug["fire_status"] == status
    assert control_value.fire == int(status["can_fire_now"])


def test_fire_status_preserves_aim_error_threshold():
    ctx = _context(goal=None, enemy=(345.0, 100.0))
    plan = LocalPlan("COMBAT", "outer_band", "strafe_outer_band", ctx.player_pos, None, None, (), 0)

    status = build_fire_status(ctx, AgentState(), plan, BaselineConfig(), (0.0, 1.0))

    assert status["aim_ok"] is False
    assert status["can_fire_now"] is False
    assert status["fire_reason"] == "aim_not_aligned"


def test_agent_returns_to_global_navigation_after_combat_interrupt():
    config = BaselineConfig(combat_exit_grace_steps=2)
    agent = HierarchicalBaselineAgent(config)
    action, combat_debug = agent.act(_goal_obs(enemy_alive=True), _goal_snapshot(enemy_alive=True))
    assert combat_debug["intent"] == "COMBAT"
    assert action["aim_dx"] == 1.0

    final_debug = combat_debug
    for _ in range(config.combat_exit_grace_steps + 1):
        _, final_debug = agent.act(_goal_obs(enemy_alive=False), _goal_snapshot(enemy_alive=False))

    assert final_debug["intent"] == "GLOBAL_NAV"
    assert agent.state.global_plan is not None


def test_public_functions_exported_from_init():
    expected = {
        context: {"build_context"},
        planning: {"create_combat_anchor", "create_global_plan_if_needed", "create_local_path", "create_local_plan"},
        selection: {"select_combat_profile", "select_intent", "select_tactical_mode"},
        control: {"build_fire_status", "control_aim", "control_fire", "control_movement", "controller"},
        action: {"build_action"},
        debug: {"build_debug", "format_debug"},
    }
    for package, names in expected.items():
        assert set(package.__all__) == names
        assert all(callable(getattr(package, name)) for name in names)


def test_private_helpers_not_exported_from_init():
    for package in (context, planning, selection, control, action, debug):
        assert not any(name.startswith("_") for name in package.__all__)


def test_only_agent_module_defines_behavior_class():
    package_root = REPO_ROOT / "experiment" / "baselines" / "hierarchical_baseline"
    classes: list[tuple[str, str]] = []
    for path in package_root.rglob("*.py"):
        if path.name == "types.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes.extend((path.name, node.name) for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

    assert classes == [("agent.py", "HierarchicalBaselineAgent")]


def test_hierarchical_autoplay_runs_goal_and_combat_scenarios():
    runner = "experiment/baselines/hierarchical_baseline/run_hierarchical_autoplay.py"
    for config_path, expected_intent in (
        ("configs/env/autoplay_goal_loop.yaml", "GLOBAL_NAV"),
        ("configs/env/autoplay_enemy_right.yaml", "COMBAT"),
    ):
        result = subprocess.run(
            [sys.executable, runner, "--config", config_path, "--steps", "1", "--fps", "0", "--print-debug"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert f"intent={expected_intent}" in result.stdout
        assert "global_plan_reason=" in result.stdout
        assert "aim_dir=" in result.stdout


def test_hierarchical_debug_script_prints_required_trace_fields():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hierarchical_baseline_debug.py",
            "--config",
            "configs/env/autoplay_goal_loop.yaml",
            "--steps",
            "2",
            "--fps",
            "0",
            "--print-debug",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for field in (
        "step=",
        "env_step=",
        "player=",
        "goal=",
        "distance_to_goal=",
        "intent=",
        "global_plan_reason=",
        "tactical_mode=",
        "combat_profile=",
        "anchor=",
        "target_cell=",
        "next_cell=",
        "move_bin=",
        "aim_dir=",
        "fire=",
        "fire_reason=",
        "fire_window_state=",
        "fire_ready=",
        "target_in_range=",
        "can_fire_now=",
        "range_policy_reason=",
        "hold_movement_policy=",
        "hold_predicted_in_range=",
        "hold_stop_used=",
        "incoming_bullet_stop_blocked=",
        "reset_soft_backoff_active=",
        "predicted_next_dist_ratio=",
        "events=",
        "mode_age=",
        "mode_locked=",
        "anchor_age=",
        "anchor_reused=",
        "fallback_previous_plan=",
        "combat_range_state=",
        "range_state=",
        "target_range_band=",
        "dist_ratio=",
        "strafe_direction=",
        "perpendicular_strafe=",
        "outer_band_strafe_active=",
        "strafe_lock_steps_remaining=",
        "strafe_flip_reason=",
        "bullet_strafe_lock_active=",
        "retreat_diagonal_allowed=",
        "movement_policy_reason=",
        "bullet_dodge_active=",
        "dodge_reason=",
        "bullet_safety_margin=",
        "dodge_lock_active=",
        "dodge_lock_steps_remaining=",
        "dodge_lock_move_bin=",
        "cooldown_strafe_fallback_used=",
        "dodge_candidates=",
        "selected_dodge_move=",
        "dodge_blocked_reasons=",
        "enemy_opposite_component_used=",
        "range_hysteresis_locked=",
        "combat_exit_blocked_reason=",
    ):
        assert field in result.stdout


def test_goal_loop_combat_range_controller_smoke():
    env_config = load_env_config("configs/env/autoplay_goal_loop.yaml")
    env = CPCEnv.from_config(env_config)
    obs = env.reset(seed=17)
    agent = HierarchicalBaselineAgent()
    combat_started = False
    combat_steps = 0

    for _ in range(300):
        action, debug_data = agent.act(obs, env.get_snapshot())
        enemy_alive = debug_data["context"].get("enemy_id") is not None
        if debug_data["intent"] == "COMBAT":
            combat_started = True
        if combat_started and enemy_alive:
            combat_steps += 1
            assert debug_data["intent"] == "COMBAT"
        fire_window_state = debug_data.get("fire_window_state")
        if combat_started and enemy_alive and fire_window_state == "ENTER":
            assert debug_data["fire_ready"] is True
            assert debug_data["target_in_range"] is False
            assert action["move_bin"] != 0
        if combat_started and enemy_alive and fire_window_state == "HOLD":
            assert debug_data["fire_ready"] is True
            assert debug_data["target_in_range"] is True
            if debug_data["can_fire_now"]:
                assert action["fire"] == 1
                if action["move_bin"] == 0:
                    assert debug_data["hold_stop_used"] is True
                else:
                    assert debug_data["hold_predicted_in_range"] is True
                    assert debug_data["predicted_next_dist_ratio"] <= 0.98
        if combat_started and enemy_alive and fire_window_state == "RESET":
            assert debug_data["fire_ready"] is False
            if debug_data["target_in_range"] or debug_data["context"].get("incoming_bullet"):
                assert action["move_bin"] != 0
        if combat_started and enemy_alive and fire_window_state == "TOO_CLOSE":
            assert action["move_bin"] != 0
        if action["fire"]:
            fire_debug = debug_data["control"]["fire"]
            assert debug_data["control"]["fire_status"]["can_fire_now"] is True
            assert fire_debug["line_of_sight"] is True
            assert fire_debug["aim_ok"] is True
            assert fire_debug["cooldown_ready"] is True
            assert fire_debug["enemy_in_range"] is True

        result = env.step(action)
        obs = result[0]
        done = bool(result[2]) if len(result) == 4 else bool(result[2] or result[3])
        if done:
            break

    assert combat_started is True
    assert combat_steps > 0


def _context(
    *,
    goal: tuple[float, float] | None,
    enemy: tuple[float, float] | None,
    goal_count: int = 0,
) -> AgentContext:
    enemy_info = EnemyInfo("enemy", enemy, 100.0, True) if enemy is not None else None
    enemy_dist = None if enemy is None else ((enemy[0] - 100.0) ** 2 + (enemy[1] - 100.0) ** 2) ** 0.5
    return AgentContext(
        player_pos=(100.0, 100.0),
        player_hp=100.0,
        player_alive=True,
        goal_pos=goal,
        goal_reached_count=goal_count,
        nearest_enemy=enemy_info,
        enemy_dist=enemy_dist,
        enemy_in_range=bool(enemy_dist is not None and enemy_dist <= 260.0),
        enemy_in_detection_range=bool(enemy_dist is not None and enemy_dist <= 360.0),
        line_of_sight=enemy is not None,
        weapon_range=260.0,
        cooldown_ready=True,
        bullet_count=0,
        incoming_bullet=False,
        events=(),
        local_grid=None,
        obstacles=(),
    )


def _goal_obs(*, enemy_alive: bool = False) -> dict:
    obs = {
        "self_pos": {"x": 100.0, "y": 100.0},
        "self_hp": 100.0,
        "goal_enabled": True,
        "goal_position": [500.0, 500.0],
        "goal_reached_count": 0,
        "can_fire": True,
    }
    if enemy_alive:
        obs.update({"enemy_pos": {"x": 200.0, "y": 100.0}, "enemy_hp": 100.0})
    return obs


def _goal_snapshot(*, enemy_alive: bool) -> dict:
    return {
        "agents": {
            "self": {"position": {"x": 100.0, "y": 100.0}, "hp": 100.0, "alive": True},
            "enemy": {
                "id": "enemy",
                "position": {"x": 200.0, "y": 100.0},
                "hp": 100.0 if enemy_alive else 0.0,
                "alive": enemy_alive,
            },
        },
        "goal": {"enabled": True, "position": {"x": 500.0, "y": 500.0}, "reached_count": 0},
        "combat": {"fire_range": 260.0},
        "weapon": {"cooldown_remaining_steps": 0},
        "events": [],
        "obstacles": [],
    }
