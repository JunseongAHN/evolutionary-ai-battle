from __future__ import annotations

from ..selection import select_combat_profile, select_tactical_mode
from ..types import AgentContext, AgentState, BaselineConfig, GlobalPlan, LocalPlan
from .create_combat_anchor import create_combat_anchor
from .create_local_path import create_local_path


def create_local_plan(
    ctx: AgentContext,
    state: AgentState,
    intent: str,
    global_plan: GlobalPlan | None,
    config: BaselineConfig,
) -> tuple[LocalPlan, dict]:
    if intent == "IDLE":
        plan = LocalPlan(intent, None, None, None, None, None, (), 0)
        return plan, {"reason": "idle", "move_bin": 0}

    mode_debug: dict = {}
    profile_debug: dict = {}
    anchor_debug: dict = {}
    if intent == "GLOBAL_NAV":
        anchor = global_plan.waypoints[-1] if global_plan and global_plan.waypoints else ctx.player_pos
        tactical_mode = None
        combat_profile = None
        anchor_debug = {"anchor": list(anchor), "reason": "global_goal_waypoint"}
    else:
        tactical_mode, mode_debug = select_tactical_mode(ctx, state, config)
        combat_profile, profile_debug = select_combat_profile(ctx, state, tactical_mode, config)
        anchor, anchor_debug = create_combat_anchor(ctx, state, combat_profile, config)

    path, target_cell, next_cell, move_bin, path_debug = create_local_path(
        ctx, state, anchor, combat_profile, config
    )
    fallback_previous_plan = False
    previous_plan = state.previous_local_plan
    if (
        target_cell is None
        and previous_plan is not None
        and previous_plan.intent == intent
        and previous_plan.target_cell is not None
        and previous_plan.next_cell is not None
    ):
        path = previous_plan.path
        target_cell = previous_plan.target_cell
        next_cell = previous_plan.next_cell
        move_bin = previous_plan.move_bin
        fallback_previous_plan = True
        path_debug = {
            **path_debug,
            "reason": "reuse_previous_valid_local_plan",
            "target_cell": list(target_cell),
            "next_cell": list(next_cell),
            "path": [list(cell) for cell in path],
            "move_bin": move_bin,
        }
    plan = LocalPlan(
        intent=intent,
        tactical_mode=tactical_mode,
        combat_profile=combat_profile,
        anchor=anchor,
        target_cell=target_cell,
        next_cell=next_cell,
        path=path,
        move_bin=move_bin,
    )
    return plan, {
        "reason": "global_navigation_plan" if intent == "GLOBAL_NAV" else "combat_tactical_stack",
        "intent": intent,
        "tactical_mode": tactical_mode,
        "combat_profile": combat_profile,
        "anchor": list(anchor),
        "target_cell": list(target_cell) if target_cell is not None else None,
        "next_cell": list(next_cell) if next_cell is not None else None,
        "path": [list(cell) for cell in path],
        "move_bin": move_bin,
        "mode_age": mode_debug.get("mode_age", 0),
        "mode_locked": bool(mode_debug.get("mode_locked", False)),
        "combat_range_state": mode_debug.get("combat_range_state"),
        "range_state": mode_debug.get("range_state"),
        "range_hysteresis_locked": bool(mode_debug.get("range_hysteresis_locked", False)),
        "target_range_band": mode_debug.get("target_range_band"),
        "dist_ratio": mode_debug.get("dist_ratio"),
        "anchor_age": anchor_debug.get("anchor_age", 0),
        "anchor_reused": bool(anchor_debug.get("anchor_reused", False)),
        "strafe_age": anchor_debug.get("strafe_age", 0),
        "strafe_direction": anchor_debug.get("strafe_direction"),
        "strafe_direction_sign": anchor_debug.get("strafe_direction_sign", state.strafe_direction),
        "fallback_previous_plan": fallback_previous_plan,
        "mode": mode_debug,
        "profile": profile_debug,
        "anchor_debug": anchor_debug,
        "path_debug": path_debug,
    }


__all__ = ["create_local_plan"]
