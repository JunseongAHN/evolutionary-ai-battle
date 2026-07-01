from __future__ import annotations

from ..types import AgentContext, AgentState, BaselineConfig, GlobalPlan


def create_global_plan_if_needed(
    ctx: AgentContext,
    state: AgentState,
    config: BaselineConfig,
) -> tuple[GlobalPlan | None, dict]:
    del config
    if ctx.goal_pos is None:
        return None, {"reason": "no_goal", "created": False}
    current = state.global_plan
    if current is None:
        reason = "missing_global_plan"
    elif current.goal_pos != ctx.goal_pos:
        reason = "goal_position_changed"
    elif current.goal_reached_count != ctx.goal_reached_count:
        reason = "goal_reached_count_changed"
    else:
        return None, {"reason": "reuse_existing_plan", "created": False}
    plan = GlobalPlan(
        goal_pos=ctx.goal_pos,
        goal_reached_count=ctx.goal_reached_count,
        waypoints=(ctx.goal_pos,),
    )
    return plan, {
        "reason": reason,
        "created": True,
        "goal_pos": list(ctx.goal_pos),
        "waypoints": [list(point) for point in plan.waypoints],
    }


__all__ = ["create_global_plan_if_needed"]
