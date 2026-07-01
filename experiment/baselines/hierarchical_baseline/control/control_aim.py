from __future__ import annotations

import math

from ..types import AgentContext, AgentState, BaselineConfig, LocalPlan


def control_aim(
    ctx: AgentContext,
    state: AgentState,
    local_plan: LocalPlan,
    config: BaselineConfig,
) -> tuple[tuple[float, float], dict]:
    del state, config
    if ctx.nearest_enemy is not None:
        target = ctx.nearest_enemy.position
        source = "enemy_position"
    elif local_plan.anchor is not None:
        target = local_plan.anchor
        source = "local_plan_anchor"
    else:
        return (1.0, 0.0), {"aim_dir": [1.0, 0.0], "reason": "default_right"}
    dx, dy = target[0] - ctx.player_pos[0], target[1] - ctx.player_pos[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return (1.0, 0.0), {"aim_dir": [1.0, 0.0], "reason": "target_at_player"}
    direction = dx / length, dy / length
    return direction, {"aim_dir": list(direction), "reason": source}


__all__ = ["control_aim"]
