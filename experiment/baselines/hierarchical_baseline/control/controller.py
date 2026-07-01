from __future__ import annotations

from ..types import AgentContext, AgentState, BaselineConfig, Control, LocalPlan
from .control_aim import control_aim
from .control_fire import build_fire_status, control_fire
from .control_movement import control_movement


def controller(
    ctx: AgentContext,
    state: AgentState,
    local_plan: LocalPlan,
    config: BaselineConfig,
) -> tuple[Control, dict]:
    aim, aim_debug = control_aim(ctx, state, local_plan, config)
    fire_status = build_fire_status(ctx, state, local_plan, config, aim)
    move_bin, movement_debug = control_movement(
        ctx,
        state,
        local_plan,
        config,
        fire_status,
    )
    fire, fire_debug = control_fire(ctx, state, local_plan, config, fire_status)
    control = Control(move_bin=move_bin, aim_dx=aim[0], aim_dy=aim[1], fire=fire)
    return control, {
        "movement": movement_debug,
        "aim": aim_debug,
        "fire_status": fire_status,
        "fire": fire_debug,
    }


__all__ = ["controller"]
