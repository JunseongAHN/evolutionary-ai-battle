from __future__ import annotations

import math

from ..types import AgentContext, AgentState, BaselineConfig


def create_combat_anchor(
    ctx: AgentContext,
    state: AgentState,
    combat_profile: str,
    config: BaselineConfig,
) -> tuple[tuple[float, float], dict]:
    if ctx.nearest_enemy is None:
        return ctx.player_pos, {
            "anchor": list(ctx.player_pos),
            "reason": "no_enemy_use_current_position",
            "anchor_age": 0,
            "anchor_reused": False,
            "strafe_age": 0,
            "strafe_direction": None,
            "strafe_direction_sign": state.strafe_direction,
        }

    px, py = ctx.player_pos
    ex, ey = ctx.nearest_enemy.position
    dx, dy = ex - px, ey - py
    distance = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / distance, dy / distance
    target_ratio = (config.outer_range_min_ratio + config.outer_range_max_ratio) * 0.5
    target_distance = ctx.weapon_range * target_ratio

    strafe_lock_steps = max(10, min(20, int(config.strafe_lock_steps)))
    direction = 1 if state.strafe_direction >= 0 else -1
    strafe_age = 0
    direction_reused = False

    if combat_profile == "approach_outer_band":
        anchor = (ex - ux * target_distance, ey - uy * target_distance)
        reason = "approach_outer_range_band"
    elif combat_profile == "backoff_to_outer_band":
        anchor = (px - ux * config.cell_size * 2.0, py - uy * config.cell_size * 2.0)
        reason = "backoff_toward_outer_range_band"
    elif combat_profile == "strafe_outer_band":
        if state.previous_tactical_mode == "outer_band":
            if state.strafe_age >= strafe_lock_steps:
                direction *= -1
                strafe_age = 1
                reason = "rotate_strafe_direction_after_lock"
            else:
                strafe_age = max(0, int(state.strafe_age)) + 1
                direction_reused = True
                reason = "reuse_persistent_strafe_direction"
        else:
            strafe_age = 1
            reason = "enter_outer_band_with_strafe_direction"

        tangent_distance = config.cell_size * 2.0
        anchor = (
            px + (-uy * direction * tangent_distance),
            py + (ux * direction * tangent_distance),
        )
    else:
        anchor = ctx.player_pos
        reason = f"stationary_anchor_for_{combat_profile}"

    return anchor, {
        "anchor": list(anchor),
        "reason": reason,
        "anchor_age": strafe_age,
        "anchor_reused": direction_reused,
        "strafe_age": strafe_age,
        "strafe_lock_steps": strafe_lock_steps,
        "strafe_direction": "right" if direction > 0 else "left",
        "strafe_direction_sign": direction,
        "target_distance": target_distance,
    }


__all__ = ["create_combat_anchor"]
