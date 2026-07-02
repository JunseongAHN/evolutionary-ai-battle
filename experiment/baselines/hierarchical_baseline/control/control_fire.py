from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..types import AgentContext, AgentState, BaselineConfig, LocalPlan


def build_fire_status(
    ctx: AgentContext,
    state: AgentState,
    local_plan: LocalPlan,
    config: BaselineConfig,
    aim: tuple[float, float] | None = None,
) -> dict[str, Any]:
    del state
    enemy = ctx.nearest_enemy
    live_combat_target = bool(
        local_plan.intent == "COMBAT"
        and ctx.player_alive
        and enemy is not None
        and enemy.alive
    )
    fire_ready = bool(live_combat_target and ctx.cooldown_ready)
    target_in_range = bool(live_combat_target and ctx.enemy_in_range)
    aim_error = _aim_error(ctx, aim) if live_combat_target else None
    aim_ok = bool(
        live_combat_target
        and aim_error is not None
        and aim_error <= config.fire_aim_error_threshold
    )
    los_ok = bool(live_combat_target and ctx.line_of_sight)
    poke_edge_fire = bool(
        str(config.combat_movement_profile or "").strip().lower() == "poke_out"
        and fire_ready
        and not target_in_range
        and ctx.enemy_dist is not None
        and ctx.enemy_dist <= ctx.weapon_range + max(0.0, float(config.move_step_distance))
        and aim_ok
        and los_ok
    )
    can_fire_now = bool(
        fire_ready
        and (target_in_range or poke_edge_fire)
        and aim_ok
        and los_ok
    )

    if local_plan.intent != "COMBAT":
        fire_reason = "not_combat_intent"
    elif not ctx.player_alive:
        fire_reason = "player_dead"
    elif enemy is None or not enemy.alive:
        fire_reason = "no_live_enemy"
    elif not target_in_range and not poke_edge_fire:
        fire_reason = "target_out_of_range"
    elif not los_ok:
        fire_reason = "line_of_sight_blocked"
    elif not aim_ok:
        fire_reason = "aim_not_aligned"
    elif not fire_ready:
        fire_reason = "cooldown_not_ready"
    elif poke_edge_fire:
        fire_reason = "poke_edge_fire"
    else:
        fire_reason = "all_conditions_met"

    return {
        "fire_ready": fire_ready,
        "target_in_range": target_in_range,
        "aim_ok": aim_ok,
        "los_ok": los_ok,
        "fire_reason": fire_reason,
        "can_fire_now": can_fire_now,
        "poke_edge_fire": poke_edge_fire,
        "aim_error": aim_error,
    }


def control_fire(
    ctx: AgentContext,
    state: AgentState,
    local_plan: LocalPlan,
    config: BaselineConfig,
    fire_status: Mapping[str, Any] | None = None,
) -> tuple[int, dict]:
    status = dict(
        fire_status
        if fire_status is not None
        else build_fire_status(ctx, state, local_plan, config)
    )
    fire = int(bool(status.get("can_fire_now", False)))
    return fire, {
        **status,
        "fire": fire,
        "reason": status.get("fire_reason"),
        "line_of_sight": bool(status.get("los_ok", False)),
        "enemy_in_range": bool(status.get("target_in_range", False)),
        "cooldown_ready": bool(status.get("fire_ready", False)),
    }


def _aim_error(ctx: AgentContext, aim: tuple[float, float] | None) -> float | None:
    if ctx.nearest_enemy is None:
        return None
    target_dx = ctx.nearest_enemy.position[0] - ctx.player_pos[0]
    target_dy = ctx.nearest_enemy.position[1] - ctx.player_pos[1]
    target_length = math.hypot(target_dx, target_dy)
    if target_length <= 1e-6:
        return 0.0
    if aim is None:
        return 0.0
    aim_length = math.hypot(*aim)
    if aim_length <= 1e-6:
        return math.pi
    dot = (
        (target_dx / target_length) * (aim[0] / aim_length)
        + (target_dy / target_length) * (aim[1] / aim_length)
    )
    return math.acos(max(-1.0, min(1.0, dot)))


__all__ = ["build_fire_status", "control_fire"]
