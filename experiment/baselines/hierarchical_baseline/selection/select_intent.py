from __future__ import annotations

from ..types import AgentContext, AgentState, BaselineConfig, GlobalPlan


COMBAT_EVENT_TYPES = {
    "bullet_spawned",
    "bullet_hit",
    "damage",
    "enemy_spawned",
}


def select_intent(
    ctx: AgentContext,
    state: AgentState,
    global_plan: GlobalPlan | None,
    config: BaselineConfig,
) -> tuple[str, dict]:
    event_types = {
        str(event.get("type", event.get("event_type", "")))
        for event in ctx.events
    }
    recent_combat = bool(event_types & COMBAT_EVENT_TYPES)
    enemy_alive = ctx.nearest_enemy is not None and ctx.nearest_enemy.alive
    enemy_alive_visible = bool(enemy_alive and ctx.enemy_in_detection_range)
    enemy_alive_tracked = bool(enemy_alive and state.previous_intent == "COMBAT")
    direct_threat = bool(
        enemy_alive_visible
        or enemy_alive_tracked
        or ctx.incoming_bullet
        or recent_combat
    )
    hysteresis = bool(
        not enemy_alive
        and state.previous_intent == "COMBAT"
        and state.no_enemy_steps < max(0, int(config.combat_exit_grace_steps))
        and (ctx.incoming_bullet or recent_combat)
    )
    if direct_threat or hysteresis:
        if enemy_alive_visible:
            reason = "enemy_alive_visible"
            combat_exit_blocked_reason = "enemy_alive_visible"
        elif enemy_alive_tracked:
            reason = "enemy_alive_tracked_outside_detection"
            combat_exit_blocked_reason = "enemy_alive_tracked"
        elif ctx.incoming_bullet:
            reason = "incoming_bullet_threat"
            combat_exit_blocked_reason = "incoming_bullet"
        elif recent_combat:
            reason = "recent_combat_event"
            combat_exit_blocked_reason = "recent_combat_event"
        else:
            reason = "combat_exit_hysteresis"
            combat_exit_blocked_reason = "combat_exit_hysteresis"
        intent = "COMBAT"
    elif global_plan is not None:
        reason = "goal_plan_available"
        intent = "GLOBAL_NAV"
        combat_exit_blocked_reason = None
    else:
        reason = "no_goal_or_combat"
        intent = "IDLE"
        combat_exit_blocked_reason = None
    return intent, {
        "intent": intent,
        "reason": reason,
        "enemy_in_detection_range": ctx.enemy_in_detection_range,
        "incoming_bullet": ctx.incoming_bullet,
        "recent_combat_event": recent_combat,
        "hysteresis": hysteresis,
        "enemy_alive_visible": enemy_alive_visible,
        "enemy_alive_tracked": enemy_alive_tracked,
        "combat_exit_blocked_reason": combat_exit_blocked_reason,
    }


__all__ = ["select_intent"]
