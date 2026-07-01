from __future__ import annotations

from typing import Any


def build_debug(
    context_debug: dict[str, Any],
    global_debug: dict[str, Any],
    intent_debug: dict[str, Any],
    local_debug: dict[str, Any],
    control_debug: dict[str, Any],
    action_debug: dict[str, Any],
) -> dict[str, Any]:
    fire_debug = control_debug.get("fire", {})
    fire_status = control_debug.get("fire_status", fire_debug)
    movement_debug = control_debug.get("movement", {})
    return {
        "context": context_debug,
        "global_plan": global_debug,
        "intent_selection": intent_debug,
        "local_plan": local_debug,
        "control": control_debug,
        "action": action_debug,
        "intent": intent_debug.get("intent"),
        "global_plan_reason": global_debug.get("reason"),
        "goal_pos": context_debug.get("goal_pos"),
        "tactical_mode": local_debug.get("tactical_mode"),
        "combat_profile": local_debug.get("combat_profile"),
        "anchor": local_debug.get("anchor"),
        "target_cell": local_debug.get("target_cell"),
        "next_cell": local_debug.get("next_cell"),
        "path": local_debug.get("path", []),
        "move_bin": action_debug.get("move_bin", 0),
        "aim_dir": action_debug.get("aim_dir", [1.0, 0.0]),
        "fire": action_debug.get("fire", 0),
        "fire_reason": fire_debug.get("reason"),
        "fire_window_state": movement_debug.get("fire_window_state"),
        "fire_ready": bool(fire_status.get("fire_ready", False)),
        "target_in_range": bool(fire_status.get("target_in_range", False)),
        "can_fire_now": bool(fire_status.get("can_fire_now", False)),
        "range_policy_reason": movement_debug.get("range_policy_reason"),
        "hold_movement_policy": movement_debug.get("hold_movement_policy"),
        "hold_predicted_in_range": movement_debug.get("hold_predicted_in_range"),
        "hold_stop_used": bool(movement_debug.get("hold_stop_used", False)),
        "incoming_bullet_stop_blocked": bool(
            movement_debug.get("incoming_bullet_stop_blocked", False)
        ),
        "reset_soft_backoff_active": bool(
            movement_debug.get("reset_soft_backoff_active", False)
        ),
        "predicted_next_dist_ratio": movement_debug.get("predicted_next_dist_ratio"),
        "events": context_debug.get("event_types", []),
        "mode_age": local_debug.get("mode_age", 0),
        "mode_locked": bool(local_debug.get("mode_locked", False)),
        "anchor_age": local_debug.get("anchor_age", 0),
        "anchor_reused": bool(local_debug.get("anchor_reused", False)),
        "fallback_previous_plan": bool(local_debug.get("fallback_previous_plan", False)),
        "combat_range_state": local_debug.get("combat_range_state"),
        "range_state": local_debug.get("range_state"),
        "target_range_band": local_debug.get("target_range_band"),
        "dist_ratio": local_debug.get("dist_ratio"),
        "strafe_direction": movement_debug.get(
            "strafe_direction", local_debug.get("strafe_direction")
        ),
        "perpendicular_strafe": bool(movement_debug.get("perpendicular_strafe", False)),
        "outer_band_strafe_active": bool(
            movement_debug.get("outer_band_strafe_active", False)
        ),
        "strafe_lock_steps_remaining": movement_debug.get(
            "strafe_lock_steps_remaining", 0
        ),
        "strafe_flip_reason": movement_debug.get("strafe_flip_reason"),
        "bullet_strafe_lock_active": bool(
            movement_debug.get("bullet_strafe_lock_active", False)
        ),
        "retreat_diagonal_allowed": bool(
            movement_debug.get("retreat_diagonal_allowed", False)
        ),
        "movement_policy_reason": movement_debug.get("movement_policy_reason"),
        "bullet_dodge_active": bool(movement_debug.get("bullet_dodge_active", False)),
        "dodge_reason": movement_debug.get("dodge_reason"),
        "bullet_safety_margin": movement_debug.get("bullet_safety_margin"),
        "dodge_lock_active": bool(movement_debug.get("dodge_lock_active", False)),
        "dodge_lock_steps_remaining": movement_debug.get(
            "dodge_lock_steps_remaining", 0
        ),
        "dodge_lock_move_bin": movement_debug.get("dodge_lock_move_bin"),
        "cooldown_strafe_fallback_used": bool(
            movement_debug.get("cooldown_strafe_fallback_used", False)
        ),
        "dodge_candidates": movement_debug.get("dodge_candidates", []),
        "selected_dodge_move": movement_debug.get("selected_dodge_move"),
        "dodge_blocked_reasons": movement_debug.get("dodge_blocked_reasons", {}),
        "enemy_opposite_component_used": bool(
            movement_debug.get("enemy_opposite_component_used", False)
        ),
        "range_hysteresis_locked": bool(local_debug.get("range_hysteresis_locked", False)),
        "combat_exit_blocked_reason": intent_debug.get("combat_exit_blocked_reason"),
    }


__all__ = ["build_debug"]
