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
        "combat_movement_profile": movement_debug.get("combat_movement_profile"),
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
        "micro_intent": movement_debug.get("micro_intent"),
        "poke_state": movement_debug.get("poke_state"),
        "poke_state_age": movement_debug.get("poke_state_age", 0),
        "poke_exit_lock_steps_remaining": movement_debug.get(
            "poke_exit_lock_steps_remaining", 0
        ),
        "poke_exit_vector": movement_debug.get("poke_exit_vector"),
        "poke_exit_move_bin": movement_debug.get("poke_exit_move_bin"),
        "poke_exit_reason": movement_debug.get("poke_exit_reason"),
        "primary_enemy_bullet_id": movement_debug.get("primary_enemy_bullet_id"),
        "primary_enemy_bullet_velocity": movement_debug.get(
            "primary_enemy_bullet_velocity"
        ),
        "dist_to_enemy": movement_debug.get("dist_to_enemy"),
        "poke_enter_ratio": movement_debug.get("poke_enter_ratio"),
        "poke_exit_ratio": movement_debug.get("poke_exit_ratio"),
        "kiting_policy_reason": movement_debug.get("kiting_policy_reason"),
        "stay_allowed": bool(movement_debug.get("stay_allowed", False)),
        "stay_blocked_reason": movement_debug.get("stay_blocked_reason"),
        "reset_dodge_override_used": bool(
            movement_debug.get("reset_dodge_override_used", False)
        ),
        "incoming_bullet_danger": bool(
            movement_debug.get("incoming_bullet_danger", False)
        ),
        "selected_escape_move": movement_debug.get("selected_escape_move"),
        "selected_escape_type": movement_debug.get("selected_escape_type"),
        "selected_escape_predicted_min_distance": movement_debug.get(
            "selected_escape_predicted_min_distance"
        ),
        "perpendicular_rejected_reason": movement_debug.get(
            "perpendicular_rejected_reason"
        ),
        "diagonal_rejected_reason": movement_debug.get("diagonal_rejected_reason"),
        "backoff_rejected_reason": movement_debug.get("backoff_rejected_reason"),
        "predicted_min_bullet_distance_for_stay": movement_debug.get(
            "predicted_min_bullet_distance_for_stay"
        ),
        "repeated_line_break_used": bool(
            movement_debug.get("repeated_line_break_used", False)
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
        "enemy_aim_noise_deg": context_debug.get("enemy_aim_noise_deg", 0.0),
        "applied_enemy_aim_noise_rad": context_debug.get(
            "applied_enemy_aim_noise_rad"
        ),
    }


__all__ = ["build_debug"]
