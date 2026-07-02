from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..types import AgentContext, AgentState, BaselineConfig, LocalPlan
from .control_fire import build_fire_status


_MOVE_VECTORS = {
    0: (0, 0),
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
    5: (-1, -1),
    6: (1, -1),
    7: (-1, 1),
    8: (1, 1),
}


def control_movement(
    ctx: AgentContext,
    state: AgentState,
    local_plan: LocalPlan,
    config: BaselineConfig,
    fire_status: Mapping[str, Any] | None = None,
) -> tuple[int, dict]:
    original_move_bin = int(local_plan.move_bin) if 0 <= int(local_plan.move_bin) <= 8 else 0
    move_bin = original_move_bin
    status = dict(
        fire_status
        if fire_status is not None
        else build_fire_status(ctx, state, local_plan, config)
    )
    fire_ready = bool(status.get("fire_ready", False))
    target_in_range = bool(status.get("target_in_range", False))
    can_fire_now = bool(status.get("can_fire_now", False))
    direction = 1 if state.strafe_direction >= 0 else -1
    strafe_age = 0
    strafe_blocked = False
    strafe_flip_reason = None
    outer_band_strafe_active = False
    perpendicular_strafe = False
    bullet_strafe_lock_active = False
    dodge_lock_steps_remaining = 0
    dodge_lock_move_bin = None
    retreat_diagonal_allowed = False
    fire_window_state = None
    range_policy_reason = "non_combat_local_plan"
    movement_policy_reason = "non_combat_local_plan"
    hold_movement_policy = None
    hold_predicted_in_range = None
    hold_stop_used = False
    incoming_bullet_stop_blocked = False
    reset_soft_backoff_active = False
    blocked_reasons: dict[str, list[str]] = {}
    selected_retreat_move: dict[str, Any] | None = None
    selected_escape_move: dict[str, Any] | None = None
    micro_intent = "GLOBAL_NAV"
    kiting_policy_reason = "non_combat_local_plan"
    stay_allowed = False
    stay_blocked_reason = "non_combat_policy"
    reset_dodge_override_used = False
    incoming_bullet_danger = False
    repeated_line_break_used = False
    selected_escape_type = None
    selected_escape_predicted_min_distance = None
    perpendicular_rejected_reason = None
    diagonal_rejected_reason = None
    backoff_rejected_reason = None
    predicted_min_bullet_distance_for_stay = None
    combat_movement_profile = _combat_movement_profile(local_plan, config)

    enemy = ctx.nearest_enemy
    combat_enemy = bool(
        local_plan.intent == "COMBAT"
        and enemy is not None
        and enemy.alive
        and ctx.enemy_dist is not None
        and ctx.weapon_range > 1e-6
    )
    dist_ratio = (
        float(ctx.enemy_dist) / float(ctx.weapon_range)
        if combat_enemy and ctx.enemy_dist is not None
        else None
    )

    if combat_enemy and dist_ratio is not None and combat_movement_profile == "poke_out":
        return _control_poke_out_movement(
            ctx,
            state,
            local_plan,
            config,
            status,
            original_move_bin,
            dist_ratio,
            combat_movement_profile,
        )

    if combat_enemy and dist_ratio is not None:
        if dist_ratio < 0.75:
            fire_window_state = "TOO_CLOSE"
            range_policy_reason = "dist_ratio_below_0.75"
        elif fire_ready and not target_in_range:
            fire_window_state = "ENTER"
            range_policy_reason = "fire_ready_target_out_of_range"
        elif fire_ready and target_in_range:
            fire_window_state = "HOLD"
            range_policy_reason = "fire_ready_target_in_range"
        else:
            fire_window_state = "RESET"
            range_policy_reason = "fire_cooldown_reset"

        stay_prediction = _predict_bullet_clearance_for_move(ctx, 0, config)
        predicted_min_bullet_distance_for_stay = stay_prediction["predicted_min_bullet_distance"]
        incoming_bullet_danger = bool(
            _incoming_bullet_records(ctx) and not stay_prediction["safe"]
        )
        if incoming_bullet_danger:
            micro_intent = "BULLET_ESCAPE"
            stay_blocked_reason = "incoming_bullet_danger"
            incoming_bullet_stop_blocked = True
            selected_escape_move, escape_blocked, escape_debug = _select_bullet_safe_escape_move(
                ctx,
                state,
                config,
                max_dist_ratio=0.98 if fire_window_state == "HOLD" else None,
            )
            blocked_reasons.update(escape_blocked)
            selected_escape_type = escape_debug["selected_escape_type"]
            selected_escape_predicted_min_distance = escape_debug[
                "selected_escape_predicted_min_distance"
            ]
            perpendicular_rejected_reason = escape_debug["perpendicular_rejected_reason"]
            diagonal_rejected_reason = escape_debug["diagonal_rejected_reason"]
            backoff_rejected_reason = escape_debug["backoff_rejected_reason"]
            move_bin = (
                int(selected_escape_move["move_bin"])
                if selected_escape_move is not None
                else 0
            )
            if selected_escape_move is not None:
                kiting_policy_reason = str(selected_escape_move["reason"])
                movement_policy_reason = f"fire_window_{fire_window_state.lower()}_bullet_escape"
                direction = _strafe_direction_for_move(ctx, move_bin, direction)
                perpendicular_strafe = _is_perpendicular_strafe(ctx, move_bin)
                outer_band_strafe_active = perpendicular_strafe
                bullet_strafe_lock_active = True
                dodge_lock_steps_remaining = max(
                    0, max(2, min(3, int(config.bullet_dodge_steps))) - 1
                )
                dodge_lock_move_bin = move_bin if dodge_lock_steps_remaining > 0 else None
                if fire_window_state == "HOLD":
                    hold_movement_policy = "incoming_bullet_safe_escape"
                    predicted_ratio = _predicted_next_dist_ratio(ctx, move_bin, config)
                    hold_predicted_in_range = bool(
                        predicted_ratio is not None and predicted_ratio <= 0.98
                    )
            else:
                kiting_policy_reason = "incoming_bullet_no_feasible_escape"
                movement_policy_reason = f"fire_window_{fire_window_state.lower()}_escape_blocked"
            reset_dodge_override_used = fire_window_state == "RESET"
        elif fire_window_state == "TOO_CLOSE":
            micro_intent = "TOO_CLOSE_RETREAT"
            kiting_policy_reason = "too_close_retreat"
            stay_blocked_reason = "too_close"
            retreat_diagonal_allowed = True
            selected_retreat_move, retreat_blocked = _select_retreat_move(
                ctx,
                direction,
                config,
            )
            blocked_reasons.update(retreat_blocked)
            move_bin = (
                int(selected_retreat_move["move_bin"])
                if selected_retreat_move is not None
                else 0
            )
            movement_policy_reason = (
                "fire_window_too_close_retreat"
                if selected_retreat_move is not None
                else "fire_window_too_close_blocked"
            )
        elif fire_window_state == "ENTER":
            micro_intent = "ENTER_RANGE"
            kiting_policy_reason = "fire_ready_approach"
            stay_blocked_reason = "target_out_of_range_fire_ready"
            move_bin, approach_blocked = _select_approach_move(ctx, config)
            if approach_blocked:
                blocked_reasons["approach"] = approach_blocked
            movement_policy_reason = (
                "fire_window_enter_approach"
                if move_bin != 0
                else "fire_window_enter_blocked"
            )
        elif fire_window_state == "HOLD":
            micro_intent = "FIRE_HOLD_KITE"
            kiting_policy_reason = "hold_safe_movement"
            hold_result = _select_hold_movement(ctx, state, config)
            move_bin = int(hold_result["move_bin"])
            direction = int(hold_result["direction"])
            strafe_age = int(hold_result["strafe_age"])
            strafe_flip_reason = hold_result["strafe_flip_reason"]
            strafe_blocked = bool(hold_result["blocked_reasons"])
            blocked_reasons.update(hold_result["blocked_reasons"])
            perpendicular_strafe = bool(hold_result["perpendicular_strafe"])
            outer_band_strafe_active = perpendicular_strafe
            bullet_strafe_lock_active = bool(hold_result["bullet_lock_active"])
            dodge_lock_steps_remaining = int(hold_result["lock_steps_remaining"])
            dodge_lock_move_bin = hold_result["lock_move_bin"]
            hold_movement_policy = hold_result["policy"]
            hold_predicted_in_range = hold_result["predicted_in_range"]
            if move_bin == 0:
                stay_allowed, stay_blocked_reason = _stay_is_allowed(
                    ctx,
                    state,
                    status,
                    config,
                )
                if not stay_allowed:
                    line_break, line_break_blocked = _select_line_break_move(
                        ctx,
                        direction,
                        config,
                        max_dist_ratio=0.98,
                    )
                    blocked_reasons.update(line_break_blocked)
                    if line_break != 0:
                        move_bin = line_break
                        repeated_line_break_used = True
                        hold_movement_policy = "repeated_hold_line_break"
                        hold_predicted_in_range = True
                        kiting_policy_reason = "repeated_hold_line_break"
            else:
                stay_allowed, stay_blocked_reason = _stay_is_allowed(
                    ctx,
                    state,
                    status,
                    config,
                )
            hold_stop_used = move_bin == 0
            movement_policy_reason = "fire_window_hold_movement"
        else:
            micro_intent = "COOLDOWN_KITE"
            stay_allowed, stay_blocked_reason = _stay_is_allowed(
                ctx,
                state,
                status,
                config,
            )
            if dist_ratio > 1.35:
                micro_intent = "COOLDOWN_RECOVER_RANGE"
                kiting_policy_reason = "cooldown_extreme_range_approach"
                move_bin, approach_blocked = _select_approach_move(ctx, config)
                if approach_blocked:
                    blocked_reasons["cooldown_extreme_approach"] = approach_blocked
                movement_policy_reason = (
                    "fire_window_reset_extreme_approach"
                    if move_bin != 0
                    else "fire_window_reset_extreme_approach_blocked"
                )
            elif target_in_range:
                retreat_diagonal_allowed = True
                move_bin, direction, kite_blocked = _select_cooldown_kite_move(
                    ctx,
                    direction,
                    config,
                    max_dist_ratio=1.05,
                )
                blocked_reasons.update(kite_blocked)
                if move_bin != 0:
                    kiting_policy_reason = "cooldown_perpendicular_or_diagonal"
                    perpendicular_strafe = _is_perpendicular_strafe(ctx, move_bin)
                    outer_band_strafe_active = perpendicular_strafe
                    movement_policy_reason = "fire_window_reset_kite"
                if move_bin == 0:
                    selected_retreat_move, retreat_blocked = _select_soft_backoff_move(
                        ctx,
                        direction,
                        config,
                        max_dist_ratio=1.05,
                    )
                    blocked_reasons.update(retreat_blocked)
                    move_bin = (
                        int(selected_retreat_move["move_bin"])
                        if selected_retreat_move is not None
                        else 0
                    )
                    reset_soft_backoff_active = move_bin != 0
                    kiting_policy_reason = "cooldown_soft_backoff_fallback"
                    movement_policy_reason = (
                        "fire_window_reset_backoff"
                        if move_bin != 0
                        else "fire_window_reset_blocked"
                    )
            else:
                if stay_allowed:
                    move_bin = 0
                    kiting_policy_reason = "cooldown_short_hold_near_range"
                    movement_policy_reason = "fire_window_reset_hold_out_of_range"
                else:
                    move_bin, direction, kite_blocked = _select_cooldown_kite_move(
                        ctx,
                        direction,
                        config,
                        max_dist_ratio=max(1.05, dist_ratio + 0.01),
                    )
                    blocked_reasons.update(kite_blocked)
                    perpendicular_strafe = move_bin != 0
                    outer_band_strafe_active = perpendicular_strafe
                    repeated_line_break_used = move_bin != 0
                    kiting_policy_reason = "cooldown_line_break"
                    movement_policy_reason = (
                        "fire_window_reset_kite_out_of_range"
                        if move_bin != 0
                        else "fire_window_reset_blocked"
                    )
    elif local_plan.intent == "COMBAT":
        range_policy_reason = "combat_without_live_enemy"
        movement_policy_reason = "combat_without_live_enemy_local_plan"

    strafe_lock_steps = max(10, min(20, int(config.strafe_lock_steps)))
    strafe_lock_steps_remaining = (
        max(0, strafe_lock_steps - strafe_age) if outer_band_strafe_active else 0
    )
    return move_bin, {
        "move_bin": move_bin,
        "reason": movement_policy_reason,
        "movement_policy_reason": movement_policy_reason,
        "range_policy_reason": range_policy_reason,
        "fire_window_state": fire_window_state,
        "fire_ready": fire_ready,
        "target_in_range": target_in_range,
        "can_fire_now": can_fire_now,
        "fire": int(can_fire_now),
        "combat_movement_profile": combat_movement_profile,
        "hold_movement_policy": hold_movement_policy,
        "hold_predicted_in_range": hold_predicted_in_range,
        "hold_stop_used": hold_stop_used,
        "incoming_bullet_stop_blocked": incoming_bullet_stop_blocked,
        "reset_soft_backoff_active": reset_soft_backoff_active,
        "micro_intent": micro_intent,
        "kiting_policy_reason": kiting_policy_reason,
        "stay_allowed": stay_allowed,
        "stay_blocked_reason": stay_blocked_reason,
        "reset_dodge_override_used": reset_dodge_override_used,
        "incoming_bullet_danger": incoming_bullet_danger,
        "selected_escape_move": selected_escape_move,
        "selected_escape_type": selected_escape_type,
        "selected_escape_predicted_min_distance": selected_escape_predicted_min_distance,
        "perpendicular_rejected_reason": perpendicular_rejected_reason,
        "diagonal_rejected_reason": diagonal_rejected_reason,
        "backoff_rejected_reason": backoff_rejected_reason,
        "predicted_min_bullet_distance_for_stay": predicted_min_bullet_distance_for_stay,
        "repeated_line_break_used": repeated_line_break_used,
        "combat_stay_steps": (
            int(state.combat_stay_steps) + 1
            if combat_enemy and move_bin == 0
            else 0
        ),
        "original_move_bin": original_move_bin,
        "dist_ratio": dist_ratio,
        "predicted_next_dist_ratio": _predicted_next_dist_ratio(ctx, move_bin, config),
        "outer_band_strafe_active": outer_band_strafe_active,
        "perpendicular_strafe": perpendicular_strafe,
        "strafe_lock_steps_remaining": strafe_lock_steps_remaining,
        "strafe_flip_reason": strafe_flip_reason,
        "bullet_strafe_lock_active": bullet_strafe_lock_active,
        "retreat_diagonal_allowed": retreat_diagonal_allowed,
        "dodge_lock_active": bullet_strafe_lock_active,
        "dodge_lock_steps_remaining": dodge_lock_steps_remaining,
        "dodge_lock_move_bin": dodge_lock_move_bin,
        "bullet_dodge_active": bullet_strafe_lock_active,
        "dodge_reason": (
            str(selected_escape_move["reason"])
            if selected_escape_move is not None
            else "bullet_perpendicular_strafe_lock"
            if bullet_strafe_lock_active
            else None
        ),
        "bullet_safety_margin": max(0.0, float(config.bullet_safety_margin)),
        "cooldown_strafe_fallback_used": False,
        "dodge_candidates": [],
        "selected_dodge_move": selected_escape_move or selected_retreat_move,
        "dodge_blocked_reasons": blocked_reasons,
        "enemy_opposite_component_used": bool(
            (selected_escape_move or selected_retreat_move or {}).get(
                "enemy_opposite_component", False
            )
        ),
        "strafe_direction": "right" if direction > 0 else "left",
        "strafe_direction_sign": direction,
        "strafe_age": strafe_age,
        "strafe_blocked": strafe_blocked,
    }


def _combat_movement_profile(local_plan: LocalPlan, config: BaselineConfig) -> str:
    configured = str(config.combat_movement_profile or "default").strip().lower()
    if configured == "poke_out" or str(local_plan.combat_profile or "").strip().lower() == "poke_out":
        return "poke_out"
    return configured or "default"


def _control_poke_out_movement(
    ctx: AgentContext,
    state: AgentState,
    local_plan: LocalPlan,
    config: BaselineConfig,
    fire_status: Mapping[str, Any],
    original_move_bin: int,
    dist_ratio: float,
    combat_movement_profile: str,
) -> tuple[int, dict]:
    del local_plan
    fire_ready = bool(fire_status.get("fire_ready", False))
    target_in_range = bool(fire_status.get("target_in_range", False))
    can_fire_now = bool(fire_status.get("can_fire_now", False))
    poke_edge_fire = bool(fire_status.get("poke_edge_fire", False))
    fire = int(can_fire_now)
    dist_to_enemy = float(ctx.enemy_dist) if ctx.enemy_dist is not None else None
    enter_ratio = max(0.0, float(config.poke_enter_ratio))
    exit_ratio = max(enter_ratio, float(config.poke_exit_ratio))
    exit_margin = max(0.0, float(config.poke_exit_margin))
    configured_lock_steps = max(0, int(config.poke_exit_lock_steps))
    previous_poke_state = state.poke_state or "POKE_ENTER_RANGE"
    previous_exit_remaining = max(0, int(state.poke_exit_lock_steps_remaining))
    primary_bullet = get_primary_enemy_bullet(ctx)
    enemy_bullet_spawned = _enemy_bullet_spawned(ctx)
    bullet_for_exit = primary_bullet if bool(config.poke_exit_uses_bullet_velocity) else None
    exit_complete = bool(
        dist_to_enemy is not None
        and (
            dist_ratio >= exit_ratio
            or dist_to_enemy >= float(ctx.weapon_range) + exit_margin
        )
    )
    exit_lock_active = bool(
        previous_poke_state == "POKE_EXIT_BULLET_DIR"
        and not exit_complete
    )
    exit_lock_expired = bool(
        previous_poke_state == "POKE_EXIT_BULLET_DIR"
        and previous_exit_remaining <= 0
        and exit_complete
    )

    blocked_reasons: dict[str, list[str]] = {}
    selected_exit_move: dict[str, Any] | None = None
    poke_exit_vector: tuple[float, float] | None = None
    poke_exit_reason: str | None = None
    primary_bullet_velocity = (
        list(primary_bullet["velocity"]) if primary_bullet is not None else None
    )
    primary_bullet_id = (
        str(primary_bullet.get("bullet_id", "")) if primary_bullet is not None else None
    )

    def exit_move(
        reason: str,
    ) -> tuple[int, dict[str, Any] | None, tuple[float, float], str, dict[str, list[str]]]:
        vector, vector_reason = get_poke_exit_vector(ctx, bullet_for_exit)
        selected_move, move_blocked = move_bin_from_vector(ctx, vector, config)
        selected_move["name"] = "poke_exit"
        selected_move["reason"] = reason
        selected_move["exit_reason"] = vector_reason
        selected_move["enemy_opposite_component"] = _enemy_opposite_component(
            ctx, int(selected_move["move_bin"])
        )
        return (
            int(selected_move["move_bin"]),
            selected_move,
            vector,
            vector_reason,
            {"poke_exit": move_blocked} if move_blocked else {},
        )

    def edge_fire_move() -> tuple[int, dict[str, Any] | None, tuple[float, float], str, dict[str, list[str]]]:
        selected_move, move_blocked = _select_poke_edge_fire_move(ctx, config)
        if selected_move is None:
            return exit_move("poke_edge_fire_fallback_exit")
        vector, vector_reason = get_poke_exit_vector(ctx, bullet_for_exit)
        selected_move["name"] = "poke_edge_fire"
        selected_move["reason"] = "poke_edge_fire_start_exit"
        selected_move["exit_reason"] = "edge_hold_outside_range"
        selected_move["enemy_opposite_component"] = _enemy_opposite_component(
            ctx, int(selected_move["move_bin"])
        )
        return (
            int(selected_move["move_bin"]),
            selected_move,
            vector,
            vector_reason,
            {"poke_edge_fire": move_blocked} if move_blocked else {},
        )

    if can_fire_now and (target_in_range or poke_edge_fire):
        if poke_edge_fire and not target_in_range:
            move_bin, selected_exit_move, poke_exit_vector, poke_exit_reason, blocked_reasons = edge_fire_move()
            movement_policy_reason = "poke_edge_fire_start_exit"
            range_policy_reason = "poke_edge_fire_outside_range"
        else:
            move_bin, selected_exit_move, poke_exit_vector, poke_exit_reason, blocked_reasons = exit_move(
                "poke_fire_in_range_start_exit"
            )
            movement_policy_reason = "poke_fire_in_range_start_exit"
            range_policy_reason = "poke_fire_in_range"
        micro_intent = "POKE_FIRE"
        poke_state = "POKE_EXIT_BULLET_DIR"
        poke_exit_lock_steps_remaining = configured_lock_steps
    elif primary_bullet is not None or enemy_bullet_spawned:
        move_bin, selected_exit_move, poke_exit_vector, poke_exit_reason, blocked_reasons = exit_move(
            "poke_exit_along_bullet_dir"
        )
        micro_intent = "POKE_EXIT_BULLET_DIR"
        poke_state = "POKE_EXIT_BULLET_DIR"
        movement_policy_reason = (
            "poke_exit_along_bullet_dir"
            if poke_exit_reason == "enemy_bullet_velocity"
            else "poke_exit_away_from_enemy"
        )
        range_policy_reason = (
            "poke_enemy_bullet_active"
            if primary_bullet is not None
            else "poke_enemy_bullet_spawned"
        )
        poke_exit_lock_steps_remaining = max(
            previous_exit_remaining - 1,
            configured_lock_steps - 1,
            0,
        )
    elif exit_lock_active:
        move_bin, selected_exit_move, poke_exit_vector, poke_exit_reason, blocked_reasons = exit_move(
            "poke_exit_lock"
        )
        micro_intent = "POKE_EXIT_BULLET_DIR"
        poke_state = "POKE_EXIT_BULLET_DIR"
        movement_policy_reason = (
            "poke_exit_along_bullet_dir"
            if poke_exit_reason == "enemy_bullet_velocity"
            else "poke_exit_away_from_enemy"
        )
        range_policy_reason = "poke_exit_lock_active"
        poke_exit_lock_steps_remaining = max(0, previous_exit_remaining - 1)
    elif target_in_range and not fire_ready and not exit_lock_expired:
        move_bin, selected_exit_move, poke_exit_vector, poke_exit_reason, blocked_reasons = exit_move(
            "poke_exit_cooldown_inside_range"
        )
        micro_intent = "POKE_EXIT_BULLET_DIR"
        poke_state = "POKE_EXIT_BULLET_DIR"
        movement_policy_reason = (
            "poke_exit_along_bullet_dir"
            if poke_exit_reason == "enemy_bullet_velocity"
            else "poke_exit_away_from_enemy"
        )
        range_policy_reason = "poke_cooldown_inside_range"
        poke_exit_lock_steps_remaining = max(configured_lock_steps - 1, 0)
    else:
        move_bin, approach_blocked = _select_approach_move(ctx, config)
        if approach_blocked:
            blocked_reasons["poke_enter_range"] = approach_blocked
        approach_predicted_ratio = _predicted_next_dist_ratio(ctx, move_bin, config)
        if (
            not can_fire_now
            and approach_predicted_ratio is not None
            and approach_predicted_ratio < 1.0
        ):
            wait_move, wait_blocked = _select_poke_edge_fire_move(ctx, config)
            if wait_move is not None:
                move_bin = int(wait_move["move_bin"])
                selected_exit_move = {
                    **wait_move,
                    "name": "poke_edge_wait",
                    "reason": "poke_wait_cooldown_outside_range",
                    "exit_reason": "edge_wait_outside_range",
                    "enemy_opposite_component": _enemy_opposite_component(
                        ctx, int(wait_move["move_bin"])
                    ),
                }
                blocked_reasons["poke_edge_wait"] = wait_blocked
        micro_intent = "POKE_ENTER_RANGE"
        poke_state = "POKE_ENTER_RANGE"
        movement_policy_reason = (
            "poke_enter_range" if move_bin != 0 else "poke_enter_range_blocked"
        )
        if selected_exit_move is not None and selected_exit_move.get("name") == "poke_edge_wait":
            movement_policy_reason = "poke_wait_cooldown_outside_range"
        range_policy_reason = (
            "poke_exit_complete_return_enter"
            if exit_complete or exit_lock_expired
            else "poke_outside_range_enter"
        )
        if movement_policy_reason == "poke_wait_cooldown_outside_range":
            range_policy_reason = "poke_wait_cooldown_outside_range"
        poke_exit_lock_steps_remaining = 0

    poke_state_age = (
        max(0, int(state.poke_state_age)) + 1
        if poke_state == previous_poke_state
        else 1
    )
    predicted_next_dist_ratio = _predicted_next_dist_ratio(ctx, move_bin, config)
    return move_bin, {
        "move_bin": move_bin,
        "reason": movement_policy_reason,
        "movement_policy_reason": movement_policy_reason,
        "range_policy_reason": range_policy_reason,
        "fire_window_state": poke_state,
        "fire_ready": fire_ready,
        "target_in_range": target_in_range,
        "can_fire_now": can_fire_now,
        "poke_edge_fire": poke_edge_fire,
        "fire": fire,
        "combat_movement_profile": combat_movement_profile,
        "micro_intent": micro_intent,
        "kiting_policy_reason": movement_policy_reason,
        "poke_state": poke_state,
        "poke_state_age": poke_state_age,
        "poke_exit_lock_steps_remaining": poke_exit_lock_steps_remaining,
        "poke_exit_vector": list(poke_exit_vector) if poke_exit_vector is not None else None,
        "poke_exit_move_bin": (
            int(selected_exit_move["move_bin"]) if selected_exit_move is not None else None
        ),
        "poke_exit_reason": poke_exit_reason,
        "primary_enemy_bullet_id": primary_bullet_id,
        "primary_enemy_bullet_velocity": primary_bullet_velocity,
        "dist_to_enemy": dist_to_enemy,
        "dist_ratio": dist_ratio,
        "poke_enter_ratio": enter_ratio,
        "poke_exit_ratio": exit_ratio,
        "poke_exit_margin": exit_margin,
        "poke_fire_while_exiting": bool(config.poke_fire_while_exiting),
        "poke_exit_uses_bullet_velocity": bool(config.poke_exit_uses_bullet_velocity),
        "hold_movement_policy": None,
        "hold_predicted_in_range": None,
        "hold_stop_used": False,
        "incoming_bullet_stop_blocked": bool(primary_bullet is not None),
        "reset_soft_backoff_active": False,
        "stay_allowed": False,
        "stay_blocked_reason": "poke_out_exit_or_enter" if move_bin != 0 else "poke_out_blocked",
        "reset_dodge_override_used": False,
        "incoming_bullet_danger": bool(primary_bullet is not None),
        "selected_escape_move": selected_exit_move,
        "selected_escape_type": "poke_bullet_direction" if selected_exit_move is not None else None,
        "selected_escape_predicted_min_distance": None,
        "perpendicular_rejected_reason": None,
        "diagonal_rejected_reason": None,
        "backoff_rejected_reason": None,
        "predicted_min_bullet_distance_for_stay": None,
        "repeated_line_break_used": False,
        "combat_stay_steps": int(state.combat_stay_steps) + 1 if move_bin == 0 else 0,
        "original_move_bin": original_move_bin,
        "predicted_next_dist_ratio": predicted_next_dist_ratio,
        "outer_band_strafe_active": False,
        "perpendicular_strafe": False,
        "strafe_lock_steps_remaining": 0,
        "strafe_flip_reason": None,
        "bullet_strafe_lock_active": False,
        "retreat_diagonal_allowed": False,
        "dodge_lock_active": False,
        "dodge_lock_steps_remaining": 0,
        "dodge_lock_move_bin": None,
        "bullet_dodge_active": bool(micro_intent == "POKE_EXIT_BULLET_DIR"),
        "dodge_reason": movement_policy_reason if micro_intent == "POKE_EXIT_BULLET_DIR" else None,
        "bullet_safety_margin": max(0.0, float(config.bullet_safety_margin)),
        "cooldown_strafe_fallback_used": False,
        "dodge_candidates": [],
        "selected_dodge_move": selected_exit_move,
        "dodge_blocked_reasons": blocked_reasons,
        "enemy_opposite_component_used": bool(
            (selected_exit_move or {}).get("enemy_opposite_component", False)
        ),
        "strafe_direction": "right" if state.strafe_direction >= 0 else "left",
        "strafe_direction_sign": 1 if state.strafe_direction >= 0 else -1,
        "strafe_age": 0,
        "strafe_blocked": False,
    }


def get_primary_enemy_bullet(ctx: AgentContext) -> dict[str, Any] | None:
    bullets = _incoming_bullet_records(ctx)
    return bullets[0] if bullets else None


def _enemy_bullet_spawned(ctx: AgentContext) -> bool:
    return any(
        event.get("type") == "bullet_spawned"
        and event.get("owner_id") == "enemy"
        for event in ctx.events
    )


def get_poke_exit_vector(
    ctx: AgentContext,
    bullet: Mapping[str, Any] | None,
) -> tuple[tuple[float, float], str]:
    if bullet is not None:
        velocity = bullet.get("velocity")
        if velocity is not None:
            vx, vy = float(velocity[0]), float(velocity[1])
            speed = math.hypot(vx, vy)
            if speed > 1e-6:
                return (vx / speed, vy / speed), "enemy_bullet_velocity"
    if ctx.nearest_enemy is None:
        return (0.0, 0.0), "missing_enemy"
    away = (
        ctx.player_pos[0] - ctx.nearest_enemy.position[0],
        ctx.player_pos[1] - ctx.nearest_enemy.position[1],
    )
    distance = math.hypot(*away)
    if distance <= 1e-6:
        return (0.0, 0.0), "zero_away_from_enemy"
    return (away[0] / distance, away[1] / distance), "away_from_enemy"


def move_bin_from_vector(
    ctx: AgentContext,
    desired: tuple[float, float],
    config: BaselineConfig,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    desired_length = math.hypot(*desired)
    if desired_length <= 1e-6:
        blocked: dict[str, list[str]] = {}
        for move_bin in range(1, 9):
            reasons = _candidate_blocked_reasons(ctx, move_bin, config)
            if not reasons:
                move_vector = _normalized_move_vector(move_bin)
                return (
                    {
                        "move_bin": move_bin,
                        "vector": list(move_vector),
                        "desired_vector": [0.0, 0.0],
                        "dot": 0.0,
                        "feasible": True,
                    },
                    blocked,
                )
            blocked[f"move_bin_{move_bin}"] = reasons
        return (
            {
                "move_bin": 0,
                "vector": [0.0, 0.0],
                "desired_vector": [0.0, 0.0],
                "dot": 0.0,
                "feasible": False,
            },
            blocked,
        )

    desired_unit = desired[0] / desired_length, desired[1] / desired_length
    exact_move = _vector_to_move_bin(*desired_unit)
    records: list[dict[str, Any]] = []
    for move_bin in range(1, 9):
        move_vector = _normalized_move_vector(move_bin)
        dot = move_vector[0] * desired_unit[0] + move_vector[1] * desired_unit[1]
        records.append(
            {
                "move_bin": move_bin,
                "vector": list(move_vector),
                "desired_vector": list(desired_unit),
                "dot": dot,
                "feasible": False,
                "blocked_reasons": _candidate_blocked_reasons(ctx, move_bin, config),
            }
        )

    blocked: dict[str, list[str]] = {}
    exact_records = [record for record in records if record["move_bin"] == exact_move]
    positive_records = sorted(
        [
            record
            for record in records
            if record["move_bin"] != exact_move and float(record["dot"]) > 1e-6
        ],
        key=lambda record: (-float(record["dot"]), int(record["move_bin"])),
    )
    fallback_records = sorted(
        [
            record
            for record in records
            if record["move_bin"] != exact_move and float(record["dot"]) <= 1e-6
        ],
        key=lambda record: (-float(record["dot"]), int(record["move_bin"])),
    )

    for record in exact_records + positive_records + fallback_records:
        reasons = list(record["blocked_reasons"])
        if not reasons:
            return {**record, "feasible": True}, blocked
        blocked[f"move_bin_{record['move_bin']}"] = reasons

    return (
        {
            "move_bin": 0,
            "vector": [0.0, 0.0],
            "desired_vector": list(desired_unit),
            "dot": 0.0,
            "feasible": False,
        },
        blocked,
    )


def _enemy_opposite_component(ctx: AgentContext, move_bin: int) -> bool:
    move_vector = _normalized_move_vector(move_bin)
    enemy_direction = _enemy_direction(ctx)
    return move_vector[0] * enemy_direction[0] + move_vector[1] * enemy_direction[1] < 0.0


def _incoming_bullet_is_dangerous(
    ctx: AgentContext,
    config: BaselineConfig,
) -> bool:
    bullets = _incoming_bullet_records(ctx)
    return bool(bullets and not _predict_bullet_clearance_for_move(ctx, 0, config)["safe"])


def _select_bullet_safe_escape_move(
    ctx: AgentContext,
    state: AgentState,
    config: BaselineConfig,
    *,
    max_dist_ratio: float | None,
) -> tuple[dict[str, Any] | None, dict[str, list[str]], dict[str, Any]]:
    bullets = _incoming_bullet_records(ctx)
    velocity = bullets[0]["velocity"] if bullets else None
    if velocity is None or math.hypot(*velocity) <= 1e-6:
        return (
            None,
            {"bullet_escape": ["missing_bullet_velocity"]},
            {
                "selected_escape_type": None,
                "selected_escape_predicted_min_distance": None,
                "perpendicular_rejected_reason": "missing_bullet_velocity",
                "diagonal_rejected_reason": "not_evaluated",
                "backoff_rejected_reason": "not_evaluated",
            },
        )

    speed = math.hypot(*velocity)
    bullet_direction = velocity[0] / speed, velocity[1] / speed
    normal = -bullet_direction[1], bullet_direction[0]
    bullet_position = bullets[0]["position"]
    side = (
        (ctx.player_pos[0] - bullet_position[0]) * normal[0]
        + (ctx.player_pos[1] - bullet_position[1]) * normal[1]
    )
    if abs(side) <= 1e-6:
        preferred_sign = 1 if state.strafe_direction >= 0 else -1
    else:
        preferred_sign = 1 if side > 0.0 else -1
    preferred_normal = normal[0] * preferred_sign, normal[1] * preferred_sign
    opposite_normal = -preferred_normal[0], -preferred_normal[1]
    enemy_direction = _enemy_direction(ctx)
    retreat = -enemy_direction[0], -enemy_direction[1]

    locked: list[tuple[str, tuple[float, float]]] = []
    if state.dodge_lock_steps_remaining > 0 and state.dodge_lock_move_bin is not None:
        locked.append(
            (
                "incoming_bullet_locked_escape",
                _normalized_move_vector(int(state.dodge_lock_move_bin)),
            )
        )
    groups = (
        (
            "perpendicular",
            locked
            + [
                ("incoming_bullet_perpendicular", preferred_normal),
                ("incoming_bullet_perpendicular_opposite", opposite_normal),
            ],
            "perpendicular_safe",
        ),
        (
            "diagonal_away",
            [
                (
                    "incoming_bullet_diagonal_away",
                    (preferred_normal[0] + retreat[0], preferred_normal[1] + retreat[1]),
                ),
                (
                    "incoming_bullet_diagonal_away_opposite",
                    (opposite_normal[0] + retreat[0], opposite_normal[1] + retreat[1]),
                ),
            ],
            "perpendicular_rejected_diagonal_safe",
        ),
        (
            "soft_backoff",
            [
                ("incoming_bullet_soft_backoff", retreat),
                (
                    "incoming_bullet_soft_backoff_diagonal",
                    (retreat[0] + preferred_normal[0], retreat[1] + preferred_normal[1]),
                ),
                (
                    "incoming_bullet_soft_backoff_diagonal_opposite",
                    (retreat[0] + opposite_normal[0], retreat[1] + opposite_normal[1]),
                ),
            ],
            "fallback_soft_backoff",
        ),
    )

    blocked: dict[str, list[str]] = {}
    evaluated: list[dict[str, Any]] = []
    rejected: dict[str, str | None] = {
        "perpendicular": None,
        "diagonal_away": None,
        "soft_backoff": None,
    }
    seen: set[int] = set()
    for escape_type, definitions, policy_reason in groups:
        group_candidates: list[dict[str, Any]] = []
        for name, vector in definitions:
            move_bin = _vector_to_move_bin(*vector)
            if move_bin == 0 or move_bin in seen:
                continue
            seen.add(move_bin)
            candidate, reasons = _evaluate_bullet_escape_candidate(
                ctx,
                move_bin,
                name,
                escape_type,
                enemy_direction,
                config,
                max_dist_ratio=max_dist_ratio,
            )
            if reasons:
                blocked[name] = reasons
                continue
            group_candidates.append(candidate)
            evaluated.append(candidate)
            if candidate["bullet_safe"]:
                candidate["reason"] = policy_reason
                return candidate, blocked, _escape_decision_debug(candidate, rejected)
        rejected[escape_type] = _escape_group_rejected_reason(group_candidates)

    for move_bin in range(1, 9):
        if move_bin in seen:
            continue
        candidate, reasons = _evaluate_bullet_escape_candidate(
            ctx,
            move_bin,
            f"least_bad_{move_bin}",
            "least_bad",
            enemy_direction,
            config,
            max_dist_ratio=max_dist_ratio,
        )
        if reasons:
            blocked[f"least_bad_{move_bin}"] = reasons
        else:
            evaluated.append(candidate)

    if evaluated:
        candidate = max(
            evaluated,
            key=lambda value: float(value["predicted_min_bullet_distance"]),
        )
        candidate["escape_type"] = "least_bad"
        candidate["reason"] = "least_bad_escape"
        return candidate, blocked, _escape_decision_debug(candidate, rejected)
    return None, blocked, _escape_decision_debug(None, rejected)


def _evaluate_bullet_escape_candidate(
    ctx: AgentContext,
    move_bin: int,
    name: str,
    escape_type: str,
    enemy_direction: tuple[float, float],
    config: BaselineConfig,
    *,
    max_dist_ratio: float | None,
) -> tuple[dict[str, Any], list[str]]:
    reasons = _candidate_blocked_reasons(ctx, move_bin, config)
    predicted_ratio = _predicted_next_dist_ratio(ctx, move_bin, config)
    if (
        max_dist_ratio is not None
        and (predicted_ratio is None or predicted_ratio > max_dist_ratio)
    ):
        reasons.append("predicted_outside_fire_range_margin")
    prediction = _predict_bullet_clearance_for_move(ctx, move_bin, config)
    if prediction["predicted_min_bullet_distance"] is None:
        reasons.append("missing_predicted_bullet_distance")
    move_vector = _normalized_move_vector(move_bin)
    candidate = {
        "name": name,
        "reason": name,
        "escape_type": escape_type,
        "move_bin": move_bin,
        "vector": list(move_vector),
        "predicted_min_bullet_distance": prediction["predicted_min_bullet_distance"],
        "dangerous_bullet_id": prediction["dangerous_bullet_id"],
        "bullet_safe": prediction["safe"],
        "bullet_safety_threshold": prediction["safety_threshold"],
        "enemy_opposite_component": (
            move_vector[0] * enemy_direction[0]
            + move_vector[1] * enemy_direction[1]
            < 0.0
        ),
    }
    return candidate, reasons


def _escape_group_rejected_reason(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "all_candidates_blocked"
    best = max(
        float(candidate["predicted_min_bullet_distance"])
        for candidate in candidates
    )
    threshold = max(float(candidate["bullet_safety_threshold"]) for candidate in candidates)
    return f"predicted_clearance_below_margin:{best:.3f}<{threshold:.3f}"


def _escape_decision_debug(
    candidate: dict[str, Any] | None,
    rejected: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "selected_escape_type": candidate.get("escape_type") if candidate else None,
        "selected_escape_predicted_min_distance": (
            candidate.get("predicted_min_bullet_distance") if candidate else None
        ),
        "perpendicular_rejected_reason": rejected["perpendicular"],
        "diagonal_rejected_reason": rejected["diagonal_away"],
        "backoff_rejected_reason": rejected["soft_backoff"],
    }


def _stay_is_allowed(
    ctx: AgentContext,
    state: AgentState,
    fire_status: Mapping[str, Any],
    config: BaselineConfig,
) -> tuple[bool, str | None]:
    if _incoming_bullet_is_dangerous(ctx, config):
        return False, "incoming_bullet_danger"
    stay_prediction = _predict_bullet_clearance_for_move(ctx, 0, config)
    if _incoming_bullet_records(ctx) and not stay_prediction["safe"]:
        return False, "predicted_bullet_distance_below_margin"
    if state.combat_stay_steps >= 1:
        return False, "repeated_stay_limit"
    useful_firing_line = bool(
        fire_status.get("fire_ready", False)
        or (
            fire_status.get("target_in_range", False)
            and fire_status.get("los_ok", False)
            and fire_status.get("aim_ok", False)
        )
    )
    if not useful_firing_line:
        return False, "no_useful_firing_line"
    return True, None


def _select_line_break_move(
    ctx: AgentContext,
    direction: int,
    config: BaselineConfig,
    *,
    max_dist_ratio: float,
) -> tuple[int, dict[str, list[str]]]:
    enemy_direction = _enemy_direction(ctx)
    definitions = (
        ("line_break_tangent", _enemy_tangent(ctx, direction)),
        ("line_break_tangent_opposite", _enemy_tangent(ctx, -direction)),
        (
            "line_break_diagonal_in",
            (
                _enemy_tangent(ctx, direction)[0] + enemy_direction[0],
                _enemy_tangent(ctx, direction)[1] + enemy_direction[1],
            ),
        ),
        (
            "line_break_diagonal_in_opposite",
            (
                _enemy_tangent(ctx, -direction)[0] + enemy_direction[0],
                _enemy_tangent(ctx, -direction)[1] + enemy_direction[1],
            ),
        ),
    )
    return _first_range_safe_move(ctx, definitions, config, max_dist_ratio=max_dist_ratio)


def _select_cooldown_kite_move(
    ctx: AgentContext,
    direction: int,
    config: BaselineConfig,
    *,
    max_dist_ratio: float,
) -> tuple[int, int, dict[str, list[str]]]:
    enemy_direction = _enemy_direction(ctx)
    retreat = -enemy_direction[0], -enemy_direction[1]
    definitions = (
        ("cooldown_tangent", _enemy_tangent(ctx, direction)),
        (
            "cooldown_diagonal_away",
            (
                _enemy_tangent(ctx, direction)[0] + retreat[0],
                _enemy_tangent(ctx, direction)[1] + retreat[1],
            ),
        ),
        ("cooldown_tangent_opposite", _enemy_tangent(ctx, -direction)),
        (
            "cooldown_diagonal_away_opposite",
            (
                _enemy_tangent(ctx, -direction)[0] + retreat[0],
                _enemy_tangent(ctx, -direction)[1] + retreat[1],
            ),
        ),
    )
    move_bin, blocked = _first_range_safe_move(
        ctx,
        definitions,
        config,
        max_dist_ratio=max_dist_ratio,
    )
    selected_direction = _strafe_direction_for_move(ctx, move_bin, direction)
    return move_bin, selected_direction, blocked


def _first_range_safe_move(
    ctx: AgentContext,
    definitions: tuple[tuple[str, tuple[float, float]], ...],
    config: BaselineConfig,
    *,
    max_dist_ratio: float,
) -> tuple[int, dict[str, list[str]]]:
    blocked: dict[str, list[str]] = {}
    seen: set[int] = set()
    for name, vector in definitions:
        move_bin = _vector_to_move_bin(*vector)
        if move_bin == 0 or move_bin in seen:
            continue
        seen.add(move_bin)
        reasons = _candidate_blocked_reasons(ctx, move_bin, config)
        predicted_ratio = _predicted_next_dist_ratio(ctx, move_bin, config)
        if predicted_ratio is None or predicted_ratio > max_dist_ratio:
            reasons.append("predicted_outside_kiting_band")
        if not reasons:
            return move_bin, blocked
        blocked[name] = reasons
    return 0, blocked


def _predict_bullet_clearance_for_move(
    ctx: AgentContext,
    move_bin: int,
    config: BaselineConfig,
) -> dict[str, Any]:
    bullets = _incoming_bullet_records(ctx)
    if not bullets:
        return {
            "safe": True,
            "predicted_min_bullet_distance": None,
            "dangerous_bullet_id": None,
            "safety_threshold": (
                float(ctx.player_radius)
                + float(config.bullet_radius)
                + float(config.bullet_safety_margin)
            ),
        }

    move_x, move_y = _normalized_move_vector(move_bin)
    horizon = max(2, min(4, int(config.bullet_prediction_horizon_steps)))
    dt = max(1e-6, float(ctx.env_dt))
    min_distance = math.inf
    worst_margin = math.inf
    dangerous_bullet_id = None
    dangerous_threshold = 0.0
    safe = True
    for bullet in bullets:
        bullet_radius = float(bullet.get("radius", config.bullet_radius))
        threshold = (
            max(0.0, float(ctx.player_radius))
            + max(0.0, bullet_radius)
            + max(0.0, float(config.bullet_safety_margin))
        )
        position = bullet["position"]
        velocity = bullet["velocity"]
        bullet_min_distance = math.inf
        for step in range(1, horizon + 1):
            predicted_player = (
                ctx.player_pos[0] + move_x * config.move_step_distance * step,
                ctx.player_pos[1] + move_y * config.move_step_distance * step,
            )
            bullet_start = (
                position[0] + velocity[0] * dt * (step - 1),
                position[1] + velocity[1] * dt * (step - 1),
            )
            bullet_end = (
                position[0] + velocity[0] * dt * step,
                position[1] + velocity[1] * dt * step,
            )
            bullet_min_distance = min(
                bullet_min_distance,
                _point_segment_distance(predicted_player, bullet_start, bullet_end),
            )
        min_distance = min(min_distance, bullet_min_distance)
        margin = bullet_min_distance - threshold
        if margin < worst_margin:
            worst_margin = margin
            dangerous_bullet_id = bullet.get("bullet_id") or None
            dangerous_threshold = threshold
        safe = safe and bullet_min_distance >= threshold
    return {
        "safe": safe,
        "predicted_min_bullet_distance": min_distance,
        "dangerous_bullet_id": dangerous_bullet_id,
        "safety_threshold": dangerous_threshold,
    }


def _incoming_bullet_records(ctx: AgentContext) -> tuple[dict[str, Any], ...]:
    if ctx.incoming_bullets:
        return ctx.incoming_bullets
    if ctx.incoming_bullet_position is None or ctx.incoming_bullet_velocity is None:
        return ()
    return (
        {
            "bullet_id": "",
            "position": ctx.incoming_bullet_position,
            "velocity": ctx.incoming_bullet_velocity,
            "radius": (
                ctx.incoming_bullet_radius
                if ctx.incoming_bullet_radius is not None
                else 12.0
            ),
        },
    )


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.dist(point, start)
    t = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / length_sq,
        ),
    )
    closest = start[0] + t * dx, start[1] + t * dy
    return math.dist(point, closest)


def _select_hold_movement(
    ctx: AgentContext,
    state: AgentState,
    config: BaselineConfig,
) -> dict[str, Any]:
    direction = 1 if state.strafe_direction >= 0 else -1
    interval = max(10, min(20, int(config.strafe_lock_steps)))
    strafe_age = max(0, int(state.strafe_age)) + 1
    flip_reason = None
    if state.strafe_age >= interval:
        direction *= -1
        strafe_age = 1
        flip_reason = "interval_expired"

    current_lock_steps = max(0, int(state.dodge_lock_steps_remaining))
    current_lock_move = state.dodge_lock_move_bin
    candidates: list[tuple[str, int, int]] = []
    if (
        current_lock_steps > 0
        and current_lock_move is not None
        and _is_perpendicular_strafe(ctx, int(current_lock_move))
    ):
        candidates.append(("incoming_bullet_locked_strafe", int(current_lock_move), direction))
    candidates.extend(
        (
            (
                "incoming_bullet_perpendicular"
                if ctx.incoming_bullet
                else "safe_perpendicular_strafe",
                _vector_to_move_bin(*_enemy_tangent(ctx, candidate_direction)),
                candidate_direction,
            )
            for candidate_direction in (direction, -direction)
        )
    )

    blocked: dict[str, list[str]] = {}
    selected_policy = None
    selected_move = 0
    selected_direction = direction
    selected_ratio = _predicted_next_dist_ratio(ctx, 0, config)
    preferred_flip_reason = "hold_preferred_strafe_blocked"
    seen: set[int] = set()
    for policy, candidate_move, candidate_direction in candidates:
        if candidate_move in seen:
            continue
        seen.add(candidate_move)
        reasons = _candidate_blocked_reasons(ctx, candidate_move, config)
        reasons += _near_boundary_reasons(ctx, candidate_move, config)
        predicted_ratio = _predicted_next_dist_ratio(ctx, candidate_move, config)
        if predicted_ratio is None or predicted_ratio > 0.98:
            reasons.append("predicted_outside_hold_margin")
        if reasons:
            blocked[policy] = reasons
            if candidate_direction == direction:
                preferred_flip_reason = (
                    "near_boundary"
                    if any(reason.startswith("near_boundary") for reason in reasons)
                    else "map_or_obstacle_blocked"
                )
            continue
        selected_policy = policy
        selected_move = candidate_move
        selected_direction = candidate_direction
        selected_ratio = predicted_ratio
        if candidate_direction != direction:
            flip_reason = preferred_flip_reason
            strafe_age = 1
        break

    if selected_move == 0 and ctx.incoming_bullet:
        soft_backoff, soft_blocked = _select_soft_backoff_move(
            ctx,
            direction,
            config,
            max_dist_ratio=0.98,
        )
        blocked.update(soft_blocked)
        if soft_backoff is not None:
            selected_policy = "incoming_bullet_soft_backoff"
            selected_move = int(soft_backoff["move_bin"])
            selected_ratio = _predicted_next_dist_ratio(ctx, selected_move, config)
        else:
            emergency_move, emergency_blocked = _select_approach_move(ctx, config)
            emergency_ratio = _predicted_next_dist_ratio(ctx, emergency_move, config)
            if (
                emergency_move != 0
                and emergency_ratio is not None
                and emergency_ratio <= 0.98
            ):
                selected_policy = "incoming_bullet_emergency_in_range_move"
                selected_move = emergency_move
                selected_ratio = emergency_ratio
            elif emergency_blocked:
                blocked["incoming_bullet_emergency"] = emergency_blocked

    if selected_move == 0:
        selected_policy = "hold_no_safe_in_range_move"

    selected_from_lock = selected_policy == "incoming_bullet_locked_strafe"
    bullet_lock_active = bool(selected_move != 0 and (ctx.incoming_bullet or selected_from_lock))
    if selected_from_lock:
        lock_steps_remaining = current_lock_steps - 1
    elif ctx.incoming_bullet and selected_move != 0:
        lock_steps_remaining = max(2, min(3, int(config.bullet_dodge_steps))) - 1
    else:
        lock_steps_remaining = 0

    return {
        "move_bin": selected_move,
        "policy": selected_policy,
        "predicted_in_range": bool(
            selected_ratio is not None and selected_ratio <= 0.98
        ),
        "direction": selected_direction,
        "strafe_age": strafe_age if selected_move != 0 else max(0, int(state.strafe_age)),
        "strafe_flip_reason": flip_reason,
        "perpendicular_strafe": _is_perpendicular_strafe(ctx, selected_move),
        "bullet_lock_active": bullet_lock_active,
        "lock_steps_remaining": max(0, lock_steps_remaining),
        "lock_move_bin": selected_move if lock_steps_remaining > 0 else None,
        "blocked_reasons": blocked,
    }


def _select_incoming_emergency_move(
    ctx: AgentContext,
    direction: int,
    config: BaselineConfig,
) -> tuple[int, dict[str, list[str]]]:
    tangent_moves = [
        _vector_to_move_bin(*_enemy_tangent(ctx, direction)),
        _vector_to_move_bin(*_enemy_tangent(ctx, -direction)),
    ]
    enemy_direction = _enemy_direction(ctx)
    retreat = -enemy_direction[0], -enemy_direction[1]
    retreat_moves = [
        _safe_retreat_move_bin(retreat, enemy_direction),
        _safe_retreat_move_bin(
            (retreat[0] + _enemy_tangent(ctx, direction)[0], retreat[1] + _enemy_tangent(ctx, direction)[1]),
            enemy_direction,
        ),
        _safe_retreat_move_bin(
            (retreat[0] + _enemy_tangent(ctx, -direction)[0], retreat[1] + _enemy_tangent(ctx, -direction)[1]),
            enemy_direction,
        ),
    ]
    approach_move = _vector_to_move_bin(*enemy_direction)
    ordered = (
        tangent_moves + retreat_moves + [approach_move]
        if ctx.enemy_in_range
        else tangent_moves + [approach_move] + retreat_moves
    )
    ordered.extend(range(1, 9))
    blocked: dict[str, list[str]] = {}
    seen: set[int] = set()
    for candidate_move in ordered:
        if candidate_move == 0 or candidate_move in seen:
            continue
        seen.add(candidate_move)
        reasons = _candidate_blocked_reasons(ctx, candidate_move, config)
        if not reasons:
            return candidate_move, blocked
        blocked[f"incoming_emergency_{candidate_move}"] = reasons
    return 0, blocked


def _select_outer_band_strafe(
    ctx: AgentContext,
    state: AgentState,
    config: BaselineConfig,
) -> tuple[int, int, int, str | None, bool, bool, int, int | None, dict[str, list[str]], str]:
    direction = 1 if state.strafe_direction >= 0 else -1
    interval = max(10, min(20, int(config.strafe_lock_steps)))
    current_lock_steps = max(0, int(state.dodge_lock_steps_remaining))
    current_lock_move_bin = state.dodge_lock_move_bin
    blocked_reasons: dict[str, list[str]] = {}
    forced_flip_reason = None

    if (
        current_lock_steps > 0
        and current_lock_move_bin is not None
        and _is_perpendicular_strafe(ctx, int(current_lock_move_bin))
    ):
        hard_reasons = _candidate_blocked_reasons(ctx, int(current_lock_move_bin), config)
        boundary_reasons = _near_boundary_reasons(ctx, int(current_lock_move_bin), config)
        if not hard_reasons and not boundary_reasons:
            move_bin = int(current_lock_move_bin)
            direction = _strafe_direction_for_move(ctx, move_bin, direction)
            remaining = current_lock_steps - 1
            return (
                move_bin,
                direction,
                max(1, int(state.strafe_age) + 1),
                None,
                False,
                True,
                remaining,
                move_bin if remaining > 0 else None,
                blocked_reasons,
                "outer_band_bullet_strafe_lock",
            )
        blocked_reasons["bullet_strafe_lock"] = hard_reasons + boundary_reasons
        direction *= -1
        forced_flip_reason = (
            "near_boundary" if boundary_reasons and not hard_reasons else "map_or_obstacle_blocked"
        )

    strafe_age = max(0, int(state.strafe_age)) + 1
    flip_reason = forced_flip_reason
    if forced_flip_reason is not None:
        strafe_age = 1
    elif state.strafe_age >= interval:
        direction *= -1
        strafe_age = 1
        flip_reason = "interval_expired"

    move_bin, selected_direction, candidate_flip_reason, candidate_blocked = _choose_strafe_move(
        ctx,
        direction,
        config,
    )
    blocked_reasons.update(candidate_blocked)
    if selected_direction != direction:
        direction = selected_direction
        strafe_age = 1
        flip_reason = candidate_flip_reason
    elif candidate_flip_reason is not None:
        flip_reason = candidate_flip_reason

    bullet_lock_active = bool(ctx.incoming_bullet and move_bin != 0)
    if bullet_lock_active:
        lock_duration = max(2, min(3, int(config.bullet_dodge_steps)))
        lock_remaining = lock_duration - 1
        lock_move_bin = move_bin
        policy_reason = "outer_band_bullet_strafe_lock"
    else:
        lock_remaining = 0
        lock_move_bin = None
        policy_reason = "outer_band_persistent_strafe"

    return (
        move_bin,
        direction,
        strafe_age,
        flip_reason,
        bool(blocked_reasons),
        bullet_lock_active,
        lock_remaining,
        lock_move_bin,
        blocked_reasons,
        policy_reason,
    )


def _choose_strafe_move(
    ctx: AgentContext,
    direction: int,
    config: BaselineConfig,
) -> tuple[int, int, str | None, dict[str, list[str]]]:
    blocked: dict[str, list[str]] = {}
    preferred_move = _vector_to_move_bin(*_enemy_tangent(ctx, direction))
    preferred_hard = _candidate_blocked_reasons(ctx, preferred_move, config)
    preferred_boundary = _near_boundary_reasons(ctx, preferred_move, config)
    if not preferred_hard and not preferred_boundary:
        return preferred_move, direction, None, blocked

    blocked["preferred_strafe"] = preferred_hard + preferred_boundary
    opposite_direction = -direction
    opposite_move = _vector_to_move_bin(*_enemy_tangent(ctx, opposite_direction))
    opposite_hard = _candidate_blocked_reasons(ctx, opposite_move, config)
    opposite_boundary = _near_boundary_reasons(ctx, opposite_move, config)
    if not opposite_hard and not opposite_boundary:
        reason = "near_boundary" if preferred_boundary and not preferred_hard else "map_or_obstacle_blocked"
        return opposite_move, opposite_direction, reason, blocked

    blocked["opposite_strafe"] = opposite_hard + opposite_boundary
    if not preferred_hard:
        return preferred_move, direction, "near_boundary_no_clear_alternative", blocked
    if not opposite_hard:
        return opposite_move, opposite_direction, "map_or_obstacle_blocked", blocked
    return 0, direction, "both_strafe_directions_blocked", blocked


def _select_retreat_move(
    ctx: AgentContext,
    strafe_direction: int,
    config: BaselineConfig,
) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
    if ctx.nearest_enemy is None:
        return None, {"retreat": ["no_enemy"]}
    enemy_direction = _enemy_direction(ctx)
    retreat = -enemy_direction[0], -enemy_direction[1]
    tangent = _enemy_tangent(ctx, strafe_direction)
    definitions = (
        ("pure_retreat", retreat),
        ("retreat_diagonal", (retreat[0] + tangent[0], retreat[1] + tangent[1])),
        ("retreat_diagonal_opposite", (retreat[0] - tangent[0], retreat[1] - tangent[1])),
    )
    blocked: dict[str, list[str]] = {}
    for name, vector in definitions:
        move_bin = _safe_retreat_move_bin(vector, enemy_direction)
        reasons = _candidate_blocked_reasons(ctx, move_bin, config)
        if not reasons:
            return {
                "name": name,
                "move_bin": move_bin,
                "vector": list(_normalized_move_vector(move_bin)),
                "feasible": True,
                "enemy_opposite_component": True,
            }, blocked
        blocked[name] = reasons
    return None, blocked


def _select_soft_backoff_move(
    ctx: AgentContext,
    strafe_direction: int,
    config: BaselineConfig,
    *,
    max_dist_ratio: float,
) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
    if ctx.nearest_enemy is None:
        return None, {"soft_backoff": ["no_enemy"]}
    enemy_direction = _enemy_direction(ctx)
    retreat = -enemy_direction[0], -enemy_direction[1]
    tangent = _enemy_tangent(ctx, strafe_direction)
    definitions = (
        ("soft_pure_retreat", retreat),
        ("soft_retreat_diagonal", (retreat[0] + tangent[0], retreat[1] + tangent[1])),
        ("soft_retreat_diagonal_opposite", (retreat[0] - tangent[0], retreat[1] - tangent[1])),
    )
    blocked: dict[str, list[str]] = {}
    for name, vector in definitions:
        move_bin = _safe_retreat_move_bin(vector, enemy_direction)
        reasons = _candidate_blocked_reasons(ctx, move_bin, config)
        predicted_ratio = _predicted_next_dist_ratio(ctx, move_bin, config)
        if predicted_ratio is None or predicted_ratio > max_dist_ratio:
            reasons.append("predicted_backoff_too_far")
        if not reasons:
            return {
                "name": name,
                "move_bin": move_bin,
                "vector": list(_normalized_move_vector(move_bin)),
                "feasible": True,
                "enemy_opposite_component": True,
            }, blocked
        blocked[name] = reasons
    return None, blocked


def _select_approach_move(
    ctx: AgentContext,
    config: BaselineConfig,
) -> tuple[int, list[str]]:
    if ctx.nearest_enemy is None:
        return 0, ["no_enemy"]
    desired = (
        ctx.nearest_enemy.position[0] - ctx.player_pos[0],
        ctx.nearest_enemy.position[1] - ctx.player_pos[1],
    )
    return _closest_feasible_move(ctx, desired, config)


def _select_poke_edge_fire_move(
    ctx: AgentContext,
    config: BaselineConfig,
) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
    if ctx.nearest_enemy is None:
        return None, {"poke_edge_fire": ["no_enemy"]}

    away = (
        ctx.player_pos[0] - ctx.nearest_enemy.position[0],
        ctx.player_pos[1] - ctx.nearest_enemy.position[1],
    )
    away_length = math.hypot(*away)
    away_unit = (0.0, 0.0) if away_length <= 1e-6 else (away[0] / away_length, away[1] / away_length)

    blocked: dict[str, list[str]] = {}
    candidates: list[dict[str, Any]] = []
    for move_bin in range(1, 9):
        reasons = _candidate_blocked_reasons(ctx, move_bin, config)
        predicted_ratio = _predicted_next_dist_ratio(ctx, move_bin, config)
        move_vector = _normalized_move_vector(move_bin)
        away_dot = move_vector[0] * away_unit[0] + move_vector[1] * away_unit[1]
        if predicted_ratio is None:
            reasons.append("missing_predicted_dist_ratio")
        elif predicted_ratio < 1.0:
            reasons.append("would_enter_direct_damage_range")
        elif away_dot < -1e-6:
            reasons.append("moves_toward_enemy")

        record = {
            "move_bin": move_bin,
            "vector": list(move_vector),
            "desired_vector": list(away_unit),
            "dot": away_dot,
            "predicted_next_dist_ratio": predicted_ratio,
            "feasible": False,
            "blocked_reasons": reasons,
        }
        if reasons:
            blocked[f"move_bin_{move_bin}"] = reasons
            continue
        candidates.append(record)

    if not candidates:
        return None, blocked

    candidates.sort(
        key=lambda record: (
            float(record["predicted_next_dist_ratio"]),
            -float(record["dot"]),
            int(record["move_bin"]),
        )
    )
    return {**candidates[0], "feasible": True}, blocked


def _closest_feasible_move(
    ctx: AgentContext,
    desired: tuple[float, float],
    config: BaselineConfig,
) -> tuple[int, list[str]]:
    if math.hypot(*desired) <= 1e-6:
        return 0, ["zero_direction"]
    desired_angle = math.atan2(desired[1], desired[0])
    candidates: list[tuple[float, int, list[str]]] = []
    for move_bin in range(1, 9):
        vx, vy = _normalized_move_vector(move_bin)
        angle = math.atan2(vy, vx)
        error = abs(math.atan2(math.sin(desired_angle - angle), math.cos(desired_angle - angle)))
        candidates.append((error, move_bin, _candidate_blocked_reasons(ctx, move_bin, config)))
    candidates.sort(key=lambda value: value[0])
    for _, move_bin, reasons in candidates:
        if not reasons:
            return move_bin, []
    return 0, candidates[0][2] if candidates else ["no_candidates"]


def _enemy_direction(ctx: AgentContext) -> tuple[float, float]:
    if ctx.nearest_enemy is None:
        return 0.0, 0.0
    dx = ctx.nearest_enemy.position[0] - ctx.player_pos[0]
    dy = ctx.nearest_enemy.position[1] - ctx.player_pos[1]
    distance = max(math.hypot(dx, dy), 1e-6)
    return dx / distance, dy / distance


def _predicted_next_dist_ratio(
    ctx: AgentContext,
    move_bin: int,
    config: BaselineConfig,
) -> float | None:
    if ctx.nearest_enemy is None or ctx.weapon_range <= 1e-6:
        return None
    vx, vy = _normalized_move_vector(move_bin)
    next_position = (
        ctx.player_pos[0] + vx * config.move_step_distance,
        ctx.player_pos[1] + vy * config.move_step_distance,
    )
    return math.dist(next_position, ctx.nearest_enemy.position) / ctx.weapon_range


def _enemy_tangent(ctx: AgentContext, direction: int) -> tuple[float, float]:
    dx, dy = _enemy_direction(ctx)
    return -dy * direction, dx * direction


def _is_perpendicular_strafe(ctx: AgentContext, move_bin: int) -> bool:
    vx, vy = _normalized_move_vector(move_bin)
    ex, ey = _enemy_direction(ctx)
    return move_bin != 0 and abs(vx * ex + vy * ey) <= 0.5


def _strafe_direction_for_move(ctx: AgentContext, move_bin: int, fallback: int) -> int:
    vx, vy = _normalized_move_vector(move_bin)
    right = _enemy_tangent(ctx, 1)
    left = _enemy_tangent(ctx, -1)
    right_dot = vx * right[0] + vy * right[1]
    left_dot = vx * left[0] + vy * left[1]
    return 1 if right_dot > left_dot else -1 if left_dot > right_dot else fallback


def _near_boundary_reasons(
    ctx: AgentContext,
    move_bin: int,
    config: BaselineConfig,
) -> list[str]:
    vx, vy = _normalized_move_vector(move_bin)
    target_x = ctx.player_pos[0] + vx * config.move_step_distance
    target_y = ctx.player_pos[1] + vy * config.move_step_distance
    margin = max(0.0, float(ctx.player_radius)) + config.move_step_distance
    reasons: list[str] = []
    if ctx.map_width is not None:
        if vx < 0.0 and target_x < margin:
            reasons.append("near_boundary_left")
        elif vx > 0.0 and target_x > ctx.map_width - margin:
            reasons.append("near_boundary_right")
    if ctx.map_height is not None:
        if vy < 0.0 and target_y < margin:
            reasons.append("near_boundary_top")
        elif vy > 0.0 and target_y > ctx.map_height - margin:
            reasons.append("near_boundary_bottom")
    return reasons


def _candidate_blocked_reasons(
    ctx: AgentContext,
    move_bin: int,
    config: BaselineConfig,
) -> list[str]:
    if move_bin == 0:
        return ["no_safe_discrete_direction"]
    vx, vy = _normalized_move_vector(move_bin)
    target = (
        ctx.player_pos[0] + vx * config.move_step_distance,
        ctx.player_pos[1] + vy * config.move_step_distance,
    )
    radius = max(0.0, float(ctx.player_radius))
    reasons: list[str] = []
    if ctx.map_width is not None and not (radius <= target[0] <= ctx.map_width - radius):
        reasons.append("map_bounds_x")
    if ctx.map_height is not None and not (radius <= target[1] <= ctx.map_height - radius):
        reasons.append("map_bounds_y")
    for obstacle in ctx.obstacles:
        if str(obstacle.get("type", "circle")) != "circle":
            continue
        if _segment_hits_circle(
            ctx.player_pos,
            target,
            (float(obstacle.get("x", 0.0)), float(obstacle.get("y", 0.0))),
            radius + float(obstacle.get("radius", 0.0)),
        ):
            reasons.append(f"obstacle:{obstacle.get('id', 'unknown')}")
            break
    if _move_blocked(ctx.local_grid, move_bin):
        reasons.append("local_grid_obstacle")
    return reasons


def _segment_hits_circle(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.dist(start, center) <= radius
    t = max(0.0, min(1.0, ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length_sq))
    closest = start[0] + t * dx, start[1] + t * dy
    return math.dist(closest, center) <= radius


def _safe_retreat_move_bin(
    desired: tuple[float, float],
    enemy_direction: tuple[float, float],
) -> int:
    desired_angle = math.atan2(desired[1], desired[0])
    options: list[tuple[float, int]] = []
    for move_bin in range(1, 9):
        vx, vy = _normalized_move_vector(move_bin)
        if vx * enemy_direction[0] + vy * enemy_direction[1] >= -1e-6:
            continue
        angle = math.atan2(vy, vx)
        error = abs(math.atan2(math.sin(desired_angle - angle), math.cos(desired_angle - angle)))
        options.append((error, move_bin))
    return min(options)[1] if options else 0


def _normalized_move_vector(move_bin: int) -> tuple[float, float]:
    dx, dy = _MOVE_VECTORS.get(move_bin, (0, 0))
    length = math.hypot(dx, dy)
    return (0.0, 0.0) if length <= 1e-6 else (dx / length, dy / length)


def _move_blocked(grid: Any, move_bin: int) -> bool:
    if grid is None or move_bin == 0:
        return False
    cells = grid.get("cells") if isinstance(grid, Mapping) else getattr(grid, "cells", None)
    if hasattr(cells, "tolist"):
        cells = cells.tolist()
    if not cells or not cells[0]:
        return False
    center = grid.get("center_cell") if isinstance(grid, Mapping) else getattr(grid, "center_cell", None)
    row, col = (int(center[0]), int(center[1])) if center is not None else (len(cells) // 2, len(cells[0]) // 2)
    dx, dy = _MOVE_VECTORS[move_bin]
    target_row, target_col = row + dy, col + dx
    if not (0 <= target_row < len(cells) and 0 <= target_col < len(cells[0])):
        return True
    names = grid.get("channel_names") if isinstance(grid, Mapping) else getattr(grid, "channel_names", ())
    try:
        obstacle_channel = list(names).index("obstacle")
    except ValueError:
        obstacle_channel = 0
    return float(cells[target_row][target_col][obstacle_channel]) > 0.0


def _vector_to_move_bin(dx: float, dy: float) -> int:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0
    angle = math.atan2(dy, dx)
    options = []
    for move_bin, (vx, vy) in _MOVE_VECTORS.items():
        if move_bin == 0:
            continue
        error = abs(math.atan2(math.sin(angle - math.atan2(vy, vx)), math.cos(angle - math.atan2(vy, vx))))
        options.append((error, move_bin))
    return min(options)[1]


__all__ = ["control_movement"]
