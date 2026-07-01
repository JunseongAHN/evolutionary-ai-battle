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

    if combat_enemy and dist_ratio is not None:
        if dist_ratio < 0.75:
            fire_window_state = "TOO_CLOSE"
            range_policy_reason = "dist_ratio_below_0.75"
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
        elif fire_ready and not target_in_range:
            fire_window_state = "ENTER"
            range_policy_reason = "fire_ready_target_out_of_range"
            move_bin, approach_blocked = _select_approach_move(ctx, config)
            if approach_blocked:
                blocked_reasons["approach"] = approach_blocked
            movement_policy_reason = (
                "fire_window_enter_approach"
                if move_bin != 0
                else "fire_window_enter_blocked"
            )
        elif fire_ready and target_in_range:
            fire_window_state = "HOLD"
            range_policy_reason = "fire_ready_target_in_range"
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
            hold_stop_used = move_bin == 0
            incoming_bullet_stop_blocked = bool(ctx.incoming_bullet and move_bin != 0)
            movement_policy_reason = "fire_window_hold_movement"
        else:
            fire_window_state = "RESET"
            range_policy_reason = "fire_cooldown_reset"
            if dist_ratio > 1.35:
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
                movement_policy_reason = (
                    "fire_window_reset_backoff"
                    if move_bin != 0
                    else "fire_window_reset_backoff_blocked"
                )
                reset_soft_backoff_active = move_bin != 0
                if move_bin == 0:
                    move_bin, direction, _, strafe_blocked_reasons = _choose_strafe_move(
                        ctx,
                        direction,
                        config,
                    )
                    blocked_reasons.update(strafe_blocked_reasons)
                    strafe_age = max(1, int(state.strafe_age) + 1)
                    perpendicular_strafe = move_bin != 0
                    outer_band_strafe_active = perpendicular_strafe
                    incoming_bullet_stop_blocked = bool(ctx.incoming_bullet and move_bin != 0)
                    movement_policy_reason = (
                        "fire_window_reset_backoff_blocked_strafe"
                        if move_bin != 0
                        else "fire_window_reset_blocked"
                    )
            else:
                if not target_in_range and not ctx.incoming_bullet:
                    move_bin = 0
                    movement_policy_reason = "fire_window_reset_hold_out_of_range"
                else:
                    move_bin, direction, _, strafe_blocked_reasons = _choose_strafe_move(
                        ctx,
                        direction,
                        config,
                    )
                    blocked_reasons.update(strafe_blocked_reasons)
                    strafe_age = max(1, int(state.strafe_age) + 1)
                    perpendicular_strafe = move_bin != 0
                    outer_band_strafe_active = perpendicular_strafe
                    incoming_bullet_stop_blocked = bool(ctx.incoming_bullet and move_bin != 0)
                    movement_policy_reason = (
                        "fire_window_reset_incoming_strafe"
                        if ctx.incoming_bullet and move_bin != 0
                        else "fire_window_reset_blocked"
                    )
    elif local_plan.intent == "COMBAT":
        range_policy_reason = "combat_without_live_enemy"
        movement_policy_reason = "combat_without_live_enemy_local_plan"

    if combat_enemy and ctx.incoming_bullet and move_bin == 0:
        emergency_move, emergency_blocked = _select_incoming_emergency_move(
            ctx,
            direction,
            config,
        )
        blocked_reasons.update(emergency_blocked)
        if emergency_move != 0:
            move_bin = emergency_move
            movement_policy_reason = f"{movement_policy_reason}_incoming_emergency"
            if fire_window_state == "HOLD":
                hold_movement_policy = "incoming_bullet_emergency_feasible_move"
                emergency_predicted_ratio = _predicted_next_dist_ratio(
                    ctx,
                    move_bin,
                    config,
                )
                hold_predicted_in_range = bool(
                    emergency_predicted_ratio is not None
                    and emergency_predicted_ratio <= 0.98
                )
                hold_stop_used = False

    if combat_enemy and ctx.incoming_bullet and move_bin != 0:
        incoming_bullet_stop_blocked = True

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
        "hold_movement_policy": hold_movement_policy,
        "hold_predicted_in_range": hold_predicted_in_range,
        "hold_stop_used": hold_stop_used,
        "incoming_bullet_stop_blocked": incoming_bullet_stop_blocked,
        "reset_soft_backoff_active": reset_soft_backoff_active,
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
            "bullet_perpendicular_strafe_lock"
            if bullet_strafe_lock_active
            else None
        ),
        "bullet_safety_margin": max(0.0, float(config.bullet_safety_margin)),
        "cooldown_strafe_fallback_used": False,
        "dodge_candidates": [],
        "selected_dodge_move": selected_retreat_move,
        "dodge_blocked_reasons": blocked_reasons,
        "enemy_opposite_component_used": bool(selected_retreat_move),
        "strafe_direction": "right" if direction > 0 else "left",
        "strafe_direction_sign": direction,
        "strafe_age": strafe_age,
        "strafe_blocked": strafe_blocked,
    }


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
