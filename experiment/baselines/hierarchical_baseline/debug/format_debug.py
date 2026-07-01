from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def format_debug(debug: Mapping[str, Any]) -> str:
    aim = debug.get("aim_dir") or [1.0, 0.0]
    return " | ".join(
        [
            f"intent={debug.get('intent')}",
            f"global_plan_reason={debug.get('global_plan_reason')}",
            f"goal_pos={debug.get('goal_pos')}",
            f"tactical_mode={debug.get('tactical_mode')}",
            f"combat_profile={debug.get('combat_profile')}",
            f"anchor={debug.get('anchor')}",
            f"target_cell={debug.get('target_cell')}",
            f"next_cell={debug.get('next_cell')}",
            f"move_bin={debug.get('move_bin')}",
            f"aim_dir=({float(aim[0]):.3f},{float(aim[1]):.3f})",
            f"fire={debug.get('fire')}",
            f"fire_reason={debug.get('fire_reason')}",
            f"fire_window_state={debug.get('fire_window_state')}",
            f"fire_ready={debug.get('fire_ready')}",
            f"target_in_range={debug.get('target_in_range')}",
            f"can_fire_now={debug.get('can_fire_now')}",
            f"range_policy_reason={debug.get('range_policy_reason')}",
            f"hold_movement_policy={debug.get('hold_movement_policy')}",
            f"hold_predicted_in_range={debug.get('hold_predicted_in_range')}",
            f"hold_stop_used={debug.get('hold_stop_used')}",
            f"incoming_bullet_stop_blocked={debug.get('incoming_bullet_stop_blocked')}",
            f"reset_soft_backoff_active={debug.get('reset_soft_backoff_active')}",
            f"predicted_next_dist_ratio={_format_ratio(debug.get('predicted_next_dist_ratio'))}",
            f"mode_age={debug.get('mode_age')}",
            f"mode_locked={debug.get('mode_locked')}",
            f"anchor_age={debug.get('anchor_age')}",
            f"anchor_reused={debug.get('anchor_reused')}",
            f"fallback_previous_plan={debug.get('fallback_previous_plan')}",
            f"combat_range_state={debug.get('combat_range_state')}",
            f"range_state={debug.get('range_state')}",
            f"target_range_band={debug.get('target_range_band')}",
            f"dist_ratio={_format_ratio(debug.get('dist_ratio'))}",
            f"strafe_direction={debug.get('strafe_direction')}",
            f"perpendicular_strafe={debug.get('perpendicular_strafe')}",
            f"outer_band_strafe_active={debug.get('outer_band_strafe_active')}",
            f"strafe_lock_steps_remaining={debug.get('strafe_lock_steps_remaining')}",
            f"strafe_flip_reason={debug.get('strafe_flip_reason')}",
            f"bullet_strafe_lock_active={debug.get('bullet_strafe_lock_active')}",
            f"retreat_diagonal_allowed={debug.get('retreat_diagonal_allowed')}",
            f"movement_policy_reason={debug.get('movement_policy_reason')}",
            f"bullet_dodge_active={debug.get('bullet_dodge_active')}",
            f"dodge_reason={debug.get('dodge_reason')}",
            f"bullet_safety_margin={debug.get('bullet_safety_margin')}",
            f"dodge_lock_active={debug.get('dodge_lock_active')}",
            f"dodge_lock_steps_remaining={debug.get('dodge_lock_steps_remaining')}",
            f"dodge_lock_move_bin={debug.get('dodge_lock_move_bin')}",
            f"cooldown_strafe_fallback_used={debug.get('cooldown_strafe_fallback_used')}",
            f"dodge_candidates={debug.get('dodge_candidates')}",
            f"selected_dodge_move={debug.get('selected_dodge_move')}",
            f"dodge_blocked_reasons={debug.get('dodge_blocked_reasons')}",
            f"enemy_opposite_component_used={debug.get('enemy_opposite_component_used')}",
            f"range_hysteresis_locked={debug.get('range_hysteresis_locked')}",
            f"combat_exit_blocked_reason={debug.get('combat_exit_blocked_reason')}",
            f"events={','.join(str(value) for value in debug.get('events', []))}",
        ]
    )


def _format_ratio(value: Any) -> str:
    return "None" if value is None else f"{float(value):.3f}"


__all__ = ["format_debug"]
