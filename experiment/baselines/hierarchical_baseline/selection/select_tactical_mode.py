from __future__ import annotations

from ..types import AgentContext, AgentState, BaselineConfig


def select_tactical_mode(
    ctx: AgentContext,
    state: AgentState,
    config: BaselineConfig,
) -> tuple[str, dict]:
    previous_mode = state.previous_tactical_mode
    previous_age = max(0, int(state.tactical_mode_age))
    distance_ratio = None if ctx.enemy_dist is None else ctx.enemy_dist / max(ctx.weapon_range, 1e-6)
    if ctx.nearest_enemy is None or ctx.enemy_dist is None:
        candidate, reason = "reposition", "no_live_enemy"
    elif distance_ratio < config.backoff_range_ratio:
        candidate, reason = "backoff", "below_backoff_threshold"
    elif distance_ratio > config.outer_range_max_ratio:
        candidate, reason = "approach", "above_outer_range_band"
    else:
        candidate, reason = "outer_band", "strafe_between_range_thresholds"

    hysteresis_steps = max(0, int(config.range_hysteresis_steps))
    range_hysteresis_locked = bool(
        previous_mode in {"outer_band", "backoff"}
        and candidate in {"outer_band", "backoff"}
        and candidate != previous_mode
        and previous_age < hysteresis_steps
    )
    if range_hysteresis_locked:
        mode = previous_mode
        reason = f"hold_{previous_mode}_for_range_hysteresis"
    else:
        mode = candidate

    mode_age = previous_age + 1 if mode == previous_mode else 1
    return mode, {
        "tactical_mode": mode,
        "reason": reason,
        "mode_age": mode_age,
        "mode_locked": range_hysteresis_locked,
        "range_hysteresis_locked": range_hysteresis_locked,
        "range_hysteresis_steps": hysteresis_steps,
        "combat_range_state": mode.upper(),
        "range_state": mode.upper(),
        "target_range_band": [config.outer_range_min_ratio, config.outer_range_max_ratio],
        "distance_ratio": distance_ratio,
        "dist_ratio": distance_ratio,
        "thresholds": {
            "approach_above": config.outer_range_max_ratio,
            "outer_min": config.outer_range_min_ratio,
            "outer_max": config.outer_range_max_ratio,
            "backoff_below": config.backoff_range_ratio,
            "backoff_max": config.backoff_max_ratio,
        },
    }


__all__ = ["select_tactical_mode"]
