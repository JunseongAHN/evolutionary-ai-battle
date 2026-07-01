from __future__ import annotations

from ..types import AgentContext, AgentState, BaselineConfig


def select_combat_profile(
    ctx: AgentContext,
    state: AgentState,
    tactical_mode: str,
    config: BaselineConfig,
) -> tuple[str, dict]:
    del ctx, state, config
    profile = {
        "approach": "approach_outer_band",
        "outer_band": "strafe_outer_band",
        "backoff": "backoff_to_outer_band",
        "reposition": "restore_line_of_sight",
    }.get(tactical_mode, "safe_reposition")
    return profile, {"combat_profile": profile, "reason": f"profile_for_{tactical_mode}"}


__all__ = ["select_combat_profile"]
