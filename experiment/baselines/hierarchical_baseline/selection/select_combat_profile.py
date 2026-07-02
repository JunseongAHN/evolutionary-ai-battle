from __future__ import annotations

from ..types import AgentContext, AgentState, BaselineConfig


def select_combat_profile(
    ctx: AgentContext,
    state: AgentState,
    tactical_mode: str,
    config: BaselineConfig,
) -> tuple[str, dict]:
    del ctx, state
    configured_profile = str(config.combat_movement_profile or "default").strip().lower()
    if configured_profile == "poke_out":
        return "poke_out", {
            "combat_profile": "poke_out",
            "combat_movement_profile": "poke_out",
            "reason": "configured_combat_movement_profile",
        }

    profile = {
        "approach": "approach_outer_band",
        "outer_band": "strafe_outer_band",
        "backoff": "backoff_to_outer_band",
        "reposition": "restore_line_of_sight",
    }.get(tactical_mode, "safe_reposition")
    return profile, {"combat_profile": profile, "reason": f"profile_for_{tactical_mode}"}


__all__ = ["select_combat_profile"]
