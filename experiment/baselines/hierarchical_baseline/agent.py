from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .action import build_action
from .context import build_context
from .control import controller
from .debug import build_debug
from .planning import create_global_plan_if_needed, create_local_plan
from .selection import select_intent
from .types import AgentState, BaselineConfig, default_agent_state, default_config


class HierarchicalBaselineAgent:
    def __init__(self, config: BaselineConfig | Mapping[str, Any] | None = None):
        if config is None:
            self.config = default_config()
        elif isinstance(config, BaselineConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = BaselineConfig(**dict(config))
        else:
            raise TypeError("config must be BaselineConfig, a mapping, or None")
        self.state = default_agent_state()

    def act(self, obs: Any, snapshot: Any | None = None) -> tuple[dict[str, int | float], dict[str, Any]]:
        ctx, ctx_debug = build_context(obs, snapshot, self.state, self.config)

        new_global_plan, global_debug = create_global_plan_if_needed(ctx, self.state, self.config)
        if ctx.goal_pos is None:
            self.state.global_plan = None
        elif new_global_plan is not None:
            self.state.global_plan = new_global_plan

        intent, intent_debug = select_intent(ctx, self.state, self.state.global_plan, self.config)
        local_plan, local_debug = create_local_plan(
            ctx,
            self.state,
            intent,
            self.state.global_plan,
            self.config,
        )
        control, control_debug = controller(ctx, self.state, local_plan, self.config)
        action, action_debug = build_action(control, self.config)
        movement_debug = control_debug.get("movement", {})

        direct_threat = bool(
            (
                ctx.nearest_enemy is not None
                and ctx.nearest_enemy.alive
                and (ctx.enemy_in_detection_range or intent == "COMBAT")
            )
            or ctx.incoming_bullet
        )
        previous_local_plan = (
            local_plan
            if local_plan.target_cell is not None and local_plan.next_cell is not None
            else self.state.previous_local_plan
        )
        self.state = AgentState(
            global_plan=self.state.global_plan,
            agent_mode=intent,
            previous_intent=intent,
            previous_tactical_mode=local_plan.tactical_mode,
            previous_target_cell=local_plan.target_cell,
            previous_anchor=local_plan.anchor,
            last_goal_reached_count=ctx.goal_reached_count,
            combat_steps=self.state.combat_steps + 1 if intent == "COMBAT" else 0,
            no_enemy_steps=0 if direct_threat else self.state.no_enemy_steps + 1,
            tactical_mode_age=int(local_debug.get("mode_age", 0)),
            anchor_age=int(local_debug.get("anchor_age", 0)),
            previous_local_plan=previous_local_plan,
            strafe_direction=int(
                movement_debug.get(
                    "strafe_direction_sign",
                    local_debug.get("strafe_direction_sign", self.state.strafe_direction),
                )
            ),
            strafe_age=(
                int(movement_debug.get("strafe_age", local_debug.get("strafe_age", 0)))
                if movement_debug.get("outer_band_strafe_active", False)
                else 0
            ),
            dodge_lock_steps_remaining=max(
                0, int(movement_debug.get("dodge_lock_steps_remaining", 0))
            ),
            dodge_lock_move_bin=(
                int(movement_debug["dodge_lock_move_bin"])
                if movement_debug.get("dodge_lock_move_bin") is not None
                else None
            ),
            combat_stay_steps=max(
                0, int(movement_debug.get("combat_stay_steps", 0))
            ),
            poke_state=movement_debug.get("poke_state"),
            poke_state_age=max(0, int(movement_debug.get("poke_state_age", 0))),
            poke_exit_lock_steps_remaining=max(
                0, int(movement_debug.get("poke_exit_lock_steps_remaining", 0))
            ),
        )

        debug = build_debug(
            ctx_debug,
            global_debug,
            intent_debug,
            local_debug,
            control_debug,
            action_debug,
        )
        debug["agent_state"] = {
            "agent_mode": self.state.agent_mode,
            "combat_steps": self.state.combat_steps,
            "no_enemy_steps": self.state.no_enemy_steps,
            "last_goal_reached_count": self.state.last_goal_reached_count,
            "mode_age": self.state.tactical_mode_age,
            "anchor_age": self.state.anchor_age,
            "strafe_direction": self.state.strafe_direction,
            "strafe_age": self.state.strafe_age,
            "dodge_lock_steps_remaining": self.state.dodge_lock_steps_remaining,
            "dodge_lock_move_bin": self.state.dodge_lock_move_bin,
            "combat_stay_steps": self.state.combat_stay_steps,
            "poke_state": self.state.poke_state,
            "poke_state_age": self.state.poke_state_age,
            "poke_exit_lock_steps_remaining": self.state.poke_exit_lock_steps_remaining,
        }
        return action, debug


__all__ = ["HierarchicalBaselineAgent"]
