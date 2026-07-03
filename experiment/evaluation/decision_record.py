from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .cpc_intent import AppliedAction, CombatAction, Layer1Output, Layer2Output


@dataclass(frozen=True)
class TacticalFrame:
    who: str
    task: str
    target: str
    where: tuple[float, float]
    how: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "who": self.who,
            "task": self.task,
            "target": self.target,
            "where": list(self.where),
            "how": self.how,
            "reason": self.reason,
        }

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.who, self.task, self.target


@dataclass(frozen=True)
class DecisionRecord:
    self_frame: TacticalFrame
    team_frame: TacticalFrame
    selected_frame: TacticalFrame
    primitive_action: dict[str, Any]
    outcome: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "self_frame": self.self_frame.as_dict(),
            "team_frame": self.team_frame.as_dict(),
            "selected_frame": self.selected_frame.as_dict(),
            "primitive_action": dict(self.primitive_action),
            "outcome": dict(self.outcome),
        }


def derive_tactical_frames(
    layer1: Layer1Output,
    layer2: Layer2Output,
    *,
    goal_position: tuple[float, float],
    teammate_position: tuple[float, float],
) -> tuple[TacticalFrame, TacticalFrame, TacticalFrame]:
    """Project existing CPC outputs into logging-only self/team tactical frames."""
    intent = layer1.cpc_intent
    selected_task = {
        "SOLO_OBJECTIVE": "advance_goal",
        "FOCUS_FIRE": "fight_enemy",
        "REGROUP": "regroup_teammate",
        "SUPPORT_TEAMMATE": "fight_enemy",
    }[intent]
    selected_who = "team" if intent in {"REGROUP", "SUPPORT_TEAMMATE"} else "self"
    selected_how = "poke_out" if selected_task == "fight_enemy" else "navigate"
    selected_target = layer2.target_ref.id or {
        "goal": "goal-0",
        "teammate": "teammate",
    }.get(layer2.target_ref.kind, "anchor")
    selected_reason = {
        "SOLO_OBJECTIVE": "goal_available",
        "FOCUS_FIRE": "enemy_blocks_goal",
        "REGROUP": "teammate_too_far",
        "SUPPORT_TEAMMATE": "enemy_attacking_teammate",
    }[intent]
    selected_frame = TacticalFrame(
        selected_who,
        selected_task,
        selected_target,
        layer2.anchor_position,
        selected_how,
        selected_reason,
    )

    self_frame = (
        selected_frame
        if selected_who == "self"
        else TacticalFrame(
            "self",
            "advance_goal",
            "goal-0",
            goal_position,
            "navigate",
            "goal_available",
        )
    )
    team_frame = (
        selected_frame
        if selected_who == "team"
        else TacticalFrame(
            "team",
            "hold_anchor",
            "anchor",
            teammate_position,
            "hold",
            "hold_safe_anchor",
        )
    )
    return self_frame, team_frame, selected_frame


def build_decision_record(
    *,
    self_frame: TacticalFrame,
    team_frame: TacticalFrame,
    selected_frame: TacticalFrame,
    combat_action: CombatAction,
    applied_action: AppliedAction,
    damage_dealt: float,
    damage_taken: float,
    goal_reached: bool,
    teammate_distance: float,
    bot_hp: float,
    player_hp: float,
    enemy_hp: float,
) -> DecisionRecord:
    return DecisionRecord(
        self_frame=self_frame,
        team_frame=team_frame,
        selected_frame=selected_frame,
        primitive_action={
            "move_bin": combat_action.move_bin,
            "aim": list(combat_action.aim),
            "fire_requested": combat_action.fire_requested,
            "fire_applied": applied_action.fire_applied,
        },
        outcome={
            "damage_dealt": float(damage_dealt),
            "damage_taken": float(damage_taken),
            "goal_reached": bool(goal_reached),
            "teammate_distance": float(teammate_distance),
            "bot_hp": float(bot_hp),
            "player_hp": float(player_hp),
            "enemy_hp": float(enemy_hp),
        },
    )


def format_selected_frame(frame: TacticalFrame | dict[str, Any]) -> str:
    value = frame.as_dict() if isinstance(frame, TacticalFrame) else dict(frame)
    where = json.dumps(value.get("where", []), separators=(",", ":"))
    return (
        f"SELECTED {value.get('who', '-')}/{value.get('task', '-')}:"
        f"{value.get('target', '-')} WHERE {where} HOW {value.get('how', '-')} "
        f"REASON {value.get('reason', '-')}"
    )


__all__ = [
    "DecisionRecord",
    "TacticalFrame",
    "build_decision_record",
    "derive_tactical_frames",
    "format_selected_frame",
]
