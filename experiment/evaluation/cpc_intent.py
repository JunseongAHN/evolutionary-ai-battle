from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


CPC_INTENTS = (
    "SOLO_OBJECTIVE",
    "SUPPORT_TEAMMATE",
    "REGROUP",
    "FOCUS_FIRE",
)


@dataclass(frozen=True)
class CpcIntentInputs:
    selfish_level: float
    teammate_distance: float
    human_recent_damage: float
    human_hp: float
    human_isolated: bool
    enemy_threatening_human: bool
    bot_goal_distance: float | None
    bot_enemy_distance: float | None


@dataclass(frozen=True)
class Layer1Output:
    cpc_intent: str


@dataclass(frozen=True)
class TargetRef:
    kind: str
    id: str | None = None

    def as_dict(self) -> dict[str, str]:
        value = {"kind": self.kind}
        if self.id is not None:
            value["id"] = self.id
        return value


@dataclass(frozen=True)
class Layer2Output:
    target_ref: TargetRef
    anchor_position: tuple[float, float]


@dataclass(frozen=True)
class CombatAction:
    move_bin: int
    aim: tuple[float, float]
    fire_requested: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "move_bin": self.move_bin,
            "aim": list(self.aim),
            "fire_requested": self.fire_requested,
        }


@dataclass(frozen=True)
class AppliedAction:
    move_bin_applied: int
    fire_applied: int
    blocked_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "move_bin_applied": self.move_bin_applied,
            "fire_applied": self.fire_applied,
        }
        if self.blocked_reason is not None:
            value["blocked_reason"] = self.blocked_reason
        return value


class CpcIntentArbiter:
    """Layer 1: select WHY/WHEN intent; selfishness is contained here."""

    def __init__(self) -> None:
        self._previous_intent: str | None = None

    def select(self, inputs: CpcIntentInputs) -> Layer1Output:
        selfish = _clamp(inputs.selfish_level, 0.0, 1.0)

        # These are the existing thresholds, retained without behavior tuning.
        max_teammate_distance = 180.0 + 520.0 * selfish
        pressure_damage_threshold = 0.5 + 19.5 * selfish
        pressure_hp_threshold = 75.0 - 65.0 * selfish
        support_distance_threshold = 180.0 + 500.0 * selfish
        focus_distance_threshold = 340.0 - 260.0 * selfish
        human_under_pressure = bool(
            inputs.enemy_threatening_human
            and (
                inputs.human_recent_damage >= pressure_damage_threshold
                or inputs.human_hp <= pressure_hp_threshold
            )
        )

        if inputs.bot_enemy_distance is not None and human_under_pressure:
            return self._remember("SUPPORT_TEAMMATE")

        regroup_release_distance = max_teammate_distance * 0.8
        regroup_active = bool(
            inputs.human_isolated
            and (
                inputs.teammate_distance > max_teammate_distance
                or (
                    self._previous_intent == "REGROUP"
                    and inputs.teammate_distance > regroup_release_distance
                )
            )
        )
        if regroup_active:
            return self._remember("REGROUP")

        if (
            inputs.bot_enemy_distance is not None
            and inputs.enemy_threatening_human
            and inputs.teammate_distance > support_distance_threshold
        ):
            return self._remember("SUPPORT_TEAMMATE")

        if (
            inputs.bot_enemy_distance is not None
            and inputs.enemy_threatening_human
            and inputs.teammate_distance <= focus_distance_threshold
        ):
            return self._remember("FOCUS_FIRE")

        if inputs.bot_enemy_distance is not None and inputs.bot_enemy_distance <= 360.0:
            return self._remember("FOCUS_FIRE")

        return self._remember("SOLO_OBJECTIVE")

    def _remember(self, intent: str) -> Layer1Output:
        self._previous_intent = intent
        return Layer1Output(intent)


class CpcTargetResolver:
    """Layer 2: resolve WHO/WHERE and create the sole CPC anchor."""

    def resolve(
        self,
        layer1: Layer1Output,
        *,
        bot_position: Mapping[str, float],
        human_position: Mapping[str, float],
        enemy_position: Mapping[str, float] | None,
        goal_position: tuple[float, float] | None,
        enemy_id: str | None,
        weapon_range: float,
        map_width: float,
        map_height: float,
    ) -> Layer2Output:
        intent = layer1.cpc_intent
        if intent == "REGROUP":
            return Layer2Output(TargetRef("teammate"), _point(human_position))
        if intent == "SUPPORT_TEAMMATE" and enemy_position is not None and enemy_id is not None:
            return Layer2Output(
                TargetRef("enemy", enemy_id),
                _support_anchor(human_position, enemy_position, map_width, map_height),
            )
        if intent == "FOCUS_FIRE" and enemy_position is not None and enemy_id is not None:
            return Layer2Output(
                TargetRef("enemy", enemy_id),
                _combat_anchor(bot_position, enemy_position, weapon_range, map_width, map_height),
            )
        return Layer2Output(
            TargetRef("goal"),
            goal_position if goal_position is not None else _point(bot_position),
        )


def _combat_anchor(
    bot: Mapping[str, float],
    enemy: Mapping[str, float],
    weapon_range: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    dx = float(bot["x"]) - float(enemy["x"])
    dy = float(bot["y"]) - float(enemy["y"])
    length = math.hypot(dx, dy) or 1.0
    distance = max(20.0, float(weapon_range) * 0.95)
    return _bounded(
        float(enemy["x"]) + dx / length * distance,
        float(enemy["y"]) + dy / length * distance,
        width,
        height,
    )


def _support_anchor(
    human: Mapping[str, float],
    enemy: Mapping[str, float],
    width: float,
    height: float,
) -> tuple[float, float]:
    dx = float(enemy["x"]) - float(human["x"])
    dy = float(enemy["y"]) - float(human["y"])
    length = math.hypot(dx, dy) or 1.0
    return _bounded(
        float(human["x"]) + dx / length * 60.0,
        float(human["y"]) + dy / length * 60.0,
        width,
        height,
    )


def _point(value: Mapping[str, float]) -> tuple[float, float]:
    return float(value["x"]), float(value["y"])


def _bounded(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    return _clamp(x, 0.0, width), _clamp(y, 0.0, height)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


__all__ = [
    "AppliedAction",
    "CPC_INTENTS",
    "CombatAction",
    "CpcIntentArbiter",
    "CpcIntentInputs",
    "CpcTargetResolver",
    "Layer1Output",
    "Layer2Output",
    "TargetRef",
]
