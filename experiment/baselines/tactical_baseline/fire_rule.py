from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from experiment.training.cpc_actions import AIM_BINS, circular_bin_distance, vec_to_aim_bin
except ModuleNotFoundError:
    from training.cpc_actions import AIM_BINS, circular_bin_distance, vec_to_aim_bin


class FireRule:
    """Deterministic tactical fire gate.

    Missing optional line-of-sight and ammo fields do not block fire. Missing
    required enemy, range, aim, or cooldown facts are treated conservatively and
    return ``fire=0``.
    """

    def __init__(
        self,
        aim_error_threshold: float = 0.15,
        require_in_range: bool = True,
        require_line_of_sight: bool = True,
        require_cooldown_ready: bool = True,
        num_aim_bins: int = AIM_BINS,
    ) -> None:
        self.aim_error_threshold = float(aim_error_threshold)
        self.require_in_range = bool(require_in_range)
        self.require_line_of_sight = bool(require_line_of_sight)
        self.require_cooldown_ready = bool(require_cooldown_ready)
        self.num_aim_bins = max(1, int(num_aim_bins))

    def decide_fire(self, obs: Any, state_snapshot: Any | None = None) -> tuple[int, dict[str, Any]]:
        observation = _mapping(obs)
        snapshot = _normalize_snapshot(state_snapshot)
        self_pos = _self_position(observation, snapshot)
        enemy_pos = _enemy_position(observation, snapshot)
        enemy_exists = _enemy_exists(observation, snapshot, enemy_pos)
        distance_to_enemy = _distance(self_pos, enemy_pos) if self_pos is not None and enemy_pos is not None else _number(
            observation.get("distance_to_enemy"),
            _mapping(snapshot.get("distances")).get("self_to_enemy"),
            _mapping(snapshot.get("range_debug")).get("distance_to_enemy"),
        )

        target_in_range, target_range_source = self._target_in_range(observation, snapshot, distance_to_enemy)
        aim_error, aim_bin_error, aim_source = self._aim_error(observation, snapshot, self_pos, enemy_pos)
        line_of_sight, los_source = self._line_of_sight(snapshot, self_pos, enemy_pos)
        cooldown_ready, cooldown_source = self._cooldown_ready(observation, snapshot)
        ammo_available, ammo_source = self._ammo_available(observation, snapshot)

        fire = 0
        reason = "all_conditions_met"
        if not enemy_exists:
            reason = "no_live_enemy"
        elif self.require_in_range and target_in_range is not True:
            reason = "target_out_of_range" if target_in_range is False else "missing_range_info"
        elif aim_error is None:
            reason = "missing_aim_error"
        elif aim_error > self.aim_error_threshold:
            reason = "aim_error_above_threshold"
        elif self.require_line_of_sight and line_of_sight is False:
            reason = "line_of_sight_blocked"
        elif self.require_cooldown_ready and cooldown_ready is not True:
            reason = "cooldown_not_ready" if cooldown_ready is False else "missing_cooldown_info"
        elif ammo_available is False:
            reason = "no_ammo"
        else:
            fire = 1

        debug = {
            "fire": int(fire),
            "enemy_exists": bool(enemy_exists),
            "aim_error": aim_error,
            "aim_bin_error": aim_bin_error,
            "target_in_range": target_in_range,
            "line_of_sight": line_of_sight,
            "cooldown_ready": cooldown_ready,
            "ammo_available": ammo_available,
            "distance_to_enemy": distance_to_enemy,
            "reason": reason,
            "sources": {
                "aim_error": aim_source,
                "target_in_range": target_range_source,
                "line_of_sight": los_source,
                "cooldown_ready": cooldown_source,
                "ammo_available": ammo_source,
            },
            "fallbacks": {
                "line_of_sight": "unknown_assumed_clear" if line_of_sight is None else None,
                "ammo": "missing_not_blocking" if ammo_available is None else None,
            },
        }
        return int(fire), debug

    def _target_in_range(
        self,
        obs: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        distance_to_enemy: float | None,
    ) -> tuple[bool | None, str]:
        if not self.require_in_range:
            return True, "disabled"
        fire_range = _number(
            _mapping(snapshot.get("combat")).get("fire_range"),
            _mapping(snapshot.get("combat")).get("bullet_range"),
            obs.get("fire_range"),
            obs.get("weapon_range"),
        )
        if distance_to_enemy is not None and fire_range is not None:
            return bool(distance_to_enemy <= fire_range), "distance_and_fire_range"
        explicit = _bool_or_none(
            obs.get("target_in_range"),
            _mapping(snapshot.get("fire_debug")).get("target_in_range"),
            _mapping(snapshot.get("range_debug")).get("in_good_range"),
        )
        if explicit is not None:
            return explicit, "explicit_flag"
        return None, "missing"

    def _aim_error(
        self,
        obs: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        self_pos: Mapping[str, float] | None,
        enemy_pos: Mapping[str, float] | None,
    ) -> tuple[float | None, int | None, str]:
        selected_aim_bin = _int_or_none(
            obs.get("selected_aim_bin"),
            obs.get("aim_bin"),
            obs.get("current_aim_bin"),
            _mapping(snapshot.get("aim_debug")).get("aim_bin"),
            _mapping(snapshot.get("fire_debug")).get("current_aim_bin"),
        )
        ideal_aim_bin = None
        if self_pos is not None and enemy_pos is not None:
            ideal_aim_bin = vec_to_aim_bin(
                {
                    "x": float(enemy_pos["x"]) - float(self_pos["x"]),
                    "y": float(enemy_pos["y"]) - float(self_pos["y"]),
                },
                self.num_aim_bins,
            )
        if ideal_aim_bin is None:
            ideal_aim_bin = _int_or_none(
                obs.get("gt_ideal_aim_bin"),
                obs.get("ideal_aim_bin"),
                _mapping(snapshot.get("aim_debug")).get("ideal_aim_bin"),
                _mapping(snapshot.get("fire_debug")).get("ideal_aim_bin"),
            )
        if selected_aim_bin is not None and ideal_aim_bin is not None:
            bin_error = circular_bin_distance(selected_aim_bin, ideal_aim_bin, self.num_aim_bins)
            return self._normalize_bin_error(bin_error), int(bin_error), "selected_vs_ideal_aim_bin"

        explicit_error = _number(
            obs.get("aim_error"),
            _mapping(snapshot.get("fire_debug")).get("aim_error"),
            _mapping(snapshot.get("aim_debug")).get("aim_bin_error"),
        )
        if explicit_error is not None:
            if float(explicit_error).is_integer():
                bin_error = int(explicit_error)
                return self._normalize_bin_error(bin_error), bin_error, "explicit_bin_error"
            return float(explicit_error), None, "explicit_normalized_error"

        aim_aligned = _bool_or_none(
            obs.get("aim_aligned"),
            _mapping(snapshot.get("fire_debug")).get("aim_aligned"),
            _mapping(snapshot.get("aim_debug")).get("is_aim_aligned"),
        )
        if aim_aligned is not None:
            return (0.0 if aim_aligned else 1.0), None, "aim_aligned_flag"
        return None, None, "missing"

    def _line_of_sight(
        self,
        snapshot: Mapping[str, Any],
        self_pos: Mapping[str, float] | None,
        enemy_pos: Mapping[str, float] | None,
    ) -> tuple[bool | None, str]:
        if not self.require_line_of_sight:
            return True, "disabled"
        explicit = _bool_or_none(
            _mapping(snapshot.get("predicates")).get("line_of_sight"),
            _mapping(snapshot.get("predicates")).get("has_line_of_sight"),
            _mapping(snapshot.get("aim_debug")).get("line_of_sight"),
            _mapping(snapshot.get("fire_debug")).get("line_of_sight"),
        )
        if explicit is not None:
            return explicit, "explicit_flag"
        if self_pos is None or enemy_pos is None:
            return None, "missing_positions"
        obstacles = _obstacles(snapshot)
        if not obstacles:
            return True, "no_obstacles"
        projectile_radius = float(_number(_mapping(snapshot.get("combat")).get("projectile_radius")) or 0.0)
        blocked = any(
            _segment_intersects_circle(
                self_pos,
                enemy_pos,
                {"x": float(obstacle.get("x", 0.0)), "y": float(obstacle.get("y", 0.0))},
                float(obstacle.get("radius", 0.0)) + projectile_radius,
            )
            for obstacle in obstacles
            if str(obstacle.get("type", "circle")) == "circle" and float(obstacle.get("radius", 0.0)) > 0.0
        )
        return (not blocked), "computed_from_circle_obstacles"

    def _cooldown_ready(
        self,
        obs: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> tuple[bool | None, str]:
        if not self.require_cooldown_ready:
            return True, "disabled"
        explicit = _bool_or_none(
            obs.get("cooldown_ready"),
            obs.get("can_fire"),
            _mapping(snapshot.get("fire_debug")).get("cooldown_ready"),
        )
        if explicit is not None:
            return explicit, "explicit_flag"
        cooldown = _number(
            _mapping(snapshot.get("weapon")).get("cooldown_remaining_steps"),
            _mapping(snapshot.get("weapon")).get("fire_cooldown"),
        )
        if cooldown is not None:
            return bool(cooldown <= 0), "weapon_cooldown"
        return None, "missing"

    def _ammo_available(self, obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[bool | None, str]:
        ammo = _number(
            obs.get("ammo"),
            obs.get("ammo_remaining"),
            _mapping(snapshot.get("weapon")).get("ammo"),
            _mapping(snapshot.get("weapon")).get("ammo_remaining"),
        )
        if ammo is None:
            return None, "missing"
        return bool(ammo > 0), "weapon_ammo"

    def _normalize_bin_error(self, bin_error: int) -> float:
        half_turn_bins = max(1.0, float(self.num_aim_bins) / 2.0)
        return float(bin_error) / half_turn_bins


def _normalize_snapshot(state_snapshot: Any | None) -> dict[str, Any]:
    if state_snapshot is None:
        return {}
    if hasattr(state_snapshot, "get_debug_state"):
        return dict(state_snapshot.get_debug_state())
    if isinstance(state_snapshot, Mapping):
        return dict(state_snapshot)
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _self_position(obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, float] | None:
    agents = _mapping(snapshot.get("agents"))
    return _position_or_none(
        _mapping(agents.get("self")).get("position"),
        obs.get("self_pos"),
        _mapping(snapshot.get("state")).get("self_pos"),
    )


def _enemy_position(obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, float] | None:
    agents = _mapping(snapshot.get("agents"))
    return _position_or_none(
        _mapping(agents.get("enemy")).get("position"),
        obs.get("enemy_pos"),
        _mapping(snapshot.get("state")).get("enemy_pos"),
    )


def _enemy_exists(
    obs: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    enemy_pos: Mapping[str, float] | None,
) -> bool:
    agents = _mapping(snapshot.get("agents"))
    enemy_agent = _mapping(agents.get("enemy"))
    hp = _number(obs.get("enemy_hp"), enemy_agent.get("hp"), _mapping(snapshot.get("state")).get("enemy_hp"))
    if hp is not None:
        return bool(hp > 0.0)
    alive = _bool_or_none(enemy_agent.get("alive"))
    if alive is not None:
        return alive
    target_enemy_id = _mapping(snapshot.get("aim_debug")).get("target_enemy_id")
    if target_enemy_id is not None:
        return True
    return enemy_pos is not None


def _position_or_none(*values: Any) -> dict[str, float] | None:
    for value in values:
        if isinstance(value, Mapping) and "x" in value and "y" in value:
            return {"x": float(value["x"]), "y": float(value["y"])}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            return {"x": float(value[0]), "y": float(value[1])}
    return None


def _number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_or_none(*values: Any) -> int | None:
    number = _number(*values)
    return None if number is None else int(number)


def _bool_or_none(*values: Any) -> bool | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
    return None


def _distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _obstacles(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    obstacles = list(snapshot.get("obstacles", []) or [])
    obstacles.extend(_mapping(snapshot.get("map")).get("obstacles", []) or [])
    return [obstacle for obstacle in obstacles if isinstance(obstacle, Mapping)]


def _segment_intersects_circle(
    start: Mapping[str, float],
    target: Mapping[str, float],
    center: Mapping[str, float],
    radius: float,
) -> bool:
    sx = float(start["x"])
    sy = float(start["y"])
    tx = float(target["x"])
    ty = float(target["y"])
    cx = float(center["x"])
    cy = float(center["y"])
    dx = tx - sx
    dy = ty - sy
    length_sq = (dx * dx) + (dy * dy)
    if length_sq <= 1e-12:
        return math.hypot(sx - cx, sy - cy) <= float(radius)
    t = max(0.0, min(1.0, (((cx - sx) * dx) + ((cy - sy) * dy)) / length_sq))
    closest_x = sx + (t * dx)
    closest_y = sy + (t * dy)
    return math.hypot(closest_x - cx, closest_y - cy) <= float(radius)


__all__ = ["FireRule"]
