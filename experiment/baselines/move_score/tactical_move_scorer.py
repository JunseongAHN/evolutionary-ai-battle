from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .move_candidate_utils import iter_move_candidates, simulate_candidate_position
from .move_score_terms import (
    enemy_spacing_score,
    enemy_threat_penalty,
    map_boundary_penalty,
    obstacle_path_collision_penalty,
    strafe_score,
)


@dataclass(frozen=True)
class MoveScoringContext:
    self_pos: dict[str, float]
    enemy_pos: dict[str, float] | None
    map_width: float
    map_height: float
    self_radius: float
    move_speed: float
    dt: float
    fire_range: float
    obstacles: list[Mapping[str, Any]]


class TacticalMoveScorer:
    """Deterministic candidate-scoring movement baseline; not a movement oracle."""

    def __init__(
        self,
        ideal_range_ratio: float = 0.7,
        collision_penalty: float = 1000.0,
        boundary_penalty: float = 1000.0,
        spacing_weight: float = 1.0,
        threat_weight: float = 2.0,
        strafe_weight: float = 1.0,
    ) -> None:
        self.ideal_range_ratio = float(ideal_range_ratio)
        self.collision_penalty = float(collision_penalty)
        self.boundary_penalty = float(boundary_penalty)
        self.spacing_weight = float(spacing_weight)
        self.threat_weight = float(threat_weight)
        self.strafe_weight = float(strafe_weight)

    def choose_move(self, obs: Mapping[str, Any], state_snapshot: Any | None = None) -> tuple[int, dict[str, Any]]:
        context = _build_context(obs, state_snapshot)
        candidate_scores: dict[int, dict[str, Any]] = {}
        ideal_range = max(0.0, context.fire_range * self.ideal_range_ratio)
        enemy_threat_range = max(context.self_radius * 2.0, context.fire_range * 0.25)

        for move_bin, move_vector in iter_move_candidates():
            candidate_x, candidate_y = simulate_candidate_position(
                context.self_pos["x"],
                context.self_pos["y"],
                move_vector[0],
                move_vector[1],
                context.move_speed,
                context.dt,
            )
            candidate_pos = {"x": candidate_x, "y": candidate_y}
            collision_term = obstacle_path_collision_penalty(
                context.self_pos,
                candidate_pos,
                context.obstacles,
                context.self_radius,
                self.collision_penalty,
            )
            boundary_term = map_boundary_penalty(
                candidate_pos,
                context.map_width,
                context.map_height,
                context.self_radius,
                self.boundary_penalty,
            )
            spacing_term = 0.0
            threat_term = 0.0
            strafe_term = 0.0
            if context.enemy_pos is not None:
                spacing_term = enemy_spacing_score(
                    candidate_pos,
                    context.enemy_pos,
                    ideal_range,
                    self.spacing_weight,
                )
                threat_term = enemy_threat_penalty(
                    candidate_pos,
                    context.enemy_pos,
                    enemy_threat_range,
                    self.threat_weight,
                )
                current_enemy_distance = _distance(context.self_pos, context.enemy_pos)
                if current_enemy_distance <= context.fire_range:
                    strafe_term = strafe_score(
                        move_vector,
                        context.self_pos,
                        context.enemy_pos,
                        self.strafe_weight,
                    )

            total = collision_term + boundary_term + spacing_term + threat_term + strafe_term
            candidate_scores[move_bin] = {
                "total": float(total),
                "collision_penalty": float(collision_term),
                "boundary_penalty": float(boundary_term),
                "spacing_score": float(spacing_term),
                "threat_penalty": float(threat_term),
                "strafe_score": float(strafe_term),
                "candidate_pos": [float(candidate_x), float(candidate_y)],
                "move_vector": [float(move_vector[0]), float(move_vector[1])],
            }

        selected_move_bin = max(sorted(candidate_scores), key=lambda move_bin: candidate_scores[move_bin]["total"])
        debug = {
            "selected_move_bin": int(selected_move_bin),
            "candidate_scores": candidate_scores,
            "reason": _reason(candidate_scores[selected_move_bin], context.enemy_pos is not None),
            "context": {
                "self_pos": [context.self_pos["x"], context.self_pos["y"]],
                "enemy_pos": (
                    [context.enemy_pos["x"], context.enemy_pos["y"]]
                    if context.enemy_pos is not None
                    else None
                ),
                "ideal_range": float(ideal_range),
                "enemy_threat_range": float(enemy_threat_range),
                "move_speed": float(context.move_speed),
                "dt": float(context.dt),
            },
        }
        return int(selected_move_bin), debug


class TacticalMoveScoreBot:
    def __init__(self, move_scorer: TacticalMoveScorer | None = None, default_aim_bin: int = 0) -> None:
        self.move_scorer = move_scorer or TacticalMoveScorer()
        self.default_aim_bin = int(default_aim_bin)

    def act(self, obs: Mapping[str, Any], state_snapshot: Any | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        selected_move_bin, debug = self.move_scorer.choose_move(obs, state_snapshot=state_snapshot)
        action = {
            "move": int(selected_move_bin),
            "aim": self.default_aim_bin,
            "fire": 0,
        }
        debug["action"] = dict(action)
        return action, debug


def _build_context(obs: Mapping[str, Any], state_snapshot: Any | None) -> MoveScoringContext:
    snapshot = _normalize_snapshot(state_snapshot)
    agents = _mapping(snapshot.get("agents"))
    self_agent = _mapping(agents.get("self"))
    enemy_agent = _mapping(agents.get("enemy"))
    map_data = _mapping(snapshot.get("map"))
    combat = _mapping(snapshot.get("combat"))

    self_pos = _position(
        self_agent.get("position"),
        obs.get("self_pos"),
        _mapping(snapshot.get("state")).get("self_pos"),
        default={"x": 0.0, "y": 0.0},
    )
    enemy_pos = _enemy_position(obs, snapshot, enemy_agent)

    return MoveScoringContext(
        self_pos=self_pos,
        enemy_pos=enemy_pos,
        map_width=float(map_data.get("width", obs.get("map_width", 1000.0))),
        map_height=float(map_data.get("height", obs.get("map_height", 1000.0))),
        self_radius=float(self_agent.get("radius", obs.get("self_radius", 12.0))),
        move_speed=float(self_agent.get("move_speed", obs.get("move_speed", 35.0))),
        dt=float(snapshot.get("dt", obs.get("dt", 1.0))),
        fire_range=float(combat.get("fire_range", obs.get("fire_range", 280.0))),
        obstacles=list(map_data.get("obstacles", snapshot.get("obstacles", [])) or []),
    )


def _normalize_snapshot(state_snapshot: Any | None) -> dict[str, Any]:
    if state_snapshot is None:
        return {}
    if hasattr(state_snapshot, "get_debug_state"):
        return dict(state_snapshot.get_debug_state())
    if isinstance(state_snapshot, Mapping):
        return dict(state_snapshot)
    return {}


def _enemy_position(
    obs: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    enemy_agent: Mapping[str, Any],
) -> dict[str, float] | None:
    enemy_hp = obs.get("enemy_hp")
    if enemy_hp is None:
        enemy_hp = enemy_agent.get("hp", _mapping(snapshot.get("state")).get("enemy_hp"))
    if enemy_hp is not None and float(enemy_hp) <= 0.0:
        return None
    value = _position_or_none(
        enemy_agent.get("position"),
        obs.get("enemy_pos"),
        _mapping(snapshot.get("state")).get("enemy_pos"),
    )
    return value


def _position(*values: Any, default: Mapping[str, float]) -> dict[str, float]:
    result = _position_or_none(*values)
    if result is None:
        return {"x": float(default["x"]), "y": float(default["y"])}
    return result


def _position_or_none(*values: Any) -> dict[str, float] | None:
    for value in values:
        if isinstance(value, Mapping) and "x" in value and "y" in value:
            return {"x": float(value["x"]), "y": float(value["y"])}
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return {"x": float(value[0]), "y": float(value[1])}
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _reason(selected: Mapping[str, Any], has_enemy: bool) -> str:
    if float(selected["collision_penalty"]) < 0.0:
        return "least bad candidate still collides with an obstacle"
    if float(selected["boundary_penalty"]) < 0.0:
        return "least bad candidate still leaves the map boundary"
    if not has_enemy:
        return "no live enemy available; selected safest deterministic move"
    if float(selected["strafe_score"]) > 0.0:
        return "selected highest score with strafe preference"
    return "selected highest score from collision, boundary, and spacing terms"
