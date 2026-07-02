from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

try:
    from experiment.core.local_occupancy_grid import build_local_occupancy_grid
except ModuleNotFoundError:
    from core.local_occupancy_grid import build_local_occupancy_grid

from ..types import AgentContext, AgentState, BaselineConfig, EnemyInfo


def build_context(
    obs: Any,
    snapshot: Any,
    state: AgentState,
    config: BaselineConfig,
    target_enemy_id: str | None = None,
) -> tuple[AgentContext, dict[str, Any]]:
    del state
    observation = dict(obs) if isinstance(obs, Mapping) else {}
    world = _snapshot(snapshot)
    player_pos, player_hp, player_alive = _extract_player(observation, world)
    goal_pos, goal_count = _extract_goal(observation, world)
    nearest_enemy = _find_nearest_enemy(observation, world, player_pos, target_enemy_id)
    enemy_dist = (
        math.dist(player_pos, nearest_enemy.position)
        if nearest_enemy is not None and nearest_enemy.alive
        else None
    )
    weapon_range = _number(
        observation.get("weapon_range"),
        observation.get("fire_range"),
        _mapping(world.get("combat")).get("fire_range"),
    ) or config.weapon_range
    obstacles = _extract_obstacles(world)
    line_of_sight = _line_of_sight(player_pos, nearest_enemy, obstacles)
    bullets = _extract_bullets(world)
    events = tuple(deepcopy(_extract_events(observation, world)))
    local_grid = _extract_local_grid(observation, world)
    cooldown_ready = _bool(
        observation.get("can_fire"),
        observation.get("cooldown_ready"),
        default=float(_mapping(world.get("weapon")).get("cooldown_remaining_steps", 0.0)) <= 0.0,
    )
    incoming_bullets = _find_incoming_bullets(
        bullets,
        player_pos,
        max_cross_track=(
            config.bullet_threat_cross_track
            + config.move_step_distance * config.bullet_prediction_horizon_steps
        ),
        default_bullet_radius=config.bullet_radius,
    )
    incoming_bullet = incoming_bullets[0] if incoming_bullets else None
    map_info = _mapping(world.get("map"))
    player_info = _mapping(world.get("player"))
    player_agent = _mapping(_mapping(world.get("agents")).get("self"))
    last_enemy_spawn = next(
        (
            event
            for event in reversed(events)
            if event.get("type") == "bullet_spawned"
            and event.get("owner_id") == "enemy"
        ),
        {},
    )
    context = AgentContext(
        player_pos=player_pos,
        player_hp=player_hp,
        player_alive=player_alive,
        goal_pos=goal_pos,
        goal_reached_count=goal_count,
        nearest_enemy=nearest_enemy,
        enemy_dist=enemy_dist,
        enemy_in_range=bool(enemy_dist is not None and enemy_dist <= weapon_range),
        enemy_in_detection_range=bool(enemy_dist is not None and enemy_dist <= config.detection_range),
        line_of_sight=line_of_sight,
        weapon_range=float(weapon_range),
        cooldown_ready=cooldown_ready,
        bullet_count=len(bullets),
        incoming_bullet=incoming_bullet is not None,
        events=events,
        local_grid=local_grid,
        obstacles=tuple(deepcopy(obstacles)),
        incoming_bullet_position=(
            incoming_bullet["position"] if incoming_bullet is not None else None
        ),
        incoming_bullet_velocity=(
            incoming_bullet["velocity"] if incoming_bullet is not None else None
        ),
        incoming_bullet_radius=(
            incoming_bullet["radius"] if incoming_bullet is not None else None
        ),
        map_width=_number(map_info.get("width")),
        map_height=_number(map_info.get("height")),
        player_radius=float(
            _number(player_info.get("radius"), player_agent.get("radius"))
            or config.player_radius
        ),
        incoming_bullets=tuple(deepcopy(incoming_bullets)),
        env_dt=float(_number(world.get("dt")) or 1.0),
    )
    return context, {
        "player_pos": list(player_pos),
        "goal_pos": list(goal_pos) if goal_pos is not None else None,
        "goal_reached_count": goal_count,
        "enemy_id": nearest_enemy.enemy_id if nearest_enemy is not None else None,
        "enemy_pos": list(nearest_enemy.position) if nearest_enemy is not None else None,
        "enemy_dist": enemy_dist,
        "enemy_in_detection_range": context.enemy_in_detection_range,
        "line_of_sight": line_of_sight,
        "bullet_count": len(bullets),
        "incoming_bullet": context.incoming_bullet,
        "incoming_bullet_position": (
            list(context.incoming_bullet_position)
            if context.incoming_bullet_position is not None
            else None
        ),
        "incoming_bullet_velocity": (
            list(context.incoming_bullet_velocity)
            if context.incoming_bullet_velocity is not None
            else None
        ),
        "incoming_bullet_radius": context.incoming_bullet_radius,
        "incoming_bullet_count": len(context.incoming_bullets),
        "map_size": [context.map_width, context.map_height],
        "player_radius": context.player_radius,
        "env_dt": context.env_dt,
        "enemy_aim_noise_deg": float(
            _number(
                world.get("enemy_aim_noise_deg"),
                _mapping(world.get("combat")).get("enemy_aim_noise_deg"),
            )
            or 0.0
        ),
        "applied_enemy_aim_noise_rad": _number(
            last_enemy_spawn.get("applied_enemy_aim_noise_rad")
        ),
        "event_types": [event.get("type", event.get("event_type")) for event in events],
    }


def _snapshot(value: Any) -> dict[str, Any]:
    if hasattr(value, "get_debug_state"):
        return dict(value.get_debug_state())
    return dict(value) if isinstance(value, Mapping) else {}


def _extract_player(obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[tuple[float, float], float, bool]:
    player = _mapping(snapshot.get("player"))
    agent = _mapping(_mapping(snapshot.get("agents")).get("self"))
    state = _mapping(snapshot.get("state"))
    position = _position(obs.get("self_pos"), player.get("position"), agent.get("position"), state.get("self_pos")) or (0.0, 0.0)
    hp = _number(obs.get("self_hp"), player.get("hp"), agent.get("hp"), state.get("self_hp")) or 0.0
    alive = _bool(player.get("alive"), agent.get("alive"), default=hp > 0.0)
    return position, float(hp), alive


def _extract_goal(obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[tuple[float, float] | None, int]:
    goal = _mapping(snapshot.get("goal"))
    enabled = _bool(obs.get("goal_enabled"), goal.get("enabled"), default=False)
    position = _position(obs.get("goal_position"), goal.get("position")) if enabled else None
    count = int(_number(obs.get("goal_reached_count"), goal.get("reached_count")) or 0)
    return position, count


def _find_nearest_enemy(
    obs: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    player_pos: tuple[float, float],
    target_enemy_id: str | None = None,
) -> EnemyInfo | None:
    candidates: list[EnemyInfo] = []
    for index, enemy in enumerate(snapshot.get("enemies", []) or []):
        if isinstance(enemy, Mapping):
            parsed = _enemy_info(enemy, str(enemy.get("id", f"enemy-{index}")))
            if parsed is not None:
                candidates.append(parsed)
    agent = _mapping(_mapping(snapshot.get("agents")).get("enemy"))
    if agent:
        parsed = _enemy_info(agent, str(agent.get("id", "enemy")))
        if parsed is not None:
            candidates.append(parsed)
    obs_position = _position(obs.get("enemy_pos"))
    if obs_position is not None:
        hp = float(_number(obs.get("enemy_hp")) or 0.0)
        candidates.append(EnemyInfo("enemy", obs_position, hp, hp > 0.0))
    alive = [
        enemy
        for enemy in candidates
        if enemy.alive and (target_enemy_id is None or enemy.enemy_id == target_enemy_id)
    ]
    if not alive:
        return None
    return min(alive, key=lambda enemy: (math.dist(player_pos, enemy.position), enemy.enemy_id))


def _enemy_info(value: Mapping[str, Any], fallback_id: str) -> EnemyInfo | None:
    position = _position(value.get("position"), value.get("pos"))
    if position is None:
        return None
    hp = float(_number(value.get("hp")) or 0.0)
    alive = _bool(value.get("alive"), default=hp > 0.0)
    return EnemyInfo(str(value.get("id", fallback_id)), position, hp, alive)


def _extract_obstacles(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    obstacles = list(snapshot.get("obstacles", []) or [])
    obstacles.extend(_mapping(snapshot.get("map")).get("obstacles", []) or [])
    unique: dict[str, dict[str, Any]] = {}
    for index, obstacle in enumerate(obstacles):
        if isinstance(obstacle, Mapping):
            key = str(obstacle.get("id", index))
            unique[key] = dict(obstacle)
    return list(unique.values())


def _line_of_sight(start: tuple[float, float], enemy: EnemyInfo | None, obstacles: list[dict[str, Any]]) -> bool:
    if enemy is None:
        return False
    return not any(_segment_hits_circle(start, enemy.position, obstacle) for obstacle in obstacles)


def _segment_hits_circle(start: tuple[float, float], end: tuple[float, float], obstacle: Mapping[str, Any]) -> bool:
    if str(obstacle.get("type", "circle")) != "circle":
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return False
    cx, cy = float(obstacle.get("x", 0.0)), float(obstacle.get("y", 0.0))
    t = max(0.0, min(1.0, ((cx - start[0]) * dx + (cy - start[1]) * dy) / length_sq))
    return math.hypot(start[0] + t * dx - cx, start[1] + t * dy - cy) <= float(obstacle.get("radius", 0.0))


def _extract_bullets(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    bullets = snapshot.get("projectiles", snapshot.get("bullets", [])) or []
    return [dict(bullet) for bullet in bullets if isinstance(bullet, Mapping)]


def _find_incoming_bullets(
    bullets: list[dict[str, Any]],
    player_pos: tuple[float, float],
    *,
    max_cross_track: float,
    default_bullet_radius: float,
) -> list[dict[str, Any]]:
    threats: list[tuple[float, float, str, dict[str, Any]]] = []
    for bullet in bullets:
        if (
            bullet.get("owner_id") in {None, "self"}
            or bullet.get("team") == "player"
            or not bool(bullet.get("alive", True))
        ):
            continue
        position = _position(bullet.get("position"), bullet.get("pos"))
        velocity = _position(bullet.get("velocity"))
        if velocity is None:
            direction = _position(bullet.get("direction"))
            speed = float(_number(bullet.get("speed")) or 0.0)
            velocity = None if direction is None else (direction[0] * speed, direction[1] * speed)
        if position is None or velocity is None:
            continue
        speed = math.hypot(*velocity)
        if speed <= 1e-6:
            continue
        vx, vy = velocity[0] / speed, velocity[1] / speed
        rx, ry = player_pos[0] - position[0], player_pos[1] - position[1]
        along_track = rx * vx + ry * vy
        cross_track = abs(rx * (-vy) + ry * vx)
        if along_track <= 0.0 or cross_track > max(0.0, float(max_cross_track)):
            continue
        parsed = {
            "bullet_id": str(bullet.get("bullet_id", "")),
            "position": position,
            "velocity": velocity,
            "radius": float(_number(bullet.get("radius")) or default_bullet_radius),
            "along_track": along_track,
            "cross_track": cross_track,
        }
        threats.append(
            (
                along_track / speed,
                cross_track,
                str(bullet.get("bullet_id", "")),
                parsed,
            )
        )
    return [item[3] for item in sorted(threats, key=lambda item: item[:3])]


def _extract_events(obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = snapshot.get("events", obs.get("recent_events", [])) or []
    return [dict(event) for event in values if isinstance(event, Mapping)]


def _extract_local_grid(obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> Any | None:
    for key in ("local_occupancy_grid", "local_grid", "occupancy_grid", "grid"):
        if obs.get(key) is not None:
            return obs[key]
    grid_snapshot: Mapping[str, Any] = snapshot
    if not snapshot.get("agents") and snapshot.get("player"):
        player = _mapping(snapshot.get("player"))
        agents: dict[str, dict[str, Any]] = {
            "self": {
                "role": "self",
                "position": _position_mapping(player.get("position")),
                "hp": float(player.get("hp", 0.0)),
                "alive": bool(player.get("alive", False)),
            }
        }
        for index, enemy in enumerate(snapshot.get("enemies", []) or []):
            if isinstance(enemy, Mapping):
                agents[f"enemy-{index}"] = {
                    "role": "enemy",
                    "position": _position_mapping(enemy.get("position")),
                    "hp": float(enemy.get("hp", 0.0)),
                    "alive": bool(enemy.get("alive", False)),
                    "radius": float(enemy.get("radius", 12.0)),
                }
        projectiles = [
            {
                **dict(bullet),
                "pos": _position_mapping(bullet.get("position", bullet.get("pos"))),
                "position": _position_mapping(bullet.get("position", bullet.get("pos"))),
                "radius": float(bullet.get("radius", 6.0)),
            }
            for bullet in snapshot.get("bullets", snapshot.get("projectiles", [])) or []
            if isinstance(bullet, Mapping)
        ]
        grid_snapshot = {
            **snapshot,
            "agents": agents,
            "bullets": projectiles,
            "projectiles": projectiles,
        }
    if grid_snapshot.get("agents"):
        try:
            return build_local_occupancy_grid(grid_snapshot, agent_id="self")
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _position_mapping(value: Any) -> dict[str, float]:
    position = _position(value) or (0.0, 0.0)
    return {"x": position[0], "y": position[1]}


def _position(*values: Any) -> tuple[float, float] | None:
    for value in values:
        if isinstance(value, Mapping) and "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _bool(*values: Any, default: bool) -> bool:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
    return bool(default)


__all__ = ["build_context"]
