from __future__ import annotations

import math

from core.schema import (
    BattleAction,
    BattleActionBody,
    AgentId,
    TacticalObservation,
    SCHEMA_VERSION,
)


def _normalize_xy(x: float, y: float) -> tuple[float, float]:
    norm = math.sqrt(x * x + y * y)
    if norm < 1e-6:
        return 0.0, 0.0
    return x / norm, y / norm


def no_op_action(
    episode_id: str,
    step: int,
    agent_id: AgentId,
) -> BattleAction:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": step,
        "agent_id": agent_id,
        "action": {
            "move_x": 0.0,
            "move_y": 0.0,
            "aim_x": 1.0,
            "aim_y": 0.0,
            "fire": 0.0,
        },
        "source": {"policy_type": "random", "policy_id": "no-op-v0"},
    }


def simple_chase_fire_policy(
    obs: TacticalObservation,
) -> BattleActionBody:
    """Scripted baseline for non-learning agents.

    Uses schema observation only.
    """
    enemies = [e for e, m in zip(obs["visible_enemies"], obs["visible_enemies_mask"]) if m and e["alive"]]

    if not obs["self"]["alive"]:
        return {
            "move_x": 0.0,
            "move_y": 0.0,
            "aim_x": 1.0,
            "aim_y": 0.0,
            "fire": 0.0,
        }

    if not enemies:
        return {
            "move_x": 0.0,
            "move_y": 0.0,
            "aim_x": 1.0,
            "aim_y": 0.0,
            "fire": 0.0,
        }

    nearest = min(enemies, key=lambda e: e["distance"])
    dx = float(nearest["relative_position"]["x"])
    dy = float(nearest["relative_position"]["y"])
    aim_x, aim_y = _normalize_xy(dx, dy)

    # Move toward enemy if far, stop if close enough.
    if nearest["distance"] > 180.0:
        move_x, move_y = aim_x, aim_y
    else:
        move_x, move_y = 0.0, 0.0

    has_los = bool(nearest.get("has_line_of_sight", True))
    fire = 1.0 if has_los and nearest["distance"] <= 260.0 else 0.0

    return {
        "move_x": move_x,
        "move_y": move_y,
        "aim_x": aim_x,
        "aim_y": aim_y,
        "fire": fire,
    }


def build_scripted_action(
    obs: TacticalObservation,
) -> BattleAction:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": obs["episode_id"],
        "step": obs["step"],
        "agent_id": obs["agent_id"],
        "action": simple_chase_fire_policy(obs),
        "source": {
            "policy_type": "linear_model",
            "policy_id": "simple-chase-fire-v0",
        },
    }
