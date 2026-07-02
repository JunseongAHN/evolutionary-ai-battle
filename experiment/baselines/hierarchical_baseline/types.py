from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineConfig:
    detection_range: float = 360.0
    combat_exit_grace_steps: int = 3
    weapon_range: float = 260.0
    ideal_range_ratio: float = 0.7
    low_hp_ratio: float = 0.35
    fire_aim_error_threshold: float = 0.15
    cell_size: float = 40.0
    obstacle_inflation_cells: int = 1
    mode_lock_steps: int = 10
    anchor_lock_steps: int = 14
    engage_range_ratio: float = 1.05
    hold_range_min_ratio: float = 0.65
    hold_range_max_ratio: float = 0.95
    kite_range_ratio: float = 0.45
    extreme_close_ratio: float = 0.30
    critical_hp_ratio: float = 0.20
    outer_range_min_ratio: float = 0.90
    outer_range_max_ratio: float = 0.98
    backoff_range_ratio: float = 0.88
    backoff_max_ratio: float = 1.05
    strafe_lock_steps: int = 14
    range_hysteresis_steps: int = 5
    bullet_dodge_steps: int = 2
    bullet_threat_cross_track: float = 24.0
    bullet_safety_margin: float = 30.0
    bullet_radius: float = 12.0
    bullet_prediction_horizon_steps: int = 3
    move_step_distance: float = 20.0
    player_radius: float = 12.0
    combat_movement_profile: str = "default"
    poke_enter_ratio: float = 0.98
    poke_exit_ratio: float = 1.15
    poke_exit_margin: float = 40.0
    poke_exit_lock_steps: int = 4
    poke_fire_while_exiting: bool = True
    poke_exit_uses_bullet_velocity: bool = True


@dataclass(frozen=True)
class EnemyInfo:
    enemy_id: str
    position: tuple[float, float]
    hp: float
    alive: bool


@dataclass(frozen=True)
class GlobalPlan:
    goal_pos: tuple[float, float]
    goal_reached_count: int
    waypoints: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AgentContext:
    player_pos: tuple[float, float]
    player_hp: float
    player_alive: bool
    goal_pos: tuple[float, float] | None
    goal_reached_count: int
    nearest_enemy: EnemyInfo | None
    enemy_dist: float | None
    enemy_in_range: bool
    enemy_in_detection_range: bool
    line_of_sight: bool
    weapon_range: float
    cooldown_ready: bool
    bullet_count: int
    incoming_bullet: bool
    events: tuple[dict[str, Any], ...]
    local_grid: Any | None
    obstacles: tuple[dict[str, Any], ...]
    incoming_bullet_position: tuple[float, float] | None = None
    incoming_bullet_velocity: tuple[float, float] | None = None
    incoming_bullet_radius: float | None = None
    map_width: float | None = None
    map_height: float | None = None
    player_radius: float = 12.0
    incoming_bullets: tuple[dict[str, Any], ...] = ()
    env_dt: float = 1.0


@dataclass(frozen=True)
class LocalPlan:
    intent: str
    tactical_mode: str | None
    combat_profile: str | None
    anchor: tuple[float, float] | None
    target_cell: tuple[int, int] | None
    next_cell: tuple[int, int] | None
    path: tuple[tuple[int, int], ...]
    move_bin: int


@dataclass(frozen=True)
class Control:
    move_bin: int
    aim_dx: float
    aim_dy: float
    fire: int


@dataclass
class AgentState:
    global_plan: GlobalPlan | None = None
    agent_mode: str = "IDLE"
    previous_intent: str | None = None
    previous_tactical_mode: str | None = None
    previous_target_cell: tuple[int, int] | None = None
    previous_anchor: tuple[float, float] | None = None
    last_goal_reached_count: int = 0
    combat_steps: int = 0
    no_enemy_steps: int = 0
    tactical_mode_age: int = 0
    anchor_age: int = 0
    previous_local_plan: LocalPlan | None = None
    strafe_direction: int = 1
    strafe_age: int = 0
    dodge_lock_steps_remaining: int = 0
    dodge_lock_move_bin: int | None = None
    combat_stay_steps: int = 0
    poke_state: str | None = None
    poke_state_age: int = 0
    poke_exit_lock_steps_remaining: int = 0


def default_config() -> BaselineConfig:
    return BaselineConfig()


def default_agent_state() -> AgentState:
    return AgentState()


__all__ = [
    "AgentContext",
    "AgentState",
    "BaselineConfig",
    "Control",
    "EnemyInfo",
    "GlobalPlan",
    "LocalPlan",
    "default_agent_state",
    "default_config",
]
