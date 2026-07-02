from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from .cpc_actions import (
        AIM_BINS,
        RawAction,
        aim_bin_to_vec,
        circular_bin_distance,
        decode_action,
        random_action,
        vec_to_aim_bin,
    )
    from .cpc_metrics import CpcMetrics
else:
    from cpc_actions import (
        AIM_BINS,
        RawAction,
        aim_bin_to_vec,
        circular_bin_distance,
        decode_action,
        random_action,
        vec_to_aim_bin,
    )
    from cpc_metrics import CpcMetrics

if __package__ in {"experiment.core", "core"}:
    from .env_config import EnvConfig, load_env_config
elif __package__:
    from experiment.core.env_config import EnvConfig, load_env_config
else:
    try:
        from experiment.core.env_config import EnvConfig, load_env_config
    except ModuleNotFoundError:
        from env_config import EnvConfig, load_env_config


Vec2 = dict[str, float]


def normalize_vec(v: Vec2) -> Vec2:
    length = math.hypot(float(v["x"]), float(v["y"]))
    if length <= 1e-6:
        return {"x": 0.0, "y": 0.0}
    return {"x": float(v["x"]) / length, "y": float(v["y"]) / length}


def dot(a: Vec2, b: Vec2) -> float:
    return (float(a["x"]) * float(b["x"])) + (float(a["y"]) * float(b["y"]))


def angle_between(a: Vec2, b: Vec2) -> float:
    a_norm = normalize_vec(a)
    b_norm = normalize_vec(b)
    value = max(-1.0, min(1.0, dot(a_norm, b_norm)))
    return math.degrees(math.acos(value))


class CPCEnv:
    stage = "local_combat"
    width = 1000.0
    height = 1000.0
    max_hp = 100.0
    move_speed = 35.0
    fire_range = 280.0
    fire_alignment = 0.65
    fire_interval_steps = 5
    projectile_speed = 140.0
    projectile_radius = 12.0
    damage = 10.0
    enemy_damage = 2.0
    enemy_move_speed = 18.0
    survival_reward = 0.001
    center = {"x": 500.0, "y": 500.0}
    safe_radius_start = 420.0
    safe_radius_end = 420.0
    shrink_safe_zone = False
    use_zone_reward = False
    aim_alignment_threshold = 0.75
    damage_dealt_ratio_weight = 1.0
    damage_taken_ratio_weight = -1.2
    bullet_hit_bonus = 0.05
    missed_shot_penalty = 0.03
    aim_bin_exact_bonus = 0.04
    aim_bin_wrong_penalty = 0.04
    good_range_bonus = 0.01
    too_close_penalty = 0.03
    too_far_penalty = 0.01
    kill_bonus = 1.0
    death_penalty = 1.0
    timeout_hp_lead_weight = 0.5
    accuracy_bonus_weight = 0.2
    aim_alignment_weight = 0.0
    aim_bin_neighbor_bonus = 0.0
    bad_aim_penalty = 0.12
    attack_intent_bonus = 0.02
    aligned_shot_bonus = 0.0
    near_aligned_shot_bonus = 0.0
    off_target_shot_penalty = 0.0
    zone_pressure_penalty = 0.20
    return_to_zone_bonus = 0.06
    move_deeper_outside_zone_penalty = 0.08
    near_edge_outward_penalty = 0.04
    default_enemy_spawn_directions = (
        "right",
        "left",
        "up",
        "down",
        "upper_right",
        "lower_right",
        "upper_left",
        "lower_left",
    )
    enemy_spawn_direction_angles = {
        "right": 0.0,
        "left": math.pi,
        "up": -math.pi / 2.0,
        "down": math.pi / 2.0,
        "upper_right": -math.pi / 4.0,
        "lower_right": math.pi / 4.0,
        "upper_left": -3.0 * math.pi / 4.0,
        "lower_left": 3.0 * math.pi / 4.0,
    }

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 50,
        randomize_enemy_spawn_direction: bool = False,
        enemy_spawn_directions: list[str] | tuple[str, ...] | None = None,
        enemy_spawn_direction: str | None = None,
        stage: str = "local_combat",
        shrink_safe_zone: bool = False,
        use_zone_reward: bool = False,
        enemy_move: bool = True,
        enemy_fire: bool = True,
        enemy_aim_noise_deg: float = 0.0,
        stationary_target_mode: bool = False,
        enemy_spawn_distance_min: float | None = None,
        enemy_spawn_distance_max: float | None = None,
        fire_interval_steps: int | None = None,
        bullet_speed: float | None = None,
        bullet_range: float | None = None,
        bullet_damage: float | None = None,
        bullet_hit_radius: float | None = None,
        config: EnvConfig | str | Path | None = None,
    ):
        self.env_config = self._load_config(config)
        if self.env_config is not None:
            if seed == 0:
                seed = self.env_config.seed
            if max_steps == 50:
                max_steps = self.env_config.max_steps
        self.seed = seed
        self.max_steps = max_steps
        self.stage = stage
        self.shrink_safe_zone = bool(shrink_safe_zone)
        self.use_zone_reward = bool(use_zone_reward)
        self.enemy_move = bool(enemy_move)
        self.enemy_fire = bool(enemy_fire)
        self.enemy_aim_noise_deg = max(0.0, float(enemy_aim_noise_deg))
        self.stationary_target_mode = bool(stationary_target_mode)
        self.enemy_spawn_distance_min = (
            float(enemy_spawn_distance_min) if enemy_spawn_distance_min is not None else None
        )
        self.enemy_spawn_distance_max = (
            float(enemy_spawn_distance_max) if enemy_spawn_distance_max is not None else None
        )
        if fire_interval_steps is not None:
            self.fire_interval_steps = int(fire_interval_steps)
        if bullet_speed is not None:
            self.projectile_speed = float(bullet_speed)
        if bullet_range is not None:
            self.fire_range = float(bullet_range)
        if bullet_damage is not None:
            self.damage = float(bullet_damage)
        if bullet_hit_radius is not None:
            self.projectile_radius = float(bullet_hit_radius)
        self.randomize_enemy_spawn_direction = bool(randomize_enemy_spawn_direction)
        self.enemy_spawn_directions = self._validate_spawn_directions(enemy_spawn_directions)
        self.enemy_spawn_direction = self._validate_spawn_direction(enemy_spawn_direction)
        self.dt = 1.0
        self.self_max_hp = float(self.max_hp)
        self.ally_max_hp = float(self.max_hp)
        self.enemy_max_hp = float(self.max_hp)
        self.player_radius = 12.0
        self.ally_radius = 12.0
        self.enemy_radius = 12.0
        self.player_spawn: Vec2 | None = None
        self.ally_spawn: Vec2 | None = None
        self.enemy_spawn: Vec2 | None = None
        self.enemy_id = "enemy"
        self.enemy_behavior = "chase"
        self.obstacles: list[dict[str, Any]] = []
        self.goal_enabled = False
        self.goal_config_position: tuple[float, float] | None = None
        self.goal_position: tuple[float, float] | None = None
        self.goal_radius = 24.0
        self.goal_respawn_on_reach = True
        self.goal_spawn_enemy_on_reach = False
        self.goal_respawn_margin = 80.0
        self.goal_max_respawns: int | None = None
        self.goal_reached_count = 0
        self.goal_respawn_count = 0
        self.enemy_spawn_count = 0
        self.last_events: list[dict[str, Any]] = []
        self.aim_turn_speed = 1.0
        if self.env_config is not None:
            self._apply_env_config(self.env_config)
        self.rng = random.Random(seed)
        self.metrics = self._new_metrics()
        self.trajectory: list[dict[str, Any]] = []
        self.step_count = 0
        self.state: dict[str, Any] = {}
        self.projectiles: list[dict[str, Any]] = []
        self.weapon: dict[str, Any] = {}
        self.enemy_weapon: dict[str, Any] = {}
        self.last_aim_debug: dict[str, Any] = {}
        self.last_zone_debug: dict[str, Any] = {}
        self.last_fire_debug: dict[str, Any] = {}
        self.current_aim_vector: dict[str, float] = {"x": 1.0, "y": 0.0}
        self.reset(seed=seed)

    @classmethod
    def from_config(cls, config: EnvConfig | str | Path) -> "CPCEnv":
        return cls(config=config)

    @staticmethod
    def _load_config(config: EnvConfig | str | Path | None) -> EnvConfig | None:
        if config is None:
            return None
        if isinstance(config, EnvConfig) or (
            config.__class__.__name__ == "EnvConfig"
            and all(hasattr(config, attr) for attr in ("seed", "max_steps", "player", "enemies"))
        ):
            return config
        return load_env_config(config)

    def _apply_env_config(self, config: EnvConfig) -> None:
        self.dt = float(config.dt)
        self.width = float(config.map_width)
        self.height = float(config.map_height)
        self.center = (
            {"x": float(config.zone.center.x), "y": float(config.zone.center.y)}
            if config.zone.center is not None
            else {"x": self.width / 2.0, "y": self.height / 2.0}
        )
        self.shrink_safe_zone = bool(config.zone.enabled)
        self.use_zone_reward = bool(config.zone.enabled)
        self.safe_radius_start = float(config.zone.safe_radius_start)
        self.safe_radius_end = float(config.zone.safe_radius_end)

        self.player_spawn = self._vec_from_config(config.player.spawn)
        self.player_radius = float(config.player.radius)
        self.self_max_hp = float(config.player.hp)
        self.max_hp = float(config.player.hp)
        self.move_speed = float(config.player.move_speed)
        self.aim_turn_speed = float(config.player.aim_turn_speed)
        self.fire_range = float(config.player.weapon_range)
        self.fire_interval_steps = int(config.player.fire_cooldown_steps)
        self.projectile_speed = float(config.player.bullet_speed)

        if config.ally is not None:
            self.ally_spawn = self._vec_from_config(config.ally.spawn)
            self.ally_radius = float(config.ally.radius)
            self.ally_max_hp = float(config.ally.hp)
        else:
            self.ally_spawn = None
            self.ally_radius = self.player_radius
            self.ally_max_hp = self.self_max_hp

        if config.enemies:
            enemy = config.enemies[0]
            self.enemy_id = enemy.id
            self.enemy_spawn = self._vec_from_config(enemy.spawn)
            self.enemy_radius = float(enemy.radius)
            self.enemy_max_hp = float(enemy.hp)
            self.enemy_move_speed = float(enemy.move_speed)
            self.enemy_behavior = enemy.behavior
            self.enemy_aim_noise_deg = max(0.0, float(enemy.enemy_aim_noise_deg))
            self.enemy_move = enemy.behavior.strip().lower() not in {"stationary", "none", "passive"}

        self.obstacles = [
            {
                "id": obstacle.id,
                "type": obstacle.type,
                "x": float(obstacle.x),
                "y": float(obstacle.y),
                "radius": float(obstacle.radius),
            }
            for obstacle in config.obstacles
        ]
        self.goal_enabled = bool(config.goal.enabled)
        self.goal_config_position = (
            (float(config.goal.position.x), float(config.goal.position.y))
            if config.goal.position is not None
            else None
        )
        self.goal_radius = max(0.0, float(config.goal.radius))
        self.goal_respawn_on_reach = bool(config.goal.respawn_on_reach)
        self.goal_spawn_enemy_on_reach = bool(config.goal.spawn_enemy_on_reach)
        self.goal_respawn_margin = max(0.0, float(config.goal.respawn_margin))
        self.goal_max_respawns = (
            None if config.goal.max_respawns is None else max(0, int(config.goal.max_respawns))
        )

    def _new_metrics(self) -> CpcMetrics:
        metrics = CpcMetrics()
        metrics.self_max_hp = float(self.self_max_hp)
        metrics.enemy_max_hp = float(self.enemy_max_hp)
        metrics.self_hp = float(self.self_max_hp)
        metrics.enemy_hp = float(self.enemy_max_hp)
        return metrics

    @staticmethod
    def _vec_from_config(value) -> Vec2:
        return {"x": float(value.x), "y": float(value.y)}

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        self.step_count = 0
        self.metrics = self._new_metrics()
        self.trajectory = []
        self.projectiles = []
        self.last_aim_debug = {}
        self.last_zone_debug = {}
        self.last_fire_debug = {}
        self.last_events = []
        self.goal_reached_count = 0
        self.goal_respawn_count = 0
        self.enemy_spawn_count = 0
        self.enemy_id = (
            self.env_config.enemies[0].id
            if self.env_config is not None and self.env_config.enemies
            else "enemy"
        )
        self.current_aim_vector = {"x": 1.0, "y": 0.0}
        self.weapon = {
            "cooldown_remaining_steps": 0,
            "fire_interval_steps": self.fire_interval_steps,
        }
        self.enemy_weapon = {
            "cooldown_remaining_steps": 0,
            "fire_interval_steps": self.fire_interval_steps,
        }
        spawn = self._spawn_positions()
        self.state = {
            "self_hp": self.self_max_hp,
            "ally_hp": self.ally_max_hp,
            "enemy_hp": self.enemy_max_hp,
            "self_pos": spawn["self_pos"],
            "ally_pos": spawn["ally_pos"],
            "enemy_pos": spawn["enemy_pos"],
        }
        self.goal_position = self._initial_goal_position()
        return self.observation()

    def step(self, action: Mapping[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        raw_action: dict[str, int | float] = {
            "move": int(action["move"]),
            "fire": int(action["fire"]),
        }
        for key in ("aim_dx", "aim_dy", "aim_angle", "aim_x", "aim_y"):
            if key in action:
                raw_action[key] = float(action[key])
        if "aim" in action:
            raw_action["aim"] = int(action["aim"])
        decoded = decode_action(raw_action)
        action_blocked_agent_dead = float(self.state.get("self_hp", 0.0)) <= 0.0
        if action_blocked_agent_dead:
            effective_decoded = decode_action({"move": 0, "aim_dx": 1.0, "aim_dy": 0.0, "fire": 0})
        else:
            effective_decoded = self._effective_decoded_action(decoded)
        previous_ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
        previous_enemy_distance = self._distance(self.state["self_pos"], self.state["enemy_pos"])
        previous_enemy_hp = float(self.state["enemy_hp"])
        previous_self_hp = float(self.state["self_hp"])

        if not action_blocked_agent_dead:
            self._move_self(effective_decoded["moveX"], effective_decoded["moveY"])
            self.current_aim_vector = normalize_vec({"x": float(effective_decoded["aimX"]), "y": float(effective_decoded["aimY"])})
        ally_under_pressure = self._ally_under_pressure()
        self._script_enemy_pressure()

        damage_dealt, damage_taken_from_bullets, bullet_events = self._update_bullets(dt=self.dt)
        if action_blocked_agent_dead:
            bullet_events.append(
                {
                    "type": "action_ignored",
                    "reason": "agent_dead",
                    "owner_id": "self",
                    "raw_action": dict(raw_action),
                    "pos": deepcopy(self.state["self_pos"]),
                }
            )
        cooldown_before = int(self.weapon["cooldown_remaining_steps"])
        fire_requested = bool(effective_decoded["fire"])
        enemy_distance = self._distance(self.state["self_pos"], self.state["enemy_pos"])
        range_debug = self._range_debug(enemy_distance)
        aim_debug = self._aim_debug(self.current_aim_vector, effective_decoded)
        fire_debug = self._fire_valid_debug(
            aim_debug=aim_debug,
            range_debug=range_debug,
            cooldown_ready=cooldown_before <= 0,
        )
        if action_blocked_agent_dead:
            fire_debug["agent_alive"] = False
            fire_debug["fire_valid"] = False
            bullet_spawned = False
            fire_blocked_reason = "agent_dead"
            spawn_event = None
        elif self.stationary_target_mode and fire_requested and not fire_debug["fire_valid"]:
            fire_blocked_reason = "cooldown" if not fire_debug["cooldown_ready"] else "invalid_fire"
            bullet_spawned = False
            spawn_event = {
                "type": "bullet_not_spawned",
                "reason": fire_blocked_reason,
                "owner_id": "self",
                "pos": deepcopy(self.state["self_pos"]),
                "cooldown_remaining_steps": int(self.weapon["cooldown_remaining_steps"]),
            }
        else:
            bullet_spawned, fire_blocked_reason, spawn_event = self._try_spawn_bullet(effective_decoded)
        if spawn_event is not None:
            bullet_events.append(spawn_event)
        self._tick_weapon_cooldowns()
        cooldown_after = int(self.weapon["cooldown_remaining_steps"])
        suppress_enemy_projectile = bullet_spawned or any(
            event.get("owner_id") == "self"
            and event.get("type") in {"bullet_hit", "bullet_hit_obstacle", "bullet_expired"}
            for event in bullet_events
        )
        damage_taken_direct, enemy_fire_events = self._resolve_enemy_pressure(
            suppress_projectile=suppress_enemy_projectile
        )
        bullet_events.extend(enemy_fire_events)
        damage_taken = damage_taken_from_bullets + damage_taken_direct
        self.step_count += 1

        ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
        moved_toward_ally = ally_distance < previous_ally_distance
        reward_components = self._reward_components(
            decoded=effective_decoded,
            previous_enemy_distance=previous_enemy_distance,
            enemy_distance=enemy_distance,
            ally_under_pressure=ally_under_pressure,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            previous_enemy_hp=previous_enemy_hp,
            previous_self_hp=previous_self_hp,
            fire_requested=fire_requested,
            can_fire=fire_debug["cooldown_ready"],
            bullet_spawned=bullet_spawned,
            fire_blocked_reason=fire_blocked_reason,
            bullet_events=bullet_events,
            range_debug=range_debug,
            aim_debug=aim_debug,
            fire_debug=fire_debug,
        )
        zone_debug = self._zone_debug(effective_decoded)
        self.last_aim_debug = aim_debug
        self.last_zone_debug = zone_debug
        self.last_fire_debug = fire_debug
        reward = sum(reward_components.values())
        enemy_hp_after_combat = float(self.state["enemy_hp"])
        self.metrics.update(
            ally_distance=ally_distance,
            ally_under_pressure=ally_under_pressure,
            moved_toward_ally=moved_toward_ally,
            fired_under_pressure=fire_requested,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            enemy_hp=float(self.state["enemy_hp"]),
            self_hp=float(self.state["self_hp"]),
            reward=reward,
            reward_components=reward_components,
            fire_requested=fire_requested,
            current_aim_bin=int(aim_debug.get("aim_bin", 0)),
            aim_alignment=aim_debug["aim_alignment"],
            aim_bin=int(aim_debug.get("aim_bin", 0)),
            ideal_aim_bin=int(aim_debug.get("ideal_aim_bin", 0)),
            aim_bin_error=int(aim_debug.get("aim_bin_error", 0)),
            aim_error=float(fire_debug["aim_error"]),
            aim_aligned=bool(fire_debug["aim_aligned"]),
            target_in_range=bool(fire_debug["target_in_range"]),
            cooldown_ready=bool(fire_debug["cooldown_ready"]),
            fire_valid=bool(fire_debug["fire_valid"]),
            valid_fire_requested=bool(fire_requested and fire_debug["fire_valid"]),
            invalid_fire_requested=bool(fire_requested and not fire_debug["fire_valid"]),
            blocked_invalid_fire=bool(
                fire_requested and not fire_debug["fire_valid"] and fire_blocked_reason == "invalid_fire"
            ),
            no_fire_when_valid=bool((not fire_requested) and fire_debug["fire_valid"]),
            shot_fired_when_valid=bool(bullet_spawned and fire_debug["fire_valid"]),
            shot_fired=bullet_spawned,
            off_target_shot=bullet_spawned and float(fire_debug["aim_error"]) > 0.15,
            bullet_hit=any(event.get("type") == "bullet_hit" for event in bullet_events),
            missed_shot=any(event.get("type") == "bullet_expired" for event in bullet_events)
            and not any(event.get("type") == "bullet_hit" for event in bullet_events),
            distance_to_enemy=enemy_distance,
            in_good_range=bool(range_debug["in_good_range"]),
            too_close=bool(range_debug["too_close"]),
            too_far=bool(range_debug["too_far"]),
            outside_safe_zone=zone_debug["outside_safe_zone"],
            near_edge_outward=reward_components.get("near_edge_outward", 0.0) < 0.0,
        )
        goal_events = self._update_goal_loop()
        events = [*bullet_events, *goal_events]
        self.last_events = deepcopy(events)
        done = self._done()
        obs = self.observation()
        info = {
            "decoded_action": effective_decoded,
            "raw_action": dict(raw_action),
            "action_debug": {
                "accepted": not action_blocked_agent_dead,
                "no_op": bool(action_blocked_agent_dead),
                "reason": "agent_dead" if action_blocked_agent_dead else "accepted",
            },
            "reward_components": reward_components,
            "metrics": self.metrics.summary(),
            "damage_delta": {
                "enemy_hp": previous_enemy_hp - enemy_hp_after_combat,
                "self_hp": previous_self_hp - float(self.state["self_hp"]),
            },
            "safe_zone": {
                "center": dict(self.center),
                "radius": self._safe_radius(),
                "distance": self._distance(self.state["self_pos"], self.center),
                "outside": zone_debug["outside_safe_zone"],
            },
            "aim_debug": aim_debug,
            "range_debug": range_debug,
            "fire_debug": fire_debug,
            "zone_debug": zone_debug,
            "fire": {
                "fire_requested": fire_requested,
                "shot_fired": bullet_spawned,
                "fire_blocked_reason": fire_blocked_reason,
                "cooldown_remaining_steps_before": cooldown_before,
                "cooldown_remaining_steps_after": cooldown_after,
                "fire_interval_steps": int(self.weapon["fire_interval_steps"]),
                "fire_valid": bool(fire_debug["fire_valid"]),
                "target_in_range": bool(fire_debug["target_in_range"]),
                "cooldown_ready": bool(fire_debug["cooldown_ready"]),
                "aim_aligned": bool(fire_debug["aim_aligned"]),
                "current_aim_bin": int(aim_debug.get("aim_bin", 0)),
                "ideal_aim_bin": int(aim_debug.get("ideal_aim_bin", 0)),
                "aim_error": float(fire_debug["aim_error"]),
            },
            "fire_selected": fire_requested,
            "shot_fired": bullet_spawned,
            "bullet_spawned": bullet_spawned,
            "bullet_count": len(self.projectiles),
            "bullet_events": bullet_events,
            "events": events,
            "bullets": deepcopy(self.projectiles),
            "projectiles": deepcopy(self.projectiles),
            "goal": self._goal_debug_state(),
        }
        self.trajectory.append(self._trajectory_step(raw_action, effective_decoded, obs, reward, done, info))
        return obs, reward, done, info

    def observation(self) -> dict[str, Any]:
        ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
        target_dir = self._target_direction()["target_dir"]
        distance_to_center = self._distance(self.state["self_pos"], self.center)
        safe_radius = self._safe_radius()
        safe_margin = safe_radius - distance_to_center
        aim_debug = self.last_aim_debug or self._aim_debug(self.current_aim_vector, {"aimX": 1.0, "aimY": 0.0, "fire": 0})
        fire_debug = self.last_fire_debug or {
            "target_in_range": bool(self._range_debug(self._distance(self.state["self_pos"], self.state["enemy_pos"]))["in_good_range"]),
            "cooldown_ready": int(self.weapon.get("cooldown_remaining_steps", 0)) <= 0,
            "aim_aligned": bool(aim_debug.get("is_aim_aligned", False)),
            "fire_valid": False,
            "aim_error": int(aim_debug.get("aim_bin_error", 0)),
        }
        return {
            "self_hp": float(self.state["self_hp"]),
            "ally_hp": float(self.state["ally_hp"]),
            "enemy_hp": float(self.state["enemy_hp"]),
            "self_pos": deepcopy(self.state["self_pos"]),
            "ally_pos": deepcopy(self.state["ally_pos"]),
            "enemy_pos": deepcopy(self.state["enemy_pos"]),
            "distance_to_ally": ally_distance,
            "ally_under_pressure": self._ally_under_pressure(),
            "self_low_hp": float(self.state["self_hp"]) <= 35.0,
            "step_count": self.step_count,
            "safe_radius": safe_radius,
            "distance_to_enemy": self._distance(self.state["self_pos"], self.state["enemy_pos"]),
            "can_fire": int(self.weapon.get("cooldown_remaining_steps", 0)) <= 0,
            "weapon_cooldown_fraction": self._weapon_cooldown_fraction(),
            "target_dir_x": target_dir["x"],
            "target_dir_y": target_dir["y"],
            "aim_alignment": float(self.last_aim_debug.get("aim_alignment", 0.0)),
            "current_aim_bin": int(aim_debug.get("aim_bin", 0)),
            "ideal_aim_bin": int(aim_debug.get("ideal_aim_bin", 0)),
            "gt_ideal_aim_bin": int(aim_debug.get("ideal_aim_bin", 0)),
            "aim_error": float(fire_debug.get("aim_error", 0.0)),
            "aim_aligned": bool(fire_debug.get("aim_aligned", False)),
            "target_in_range": bool(fire_debug.get("target_in_range", False)),
            "cooldown_ready": bool(fire_debug.get("cooldown_ready", False)),
            "fire_valid": bool(fire_debug.get("fire_valid", False)),
            "distance_to_center": distance_to_center,
            "safe_margin_fraction": safe_margin / max(1.0, safe_radius),
            "outside_safe_zone": distance_to_center > safe_radius,
            "goal_enabled": bool(self.goal_enabled),
            "goal_position": list(self.goal_position) if self.goal_position is not None else None,
            "goal_radius": float(self.goal_radius),
            "goal_reached_count": int(self.goal_reached_count),
            "distance_to_goal": self._distance_to_goal(),
        }

    def sample_action(self) -> RawAction:
        return random_action(self.rng)

    def export_trajectory(self) -> dict[str, Any]:
        return {
            "trajectoryId": f"toy-cpc-{self.seed}",
            "schemaVersion": "toy-cpc-0.1",
            "scenarioId": "toy_cpc_debug",
            "seed": self.seed,
            "steps": deepcopy(self.trajectory),
            "metrics": self.metrics.summary(),
        }

    def get_debug_state(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "step": self.step_count,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "dt": self.dt,
            "map": {
                "width": self.width,
                "height": self.height,
                "center": dict(self.center),
                "obstacles": deepcopy(self.obstacles),
                "safe_radius": self._safe_radius(),
                "shrink_safe_zone": self.shrink_safe_zone,
                "use_zone_reward": self.use_zone_reward,
                "enemy_move": self.enemy_move,
                "enemy_fire": self.enemy_fire,
                "stationary_target_mode": self.stationary_target_mode,
                "enemy_spawn_distance_min": self.enemy_spawn_distance_min,
                "enemy_spawn_distance_max": self.enemy_spawn_distance_max,
            },
            "combat": {
                "fire_range": self.fire_range,
                "bullet_range": self.fire_range,
                "fire_alignment": self.fire_alignment,
                "projectile_speed": self.projectile_speed,
                "projectile_radius": self.projectile_radius,
                "enemy_aim_noise_deg": float(self.enemy_aim_noise_deg),
            },
            "weapon": deepcopy(self.weapon),
            "state": deepcopy(self.state),
            "bullets": deepcopy(self.projectiles),
            "projectiles": deepcopy(self.projectiles),
            "obstacles": deepcopy(self.obstacles),
            "goal": self._goal_debug_state(),
            "events": deepcopy(self.last_events),
            "agents": {
                "self": {
                    "position": deepcopy(self.state["self_pos"]),
                    "hp": float(self.state["self_hp"]),
                    "radius": self.player_radius,
                    "move_speed": self.move_speed,
                    "aim_turn_speed": self.aim_turn_speed,
                    "alive": float(self.state["self_hp"]) > 0.0,
                },
                "ally": {
                    "position": deepcopy(self.state["ally_pos"]),
                    "hp": float(self.state["ally_hp"]),
                    "radius": self.ally_radius,
                    "alive": float(self.state["ally_hp"]) > 0.0,
                },
                "enemy": {
                    "id": self.enemy_id,
                    "position": deepcopy(self.state["enemy_pos"]),
                    "hp": float(self.state["enemy_hp"]),
                    "radius": self.enemy_radius,
                    "move_speed": self.enemy_move_speed,
                    "behavior": self.enemy_behavior,
                    "alive": float(self.state["enemy_hp"]) > 0.0,
                },
            },
            "distances": {
                "self_to_ally": self._distance(self.state["self_pos"], self.state["ally_pos"]),
                "self_to_enemy": self._distance(self.state["self_pos"], self.state["enemy_pos"]),
                "ally_to_enemy": self._distance(self.state["ally_pos"], self.state["enemy_pos"]),
                "self_to_center": self._distance(self.state["self_pos"], self.center),
            },
            "predicates": {
                "ally_under_pressure": self._ally_under_pressure(),
                "self_low_hp": float(self.state["self_hp"]) <= 35.0,
            },
            "safe_zone": {
                "center": dict(self.center),
                "radius": self._safe_radius(),
                "distance": self._distance(self.state["self_pos"], self.center),
                "outside": self._distance(self.state["self_pos"], self.center) > self._safe_radius(),
            },
            "aim_debug": deepcopy(self.last_aim_debug),
            "fire_debug": deepcopy(self.last_fire_debug),
            "range_debug": self._range_debug(self._distance(self.state["self_pos"], self.state["enemy_pos"])),
            "zone_debug": deepcopy(self.last_zone_debug),
            "metrics": self.metrics.summary(),
        }

    def get_snapshot(self) -> dict[str, Any]:
        """Return a detached, debug-oriented view of the current world."""
        snapshot = {
            "step": int(self.step_count),
            "map": {"width": float(self.width), "height": float(self.height)},
            "player": {
                "position": self._position_list(self.state["self_pos"]),
                "hp": float(self.state["self_hp"]),
                "alive": float(self.state["self_hp"]) > 0.0,
            },
            "enemies": [
                {
                    "id": str(self.enemy_id),
                    "position": self._position_list(self.state["enemy_pos"]),
                    "hp": float(self.state["enemy_hp"]),
                    "alive": float(self.state["enemy_hp"]) > 0.0,
                }
            ],
            "bullets": [self._snapshot_bullet(bullet) for bullet in self.projectiles],
            "obstacles": deepcopy(self.obstacles),
            "goal": {
                "enabled": bool(self.goal_enabled),
                "position": list(self.goal_position) if self.goal_position is not None else None,
                "radius": float(self.goal_radius),
                "reached_count": int(self.goal_reached_count),
            },
            "events": deepcopy(self.last_events),
        }
        return deepcopy(snapshot)

    def _update_goal_loop(self) -> list[dict[str, Any]]:
        if not self.goal_enabled or self.goal_position is None:
            return []
        if float(self.state.get("self_hp", 0.0)) <= 0.0:
            return []
        distance_to_goal = self._distance_to_goal()
        if distance_to_goal is None or distance_to_goal > self.goal_radius:
            return []

        reached_position = list(self.goal_position)
        self.goal_reached_count += 1
        events: list[dict[str, Any]] = [
            {
                "type": "goal_reached",
                "position": reached_position,
                "goal_reached_count": int(self.goal_reached_count),
            }
        ]

        can_respawn = self.goal_respawn_on_reach and (
            self.goal_max_respawns is None or self.goal_respawn_count < self.goal_max_respawns
        )
        if can_respawn:
            self.goal_position = self._random_valid_position(
                radius=self.goal_radius,
                margin=self.goal_respawn_margin,
                min_player_distance=self.goal_respawn_margin,
            )
            self.goal_respawn_count += 1
            events.append(
                {
                    "type": "goal_respawned",
                    "position": list(self.goal_position),
                    "goal_reached_count": int(self.goal_reached_count),
                }
            )
        else:
            self.goal_position = None

        if self.goal_spawn_enemy_on_reach:
            enemy_event = self._respawn_enemy_for_goal()
            if enemy_event is not None:
                events.append(enemy_event)
        return events

    def _initial_goal_position(self) -> tuple[float, float] | None:
        if not self.goal_enabled:
            return None
        if self.goal_config_position is not None and self._spawn_position_is_valid(
            self.goal_config_position,
            radius=self.goal_radius,
        ):
            return tuple(self.goal_config_position)
        return self._random_valid_position(
            radius=self.goal_radius,
            margin=self.goal_respawn_margin,
            min_player_distance=self.goal_respawn_margin,
        )

    def _respawn_enemy_for_goal(self) -> dict[str, Any] | None:
        preferred = self.goal_position
        position = self._random_valid_position(
            radius=self.enemy_radius,
            margin=max(self.enemy_radius, 8.0),
            min_player_distance=max(self.goal_respawn_margin, self.player_radius + self.enemy_radius + 1.0),
            preferred_origin=preferred,
        )
        if position is None:
            return None
        self.enemy_spawn_count += 1
        base_id = self.env_config.enemies[0].id if self.env_config and self.env_config.enemies else "enemy"
        self.enemy_id = f"{base_id}-goal-{self.enemy_spawn_count}"
        self.state["enemy_pos"] = {"x": float(position[0]), "y": float(position[1])}
        self.state["enemy_hp"] = float(self.enemy_max_hp)
        self.enemy_weapon["cooldown_remaining_steps"] = 0
        return {
            "type": "enemy_spawned",
            "enemy_id": self.enemy_id,
            "position": list(position),
            "reason": "goal_reached",
        }

    def _random_valid_position(
        self,
        *,
        radius: float,
        margin: float,
        min_player_distance: float,
        preferred_origin: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        edge_margin = max(float(radius), min(float(margin), min(self.width, self.height) / 2.0))
        player = self.state.get("self_pos", {"x": 0.0, "y": 0.0})
        origins = (preferred_origin, None) if preferred_origin is not None else (None,)
        for origin in origins:
            for require_player_distance in (True, False):
                for _ in range(256):
                    if origin is None:
                        x = self.rng.uniform(edge_margin, max(edge_margin, self.width - edge_margin))
                        y = self.rng.uniform(edge_margin, max(edge_margin, self.height - edge_margin))
                    else:
                        angle = self.rng.uniform(-math.pi, math.pi)
                        distance = self.rng.uniform(max(48.0, radius * 2.0), max(120.0, self.goal_respawn_margin * 1.5))
                        x = self._clamp(origin[0] + math.cos(angle) * distance, edge_margin, self.width - edge_margin)
                        y = self._clamp(origin[1] + math.sin(angle) * distance, edge_margin, self.height - edge_margin)
                    candidate = (float(x), float(y))
                    if not self._spawn_position_is_valid(candidate, radius=radius):
                        continue
                    if require_player_distance and self._distance(
                        {"x": candidate[0], "y": candidate[1]}, player
                    ) < min_player_distance:
                        continue
                    return candidate
        raise RuntimeError("unable to find a valid spawn position")

    def _spawn_position_is_valid(self, position: tuple[float, float], *, radius: float) -> bool:
        x, y = float(position[0]), float(position[1])
        if x - radius < 0.0 or x + radius > self.width or y - radius < 0.0 or y + radius > self.height:
            return False
        return all(
            math.hypot(x - float(obstacle["x"]), y - float(obstacle["y"]))
            > float(radius) + float(obstacle["radius"])
            for obstacle in self._circle_obstacles()
        )

    def _distance_to_goal(self) -> float | None:
        if self.goal_position is None:
            return None
        return self._distance(
            self.state["self_pos"],
            {"x": float(self.goal_position[0]), "y": float(self.goal_position[1])},
        )

    def _goal_debug_state(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.goal_enabled),
            "position": (
                {"x": float(self.goal_position[0]), "y": float(self.goal_position[1])}
                if self.goal_position is not None
                else None
            ),
            "radius": float(self.goal_radius),
            "reached_count": int(self.goal_reached_count),
            "distance": self._distance_to_goal(),
        }

    @staticmethod
    def _position_list(position: Mapping[str, Any]) -> list[float]:
        return [float(position["x"]), float(position["y"])]

    def _snapshot_bullet(self, bullet: Mapping[str, Any]) -> dict[str, Any]:
        direction = bullet.get("direction", {})
        speed = float(bullet.get("speed", 0.0))
        owner_id = bullet.get("owner_id")
        return {
            "position": self._position_list(bullet.get("pos", {"x": 0.0, "y": 0.0})),
            "velocity": [
                float(direction.get("x", 0.0)) * speed,
                float(direction.get("y", 0.0)) * speed,
            ],
            "owner_id": owner_id,
            "team": "player" if owner_id == "self" else "enemy" if owner_id == "enemy" else None,
            "radius": float(bullet.get("radius", self.projectile_radius)),
            "ttl": None,
        }

    def _move_self(self, move_x: float, move_y: float) -> None:
        pos = self.state["self_pos"]
        start = dict(pos)
        target = {
            "x": self._clamp(pos["x"] + move_x * self.move_speed, 0.0, self.width),
            "y": self._clamp(pos["y"] + move_y * self.move_speed, 0.0, self.height),
        }
        resolved = self._resolve_obstacle_blocked_move(start, target, self.player_radius)
        pos["x"] = resolved["x"]
        pos["y"] = resolved["y"]

    def _effective_decoded_action(self, decoded: dict[str, float]) -> dict[str, float]:
        if not self._should_freeze_movement():
            return decoded
        return {
            **decoded,
            "moveX": 0.0,
            "moveY": 0.0,
        }

    def _should_freeze_movement(self) -> bool:
        # Stage 1B keeps the target stationary so evaluation isolates aim + fire.
        return bool(self.stationary_target_mode and self.stage == "local_combat")

    def _script_enemy_pressure(self) -> None:
        if not self.enemy_move:
            return
        if float(self.state["enemy_hp"]) <= 0.0:
            return
        enemy = self.state["enemy_pos"]
        target = self.state["self_pos"]
        distance_to_self = self._distance(enemy, target)
        behavior = self.enemy_behavior.strip().lower()
        if behavior not in {"pursue", "approach", "aggressive"} and distance_to_self <= self.fire_range * 0.9:
            return
        if distance_to_self <= self.player_radius + self.enemy_radius:
            return

        center_distance = self._distance(enemy, self.center)
        target_pos = self.center if self.stage != "local_combat" and center_distance > self._safe_radius() else target
        dx = target_pos["x"] - enemy["x"]
        dy = target_pos["y"] - enemy["y"]
        length = max(math.hypot(dx, dy), 1e-6)
        start = dict(enemy)
        step_distance = min(float(self.enemy_move_speed), max(0.0, distance_to_self - self.player_radius - self.enemy_radius))
        resolved = self._resolve_enemy_move(
            start,
            direction=(dx / length, dy / length),
            step_distance=step_distance,
        )
        enemy["x"] = resolved["x"]
        enemy["y"] = resolved["y"]

    def _resolve_enemy_move(
        self,
        start: Vec2,
        *,
        direction: tuple[float, float],
        step_distance: float,
    ) -> Vec2:
        best_partial = dict(start)
        best_distance = 0.0
        for angle in (0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0):
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            move_x = direction[0] * cos_angle - direction[1] * sin_angle
            move_y = direction[0] * sin_angle + direction[1] * cos_angle
            target = {
                "x": self._clamp(start["x"] + move_x * step_distance, 0.0, self.width),
                "y": self._clamp(start["y"] + move_y * step_distance, 0.0, self.height),
            }
            requested_distance = self._distance(start, target)
            if requested_distance <= 1e-6:
                continue
            resolved = self._resolve_obstacle_blocked_move(start, target, self.enemy_radius)
            resolved_distance = self._distance(start, resolved)
            if resolved_distance > best_distance:
                best_partial = resolved
                best_distance = resolved_distance
            if resolved_distance >= requested_distance * 0.99:
                return resolved
        return best_partial

    def _try_spawn_bullet(self, decoded: dict[str, float]) -> tuple[bool, str | None, dict[str, Any] | None]:
        if int(decoded["fire"]) != 1:
            return False, None, None

        if int(self.weapon.get("cooldown_remaining_steps", 0)) > 0:
            return False, "cooldown", {
                "type": "bullet_not_spawned",
                "reason": "cooldown",
                "owner_id": "self",
                "pos": deepcopy(self.state["self_pos"]),
                "cooldown_remaining_steps": int(self.weapon["cooldown_remaining_steps"]),
            }

        aim_x = float(decoded["aimX"])
        aim_y = float(decoded["aimY"])
        length = math.hypot(aim_x, aim_y)
        if length <= 1e-6:
            return False, "invalid_aim", {
                "type": "bullet_not_spawned",
                "reason": "invalid_aim",
                "owner_id": "self",
                "pos": deepcopy(self.state["self_pos"]),
            }

        direction = {"x": aim_x / length, "y": aim_y / length}
        bullet = self._spawn_projectile(
            owner_id="self",
            position=self.state["self_pos"],
            direction=direction,
            damage=self.damage,
        )
        self.weapon["cooldown_remaining_steps"] = int(self.weapon["fire_interval_steps"])
        return True, None, {
            "type": "bullet_spawned",
            "bullet_id": bullet["bullet_id"],
            "owner_id": bullet["owner_id"],
            "pos": deepcopy(bullet["pos"]),
        }

    def _spawn_projectile(
        self,
        *,
        owner_id: str,
        position: Vec2,
        direction: Vec2,
        damage: float,
    ) -> dict[str, Any]:
        bullet = {
            "bullet_id": f"bullet-{self.seed}-{self.step_count}-{len(self.projectiles)}",
            "owner_id": owner_id,
            "spawn_pos": deepcopy(position),
            "pos": deepcopy(position),
            "previous_pos": deepcopy(position),
            "direction": deepcopy(direction),
            "speed": self.projectile_speed,
            "max_range": self.fire_range,
            "traveled_distance": 0.0,
            "damage": float(damage),
            "radius": self.projectile_radius,
            "alive": True,
        }
        self.projectiles.append(bullet)
        return bullet

    def _tick_weapon_cooldowns(self) -> None:
        self.weapon["cooldown_remaining_steps"] = max(
            0,
            int(self.weapon.get("cooldown_remaining_steps", 0)) - 1,
        )
        self.enemy_weapon["cooldown_remaining_steps"] = max(
            0,
            int(self.enemy_weapon.get("cooldown_remaining_steps", 0)) - 1,
        )

    def _weapon_cooldown_fraction(self) -> float:
        interval = max(1, int(self.weapon.get("fire_interval_steps", self.fire_interval_steps)))
        return max(0.0, min(1.0, int(self.weapon.get("cooldown_remaining_steps", 0)) / interval))

    def _update_bullets(self, dt: float = 1.0) -> tuple[float, float, list[dict[str, Any]]]:
        damage_dealt = 0.0
        damage_taken = 0.0
        events: list[dict[str, Any]] = []
        active_bullets: list[dict[str, Any]] = []

        for bullet in self.projectiles:
            if not bool(bullet.get("alive", True)):
                continue

            previous_position = deepcopy(bullet["pos"])
            direction = bullet["direction"]
            step_distance = float(bullet["speed"]) * float(dt)
            next_position = {
                "x": float(previous_position["x"]) + (float(direction["x"]) * step_distance),
                "y": float(previous_position["y"]) + (float(direction["y"]) * step_distance),
            }
            bullet_radius = float(bullet["radius"])
            obstacle_hit = self._first_obstacle_hit(previous_position, next_position, moving_radius=bullet_radius)
            target = self._projectile_target(bullet)
            target_hit_t = (
                self._segment_circle_hit_fraction(
                    previous_position,
                    next_position,
                    target["position"],
                    bullet_radius + float(target["radius"]),
                )
                if target is not None
                else None
            )

            if obstacle_hit is not None and (target_hit_t is None or obstacle_hit[0] <= target_hit_t):
                hit_t, obstacle = obstacle_hit
                hit_position = self._lerp_position(previous_position, next_position, hit_t)
                traveled = step_distance * hit_t
                bullet["previous_pos"] = previous_position
                bullet["pos"] = hit_position
                bullet["traveled_distance"] = float(bullet.get("traveled_distance", 0.0)) + traveled
                bullet["alive"] = False
                events.extend(
                    [
                        {
                            "type": "bullet_moved",
                            "bullet_id": bullet["bullet_id"],
                            "owner_id": bullet["owner_id"],
                            "from": previous_position,
                            "pos": deepcopy(hit_position),
                            "traveled_distance": bullet["traveled_distance"],
                        },
                        {
                            "type": "bullet_hit_obstacle",
                            "bullet_id": bullet["bullet_id"],
                            "owner_id": bullet["owner_id"],
                            "obstacle_id": obstacle.get("id"),
                            "pos": deepcopy(hit_position),
                        },
                    ]
                )
                continue

            if target is not None and target_hit_t is not None:
                hit_position = self._lerp_position(previous_position, next_position, target_hit_t)
                traveled = step_distance * target_hit_t
                bullet["previous_pos"] = previous_position
                bullet["pos"] = hit_position
                bullet["traveled_distance"] = float(bullet.get("traveled_distance", 0.0)) + traveled
                events.append(
                    {
                        "type": "bullet_moved",
                        "bullet_id": bullet["bullet_id"],
                        "owner_id": bullet["owner_id"],
                        "from": previous_position,
                        "pos": deepcopy(hit_position),
                        "traveled_distance": bullet["traveled_distance"],
                    }
                )
                damage = min(float(bullet["damage"]), float(self.state[target["hp_key"]]))
                self.state[target["hp_key"]] = max(0.0, float(self.state[target["hp_key"]]) - damage)
                if target["target_id"] == "enemy":
                    damage_dealt += damage
                elif target["target_id"] == "self":
                    damage_taken += damage
                bullet["alive"] = False
                events.append(
                    {
                        "type": "bullet_hit",
                        "bullet_id": bullet["bullet_id"],
                        "owner_id": bullet["owner_id"],
                        "target_id": target["target_id"],
                        "damage": damage,
                        "pos": deepcopy(hit_position),
                    }
                )
                continue

            bullet["previous_pos"] = previous_position
            bullet["pos"] = next_position
            bullet["traveled_distance"] = float(bullet.get("traveled_distance", 0.0)) + step_distance
            events.append(
                {
                    "type": "bullet_moved",
                    "bullet_id": bullet["bullet_id"],
                    "owner_id": bullet["owner_id"],
                    "from": previous_position,
                    "pos": deepcopy(next_position),
                    "traveled_distance": bullet["traveled_distance"],
                }
            )

            expired = (
                float(bullet["traveled_distance"]) >= float(bullet["max_range"])
                or next_position["x"] < 0.0
                or next_position["x"] > self.width
                or next_position["y"] < 0.0
                or next_position["y"] > self.height
            )
            if expired:
                bullet["alive"] = False
                events.append(
                    {
                        "type": "bullet_expired",
                        "bullet_id": bullet["bullet_id"],
                        "owner_id": bullet["owner_id"],
                        "pos": deepcopy(next_position),
                        "traveled_distance": bullet["traveled_distance"],
                    }
                )
            else:
                active_bullets.append(bullet)

        self.projectiles = active_bullets
        return damage_dealt, damage_taken, events

    def _resolve_obstacle_blocked_move(self, start: Vec2, target: Vec2, moving_radius: float) -> Vec2:
        obstacle_hit = self._first_obstacle_hit(start, target, moving_radius=moving_radius)
        if obstacle_hit is None:
            return target

        hit_t, _ = obstacle_hit
        return self._lerp_position(start, target, max(0.0, hit_t - 1e-6))

    def _first_obstacle_hit(
        self,
        start: Vec2,
        target: Vec2,
        *,
        moving_radius: float,
    ) -> tuple[float, dict[str, Any]] | None:
        closest: tuple[float, dict[str, Any]] | None = None
        for obstacle in self._circle_obstacles():
            center = {"x": float(obstacle["x"]), "y": float(obstacle["y"])}
            radius = float(obstacle["radius"]) + float(moving_radius)
            hit_t = self._segment_circle_hit_fraction(start, target, center, radius)
            if hit_t is None:
                continue
            if closest is None or hit_t < closest[0]:
                closest = (hit_t, obstacle)
        return closest

    def _circle_obstacles(self) -> list[dict[str, Any]]:
        return [
            obstacle
            for obstacle in self.obstacles
            if obstacle.get("type", "circle") == "circle" and float(obstacle.get("radius", 0.0)) > 0.0
        ]

    @staticmethod
    def _segment_circle_hit_fraction(start: Vec2, target: Vec2, center: Vec2, radius: float) -> float | None:
        sx = float(start["x"])
        sy = float(start["y"])
        dx = float(target["x"]) - sx
        dy = float(target["y"]) - sy
        cx = float(center["x"])
        cy = float(center["y"])
        fx = sx - cx
        fy = sy - cy
        radius_sq = float(radius) * float(radius)
        if (fx * fx) + (fy * fy) <= radius_sq:
            return 0.0

        a = (dx * dx) + (dy * dy)
        if a <= 1e-9:
            return None

        b = 2.0 * ((fx * dx) + (fy * dy))
        c = (fx * fx) + (fy * fy) - radius_sq
        discriminant = (b * b) - (4.0 * a * c)
        if discriminant < 0.0:
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        for hit_t in ((-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)):
            if -1e-9 <= hit_t <= 1.0 + 1e-9:
                return max(0.0, min(1.0, hit_t))
        return None

    @staticmethod
    def _lerp_position(start: Vec2, target: Vec2, t: float) -> Vec2:
        return {
            "x": float(start["x"]) + ((float(target["x"]) - float(start["x"])) * float(t)),
            "y": float(start["y"]) + ((float(target["y"]) - float(start["y"])) * float(t)),
        }

    def _projectile_target(self, bullet: dict[str, Any]) -> dict[str, Any] | None:
        owner_id = str(bullet.get("owner_id", ""))
        if owner_id == "self":
            if float(self.state.get("enemy_hp", 0.0)) <= 0.0:
                return None
            return {
                "target_id": "enemy",
                "position": self.state["enemy_pos"],
                "hp_key": "enemy_hp",
                "radius": self.enemy_radius,
            }
        if owner_id == "enemy":
            if float(self.state.get("self_hp", 0.0)) <= 0.0:
                return None
            return {
                "target_id": "self",
                "position": self.state["self_pos"],
                "hp_key": "self_hp",
                "radius": self.player_radius,
            }
        return None

    def _resolve_enemy_pressure(self, *, suppress_projectile: bool = False) -> tuple[float, list[dict[str, Any]]]:
        if not self.enemy_fire:
            return 0.0, []
        if float(self.state["enemy_hp"]) <= 0.0:
            return 0.0, []

        damage_taken = 0.0
        events: list[dict[str, Any]] = []
        if self._distance(self.state["enemy_pos"], self.state["ally_pos"]) <= self.fire_range:
            self.state["ally_hp"] = max(0.0, float(self.state["ally_hp"]) - 2.0)
        self_in_range = self._distance(self.state["enemy_pos"], self.state["self_pos"]) <= self.fire_range
        blocked_by_obstacle = self._line_blocked_by_obstacle(self.state["enemy_pos"], self.state["self_pos"])
        if self_in_range and not blocked_by_obstacle:
            damage_taken = self.enemy_damage
            self.state["self_hp"] = max(0.0, float(self.state["self_hp"]) - damage_taken)

        if suppress_projectile or int(self.enemy_weapon.get("cooldown_remaining_steps", 0)) > 0:
            return damage_taken, events
        if not self_in_range:
            return damage_taken, events

        direction = normalize_vec(
            {
                "x": float(self.state["self_pos"]["x"]) - float(self.state["enemy_pos"]["x"]),
                "y": float(self.state["self_pos"]["y"]) - float(self.state["enemy_pos"]["y"]),
            }
        )
        if abs(direction["x"]) <= 1e-6 and abs(direction["y"]) <= 1e-6:
            return damage_taken, events

        applied_noise_rad = 0.0
        if self.enemy_aim_noise_deg > 0.0:
            max_noise_rad = math.radians(self.enemy_aim_noise_deg)
            applied_noise_rad = self.rng.uniform(-max_noise_rad, max_noise_rad)
            cosine = math.cos(applied_noise_rad)
            sine = math.sin(applied_noise_rad)
            direction = normalize_vec(
                {
                    "x": direction["x"] * cosine - direction["y"] * sine,
                    "y": direction["x"] * sine + direction["y"] * cosine,
                }
            )

        bullet = self._spawn_projectile(
            owner_id="enemy",
            position=self.state["enemy_pos"],
            direction=direction,
            damage=0.0,
        )
        self.enemy_weapon["cooldown_remaining_steps"] = int(self.enemy_weapon["fire_interval_steps"])
        events.append(
            {
                "type": "bullet_spawned",
                "bullet_id": bullet["bullet_id"],
                "owner_id": bullet["owner_id"],
                "pos": deepcopy(bullet["pos"]),
                "enemy_aim_noise_deg": float(self.enemy_aim_noise_deg),
                "applied_enemy_aim_noise_rad": float(applied_noise_rad),
            }
        )
        return damage_taken, events

    def _line_blocked_by_obstacle(self, start: Vec2, target: Vec2) -> bool:
        return self._first_obstacle_hit(start, target, moving_radius=self.projectile_radius) is not None

    def _reward_components(
        self,
        *,
        decoded: dict[str, float],
        previous_enemy_distance: float,
        enemy_distance: float,
        ally_under_pressure: bool,
        damage_dealt: float,
        damage_taken: float,
        previous_enemy_hp: float,
        previous_self_hp: float,
        fire_requested: bool,
        can_fire: bool,
        bullet_spawned: bool,
        fire_blocked_reason: str | None,
        bullet_events: list[dict[str, Any]],
        range_debug: dict[str, Any],
        aim_debug: dict[str, Any],
        fire_debug: dict[str, Any],
    ) -> dict[str, float]:
        del ally_under_pressure, previous_enemy_distance, enemy_distance
        aim_bin_error = int(aim_debug["aim_bin_error"])
        self_dead = previous_self_hp > 0.0 and float(self.state["self_hp"]) <= 0.0
        enemy_dead = previous_enemy_hp > 0.0 and float(self.state["enemy_hp"]) <= 0.0
        bullet_expired = any(event.get("type") == "bullet_expired" for event in bullet_events)
        bullet_hit = any(event.get("type") == "bullet_hit" for event in bullet_events)
        shot_fired = bool(bullet_spawned)
        combat_engaged_this_step = shot_fired or float(damage_dealt) > 0.0 or float(damage_taken) > 0.0
        enemy_hp_loss_ratio = float(damage_dealt) / max(1.0, self.enemy_max_hp)
        self_hp_loss_ratio = float(damage_taken) / max(1.0, self.self_max_hp)
        episode_done = self._done()
        timeout = self.step_count >= self.max_steps and not enemy_dead and not self_dead
        summary = self.metrics.summary()
        total_damage_dealt_ratio = (float(summary.get("damage_dealt", 0.0)) + float(damage_dealt)) / max(1.0, self.enemy_max_hp)
        total_damage_taken_ratio = (float(summary.get("damage_taken", 0.0)) + float(damage_taken)) / max(1.0, self.self_max_hp)
        prior_shots = int(float(summary.get("shot_fired_count", 0.0)))
        prior_hits = int(float(summary.get("bullet_hit_count", 0.0)))
        episode_shots = prior_shots + int(shot_fired)
        hit_ratio = (prior_hits + int(bullet_hit)) / max(prior_shots + int(shot_fired), 1)
        if self.stationary_target_mode:
            fire_valid = bool(fire_debug["fire_valid"])
            shot_fired_reward = 0.03 if shot_fired and fire_valid else 0.0
            bad_aim_shot_penalty = -0.02 if shot_fired and not fire_debug["aim_aligned"] else 0.0
            bullet_hit_reward = 0.30 if bullet_hit else 0.0
            damage_dealt_reward = 1.5 * enemy_hp_loss_ratio
            return {
                "damage_dealt_ratio": damage_dealt_reward,
                "damage_taken_ratio": 0.0,
                "bullet_hit_reward": bullet_hit_reward,
                "shot_fired_reward": shot_fired_reward,
                "bad_aim_shot_penalty": bad_aim_shot_penalty,
                "missed_shot_penalty": -0.02 if shot_fired and bullet_expired and not bullet_hit else 0.0,
                "no_fire_ready_penalty": -0.01 if fire_valid and not shot_fired else 0.0,
                "cooldown_blocked_fire_penalty": -0.01 if fire_requested and not fire_debug["cooldown_ready"] else 0.0,
                "invalid_fire_penalty": -0.05 if fire_requested and not fire_valid and fire_debug["cooldown_ready"] else 0.0,
            }
        components = {
            "damage_dealt_ratio": self.damage_dealt_ratio_weight * enemy_hp_loss_ratio,
            "damage_taken_ratio": self.damage_taken_ratio_weight * self_hp_loss_ratio,
            "bullet_hit": self.bullet_hit_bonus if bullet_hit else 0.0,
            "missed_shot": -self.missed_shot_penalty if bullet_expired and not bullet_hit else 0.0,
            "aim_bin_exact": self.aim_bin_exact_bonus if shot_fired and aim_bin_error == 0 else 0.0,
            "aim_bin_wrong": -self.aim_bin_wrong_penalty if shot_fired and aim_bin_error >= 2 else 0.0,
            "aim_alignment": 0.02 * max(0.0, float(aim_debug["aim_alignment"])) if shot_fired else 0.0,
            "good_range": self.good_range_bonus if combat_engaged_this_step and range_debug["in_good_range"] else 0.0,
            "too_close": -self.too_close_penalty if range_debug["too_close"] else 0.0,
            "too_far": -self.too_far_penalty if range_debug["too_far"] and episode_shots == 0 and self.step_count > 20 else 0.0,
            "kill": self.kill_bonus if enemy_dead else 0.0,
            "death": -self.death_penalty if self_dead else 0.0,
            "timeout_hp_lead": (
                self.timeout_hp_lead_weight * (total_damage_dealt_ratio - total_damage_taken_ratio) if timeout else 0.0
            ),
            "accuracy_bonus": (
                self.accuracy_bonus_weight * hit_ratio * min(total_damage_dealt_ratio, 1.0) if timeout else 0.0
            ),
            "no_shot_episode": -0.5 if episode_done and episode_shots == 0 else 0.0,
            "death_without_shooting": -1.0 if self_dead and episode_shots == 0 else 0.0,
            "death_without_damage": -0.5 if self_dead and total_damage_dealt_ratio == 0.0 else 0.0,
        }
        if self.use_zone_reward:
            zone_debug = self._zone_debug(decoded)
            outside_safe = bool(zone_debug["outside_safe_zone"])
            move_toward_center = float(zone_debug["move_toward_center"])
            edge_ratio = float(zone_debug["distance_to_center"]) / max(1.0, float(zone_debug["safe_radius"]))
            components.update(
                {
                    "zone_pressure": -self.zone_pressure_penalty if outside_safe else 0.0,
                    "return_to_zone": self.return_to_zone_bonus if outside_safe and move_toward_center > 0.5 else 0.0,
                    "move_deeper_outside_zone": (
                        -self.move_deeper_outside_zone_penalty if outside_safe and move_toward_center < -0.2 else 0.0
                    ),
                    "near_edge_outward": (
                        -self.near_edge_outward_penalty if edge_ratio > 0.90 and move_toward_center < 0.0 else 0.0
                    ),
                }
            )
        return components

    def _aim_alignment(self, decoded: dict[str, float]) -> float:
        return float(self._aim_debug(self._aim_vector_from_decoded(decoded), decoded)["aim_alignment"])

    def _aim_debug(self, aim_vector: dict[str, float], decoded: dict[str, float]) -> dict[str, Any]:
        target = self._target_direction()
        target_dir = target["target_dir"]
        aim_dir = normalize_vec(dict(aim_vector))
        has_enemy = target["target_enemy_id"] is not None
        alignment = dot(aim_dir, target_dir) if has_enemy else 0.0
        ideal_angle = math.degrees(math.atan2(target_dir["y"], target_dir["x"])) if has_enemy else 0.0
        current_angle = math.degrees(math.atan2(aim_dir["y"], aim_dir["x"])) if has_enemy else 0.0
        angle_error = angle_between(aim_dir, target_dir) if has_enemy else 180.0
        aim_error = min(1.0, max(0.0, angle_error / 180.0))
        return {
            "target_enemy_id": target["target_enemy_id"],
            "target_dir": target_dir,
            "aim_dir": aim_dir,
            "aim_bin": int(vec_to_aim_bin(aim_dir, AIM_BINS)) if has_enemy else 0,
            "ideal_aim_bin": int(vec_to_aim_bin(target_dir, AIM_BINS)) if has_enemy else 0,
            "aim_bin_error": circular_bin_distance(
                vec_to_aim_bin(aim_dir, AIM_BINS),
                vec_to_aim_bin(target_dir, AIM_BINS),
                AIM_BINS,
            )
            if has_enemy
            else 0,
            "aim_alignment": alignment,
            "current_aim_angle_deg": current_angle,
            "ideal_aim_angle_deg": ideal_angle,
            "angle_error_deg": angle_error,
            "aim_error": aim_error,
            "is_exact_aim": bool(has_enemy and angle_error <= 1e-6),
            "is_near_aim": bool(has_enemy and angle_error <= 20.0),
            "is_bad_aim": bool(has_enemy and angle_error >= 60.0),
            "is_aim_aligned": bool(has_enemy and aim_error <= self.aim_alignment_threshold),
        }

    def _fire_valid_debug(
        self,
        *,
        aim_debug: dict[str, Any],
        range_debug: dict[str, Any],
        cooldown_ready: bool,
    ) -> dict[str, Any]:
        aim_error = float(aim_debug.get("aim_error", 0.0))
        aim_aligned = bool(aim_error <= self.aim_alignment_threshold)
        target_in_range = bool(range_debug.get("in_good_range", False))
        fire_valid = bool(target_in_range and cooldown_ready and aim_aligned)
        return {
            "current_aim_bin": int(aim_debug.get("aim_bin", 0)),
            "ideal_aim_bin": int(aim_debug.get("ideal_aim_bin", 0)),
            "aim_error": aim_error,
            "aim_aligned": aim_aligned,
            "target_in_range": target_in_range,
            "cooldown_ready": bool(cooldown_ready),
            "fire_valid": fire_valid,
        }

    def _target_direction(self) -> dict[str, Any]:
        if float(self.state.get("enemy_hp", 0.0)) <= 0.0:
            return {"target_enemy_id": None, "target_dir": {"x": 0.0, "y": 0.0}}
        return {
            "target_enemy_id": "enemy",
            "target_dir": normalize_vec(
                {
                    "x": float(self.state["enemy_pos"]["x"]) - float(self.state["self_pos"]["x"]),
                    "y": float(self.state["enemy_pos"]["y"]) - float(self.state["self_pos"]["y"]),
                }
            ),
        }

    def _zone_debug(self, decoded: dict[str, float]) -> dict[str, Any]:
        distance_to_center = self._distance(self.state["self_pos"], self.center)
        safe_radius = self._safe_radius()
        center_dir = normalize_vec(
            {
                "x": float(self.center["x"]) - float(self.state["self_pos"]["x"]),
                "y": float(self.center["y"]) - float(self.state["self_pos"]["y"]),
            }
        )
        move_dir = normalize_vec({"x": float(decoded["moveX"]), "y": float(decoded["moveY"])})
        return {
            "distance_to_center": distance_to_center,
            "safe_radius": safe_radius,
            "outside_safe_zone": distance_to_center > safe_radius,
            "safe_margin": safe_radius - distance_to_center,
            "center_dir": center_dir,
            "move_dir": move_dir,
            "move_toward_center": dot(move_dir, center_dir),
        }

    def _range_debug(self, distance_to_enemy: float) -> dict[str, Any]:
        optimal_range = 0.5 * self.fire_range
        good_range_min = 0.25 * self.fire_range
        good_range_max = 0.9 * self.fire_range
        too_close = float(distance_to_enemy) < good_range_min
        too_far = float(distance_to_enemy) > good_range_max
        return {
            "distance_to_enemy": float(distance_to_enemy),
            "optimal_range": optimal_range,
            "good_range_min": good_range_min,
            "good_range_max": good_range_max,
            "too_close": bool(too_close),
            "too_far": bool(too_far),
            "in_good_range": bool(not too_close and not too_far),
        }

    @staticmethod
    def _aim_vector_from_decoded(decoded: dict[str, float]) -> dict[str, float]:
        return normalize_vec({"x": float(decoded["aimX"]), "y": float(decoded["aimY"])})

    def _safe_radius(self) -> float:
        if not self.shrink_safe_zone:
            return self.safe_radius_start
        progress = min(1.0, self.step_count / max(1, self.max_steps - 1))
        return self.safe_radius_start + (self.safe_radius_end - self.safe_radius_start) * progress

    def _spawn_positions(self) -> dict[str, Vec2]:
        if self.player_spawn is not None or self.enemy_spawn is not None:
            self_pos = deepcopy(self.player_spawn) if self.player_spawn is not None else {"x": 430.0, "y": 500.0}
            ally_pos = (
                deepcopy(self.ally_spawn)
                if self.ally_spawn is not None
                else {"x": self_pos["x"] - 60.0, "y": self_pos["y"] + 45.0}
            )
            enemy_pos = deepcopy(self.enemy_spawn) if self.enemy_spawn is not None else {"x": self_pos["x"] + self.fire_range, "y": self_pos["y"]}
            return {
                "self_pos": self_pos,
                "ally_pos": ally_pos,
                "enemy_pos": enemy_pos,
            }

        angle = self._spawn_angle()
        if self.enemy_spawn_distance_min is not None or self.enemy_spawn_distance_max is not None:
            low = self.enemy_spawn_distance_min if self.enemy_spawn_distance_min is not None else 0.8 * self.fire_range
            high = self.enemy_spawn_distance_max if self.enemy_spawn_distance_max is not None else 1.2 * self.fire_range
        else:
            low = 0.8 * self.fire_range
            high = 1.2 * self.fire_range
        distance = self.rng.uniform(float(low), float(high))
        self_pos = {"x": 430.0, "y": 500.0}
        enemy_pos = {
            "x": self._clamp(self_pos["x"] + math.cos(angle) * distance, 0.0, self.width),
            "y": self._clamp(self_pos["y"] + math.sin(angle) * distance, 0.0, self.height),
        }
        ally_pos = {"x": self_pos["x"] - 60.0, "y": self_pos["y"] + 45.0}
        return {
            "self_pos": self_pos,
            "ally_pos": ally_pos,
            "enemy_pos": enemy_pos,
        }

    def _spawn_angle(self) -> float:
        if self.enemy_spawn_direction is not None:
            return self.enemy_spawn_direction_angles[self.enemy_spawn_direction]
        if not self.randomize_enemy_spawn_direction:
            return self.rng.uniform(-0.35, 0.35)
        direction_name = self.rng.choice(list(self.enemy_spawn_directions))
        direction = self.enemy_spawn_direction_angles[direction_name]
        return direction + self.rng.uniform(-0.18, 0.18)

    @classmethod
    def _validate_spawn_directions(cls, directions: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        if directions is None:
            return cls.default_enemy_spawn_directions
        cleaned = tuple(str(direction).strip() for direction in directions if str(direction).strip())
        if not cleaned:
            raise ValueError("enemy_spawn_directions must include at least one direction")
        unknown = [direction for direction in cleaned if direction not in cls.enemy_spawn_direction_angles]
        if unknown:
            valid = ", ".join(cls.default_enemy_spawn_directions)
            raise ValueError(f"unknown enemy spawn direction(s): {unknown}; valid directions: {valid}")
        return cleaned

    @classmethod
    def _validate_spawn_direction(cls, direction: str | None) -> str | None:
        if direction is None or str(direction).strip() == "":
            return None
        cleaned = str(direction).strip()
        if cleaned not in cls.enemy_spawn_direction_angles:
            valid = ", ".join(cls.default_enemy_spawn_directions)
            raise ValueError(f"unknown enemy spawn direction {cleaned!r}; valid directions: {valid}")
        return cleaned

    def _trajectory_step(
        self,
        raw_action: Mapping[str, int | float],
        decoded: dict[str, float],
        obs: dict[str, Any],
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "actorId": "self",
            "action": {
                "moveX": decoded["moveX"],
                "moveY": decoded["moveY"],
                "aimX": decoded["aimX"],
                "aimY": decoded["aimY"],
                "fire": decoded["fire"],
            },
            "rawAction": dict(raw_action),
            "state": obs,
            "reward": reward,
            "done": done,
            "measurements": {
                "distanceToAlly": obs["distance_to_ally"],
                "allyUnderPressure": obs["ally_under_pressure"],
                "damageDealt": info["damage_delta"]["enemy_hp"],
                "damageTaken": info["damage_delta"]["self_hp"],
            },
        }

    def _ally_under_pressure(self) -> bool:
        return (
            float(self.state["ally_hp"]) > 0.0
            and float(self.state["enemy_hp"]) > 0.0
            and self._distance(self.state["ally_pos"], self.state["enemy_pos"]) <= self.fire_range
        )

    def _done(self) -> bool:
        player_dead = float(self.state["self_hp"]) <= 0.0
        return self.step_count >= self.max_steps or player_dead

    @staticmethod
    def _distance(a: Vec2, b: Vec2) -> float:
        return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

    @staticmethod
    def _segment_distance(a: Vec2, b: Vec2, point: Vec2) -> float:
        ax = float(a["x"])
        ay = float(a["y"])
        bx = float(b["x"])
        by = float(b["y"])
        px = float(point["x"])
        py = float(point["y"])
        dx = bx - ax
        dy = by - ay
        length_sq = (dx * dx) + (dy * dy)
        if length_sq <= 1e-6:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (((px - ax) * dx) + ((py - ay) * dy)) / length_sq))
        closest_x = ax + (t * dx)
        closest_y = ay + (t * dy)
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
