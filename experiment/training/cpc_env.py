from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

if __package__:
    from .cpc_actions import AIM_BINS, RawAction, decode_action, random_action
    from .cpc_metrics import CpcMetrics
else:
    from cpc_actions import AIM_BINS, RawAction, decode_action, random_action
    from cpc_metrics import CpcMetrics


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


def aim_bin_to_vec(aim_bin: int, num_bins: int = AIM_BINS) -> Vec2:
    theta = (2.0 * math.pi * int(aim_bin)) / int(num_bins)
    return {"x": math.cos(theta), "y": math.sin(theta)}


def vec_to_aim_bin(vec: Vec2, num_bins: int = AIM_BINS) -> int:
    normalized = normalize_vec(vec)
    if normalized["x"] == 0.0 and normalized["y"] == 0.0:
        return 0
    theta = math.atan2(normalized["y"], normalized["x"])
    if theta < 0.0:
        theta += 2.0 * math.pi
    return int(round((theta / (2.0 * math.pi)) * int(num_bins))) % int(num_bins)


class CPCEnv:
    width = 1000.0
    height = 1000.0
    max_hp = 100.0
    move_speed = 35.0
    fire_range = 260.0
    fire_alignment = 0.65
    fire_interval_steps = 5
    projectile_speed = 140.0
    projectile_radius = 8.0
    damage = 10.0
    enemy_damage = 2.0
    enemy_move_speed = 18.0
    survival_reward = 0.001
    center = {"x": 500.0, "y": 500.0}
    safe_radius_start = 420.0
    safe_radius_end = 120.0
    aim_alignment_threshold = 0.75
    aim_alignment_weight = 0.12
    bad_aim_penalty = 0.05
    aligned_shot_bonus = 0.08
    off_target_shot_penalty = 0.08
    bullet_hit_bonus = 0.20
    missed_shot_penalty = 0.03
    zone_pressure_penalty = 0.20
    return_to_zone_bonus = 0.06
    move_deeper_outside_zone_penalty = 0.08
    near_edge_outward_penalty = 0.04

    def __init__(self, seed: int = 0, max_steps: int = 50, randomize_enemy_spawn_direction: bool = False):
        self.seed = seed
        self.max_steps = max_steps
        self.randomize_enemy_spawn_direction = bool(randomize_enemy_spawn_direction)
        self.rng = random.Random(seed)
        self.metrics = CpcMetrics()
        self.trajectory: list[dict[str, Any]] = []
        self.step_count = 0
        self.state: dict[str, Any] = {}
        self.projectiles: list[dict[str, Any]] = []
        self.weapon: dict[str, Any] = {}
        self.last_aim_debug: dict[str, Any] = {}
        self.last_zone_debug: dict[str, Any] = {}
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        self.step_count = 0
        self.metrics = CpcMetrics()
        self.trajectory = []
        self.projectiles = []
        self.last_aim_debug = {}
        self.last_zone_debug = {}
        self.weapon = {
            "cooldown_remaining_steps": 0,
            "fire_interval_steps": self.fire_interval_steps,
        }
        spawn = self._spawn_positions()
        self.state = {
            "self_hp": self.max_hp,
            "ally_hp": self.max_hp,
            "enemy_hp": self.max_hp,
            "self_pos": spawn["self_pos"],
            "ally_pos": spawn["ally_pos"],
            "enemy_pos": spawn["enemy_pos"],
        }
        return self.observation()

    def step(self, action: RawAction | dict[str, int]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        raw_action: RawAction = {
            "move": int(action["move"]),
            "aim": int(action["aim"]),
            "fire": int(action["fire"]),
        }
        decoded = decode_action(raw_action)
        previous_ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
        previous_enemy_distance = self._distance(self.state["self_pos"], self.state["enemy_pos"])
        previous_enemy_hp = float(self.state["enemy_hp"])
        previous_self_hp = float(self.state["self_hp"])

        self._move_self(decoded["moveX"], decoded["moveY"])
        ally_under_pressure = self._ally_under_pressure()
        self._script_enemy_pressure()

        damage_dealt, bullet_events = self._update_bullets()
        cooldown_before = int(self.weapon["cooldown_remaining_steps"])
        fire_requested = bool(decoded["fire"])
        bullet_spawned, fire_blocked_reason, spawn_event = self._try_spawn_bullet(decoded)
        if spawn_event is not None:
            bullet_events.append(spawn_event)
        self._tick_weapon_cooldowns()
        cooldown_after = int(self.weapon["cooldown_remaining_steps"])
        damage_taken = self._resolve_enemy_pressure()
        self.step_count += 1

        ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
        enemy_distance = self._distance(self.state["self_pos"], self.state["enemy_pos"])
        moved_toward_ally = ally_distance < previous_ally_distance
        reward_components = self._reward_components(
            decoded=decoded,
            previous_enemy_distance=previous_enemy_distance,
            enemy_distance=enemy_distance,
            ally_under_pressure=ally_under_pressure,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            previous_enemy_hp=previous_enemy_hp,
            previous_self_hp=previous_self_hp,
            fire_requested=fire_requested,
            bullet_spawned=bullet_spawned,
            fire_blocked_reason=fire_blocked_reason,
            bullet_events=bullet_events,
        )
        aim_debug = self._aim_debug(raw_action["aim"], decoded)
        zone_debug = self._zone_debug(decoded)
        self.last_aim_debug = aim_debug
        self.last_zone_debug = zone_debug
        reward = sum(reward_components.values())
        done = self._done()
        obs = self.observation()
        self.metrics.update(
            ally_distance=ally_distance,
            ally_under_pressure=ally_under_pressure,
            moved_toward_ally=moved_toward_ally,
            fired_under_pressure=bool(decoded["fire"]),
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            aim_alignment=aim_debug["aim_alignment"],
            off_target_shot=bullet_spawned and not aim_debug["is_aim_aligned"],
            bullet_hit=any(event.get("type") == "bullet_hit" for event in bullet_events),
            outside_safe_zone=zone_debug["outside_safe_zone"],
            near_edge_outward=reward_components["near_edge_outward"] < 0.0,
        )
        info = {
            "decoded_action": decoded,
            "raw_action": dict(raw_action),
            "reward_components": reward_components,
            "metrics": self.metrics.summary(),
            "damage_delta": {
                "enemy_hp": previous_enemy_hp - float(self.state["enemy_hp"]),
                "self_hp": previous_self_hp - float(self.state["self_hp"]),
            },
            "safe_zone": {
                "center": dict(self.center),
                "radius": self._safe_radius(),
                "distance": self._distance(self.state["self_pos"], self.center),
                "outside": zone_debug["outside_safe_zone"],
            },
            "aim_debug": aim_debug,
            "zone_debug": zone_debug,
            "fire": {
                "fire_requested": fire_requested,
                "shot_fired": bullet_spawned,
                "fire_blocked_reason": fire_blocked_reason,
                "cooldown_remaining_steps_before": cooldown_before,
                "cooldown_remaining_steps_after": cooldown_after,
                "fire_interval_steps": int(self.weapon["fire_interval_steps"]),
            },
            "fire_selected": fire_requested,
            "shot_fired": bullet_spawned,
            "bullet_spawned": bullet_spawned,
            "bullet_count": len(self.projectiles),
            "bullet_events": bullet_events,
            "bullets": deepcopy(self.projectiles),
            "projectiles": deepcopy(self.projectiles),
        }
        self.trajectory.append(self._trajectory_step(raw_action, decoded, obs, reward, done, info))
        return obs, reward, done, info

    def observation(self) -> dict[str, Any]:
        ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
        target_dir = self._target_direction()["target_dir"]
        distance_to_center = self._distance(self.state["self_pos"], self.center)
        safe_radius = self._safe_radius()
        safe_margin = safe_radius - distance_to_center
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
            "distance_to_center": distance_to_center,
            "safe_margin_fraction": safe_margin / max(1.0, safe_radius),
            "outside_safe_zone": distance_to_center > safe_radius,
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
            "step": self.step_count,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "map": {
                "width": self.width,
                "height": self.height,
                "center": dict(self.center),
                "safe_radius": self._safe_radius(),
            },
            "combat": {
                "fire_range": self.fire_range,
                "fire_alignment": self.fire_alignment,
                "projectile_speed": self.projectile_speed,
                "projectile_radius": self.projectile_radius,
            },
            "weapon": deepcopy(self.weapon),
            "state": deepcopy(self.state),
            "bullets": deepcopy(self.projectiles),
            "projectiles": deepcopy(self.projectiles),
            "agents": {
                "self": {
                    "position": deepcopy(self.state["self_pos"]),
                    "hp": float(self.state["self_hp"]),
                    "alive": float(self.state["self_hp"]) > 0.0,
                },
                "ally": {
                    "position": deepcopy(self.state["ally_pos"]),
                    "hp": float(self.state["ally_hp"]),
                    "alive": float(self.state["ally_hp"]) > 0.0,
                },
                "enemy": {
                    "position": deepcopy(self.state["enemy_pos"]),
                    "hp": float(self.state["enemy_hp"]),
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
            "zone_debug": deepcopy(self.last_zone_debug),
            "metrics": self.metrics.summary(),
        }

    def _move_self(self, move_x: float, move_y: float) -> None:
        pos = self.state["self_pos"]
        pos["x"] = self._clamp(pos["x"] + move_x * self.move_speed, 0.0, self.width)
        pos["y"] = self._clamp(pos["y"] + move_y * self.move_speed, 0.0, self.height)

    def _script_enemy_pressure(self) -> None:
        if float(self.state["enemy_hp"]) <= 0.0:
            return
        enemy = self.state["enemy_pos"]
        target = self.state["self_pos"]
        distance_to_self = self._distance(enemy, target)
        if distance_to_self <= self.fire_range * 0.9:
            return

        center_distance = self._distance(enemy, self.center)
        target_pos = self.center if center_distance > self._safe_radius() else target
        dx = target_pos["x"] - enemy["x"]
        dy = target_pos["y"] - enemy["y"]
        length = max(math.hypot(dx, dy), 1e-6)
        enemy["x"] = self._clamp(enemy["x"] + (dx / length) * self.enemy_move_speed, 0.0, self.width)
        enemy["y"] = self._clamp(enemy["y"] + (dy / length) * self.enemy_move_speed, 0.0, self.height)

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
        bullet = {
            "bullet_id": f"bullet-{self.seed}-{self.step_count}-{len(self.projectiles)}",
            "owner_id": "self",
            "spawn_pos": deepcopy(self.state["self_pos"]),
            "pos": deepcopy(self.state["self_pos"]),
            "previous_pos": deepcopy(self.state["self_pos"]),
            "direction": direction,
            "speed": self.projectile_speed,
            "max_range": self.fire_range,
            "traveled_distance": 0.0,
            "damage": self.damage,
            "radius": self.projectile_radius,
            "alive": True,
        }
        self.projectiles.append(bullet)
        self.weapon["cooldown_remaining_steps"] = int(self.weapon["fire_interval_steps"])
        return True, None, {
            "type": "bullet_spawned",
            "bullet_id": bullet["bullet_id"],
            "owner_id": bullet["owner_id"],
            "pos": deepcopy(bullet["pos"]),
        }

    def _tick_weapon_cooldowns(self) -> None:
        self.weapon["cooldown_remaining_steps"] = max(
            0,
            int(self.weapon.get("cooldown_remaining_steps", 0)) - 1,
        )

    def _weapon_cooldown_fraction(self) -> float:
        interval = max(1, int(self.weapon.get("fire_interval_steps", self.fire_interval_steps)))
        return max(0.0, min(1.0, int(self.weapon.get("cooldown_remaining_steps", 0)) / interval))

    def _update_bullets(self, dt: float = 1.0) -> tuple[float, list[dict[str, Any]]]:
        damage_dealt = 0.0
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

            if (
                float(self.state["enemy_hp"]) > 0.0
                and self._segment_distance(previous_position, next_position, self.state["enemy_pos"]) <= float(bullet["radius"])
            ):
                damage = min(float(bullet["damage"]), float(self.state["enemy_hp"]))
                self.state["enemy_hp"] = max(0.0, float(self.state["enemy_hp"]) - damage)
                damage_dealt += damage
                bullet["alive"] = False
                events.append(
                    {
                        "type": "bullet_hit",
                        "bullet_id": bullet["bullet_id"],
                        "owner_id": bullet["owner_id"],
                        "target_id": "enemy",
                        "damage": damage,
                        "pos": deepcopy(next_position),
                    }
                )
                continue

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
        return damage_dealt, events

    def _resolve_enemy_pressure(self) -> float:
        damage_taken = 0.0
        if float(self.state["enemy_hp"]) <= 0.0:
            return 0.0
        if self._distance(self.state["enemy_pos"], self.state["ally_pos"]) <= self.fire_range:
            self.state["ally_hp"] = max(0.0, float(self.state["ally_hp"]) - 2.0)
        if self._distance(self.state["enemy_pos"], self.state["self_pos"]) <= self.fire_range:
            damage_taken = self.enemy_damage
            self.state["self_hp"] = max(0.0, float(self.state["self_hp"]) - damage_taken)
        return damage_taken

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
        bullet_spawned: bool,
        fire_blocked_reason: str | None,
        bullet_events: list[dict[str, Any]],
    ) -> dict[str, float]:
        del ally_under_pressure
        aim_debug = self._aim_debug(self._aim_bin_from_decoded(decoded), decoded)
        zone_debug = self._zone_debug(decoded)
        alignment = float(aim_debug["aim_alignment"])
        in_range = enemy_distance <= self.fire_range
        relevant_range = enemy_distance <= self.fire_range * 1.25
        outside_safe = bool(zone_debug["outside_safe_zone"])
        self_dead = previous_self_hp > 0.0 and float(self.state["self_hp"]) <= 0.0
        enemy_dead = previous_enemy_hp > 0.0 and float(self.state["enemy_hp"]) <= 0.0
        bullet_expired = any(event.get("type") == "bullet_expired" for event in bullet_events)
        bullet_hit = any(event.get("type") == "bullet_hit" for event in bullet_events)
        shot_fired = bool(bullet_spawned)
        move_toward_center = float(zone_debug["move_toward_center"])
        edge_ratio = float(zone_debug["distance_to_center"]) / max(1.0, float(zone_debug["safe_radius"]))
        return {
            "damage_dealt": damage_dealt * 0.10,
            "bullet_hit": self.bullet_hit_bonus if bullet_hit else 0.0,
            "damage_taken": -damage_taken * 0.05,
            "death": -1.0 if self_dead else 0.0,
            "win": 2.0 if enemy_dead else 0.0,
            "survival": self.survival_reward if float(self.state["self_hp"]) > 0.0 else 0.0,
            "approach_enemy": 0.03 if enemy_distance < previous_enemy_distance else -0.02,
            "aim_alignment": self.aim_alignment_weight * max(0.0, alignment),
            "bad_aim": -self.bad_aim_penalty if relevant_range and alignment < 0.3 else 0.0,
            "attack_intent": 0.05 if int(decoded["fire"]) == 1 and in_range and alignment >= self.fire_alignment else 0.0,
            "aligned_shot": self.aligned_shot_bonus if shot_fired and bool(aim_debug["is_aim_aligned"]) else 0.0,
            "off_target_shot": -self.off_target_shot_penalty if shot_fired and not bool(aim_debug["is_aim_aligned"]) else 0.0,
            "zone_pressure": -self.zone_pressure_penalty if outside_safe else 0.0,
            "return_to_zone": self.return_to_zone_bonus if outside_safe and move_toward_center > 0.5 else 0.0,
            "move_deeper_outside_zone": (
                -self.move_deeper_outside_zone_penalty if outside_safe and move_toward_center < -0.2 else 0.0
            ),
            "near_edge_outward": -self.near_edge_outward_penalty if edge_ratio > 0.90 and move_toward_center < 0.0 else 0.0,
            "shot_fired": 0.0 if bullet_spawned else 0.0,
            "wasted_fire": -0.01 if fire_requested and not bullet_spawned and fire_blocked_reason != "cooldown" else 0.0,
            "missed_shot": -self.missed_shot_penalty if bullet_expired and not bullet_hit else 0.0,
        }

    def _aim_alignment(self, decoded: dict[str, float]) -> float:
        return float(self._aim_debug(self._aim_bin_from_decoded(decoded), decoded)["aim_alignment"])

    def _aim_debug(self, aim_bin: int, decoded: dict[str, float]) -> dict[str, Any]:
        target = self._target_direction()
        target_dir = target["target_dir"]
        aim_dir = normalize_vec({"x": float(decoded["aimX"]), "y": float(decoded["aimY"])})
        has_enemy = target["target_enemy_id"] is not None
        alignment = dot(aim_dir, target_dir) if has_enemy else 0.0
        return {
            "target_enemy_id": target["target_enemy_id"],
            "target_dir": target_dir,
            "aim_dir": aim_dir,
            "aim_bin": int(aim_bin),
            "ideal_aim_bin": vec_to_aim_bin(target_dir) if has_enemy else 0,
            "aim_alignment": alignment,
            "angle_error_deg": angle_between(aim_dir, target_dir) if has_enemy else 180.0,
            "is_aim_aligned": bool(has_enemy and alignment >= self.aim_alignment_threshold),
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

    @staticmethod
    def _aim_bin_from_decoded(decoded: dict[str, float]) -> int:
        return vec_to_aim_bin({"x": float(decoded["aimX"]), "y": float(decoded["aimY"])})

    def _safe_radius(self) -> float:
        progress = min(1.0, self.step_count / max(1, self.max_steps - 1))
        return self.safe_radius_start + (self.safe_radius_end - self.safe_radius_start) * progress

    def _spawn_positions(self) -> dict[str, Vec2]:
        angle = self._spawn_angle()
        distance = self.rng.uniform(0.8 * self.fire_range, 1.2 * self.fire_range)
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
        if not self.randomize_enemy_spawn_direction:
            return self.rng.uniform(-0.35, 0.35)
        base_angles = {
            "right": 0.0,
            "left": math.pi,
            "up": -math.pi / 2.0,
            "down": math.pi / 2.0,
            "upper_right": -math.pi / 4.0,
            "lower_right": math.pi / 4.0,
            "upper_left": -3.0 * math.pi / 4.0,
            "lower_left": 3.0 * math.pi / 4.0,
        }
        direction = self.rng.choice(list(base_angles.values()))
        return direction + self.rng.uniform(-0.18, 0.18)

    def _trajectory_step(
        self,
        raw_action: RawAction,
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
        team_dead = float(self.state["self_hp"]) <= 0.0 and float(self.state["ally_hp"]) <= 0.0
        enemies_dead = float(self.state["enemy_hp"]) <= 0.0
        return self.step_count >= self.max_steps or team_dead or enemies_dead

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
