from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

if __package__:
    from .cpc_actions import RawAction, decode_action, random_action
    from .cpc_metrics import CpcMetrics
else:
    from cpc_actions import RawAction, decode_action, random_action
    from cpc_metrics import CpcMetrics


Vec2 = dict[str, float]


class CPCEnv:
    width = 1000.0
    height = 1000.0
    max_hp = 100.0
    move_speed = 35.0
    fire_range = 260.0
    fire_alignment = 0.65
    damage = 10.0
    enemy_damage = 2.0
    enemy_move_speed = 18.0
    survival_reward = 0.001
    center = {"x": 500.0, "y": 500.0}
    safe_radius_start = 420.0
    safe_radius_end = 120.0
    zone_pressure_penalty = 0.08

    def __init__(self, seed: int = 0, max_steps: int = 50):
        self.seed = seed
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.metrics = CpcMetrics()
        self.trajectory: list[dict[str, Any]] = []
        self.step_count = 0
        self.state: dict[str, Any] = {}
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        self.step_count = 0
        self.metrics = CpcMetrics()
        self.trajectory = []
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

        damage_dealt = self._resolve_fire(decoded)
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
        )
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
            },
        }
        self.trajectory.append(self._trajectory_step(raw_action, decoded, obs, reward, done, info))
        return obs, reward, done, info

    def observation(self) -> dict[str, Any]:
        ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
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
            "safe_radius": self._safe_radius(),
            "distance_to_enemy": self._distance(self.state["self_pos"], self.state["enemy_pos"]),
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

    def _resolve_fire(self, decoded: dict[str, float]) -> float:
        if int(decoded["fire"]) != 1:
            return 0.0
        enemy_distance = self._distance(self.state["self_pos"], self.state["enemy_pos"])
        if enemy_distance > self.fire_range:
            return 0.0

        to_enemy_x = self.state["enemy_pos"]["x"] - self.state["self_pos"]["x"]
        to_enemy_y = self.state["enemy_pos"]["y"] - self.state["self_pos"]["y"]
        length = max(math.hypot(to_enemy_x, to_enemy_y), 1e-6)
        alignment = (decoded["aimX"] * to_enemy_x / length) + (decoded["aimY"] * to_enemy_y / length)
        if alignment < self.fire_alignment:
            return 0.0

        damage = min(self.damage, float(self.state["enemy_hp"]))
        self.state["enemy_hp"] = max(0.0, float(self.state["enemy_hp"]) - damage)
        return damage

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
    ) -> dict[str, float]:
        del ally_under_pressure
        alignment = self._aim_alignment(decoded)
        in_range = enemy_distance <= self.fire_range
        outside_safe = self._distance(self.state["self_pos"], self.center) > self._safe_radius()
        self_dead = previous_self_hp > 0.0 and float(self.state["self_hp"]) <= 0.0
        enemy_dead = previous_enemy_hp > 0.0 and float(self.state["enemy_hp"]) <= 0.0
        return {
            "damage_dealt": damage_dealt * 0.10,
            "damage_taken": -damage_taken * 0.05,
            "death": -1.0 if self_dead else 0.0,
            "win": 2.0 if enemy_dead else 0.0,
            "survival": self.survival_reward if float(self.state["self_hp"]) > 0.0 else 0.0,
            "approach_enemy": 0.03 if enemy_distance < previous_enemy_distance else -0.02,
            "aim_alignment": max(0.0, alignment) * 0.02 if in_range else 0.0,
            "attack_intent": 0.05 if int(decoded["fire"]) == 1 and in_range and alignment >= self.fire_alignment else 0.0,
            "zone_pressure": -self.zone_pressure_penalty if outside_safe else 0.0,
        }

    def _aim_alignment(self, decoded: dict[str, float]) -> float:
        to_enemy_x = self.state["enemy_pos"]["x"] - self.state["self_pos"]["x"]
        to_enemy_y = self.state["enemy_pos"]["y"] - self.state["self_pos"]["y"]
        length = max(math.hypot(to_enemy_x, to_enemy_y), 1e-6)
        return (decoded["aimX"] * to_enemy_x / length) + (decoded["aimY"] * to_enemy_y / length)

    def _safe_radius(self) -> float:
        progress = min(1.0, self.step_count / max(1, self.max_steps - 1))
        return self.safe_radius_start + (self.safe_radius_end - self.safe_radius_start) * progress

    def _spawn_positions(self) -> dict[str, Vec2]:
        angle = self.rng.uniform(-0.35, 0.35)
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
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
