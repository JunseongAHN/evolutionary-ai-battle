from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

try:
    from cpc_actions import RawAction, decode_action, random_action
    from cpc_metrics import CpcMetrics
except ImportError:
    from .cpc_actions import RawAction, decode_action, random_action
    from .cpc_metrics import CpcMetrics


Vec2 = dict[str, float]


class CPCEnv:
    width = 1000.0
    height = 1000.0
    max_hp = 100.0
    move_speed = 35.0
    fire_range = 260.0
    fire_alignment = 0.65
    damage = 10.0

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
        self.state = {
            "self_hp": self.max_hp,
            "ally_hp": self.max_hp,
            "enemy_hp": self.max_hp,
            "self_pos": {"x": 260.0, "y": 500.0},
            "ally_pos": {"x": 380.0, "y": 560.0},
            "enemy_pos": {"x": 650.0, "y": 540.0},
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
        previous_enemy_hp = float(self.state["enemy_hp"])
        previous_self_hp = float(self.state["self_hp"])

        self._move_self(decoded["moveX"], decoded["moveY"])
        ally_under_pressure = self._ally_under_pressure()

        damage_dealt = self._resolve_fire(decoded)
        damage_taken = self._resolve_enemy_pressure()
        self.step_count += 1

        ally_distance = self._distance(self.state["self_pos"], self.state["ally_pos"])
        moved_toward_ally = ally_distance < previous_ally_distance
        reward_components = self._reward_components(
            ally_distance=ally_distance,
            ally_under_pressure=ally_under_pressure,
            moved_toward_ally=moved_toward_ally,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
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
            damage_taken = 3.0
            self.state["self_hp"] = max(0.0, float(self.state["self_hp"]) - damage_taken)
        return damage_taken

    def _reward_components(
        self,
        *,
        ally_distance: float,
        ally_under_pressure: bool,
        moved_toward_ally: bool,
        damage_dealt: float,
        damage_taken: float,
    ) -> dict[str, float]:
        return {
            "survival": 0.02 if float(self.state["self_hp"]) > 0.0 else -1.0,
            "ally_support": 0.05 if ally_under_pressure and moved_toward_ally else 0.0,
            "damage": damage_dealt * 0.05,
            "pressure_response": 0.05 if ally_under_pressure and damage_dealt > 0.0 else 0.0,
            "isolation": -0.03 if ally_distance > self.metrics.isolation_threshold else 0.0,
            "self_preservation": 0.03 if bool(self.observation()["self_low_hp"]) and damage_taken <= 0.0 else 0.0,
            "damage_taken": -damage_taken * 0.03,
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
