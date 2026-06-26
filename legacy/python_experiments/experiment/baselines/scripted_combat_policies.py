from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from experiment.training.cpc_actions import AIM_BINS, RawAction, aim_bin_to_vec, circular_bin_distance, vec_to_aim_bin
    from experiment.training.cpc_env import CPCEnv, normalize_vec
except ModuleNotFoundError:
    from training.cpc_actions import AIM_BINS, RawAction, aim_bin_to_vec, circular_bin_distance, vec_to_aim_bin
    from training.cpc_env import CPCEnv, normalize_vec


VALID_MODES = ("stand_still", "keep_range", "approach_until_in_range")


@dataclass
class ScriptedAction:
    action: RawAction
    diagnostics: dict[str, Any]


class ScriptedAimAtEnemyPolicy:
    """Deterministic projectile-mechanics baseline; not a learning policy."""

    def __init__(self, mode: str = "stand_still", num_aim_bins: int = AIM_BINS) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.num_aim_bins = int(num_aim_bins)

    def act(self, observation: dict[str, Any], env: CPCEnv) -> RawAction:
        return self.act_with_diagnostics(observation, env).action

    def act_with_diagnostics(self, observation: dict[str, Any], env: CPCEnv) -> ScriptedAction:
        self_pos = dict(env.state["self_pos"])
        enemy_pos = dict(env.state["enemy_pos"])
        delta = {
            "x": float(enemy_pos["x"]) - float(self_pos["x"]),
            "y": float(enemy_pos["y"]) - float(self_pos["y"]),
        }
        distance = math.hypot(delta["x"], delta["y"])
        target_dir = normalize_vec(delta)
        selected_aim_bin = vec_to_aim_bin(target_dir, self.num_aim_bins)
        selected_aim_dir = aim_bin_to_vec(selected_aim_bin, self.num_aim_bins)
        ideal_aim_bin = vec_to_aim_bin(target_dir, self.num_aim_bins)
        aim_bin_error = circular_bin_distance(selected_aim_bin, ideal_aim_bin, self.num_aim_bins)
        can_fire = bool(observation.get("can_fire", int(env.weapon.get("cooldown_remaining_steps", 0)) <= 0))
        action: RawAction = {
            "move": self._move_bin(env, distance, target_dir),
            "aim": selected_aim_bin,
            "fire": 1 if can_fire else 0,
        }
        diagnostics = {
            "mode": self.mode,
            "self_pos": self_pos,
            "enemy_pos": enemy_pos,
            "distance_to_enemy": distance,
            "target_dir": target_dir,
            "selected_aim_bin": selected_aim_bin,
            "selected_aim_dir": selected_aim_dir,
            "ideal_aim_bin": ideal_aim_bin,
            "aim_bin_error": aim_bin_error,
            "angle_error_deg": _angle_between(selected_aim_dir, target_dir),
            "can_fire": can_fire,
        }
        return ScriptedAction(action=action, diagnostics=diagnostics)

    def _move_bin(self, env: CPCEnv, distance: float, target_dir: dict[str, float]) -> int:
        if self.mode == "stand_still":
            return 0
        if self.mode == "approach_until_in_range":
            return _nearest_move_bin(target_dir) if distance > env.fire_range * 0.7 else 0
        if self.mode == "keep_range":
            range_debug = env._range_debug(distance)
            if range_debug["too_far"]:
                return _nearest_move_bin(target_dir)
            if range_debug["too_close"]:
                return _nearest_move_bin({"x": -target_dir["x"], "y": -target_dir["y"]})
        return 0


def _nearest_move_bin(direction: dict[str, float]) -> int:
    x = float(direction["x"])
    y = float(direction["y"])
    if abs(x) < 0.25 and abs(y) < 0.25:
        return 0
    horizontal = 4 if x > 0.25 else 3 if x < -0.25 else 0
    vertical = 2 if y > 0.25 else 1 if y < -0.25 else 0
    if horizontal and vertical:
        return {
            (3, 1): 5,
            (4, 1): 6,
            (3, 2): 7,
            (4, 2): 8,
        }[(horizontal, vertical)]
    return horizontal or vertical


def _angle_between(a: dict[str, float], b: dict[str, float]) -> float:
    ax = float(a["x"])
    ay = float(a["y"])
    bx = float(b["x"])
    by = float(b["y"])
    denom = max(1e-9, math.hypot(ax, ay) * math.hypot(bx, by))
    cosine = max(-1.0, min(1.0, ((ax * bx) + (ay * by)) / denom))
    return math.degrees(math.acos(cosine))
