from __future__ import annotations

import math
from dataclasses import dataclass, field

if __package__:
    from .cpc_actions import AIM_BINS
else:
    from cpc_actions import AIM_BINS


@dataclass
class CpcMetrics:
    ally_distances: list[float] = field(default_factory=list)
    isolation_steps: int = 0
    pressure_steps: int = 0
    pressure_response_steps: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    enemy_hp: float = 100.0
    self_hp: float = 100.0
    enemy_max_hp: float = 100.0
    self_max_hp: float = 100.0
    total_return: float = 0.0
    reward_component_sums: dict[str, float] = field(default_factory=dict)
    aim_alignment_sum: float = 0.0
    aim_bin_counts: list[int] = field(default_factory=lambda: [0] * AIM_BINS)
    ideal_aim_bin_counts: list[int] = field(default_factory=lambda: [0] * AIM_BINS)
    exact_aim_match_count: int = 0
    within_1_bin_aim_count: int = 0
    bad_aim_count: int = 0
    shot_fired_count: int = 0
    shot_exact_aim_count: int = 0
    shot_near_aim_count: int = 0
    shot_off_target_count: int = 0
    off_target_shot_count: int = 0
    fire_requested_count: int = 0
    bullet_hit_count: int = 0
    missed_shot_count: int = 0
    enemy_dead: bool = False
    self_dead: bool = False
    enemy_distance_sum: float = 0.0
    good_range_steps: int = 0
    too_close_steps: int = 0
    too_far_steps: int = 0
    outside_safe_zone_steps: int = 0
    near_edge_outward_count: int = 0
    isolation_threshold: float = 220.0

    def update(
        self,
        *,
        ally_distance: float,
        ally_under_pressure: bool,
        moved_toward_ally: bool,
        fired_under_pressure: bool,
        damage_dealt: float,
        damage_taken: float,
        enemy_hp: float = 100.0,
        self_hp: float = 100.0,
        reward: float = 0.0,
        reward_components: dict[str, float] | None = None,
        fire_requested: bool = False,
        aim_alignment: float = 0.0,
        aim_bin: int = 0,
        ideal_aim_bin: int = 0,
        aim_bin_error: int = 0,
        shot_fired: bool = False,
        off_target_shot: bool = False,
        bullet_hit: bool = False,
        missed_shot: bool = False,
        distance_to_enemy: float = 0.0,
        in_good_range: bool = False,
        too_close: bool = False,
        too_far: bool = False,
        outside_safe_zone: bool = False,
        near_edge_outward: bool = False,
    ) -> None:
        self.ally_distances.append(float(ally_distance))
        if ally_distance > self.isolation_threshold:
            self.isolation_steps += 1
        if ally_under_pressure:
            self.pressure_steps += 1
            if moved_toward_ally or fired_under_pressure:
                self.pressure_response_steps += 1
        self.damage_dealt += float(damage_dealt)
        self.damage_taken += float(damage_taken)
        self.enemy_hp = float(enemy_hp)
        self.self_hp = float(self_hp)
        self.enemy_dead = self.enemy_hp <= 0.0
        self.self_dead = self.self_hp <= 0.0
        self.total_return += float(reward)
        for key, value in (reward_components or {}).items():
            self.reward_component_sums[key] = self.reward_component_sums.get(key, 0.0) + float(value)
        self.aim_alignment_sum += float(aim_alignment)
        self.aim_bin_counts[int(aim_bin) % AIM_BINS] += 1
        self.ideal_aim_bin_counts[int(ideal_aim_bin) % AIM_BINS] += 1
        self.exact_aim_match_count += int(int(aim_bin_error) == 0)
        self.within_1_bin_aim_count += int(int(aim_bin_error) <= 1)
        self.bad_aim_count += int(int(aim_bin_error) >= 3)
        self.fire_requested_count += int(bool(fire_requested))
        if shot_fired:
            self.shot_fired_count += 1
            self.shot_exact_aim_count += int(int(aim_bin_error) == 0)
            self.shot_near_aim_count += int(int(aim_bin_error) == 1)
            self.shot_off_target_count += int(int(aim_bin_error) >= 2)
        self.off_target_shot_count += int(bool(off_target_shot))
        self.bullet_hit_count += int(bool(bullet_hit))
        self.missed_shot_count += int(bool(missed_shot))
        self.enemy_distance_sum += float(distance_to_enemy)
        self.good_range_steps += int(bool(in_good_range))
        self.too_close_steps += int(bool(too_close))
        self.too_far_steps += int(bool(too_far))
        self.outside_safe_zone_steps += int(bool(outside_safe_zone))
        self.near_edge_outward_count += int(bool(near_edge_outward))

    def summary(self) -> dict[str, float]:
        steps = max(1, len(self.ally_distances))
        pressure_steps = max(1, self.pressure_steps)
        shot_steps = max(1, self.shot_fired_count)
        damage_dealt_ratio = self.damage_dealt / max(1.0, self.enemy_max_hp)
        damage_taken_ratio = self.damage_taken / max(1.0, self.self_max_hp)
        summary = {f"reward_{key}": value for key, value in sorted(self.reward_component_sums.items())}
        summary.update({
            "avg_ally_distance": sum(self.ally_distances) / steps,
            "isolation_rate": self.isolation_steps / steps,
            "teammate_under_pressure_response": self.pressure_response_steps / pressure_steps,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "damage_dealt_ratio": damage_dealt_ratio,
            "damage_taken_ratio": damage_taken_ratio,
            "damage_trade_ratio": damage_dealt_ratio - damage_taken_ratio,
            "enemy_hp_remaining_ratio": max(0.0, self.enemy_hp) / max(1.0, self.enemy_max_hp),
            "self_hp_remaining_ratio": max(0.0, self.self_hp) / max(1.0, self.self_max_hp),
            "kill_rate": float(self.enemy_dead),
            "enemy_dead": float(self.enemy_dead),
            "self_dead": float(self.self_dead),
            "survival_steps": float(len(self.ally_distances)),
            "mean_aim_alignment": self.aim_alignment_sum / steps,
            "aim_bin_0_rate": self.aim_bin_counts[0] / steps,
            "aim_bin_entropy": _entropy(self.aim_bin_counts),
            "ideal_aim_bin_distribution": {
                str(index): float(count) for index, count in enumerate(self.ideal_aim_bin_counts) if count
            },
            "exact_aim_match_rate": self.exact_aim_match_count / steps,
            "within_1_bin_aim_rate": self.within_1_bin_aim_count / steps,
            "bad_aim_rate": self.bad_aim_count / steps,
            "shot_exact_aim_rate": self.shot_exact_aim_count / shot_steps,
            "shot_near_aim_rate": self.shot_near_aim_count / shot_steps,
            "shot_off_target_rate": self.shot_off_target_count / shot_steps,
            "bullet_hit_per_shot": self.bullet_hit_count / shot_steps,
            "fire_requested_count": float(self.fire_requested_count),
            "shot_fired_count": float(self.shot_fired_count),
            "off_target_shot_count": float(self.off_target_shot_count),
            "bullet_hit_count": float(self.bullet_hit_count),
            "missed_shot_count": float(self.missed_shot_count),
            "hit_ratio": self.bullet_hit_count / shot_steps,
            "missed_shot_rate": self.missed_shot_count / shot_steps,
            "avg_distance_to_enemy": self.enemy_distance_sum / steps,
            "good_range_rate": self.good_range_steps / steps,
            "too_close_rate": self.too_close_steps / steps,
            "too_far_rate": self.too_far_steps / steps,
            "total_return": self.total_return,
            "mean_step_reward": self.total_return / steps,
            "outside_safe_zone_rate": self.outside_safe_zone_steps / steps,
            "near_edge_outward_count": float(self.near_edge_outward_count),
        })
        return summary


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy
