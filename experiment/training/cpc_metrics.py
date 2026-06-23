from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CpcMetrics:
    ally_distances: list[float] = field(default_factory=list)
    isolation_steps: int = 0
    pressure_steps: int = 0
    pressure_response_steps: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    aim_alignment_sum: float = 0.0
    off_target_shot_count: int = 0
    bullet_hit_count: int = 0
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
        aim_alignment: float = 0.0,
        off_target_shot: bool = False,
        bullet_hit: bool = False,
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
        self.aim_alignment_sum += float(aim_alignment)
        self.off_target_shot_count += int(bool(off_target_shot))
        self.bullet_hit_count += int(bool(bullet_hit))
        self.outside_safe_zone_steps += int(bool(outside_safe_zone))
        self.near_edge_outward_count += int(bool(near_edge_outward))

    def summary(self) -> dict[str, float]:
        steps = max(1, len(self.ally_distances))
        pressure_steps = max(1, self.pressure_steps)
        return {
            "avg_ally_distance": sum(self.ally_distances) / steps,
            "isolation_rate": self.isolation_steps / steps,
            "teammate_under_pressure_response": self.pressure_response_steps / pressure_steps,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "mean_aim_alignment": self.aim_alignment_sum / steps,
            "off_target_shot_count": float(self.off_target_shot_count),
            "bullet_hit_count": float(self.bullet_hit_count),
            "outside_safe_zone_rate": self.outside_safe_zone_steps / steps,
            "near_edge_outward_count": float(self.near_edge_outward_count),
        }
