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

    def summary(self) -> dict[str, float]:
        steps = max(1, len(self.ally_distances))
        pressure_steps = max(1, self.pressure_steps)
        return {
            "avg_ally_distance": sum(self.ally_distances) / steps,
            "isolation_rate": self.isolation_steps / steps,
            "teammate_under_pressure_response": self.pressure_response_steps / pressure_steps,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
        }
