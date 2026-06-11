from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from features import (
    DEFAULT_THRESHOLDS,
    LABEL_TO_INDEX,
    extract_model_features,
    feature_vector_from_features,
    manhattan_distance,
)

SCENARIO_INTENTS = {
    "direct_enemy_contact": "attack_nearest_enemy",
    "teammate_under_pressure": "support_teammate_under_pressure",
    "isolated_teammate": "reduce_isolation",
    "self_low_hp": "retreat_when_low_hp",
}

SCENARIO_ORDER = list(SCENARIO_INTENTS.keys())


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label: str


SCENARIO_SPECS = [ScenarioSpec(scenario_id=scenario_id, label=label) for scenario_id, label in SCENARIO_INTENTS.items()]


def _hash_to_seed(seed: int, scenario_id: str, sample_index: int, attempt: int) -> int:
    digest = hashlib.sha256(f"{seed}:{scenario_id}:{sample_index}:{attempt}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _clamp(value: int, low: int = 0, high: int = 19) -> int:
    return max(low, min(high, value))


def _player(player_id: str, team_id: str, role: str, x: int, y: int, hp: int, cooldown: int, alive: bool = True) -> Dict[str, object]:
    return {
        "id": player_id,
        "teamId": team_id,
        "role": role,
        "x": _clamp(x),
        "y": _clamp(y),
        "hp": int(hp),
        "alive": bool(alive),
        "weaponCooldownSteps": int(cooldown),
    }


def _sample_cooldown(rng: random.Random, sample_index: int) -> int:
    return 0 if sample_index % 2 == 0 else rng.randint(1, 5)


def _sample_enemy_hp(rng: random.Random, near_threshold: bool, sample_index: int) -> int:
    if near_threshold:
        return rng.randint(30, 50)
    if sample_index % 6 == 0:
        return rng.randint(60, 79)
    return rng.randint(80, 100)


def _scenario_rng(seed: int, scenario_id: str, sample_index: int, attempt: int) -> random.Random:
    return random.Random(_hash_to_seed(seed, scenario_id, sample_index, attempt))


def _build_direct_enemy_contact(rng: random.Random, near_threshold: bool, sample_index: int) -> Dict[str, object]:
    self_x, self_y = 5 + rng.randint(-1, 1), 5 + rng.randint(-1, 1)
    ally_x, ally_y = self_x, self_y + (6 if near_threshold else 5)
    enemy0_x, enemy0_y = self_x + (5 if near_threshold else 4), self_y
    enemy1_x = self_x + rng.randint(7, 12)
    enemy1_y = self_y + rng.randint(-4, 4)

    return {
        "players": [
            _player("team-a-0", "team-a", "self", self_x, self_y, rng.randint(36, 45) if near_threshold else rng.randint(80, 100), _sample_cooldown(rng, sample_index)),
            _player("team-a-1", "team-a", "ally", ally_x, ally_y, rng.randint(80, 100), 0),
            _player("team-b-0", "team-b", "enemy", enemy0_x, enemy0_y, _sample_enemy_hp(rng, near_threshold, sample_index), 0),
            _player("team-b-1", "team-b", "enemy", enemy1_x, enemy1_y, _sample_enemy_hp(rng, near_threshold, sample_index + 1), 0),
        ]
    }


def _build_teammate_under_pressure(rng: random.Random, near_threshold: bool, sample_index: int) -> Dict[str, object]:
    self_x, self_y = 3 + rng.randint(-1, 1), 3 + rng.randint(-1, 1)
    ally_distance = rng.randint(2, 5) if near_threshold else rng.randint(6, 10)
    ally_x, ally_y = self_x + ally_distance, self_y
    enemy0_x, enemy0_y = ally_x + (5 if near_threshold else 4), ally_y
    enemy1_x = self_x + rng.randint(5, 11)
    enemy1_y = self_y + rng.randint(-5, 5)

    return {
        "players": [
            _player("team-a-0", "team-a", "self", self_x, self_y, rng.randint(36, 48) if near_threshold else rng.randint(80, 100), _sample_cooldown(rng, sample_index)),
            _player("team-a-1", "team-a", "ally", ally_x, ally_y, rng.randint(12, 35) if near_threshold else rng.randint(15, 30), 0),
            _player("team-b-0", "team-b", "enemy", enemy0_x, enemy0_y, _sample_enemy_hp(rng, near_threshold, sample_index), 0),
            _player("team-b-1", "team-b", "enemy", enemy1_x, enemy1_y, _sample_enemy_hp(rng, near_threshold, sample_index + 1), 0),
        ]
    }


def _build_isolated_teammate(rng: random.Random, near_threshold: bool, sample_index: int) -> Dict[str, object]:
    self_x, self_y = 2 + rng.randint(0, 1), 2 + rng.randint(0, 1)
    ally_distance = rng.randint(7, 8) if near_threshold else rng.randint(10, 14)
    ally_x, ally_y = self_x + ally_distance, self_y + ally_distance
    enemy0_x, enemy0_y = self_x + rng.randint(10, 14), self_y + rng.randint(0, 4)
    enemy1_x = self_x + rng.randint(8, 13)
    enemy1_y = self_y + rng.randint(4, 9)

    return {
        "players": [
            _player("team-a-0", "team-a", "self", self_x, self_y, rng.randint(40, 60) if near_threshold else rng.randint(80, 100), _sample_cooldown(rng, sample_index)),
            _player("team-a-1", "team-a", "ally", ally_x, ally_y, rng.randint(80, 100), 0),
            _player("team-b-0", "team-b", "enemy", enemy0_x, enemy0_y, _sample_enemy_hp(rng, near_threshold, sample_index), 0),
            _player("team-b-1", "team-b", "enemy", enemy1_x, enemy1_y, _sample_enemy_hp(rng, near_threshold, sample_index + 1), 0),
        ]
    }


def _build_self_low_hp(rng: random.Random, near_threshold: bool, sample_index: int) -> Dict[str, object]:
    self_x, self_y = 5 + rng.randint(-1, 1), 5 + rng.randint(-1, 1)
    ally_x, ally_y = self_x + 3, self_y + 1
    enemy0_x, enemy0_y = self_x + (5 if near_threshold else 2), self_y
    enemy1_x = self_x + rng.randint(6, 11)
    enemy1_y = self_y + rng.randint(-4, 4)

    return {
        "players": [
            _player("team-a-0", "team-a", "self", self_x, self_y, rng.randint(32, 35) if near_threshold else rng.randint(5, 30), _sample_cooldown(rng, sample_index)),
            _player("team-a-1", "team-a", "ally", ally_x, ally_y, rng.randint(80, 100), 0),
            _player("team-b-0", "team-b", "enemy", enemy0_x, enemy0_y, _sample_enemy_hp(rng, near_threshold, sample_index), 0),
            _player("team-b-1", "team-b", "enemy", enemy1_x, enemy1_y, _sample_enemy_hp(rng, near_threshold, sample_index + 1), 0),
        ]
    }


def _build_state_for_scenario(scenario_id: str, rng: random.Random, near_threshold: bool, sample_index: int) -> Dict[str, object]:
    if scenario_id == "direct_enemy_contact":
        return _build_direct_enemy_contact(rng, near_threshold, sample_index)
    if scenario_id == "teammate_under_pressure":
        return _build_teammate_under_pressure(rng, near_threshold, sample_index)
    if scenario_id == "isolated_teammate":
        return _build_isolated_teammate(rng, near_threshold, sample_index)
    if scenario_id == "self_low_hp":
        return _build_self_low_hp(rng, near_threshold, sample_index)
    raise ValueError(f"unknown scenario_id: {scenario_id}")


def _compute_predicate_debug(state: Dict[str, object]) -> Dict[str, object]:
    players = list(state.get("players", []))
    self_player = next((player for player in players if player.get("role") == "self"), None)
    ally_player = next((player for player in players if player.get("role") == "ally"), None)
    enemy_players = [player for player in players if player.get("role") == "enemy"]

    if self_player is None or ally_player is None or len(enemy_players) < 2:
        raise ValueError("state must contain one self, one ally, and two enemy players")

    self_xy = {"x": int(self_player["x"]), "y": int(self_player["y"])}
    ally_xy = {"x": int(ally_player["x"]), "y": int(ally_player["y"])}
    enemy0_xy = {"x": int(enemy_players[0]["x"]), "y": int(enemy_players[0]["y"])}
    enemy1_xy = {"x": int(enemy_players[1]["x"]), "y": int(enemy_players[1]["y"])}

    thresholds = DEFAULT_THRESHOLDS
    ally_distance = manhattan_distance(self_xy, ally_xy)
    enemy0_distance = manhattan_distance(self_xy, enemy0_xy)
    enemy1_distance = manhattan_distance(self_xy, enemy1_xy)
    enemy_near_ally_distance = min(
        manhattan_distance(ally_xy, enemy0_xy),
        manhattan_distance(ally_xy, enemy1_xy),
    )

    self_low_hp = int(int(self_player.get("hp", 0)) <= thresholds["low_hp"])
    ally_low_hp = int(int(ally_player.get("hp", 0)) <= thresholds["low_hp"])
    enemy_nearby = int(min(enemy0_distance, enemy1_distance) <= thresholds["enemy_threat_range"])
    enemy_near_ally = int(enemy_near_ally_distance <= thresholds["enemy_threat_range"])
    ally_under_pressure = int(ally_low_hp and enemy_near_ally)
    is_isolated = int(ally_distance > thresholds["isolation_range"])
    can_fire = int(bool(self_player.get("alive", False)) and int(self_player.get("weaponCooldownSteps", 0)) <= 0)

    return {
        "selfLowHp": self_low_hp,
        "allyLowHp": ally_low_hp,
        "allyUnderPressure": ally_under_pressure,
        "isIsolated": is_isolated,
        "enemyNearby": enemy_nearby,
        "enemyNearAlly": enemy_near_ally,
        "canFire": can_fire,
        "distances": {
            "ally": ally_distance,
            "enemy0": enemy0_distance,
            "enemy1": enemy1_distance,
            "enemyNearAlly": enemy_near_ally_distance,
        },
    }


def _scenario_constraints(scenario_id: str, predicate_debug: Dict[str, object]) -> bool:
    if scenario_id == "direct_enemy_contact":
        return predicate_debug["enemyNearby"] == 1 and predicate_debug["allyUnderPressure"] == 0 and predicate_debug["isIsolated"] == 0 and predicate_debug["selfLowHp"] == 0
    if scenario_id == "teammate_under_pressure":
        return predicate_debug["allyUnderPressure"] == 1 and predicate_debug["allyLowHp"] == 1 and predicate_debug["enemyNearAlly"] == 1 and predicate_debug["selfLowHp"] == 0
    if scenario_id == "isolated_teammate":
        return predicate_debug["isIsolated"] == 1 and predicate_debug["allyUnderPressure"] == 0 and predicate_debug["enemyNearby"] == 0 and predicate_debug["selfLowHp"] == 0
    if scenario_id == "self_low_hp":
        return predicate_debug["selfLowHp"] == 1 and predicate_debug["enemyNearby"] == 1 and predicate_debug["allyUnderPressure"] == 0
    return False


def _hash_sample_order(seed: int, scenario_id: str, sample_index: int) -> float:
    return random.Random(_hash_to_seed(seed, scenario_id, sample_index, 9999)).random()


def _build_sample(seed: int, scenario_id: str, sample_index: int, split: str) -> Dict[str, object]:
    near_threshold = (sample_index % 5) == 0
    for attempt in range(500):
        rng = _scenario_rng(seed, scenario_id, sample_index, attempt)
        state = _build_state_for_scenario(scenario_id, rng, near_threshold, sample_index)
        predicate_debug = _compute_predicate_debug(state)
        if not _scenario_constraints(scenario_id, predicate_debug):
            continue

        features = extract_model_features(state, DEFAULT_THRESHOLDS)
        label = SCENARIO_INTENTS[scenario_id]
        sample = {
            "sampleId": f"{scenario_id}-{sample_index + 1:04d}",
            "schemaVersion": "intent-dataset-v0.2",
            "scenarioId": scenario_id,
            "split": split,
            "label": label,
            "labelIndex": LABEL_TO_INDEX[label],
            "features": features,
            "featureVector": feature_vector_from_features(features),
            "predicateDebug": predicate_debug,
            "state": state,
            "gt": {"intent": label},
        }
        return sample

    raise RuntimeError(f"failed to generate valid sample for {scenario_id} after many attempts")


def generate_dataset(num_per_scenario: int = 100, eval_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    all_samples: List[Dict[str, object]] = []
    grouped = {"train": [], "eval": []}

    eval_count = max(1, min(num_per_scenario - 1, int(round(num_per_scenario * eval_ratio))))
    train_count = num_per_scenario - eval_count
    if train_count <= 0:
        raise ValueError("train split would be empty; reduce eval_ratio or increase num_per_scenario")

    for scenario_id in SCENARIO_ORDER:
        scenario_samples = []
        for sample_index in range(num_per_scenario):
            sample = _build_sample(seed, scenario_id, sample_index, "train")
            scenario_samples.append((_hash_sample_order(seed, scenario_id, sample_index), sample))

        scenario_samples.sort(key=lambda item: item[0])
        eval_bucket = [sample for _, sample in scenario_samples[:eval_count]]
        train_bucket = [sample for _, sample in scenario_samples[eval_count:]]

        for sample in eval_bucket:
            sample["split"] = "eval"
            grouped["eval"].append(sample)
            all_samples.append(sample)
        for sample in train_bucket:
            grouped["train"].append(sample)
            all_samples.append(sample)

    return all_samples, grouped

