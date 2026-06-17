from __future__ import annotations

import math
import random
from copy import deepcopy

from .schema import (
    AgentId,
    BattleAction,
    BattleConfig,
    BattleEvent,
    BattleSnapshot,
    MultiAgentAction,
    MultiAgentStep,
    SCHEMA_VERSION,
    TacticalObservation,
)
from .vectorizer import DEFAULT_VECTOR_KEYS, build_observation_vector

DEFAULT_AGENT_IDS = ["team-a-0", "team-a-1", "team-b-0", "team-b-1"]
DEFAULT_TEAM_MAP = {
    "team-a-0": "team-a",
    "team-a-1": "team-a",
    "team-b-0": "team-b",
    "team-b-1": "team-b",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm(x: float, y: float) -> float:
    return math.sqrt((x * x) + (y * y))


def _normalize(x: float, y: float) -> tuple[float, float]:
    length = _norm(x, y)
    if length <= 1e-6:
        return 0.0, 0.0
    return x / length, y / length


def _distance(a: dict, b: dict) -> float:
    return _norm(a["position"]["x"] - b["position"]["x"], a["position"]["y"] - b["position"]["y"])


class PythonBattleCoreEnv:
    width = 1000.0
    height = 1000.0
    max_steps = 500
    max_hp = 100.0
    move_speed = 20.0
    fire_range = 260.0
    damage = 10.0

    def __init__(self):
        self.agent_ids: list[AgentId] = list(DEFAULT_AGENT_IDS)
        self.team_ids = ["team-a", "team-b"]
        self.agent_team_map = dict(DEFAULT_TEAM_MAP)
        self.episode_id = "episode-0"
        self.step_index = 0
        self.rng = random.Random(0)
        self.agents: dict[AgentId, dict] = {}
        self.recent_events: list[BattleEvent] = []
        self.last_damage_dealt: dict[AgentId, float] = {}
        self.last_damage_taken: dict[AgentId, float] = {}

    @property
    def config(self) -> BattleConfig:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "duo",
            "team_count": 2,
            "players_per_team": 2,
            "max_steps": self.max_steps,
            "map": {
                "width": self.width,
                "height": self.height,
                "coordinate_system": "world-2d",
            },
            "observation_spec": {
                "mode": "local_tactical",
                "vector_keys": list(DEFAULT_VECTOR_KEYS),
                "max_visible_enemies": 3,
                "max_visible_allies": 1,
                "max_visible_obstacles": 0,
                "max_recent_events": 8,
                "entity_feature_keys": {
                    "enemy": ["relative_x", "relative_y", "distance", "hp", "alive"],
                    "ally": ["relative_x", "relative_y", "distance", "hp", "alive"],
                    "obstacle": [],
                    "event": ["event_type", "age_steps", "relative_x", "relative_y", "value"],
                },
            },
            "action_spec": {
                "action_type": "continuous_2d",
                "action_keys": ["move_x", "move_y", "aim_x", "aim_y", "fire"],
                "bounds": {
                    "move_x": [-1.0, 1.0],
                    "move_y": [-1.0, 1.0],
                    "aim_x": [-1.0, 1.0],
                    "aim_y": [-1.0, 1.0],
                    "fire": [0.0, 1.0],
                },
            },
        }

    def reset(self, seed: int | None = None) -> dict[AgentId, TacticalObservation]:
        if seed is None:
            seed = 0
        self.rng.seed(seed)
        self.episode_id = f"python-core-{seed}"
        self.step_index = 0
        self.recent_events = []
        self.last_damage_dealt = {}
        self.last_damage_taken = {}
        self.agents = {
            "team-a-0": self._agent("team-a-0", 180.0, 430.0, 1.0, 0.0),
            "team-a-1": self._agent("team-a-1", 180.0, 570.0, 1.0, 0.0),
            "team-b-0": self._agent("team-b-0", 820.0, 430.0, -1.0, 0.0),
            "team-b-1": self._agent("team-b-1", 820.0, 570.0, -1.0, 0.0),
        }
        self.recent_events = [
            self._event("spawn", agent_id, team_id=self.agent_team_map[agent_id], position=self.agents[agent_id]["position"])
            for agent_id in self.agent_ids
        ]
        return self._observations()

    def step(self, actions: MultiAgentAction) -> MultiAgentStep:
        events: list[BattleEvent] = []
        self.last_damage_dealt = {}
        self.last_damage_taken = {}

        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            if not agent["alive"]:
                continue
            action = actions["actions"].get(agent_id, self._no_op_action(agent_id))["action"]
            move_x = _clamp(float(action.get("move_x", 0.0)), -1.0, 1.0)
            move_y = _clamp(float(action.get("move_y", 0.0)), -1.0, 1.0)
            move_len = _norm(move_x, move_y)
            if move_len > 1.0:
                move_x, move_y = move_x / move_len, move_y / move_len

            old_x = agent["position"]["x"]
            old_y = agent["position"]["y"]
            agent["position"]["x"] = _clamp(old_x + (move_x * self.move_speed), 0.0, self.width)
            agent["position"]["y"] = _clamp(old_y + (move_y * self.move_speed), 0.0, self.height)
            if agent["position"]["x"] != old_x or agent["position"]["y"] != old_y:
                events.append(self._event("move", agent_id, position=deepcopy(agent["position"])))

            aim_x = float(action.get("aim_x", 0.0))
            aim_y = float(action.get("aim_y", 0.0))
            aim_nx, aim_ny = _normalize(aim_x, aim_y)
            if aim_nx or aim_ny:
                agent["aim"] = {"x": aim_nx, "y": aim_ny}
                agent["facing"] = {"x": aim_nx, "y": aim_ny}

        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            if not agent["alive"]:
                continue
            action = actions["actions"].get(agent_id, self._no_op_action(agent_id))["action"]
            if float(action.get("fire", 0.0)) <= 0.5:
                continue
            target = self._nearest_alive_enemy(agent_id)
            hit = target is not None and _distance(agent, target) <= self.fire_range
            events.append(self._event(
                "fire",
                agent_id,
                target_id=target["agent_id"] if target else None,
                position=deepcopy(agent["position"]),
                metadata={"wasteful": not hit},
            ))
            if not hit or target is None:
                continue
            damage = min(self.damage, target["hp"])
            target["hp"] = max(0.0, target["hp"] - self.damage)
            self.last_damage_dealt[agent_id] = self.last_damage_dealt.get(agent_id, 0.0) + damage
            self.last_damage_taken[target["agent_id"]] = self.last_damage_taken.get(target["agent_id"], 0.0) + damage
            events.append(self._event("damage", agent_id, target["agent_id"], value=damage, position=deepcopy(target["position"])))
            if target["hp"] <= 0.0 and target["alive"]:
                target["alive"] = False
                events.append(self._event("death", agent_id, target["agent_id"], position=deepcopy(target["position"])))

        self.step_index += 1
        self.recent_events = (self.recent_events + events)[-8:]
        observations = self._observations()
        terminated = self._one_team_remaining()
        truncated = self.step_index >= self.max_steps
        step: MultiAgentStep = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step": self.step_index,
            "observations": observations,
            "actions": actions["actions"],
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "snapshot": self._snapshot(events),
                "events": events,
            },
        }
        return step

    def _agent(self, agent_id: AgentId, x: float, y: float, aim_x: float, aim_y: float) -> dict:
        return {
            "agent_id": agent_id,
            "team_id": self.agent_team_map[agent_id],
            "position": {"x": x, "y": y},
            "hp": self.max_hp,
            "alive": True,
            "facing": {"x": aim_x, "y": aim_y},
            "aim": {"x": aim_x, "y": aim_y},
        }

    def _event(
        self,
        event_type,
        actor_id: AgentId,
        target_id: AgentId | None = None,
        team_id: str | None = None,
        position: dict | None = None,
        value: float | None = None,
        metadata: dict | None = None,
    ) -> BattleEvent:
        event: BattleEvent = {
            "event_id": f"event-{self.step_index}-{len(self.recent_events)}-{actor_id}-{event_type}",
            "step": self.step_index,
            "event_type": event_type,
            "actor_id": actor_id,
        }
        if target_id is not None:
            event["target_id"] = target_id
        if team_id is not None:
            event["team_id"] = team_id
        if position is not None:
            event["position"] = position
        if value is not None:
            event["value"] = value
        if metadata is not None:
            event["metadata"] = metadata
        return event

    def _no_op_action(self, agent_id: AgentId) -> BattleAction:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step": self.step_index,
            "agent_id": agent_id,
            "action": {"move_x": 0.0, "move_y": 0.0, "aim_x": 1.0, "aim_y": 0.0, "fire": 0.0},
            "source": {"policy_type": "random", "policy_id": "no-op"},
        }

    def _nearest_alive_enemy(self, agent_id: AgentId) -> dict | None:
        agent = self.agents[agent_id]
        enemies = [
            candidate for candidate in self.agents.values()
            if candidate["alive"] and candidate["team_id"] != agent["team_id"]
        ]
        if not enemies:
            return None
        return min(enemies, key=lambda candidate: _distance(agent, candidate))

    def _visible_entities(self, agent_id: AgentId, same_team: bool) -> list[dict]:
        agent = self.agents[agent_id]
        entities = []
        for candidate in self.agents.values():
            if candidate["agent_id"] == agent_id:
                continue
            if (candidate["team_id"] == agent["team_id"]) != same_team:
                continue
            rel_x = candidate["position"]["x"] - agent["position"]["x"]
            rel_y = candidate["position"]["y"] - agent["position"]["y"]
            entities.append({
                "entity_id": candidate["agent_id"],
                "team_id": candidate["team_id"],
                "relative_position": {"x": rel_x, "y": rel_y},
                "distance": _norm(rel_x, rel_y),
                "hp": candidate["hp"],
                "alive": candidate["alive"],
                "has_line_of_sight": True,
            })
        return sorted(entities, key=lambda entity: entity["distance"])

    def _observations(self) -> dict[AgentId, TacticalObservation]:
        return {agent_id: self._observation(agent_id) for agent_id in self.agent_ids}

    def _observation(self, agent_id: AgentId) -> TacticalObservation:
        agent = self.agents[agent_id]
        enemies = self._visible_entities(agent_id, same_team=False)[:3]
        allies = self._visible_entities(agent_id, same_team=True)[:1]
        recent_events = [
            {
                "event_type": event["event_type"],
                "age_steps": max(0, self.step_index - event["step"]),
                **({"actor_id": event["actor_id"]} if "actor_id" in event else {}),
                **({"target_id": event["target_id"]} if "target_id" in event else {}),
                **({"value": event["value"]} if "value" in event else {}),
            }
            for event in self.recent_events[-8:]
        ]
        vector = build_observation_vector(
            self_hp=agent["hp"],
            max_hp=self.max_hp,
            self_alive=agent["alive"],
            self_x=agent["position"]["x"],
            self_y=agent["position"]["y"],
            map_width=self.width,
            map_height=self.height,
            nearest_enemy=enemies[0] if enemies else None,
            nearest_ally=allies[0] if allies else None,
            visible_enemy_count=len([enemy for enemy in enemies if enemy["alive"]]),
            recent_damage_taken=self.last_damage_taken.get(agent_id, 0.0),
            recent_damage_dealt=self.last_damage_dealt.get(agent_id, 0.0),
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step": self.step_index,
            "agent_id": agent_id,
            "team_id": agent["team_id"],
            "mode": "duo",
            "self": {
                "hp": agent["hp"],
                "alive": agent["alive"],
                "position": deepcopy(agent["position"]),
            },
            "vector": vector,
            "vector_keys": list(DEFAULT_VECTOR_KEYS),
            "visible_enemies": enemies,
            "visible_enemies_mask": [True] * len(enemies),
            "visible_allies": allies,
            "visible_allies_mask": [True] * len(allies),
            "visible_obstacles": [],
            "visible_obstacles_mask": [],
            "recent_events": recent_events,
            "recent_events_mask": [True] * len(recent_events),
        }

    def _snapshot(self, events: list[BattleEvent]) -> BattleSnapshot:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step": self.step_index,
            "mode": "duo",
            "agent_ids": list(self.agent_ids),
            "team_ids": list(self.team_ids),
            "agent_team_map": dict(self.agent_team_map),
            "map": {"width": self.width, "height": self.height, "obstacles": []},
            "agents": deepcopy(self.agents),
            "events": events,
        }

    def _one_team_remaining(self) -> bool:
        alive_teams = {agent["team_id"] for agent in self.agents.values() if agent["alive"]}
        return len(alive_teams) <= 1

