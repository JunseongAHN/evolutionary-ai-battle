from __future__ import annotations

from dataclasses import dataclass

from .schema import AgentId, MultiAgentStep


@dataclass(frozen=True)
class RewardWeights:
    damage_dealt: float = 0.02
    damage_taken: float = 0.02
    kill: float = 1.0
    death: float = 1.0
    alive_step: float = 0.005
    wasteful_fire: float = 0.01
    support_response: float = 0.10
    isolation_step: float = 0.005


def compute_agent_reward(
    step: MultiAgentStep,
    agent_id: AgentId,
    weights: RewardWeights = RewardWeights(),
) -> float:
    reward = 0.0

    obs = step["observations"].get(agent_id)
    if obs is not None and obs["self"]["alive"]:
        reward += weights.alive_step

    for event in step["info"]["events"]:
        event_type = event["event_type"]
        actor_id = event.get("actor_id")
        target_id = event.get("target_id")
        value = float(event.get("value", 0.0) or 0.0)

        if event_type == "damage":
            if actor_id == agent_id:
                reward += weights.damage_dealt * value
            if target_id == agent_id:
                reward -= weights.damage_taken * value
        elif event_type == "death":
            if actor_id == agent_id:
                reward += weights.kill
            if target_id == agent_id:
                reward -= weights.death
        elif event_type == "fire":
            metadata = event.get("metadata", {}) or {}
            if actor_id == agent_id and metadata.get("wasteful", False):
                reward -= weights.wasteful_fire
        elif event_type == "support_response":
            if actor_id == agent_id:
                reward += weights.support_response
        elif event_type == "isolation":
            if actor_id == agent_id:
                reward -= weights.isolation_step

    return float(reward)

