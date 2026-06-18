from __future__ import annotations

from core.schema import AgentId, BattleEvent, BattleSnapshot, TacticalObservation


def _fmt_vec2(value: dict | None) -> str:
    if not value:
        return "(?, ?)"
    return f"({float(value.get('x', 0.0)):.1f}, {float(value.get('y', 0.0)):.1f})"


def summarize_agent(snapshot: BattleSnapshot, agent_id: AgentId) -> str:
    agent = snapshot["agents"][agent_id]
    return (
        f"{agent_id} team={agent['team_id']} hp={float(agent['hp']):.1f} "
        f"alive={agent['alive']} pos={_fmt_vec2(agent.get('position'))} "
        f"aim={_fmt_vec2(agent.get('aim'))}"
    )


def summarize_snapshot(snapshot: BattleSnapshot) -> str:
    lines = [
        f"step={snapshot['step']} episode={snapshot['episode_id']} mode={snapshot['mode']}",
        f"map={float(snapshot['map']['width']):.0f}x{float(snapshot['map']['height']):.0f}",
    ]
    lines.extend(summarize_agent(snapshot, agent_id) for agent_id in snapshot["agent_ids"])
    return "\n".join(lines)


def summarize_events(events: list[BattleEvent], limit: int = 10) -> str:
    recent = events[-limit:]
    if not recent:
        return "events: none"
    lines = []
    for event in recent:
        parts = [f"[step={event['step']}]", str(event["event_type"])]
        if "actor_id" in event:
            parts.append(f"actor={event['actor_id']}")
        if "target_id" in event:
            parts.append(f"target={event['target_id']}")
        if "value" in event:
            parts.append(f"value={float(event['value']):.1f}")
        metadata = event.get("metadata", {})
        if metadata:
            if "hit" in metadata:
                parts.append(f"hit={metadata['hit']}")
            if "cooldown_blocked" in metadata:
                parts.append(f"cooldown_blocked={metadata['cooldown_blocked']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def summarize_observation(obs: TacticalObservation) -> str:
    self_obs = obs["self"]
    return (
        f"obs agent={obs['agent_id']} step={obs['step']} team={obs['team_id']} "
        f"hp={float(self_obs['hp']):.1f} alive={self_obs['alive']} "
        f"pos={_fmt_vec2(self_obs['position'])} "
        f"enemies={len(obs['visible_enemies'])} allies={len(obs['visible_allies'])} "
        f"events={len(obs['recent_events'])} vector={len(obs['vector'])}"
    )

