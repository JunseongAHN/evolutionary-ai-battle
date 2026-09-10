"""Bridge episode -> harness ``EpisodeTrajectory`` (S6 / M8).

Turns one record from ``eval.run_episodes(full_obs=True)`` into the common schema defined in
`docs/common-interface-v0.md`, so a survev episode can be validated by
``experiment.core.schema_validation.validate_episode`` and measured by
``experiment.core.harness_metrics.compute_metrics`` — the same path a toy-environment or a
future Survev.io episode would take.

Three things are worth knowing about the mapping.

**Enemy HP is absent, not zero.** The schema's ``EntityObservation`` has an ``hp`` field, but the
bridge never reports an enemy's HP (a human client is not told it), so the key is simply left out
of ``visible_enemies`` entries; allies keep theirs, because survev's group status does carry a
teammate's HP. Omission is the honest encoding of "unknown", and the validator checks which keys
are *allowed*, not which are present.

**The observation vector is a small diagnostic one, not the policy's 216-d featurization.** The
schema wants a numeric vector with matching keys; it does not require the trainer's features.
A compact, self-describing vector keeps the export independent of whichever featurizer a policy
happens to use (and of its per-agent memory state), which matters because scripted agents never
went through one.

**Scripted agents' actions are reconstructed.** Python only issues actions for the controlled
agents; the server decides for the rest. Their ``aim`` comes from the facing direction the
observation reports, ``fire`` from whether they produced a fire event during the step, and
``move`` from the direction they actually travelled in — the same quantities the engine acted on,
read back off the world instead of forward off the wire.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "cpc-common-v0"

#: `test_normal` is 264 x 264 world units; the scenario region is a 128 u box inside it.
DEFAULT_MAP_SIZE = 264.0

#: Caps for the variable-length entity arrays. Duo 2v2 can only ever show 2 enemies and 1 ally;
#: obstacles are empty on the open field and are capped generously for future scenarios.
MAX_VISIBLE_ENEMIES = 2
MAX_VISIBLE_ALLIES = 1
MAX_VISIBLE_OBSTACLES = 16
MAX_RECENT_EVENTS = 16

VECTOR_KEYS: tuple[str, ...] = (
    "self_hp",
    "self_alive",
    "self_downed",
    "self_armed",
    "self_clip",
    "self_reserve",
    "nearest_enemy_distance",
    "visible_enemy_count",
    "nearest_ally_distance",
    "ally_hp",
    "visible_loot_count",
    "shots_heard_count",
)

ACTION_KEYS: tuple[str, ...] = ("move_x", "move_y", "aim_x", "aim_y", "fire")

#: bridge event type -> schema event type. `death` is the schema's name for a kill, which is what
#: the harness combat metrics count; the rest keep their own names (the schema's event type is a
#: hint, not a closed set the validator enforces).
_EVENT_TYPES = {"fire": "fire", "damage": "damage", "kill": "death"}


def to_episode_trajectory(
    record: Mapping[str, Any],
    *,
    map_size: float = DEFAULT_MAP_SIZE,
    policy_type: str = "future_policy",
    policy_id: str | None = None,
) -> dict[str, Any]:
    """One eval record (recorded with ``full_obs=True``) as an ``EpisodeTrajectory``."""
    steps = record.get("steps") or []
    if not steps:
        raise ValueError("record has no steps; run the episode with full_obs=True")
    agent_ids = [a for a in (record.get("agent_ids") or []) if isinstance(a, str)]
    if not agent_ids:
        raise ValueError("record has no agent_ids")
    missing = [a for a in agent_ids if a not in (steps[0].get("obs") or {})]
    if missing:
        raise ValueError(f"steps carry no observation for {missing}; record with full_obs=True")

    teams = dict(record.get("teams") or {})
    controlled = set(record.get("controlled") or [])
    episode_id = _episode_id(record)
    dt = float(record.get("ticks") or 10) / 100.0  # gameTps is 100

    final_obs = record.get("final_obs") or {}
    out_steps = []
    for index, step in enumerate(steps):
        nxt = steps[index + 1] if index + 1 < len(steps) else None
        # the snapshot describes the state the step ended in, so it comes from the next step's
        # observation (or the terminal one) — the observation itself stays the pre-action state
        after = (nxt.get("obs") if nxt is not None else final_obs) or step.get("obs") or {}
        out_steps.append(
            _step(
                step,
                nxt,
                after=after,
                index=index,
                last=nxt is None,
                episode_id=episode_id,
                agent_ids=agent_ids,
                teams=teams,
                controlled=controlled,
                map_size=map_size,
                reason=record.get("reason"),
                policy_type=policy_type,
                policy_id=policy_id,
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "config": _config(record, agent_ids, teams, map_size, dt),
        "steps": out_steps,
    }


def add_final_metrics(episode: dict[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Attaches ``final_metrics`` in place (kept separate so the metric pass stays optional)."""
    episode["final_metrics"] = dict(metrics)
    return episode


# --------------------------------------------------------------------------------------
# config


def _config(
    record: Mapping[str, Any],
    agent_ids: Sequence[str],
    teams: Mapping[str, str],
    map_size: float,
    dt: float,
) -> dict[str, Any]:
    team_ids = sorted({teams[a] for a in agent_ids if a in teams})
    per_team = len(agent_ids) // len(team_ids) if team_ids else len(agent_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "duo" if per_team == 2 else "solo",
        "team_count": max(1, len(team_ids)),
        "players_per_team": 2 if per_team == 2 else 1,
        "max_steps": max(1, len(record.get("steps") or [])),
        # not in the schema's BattleConfig, but harness_metrics reads it to report times in
        # seconds instead of steps; a consumer that does not know the key ignores it
        "step_seconds": dt,
        "map": {"width": map_size, "height": map_size, "coordinate_system": "world-2d"},
        "observation_spec": {
            "mode": "local_tactical",
            "vector_keys": list(VECTOR_KEYS),
            "max_visible_enemies": MAX_VISIBLE_ENEMIES,
            "max_visible_allies": MAX_VISIBLE_ALLIES,
            "max_visible_obstacles": MAX_VISIBLE_OBSTACLES,
            "max_recent_events": MAX_RECENT_EVENTS,
            "entity_feature_keys": {
                # enemies carry no hp: a client is never told it
                "enemy": ["relative_x", "relative_y", "distance", "alive"],
                "ally": ["relative_x", "relative_y", "distance", "hp", "alive"],
                "obstacle": ["relative_x", "relative_y", "width", "height", "distance"],
                "event": ["event_type", "age_steps", "relative_x", "relative_y", "value"],
            },
        },
        "action_spec": {
            "action_type": "continuous_2d",
            "action_keys": list(ACTION_KEYS),
            "bounds": {
                "move_x": [-1.0, 1.0],
                "move_y": [-1.0, 1.0],
                "aim_x": [-1.0, 1.0],
                "aim_y": [-1.0, 1.0],
                "fire": [0.0, 1.0],
            },
        },
    }


# --------------------------------------------------------------------------------------
# steps


def _step(
    step: Mapping[str, Any],
    nxt: Mapping[str, Any] | None,
    *,
    after: Mapping[str, Any],
    index: int,
    last: bool,
    episode_id: str,
    agent_ids: Sequence[str],
    teams: Mapping[str, str],
    controlled: set[str],
    map_size: float,
    reason: Any,
    policy_type: str,
    policy_id: str | None,
) -> dict[str, Any]:
    obs_in = step.get("obs") or {}
    events = _events(step, index, teams)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": index,
        "mode": "duo" if len(agent_ids) == 4 else "solo",
        "agent_ids": list(agent_ids),
        "team_ids": sorted({teams[a] for a in agent_ids if a in teams}),
        "agent_team_map": {a: teams.get(a, "") for a in agent_ids},
        "map": {"width": map_size, "height": map_size},
        "agents": {a: _agent_snapshot(a, after.get(a) or obs_in.get(a), teams) for a in agent_ids},
        "events": events,
    }
    observations = {
        a: _observation(a, obs_in.get(a), teams, episode_id, index, len(agent_ids), step)
        for a in agent_ids
    }
    actions = {
        a: _action(a, step, nxt, episode_id, index, a in controlled, policy_type, policy_id)
        for a in agent_ids
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": index,
        "observations": observations,
        "actions": actions,
        "rewards": {a: float(v) for a, v in (step.get("rewards") or {}).items()},
        "terminated": bool(last and reason in ("elimination", "controlled_dead")),
        "truncated": bool(last and reason == "time_limit"),
        "info": {"snapshot": snapshot, "events": events},
    }


def _agent_snapshot(
    agent_id: str,
    obs: Mapping[str, Any] | None,
    teams: Mapping[str, str],
) -> dict[str, Any]:
    me = (obs or {}).get("self") or {}
    return {
        "agent_id": agent_id,
        "team_id": teams.get(agent_id, ""),
        "position": _vec(me.get("pos")),
        "hp": float(me.get("hp") or 0.0),
        "alive": not bool(me.get("dead")),
        "downed": bool(me.get("downed")),
        "facing": _vec(me.get("dir")),
        # not in the schema's AgentSnapshot; harness_metrics needs it to tell a wasted shot
        # from a melee swing (a fists "fire" produces no fire event but is not wasteful)
        "armed": str(me.get("weapon") or "") not in ("", "fists"),
    }


def _observation(
    agent_id: str,
    obs: Mapping[str, Any] | None,
    teams: Mapping[str, str],
    episode_id: str,
    index: int,
    agent_count: int,
    step: Mapping[str, Any],
) -> dict[str, Any]:
    obs = obs or {}
    me = obs.get("self") or {}
    pos = _vec(me.get("pos"))

    enemies = [
        {
            "entity_id": str(p.get("id", "")),
            "team_id": str(p.get("team", "")),
            "relative_position": _sub(_vec(p.get("pos")), pos),
            "distance": float(p.get("dist") or 0.0),
            "alive": not bool(p.get("dead")),
        }
        for p in (obs.get("players") or [])[:MAX_VISIBLE_ENEMIES]
    ]
    allies = [
        {
            "entity_id": str(m.get("id", "")),
            "team_id": teams.get(agent_id, ""),
            "relative_position": _sub(_vec(m.get("pos")), pos),
            "distance": float(m.get("dist") or 0.0),
            "hp": float(m.get("hp") or 0.0),
            "alive": not bool(m.get("dead")),
        }
        for m in (obs.get("teammates") or [])[:MAX_VISIBLE_ALLIES]
    ]
    obstacles = [
        {
            "obstacle_id": str(o.get("id", "")),
            "relative_position": _sub(_vec(o.get("pos")), pos),
            "width": float(o.get("scale") or 1.0),
            "height": float(o.get("scale") or 1.0),
            "distance": float(o.get("dist") or 0.0),
            "blocks_line_of_sight": bool(o.get("collidable")),
        }
        for o in (obs.get("obstacles") or [])[:MAX_VISIBLE_OBSTACLES]
    ]
    recent = [
        {
            "event_type": str(e.get("type", "")),
            "age_steps": 0,
            "actor_id": e.get("agent"),
            "value": float(e.get("amount") or 0.0),
        }
        for e in _agent_visible_events(step, agent_id)[:MAX_RECENT_EVENTS]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": index,
        "agent_id": agent_id,
        "team_id": teams.get(agent_id, ""),
        "mode": "duo" if agent_count == 4 else "solo",
        "self": {"hp": float(me.get("hp") or 0.0), "alive": not bool(me.get("dead")), "position": pos},
        "vector": _vector(obs, enemies, allies),
        "vector_keys": list(VECTOR_KEYS),
        "visible_enemies": enemies,
        "visible_enemies_mask": [True] * len(enemies) + [False] * (MAX_VISIBLE_ENEMIES - len(enemies)),
        "visible_allies": allies,
        "visible_allies_mask": [True] * len(allies) + [False] * (MAX_VISIBLE_ALLIES - len(allies)),
        "visible_obstacles": obstacles,
        "visible_obstacles_mask": [True] * len(obstacles) + [False] * (MAX_VISIBLE_OBSTACLES - len(obstacles)),
        "recent_events": recent,
        "recent_events_mask": [True] * len(recent) + [False] * (MAX_RECENT_EVENTS - len(recent)),
    }


def _vector(
    obs: Mapping[str, Any],
    enemies: Sequence[Mapping[str, Any]],
    allies: Sequence[Mapping[str, Any]],
) -> list[float]:
    me = obs.get("self") or {}
    weapon = str(me.get("weapon") or "")
    nearest_enemy = min((e["distance"] for e in enemies), default=-1.0)
    nearest_ally = min((a["distance"] for a in allies), default=-1.0)
    values = {
        "self_hp": float(me.get("hp") or 0.0),
        "self_alive": 0.0 if me.get("dead") else 1.0,
        "self_downed": 1.0 if me.get("downed") else 0.0,
        "self_armed": 0.0 if weapon in ("", "fists") else 1.0,
        "self_clip": float(me.get("clip") or 0.0),
        "self_reserve": float(me.get("reserve") or 0.0),
        "nearest_enemy_distance": nearest_enemy,
        "visible_enemy_count": float(len(enemies)),
        "nearest_ally_distance": nearest_ally,
        "ally_hp": float(allies[0]["hp"]) if allies else -1.0,
        "visible_loot_count": float(len(obs.get("loot") or [])),
        "shots_heard_count": float(len(obs.get("shots_heard") or [])),
    }
    return [values[key] for key in VECTOR_KEYS]


# --------------------------------------------------------------------------------------
# actions


def _action(
    agent_id: str,
    step: Mapping[str, Any],
    nxt: Mapping[str, Any] | None,
    episode_id: str,
    index: int,
    is_controlled: bool,
    policy_type: str,
    policy_id: str | None,
) -> dict[str, Any]:
    if is_controlled:
        body = _controlled_action(step, agent_id)
        source: dict[str, Any] = {"policy_type": policy_type}
        if policy_id:
            source["policy_id"] = policy_id
    else:
        body = _scripted_action(step, nxt, agent_id)
        source = {"policy_type": "future_policy", "policy_id": "survev-scripted"}
    action = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step": index,
        "agent_id": agent_id,
        "action": body,
        "source": source,
    }
    return action


def _controlled_action(step: Mapping[str, Any], agent_id: str) -> dict[str, float]:
    entry = (step.get("actions") or {}).get(agent_id) or {}
    cpc = entry.get("cpc") or {}
    move = _vec(cpc.get("move"))
    aim = _vec(cpc.get("aim"))
    fire = cpc.get("fire") or {}
    return {
        "move_x": move["x"],
        "move_y": move["y"],
        "aim_x": aim["x"],
        "aim_y": aim["y"],
        "fire": 1.0 if (fire.get("hold") or fire.get("start")) else 0.0,
    }


def _scripted_action(
    step: Mapping[str, Any],
    nxt: Mapping[str, Any] | None,
    agent_id: str,
) -> dict[str, float]:
    """Read back off the world: aim from facing, move from travel, fire from the fire events."""
    me = ((step.get("obs") or {}).get(agent_id) or {}).get("self") or {}
    aim = _normalize(_vec(me.get("dir")))
    move = {"x": 0.0, "y": 0.0}
    if nxt is not None:
        later = ((nxt.get("obs") or {}).get(agent_id) or {}).get("self") or {}
        move = _normalize(_sub(_vec(later.get("pos")), _vec(me.get("pos"))))
    fired = any(
        e.get("type") == "fire" and e.get("agent") == agent_id for e in (step.get("events") or [])
    )
    return {
        "move_x": move["x"],
        "move_y": move["y"],
        "aim_x": aim["x"],
        "aim_y": aim["y"],
        "fire": 1.0 if fired else 0.0,
    }


# --------------------------------------------------------------------------------------
# events


def _events(
    step: Mapping[str, Any],
    index: int,
    teams: Mapping[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, event in enumerate(step.get("events") or []):
        kind = str(event.get("type", ""))
        subject = event.get("agent")
        source = event.get("source")
        mapped: dict[str, Any] = {
            "event_id": f"s{index}-{kind}-{i}",
            "step": index,
            "event_type": _EVENT_TYPES.get(kind, kind),
        }
        if kind in ("damage", "down", "kill", "revive"):
            # the subject is who it happened to; the source is who caused it
            mapped["actor_id"] = source if source else subject
            mapped["target_id"] = subject
        else:
            mapped["actor_id"] = subject
        if subject in teams:
            mapped["team_id"] = teams[subject]
        if event.get("pos") is not None:
            mapped["position"] = _vec(event.get("pos"))
        if event.get("amount") is not None:
            mapped["value"] = float(event["amount"])
        metadata = {k: event[k] for k in ("weapon", "item", "count", "index") if event.get(k) is not None}
        if metadata:
            mapped["metadata"] = metadata
        out.append(mapped)
    return out


def _agent_visible_events(step: Mapping[str, Any], agent_id: str) -> list[Mapping[str, Any]]:
    """Only the events this agent took part in: the shared list is a log, not its information set."""
    return [
        e
        for e in (step.get("events") or [])
        if isinstance(e, Mapping) and agent_id in (e.get("agent"), e.get("source"))
    ]


# --------------------------------------------------------------------------------------
# small helpers


def _episode_id(record: Mapping[str, Any]) -> str:
    return f"survev-{record.get('seed', 'seed')}-ep{record.get('episode', 0)}"


def _vec(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {"x": float(value.get("x", 0.0)), "y": float(value.get("y", 0.0))}
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return {"x": float(value[0]), "y": float(value[1])}
    return {"x": 0.0, "y": 0.0}


def _sub(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    return {"x": a["x"] - b["x"], "y": a["y"] - b["y"]}


def _normalize(v: Mapping[str, float]) -> dict[str, float]:
    length = math.hypot(v["x"], v["y"])
    if length < 1e-6:
        return {"x": 0.0, "y": 0.0}
    return {"x": v["x"] / length, "y": v["y"] / length}
