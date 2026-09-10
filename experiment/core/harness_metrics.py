"""Metric vector for a common-schema ``EpisodeTrajectory``.

The counterpart of `docs/evaluation-metrics.md` for episodes that arrive as JSONL rather than
as the TypeScript `engine/traces` trajectory the browser harness evaluates. Four independent
groups per agent — combat, survival, cooperation, movement — deliberately **not** collapsed
into a scalar: a bot can deal a lot of damage while abandoning its partner, or deal none while
covering it perfectly, and both patterns have to stay visible.

Everything is read off the per-step ``info.snapshot`` (positions, hp, alive) and
``info.snapshot.events``, so it works on any producer of the schema, not just the survev bridge.

Thresholds default to the survev field scenario and are grounded in engine constants rather than
picked by feel — see ``MetricOptions``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "cpc-common-v0"


@dataclass(frozen=True)
class MetricOptions:
    """Distances in world units, times in steps. Defaults follow the survev engine.

    ``support_distance`` is the half-height of a player's rendered view rectangle
    (``(zoom + 4) / (16/9)`` = 18 u at the 1x scope), i.e. the range at which partners see each
    other whichever way they face. ``isolation_distance`` is the gunshot audible radius (48 u,
    the client's ``otherPlayers`` channel ``maxRange``): past it you cannot even hear your
    partner's fight, which is what being isolated means. ``threat_distance`` is the scripted
    bots' ``fireRange`` (30 u), the range at which an enemy is actually shooting at you.
    ``stuck_speed`` is 1 u/s, under a tenth of ``moveSpeed`` (12 u/s).
    """

    support_distance: float = 18.0
    isolation_distance: float = 48.0
    threat_distance: float = 30.0
    low_hp: float = 50.0
    response_window_steps: int = 10
    stuck_speed: float = 1.0


def compute_metrics(
    episode: Mapping[str, Any],
    options: MetricOptions | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-agent ``{combat, survival, cooperation, movement}`` for one episode.

    Raises ``ValueError`` if the episode has no steps; use
    ``schema_validation.validate_episode`` first to check the shape.
    """
    opts = options or MetricOptions()
    steps = episode.get("steps") or []
    if not steps:
        raise ValueError("episode has no steps")

    agent_ids = _agent_ids(steps)
    teams = _agent_team_map(steps)
    dt = _step_seconds(episode)

    out: dict[str, dict[str, Any]] = {}
    for agent_id in agent_ids:
        team_id = teams.get(agent_id, "")
        allies = [a for a in agent_ids if a != agent_id and teams.get(a) == team_id]
        out[agent_id] = {
            "agent_id": agent_id,
            "team_id": team_id,
            "combat": _combat(steps, agent_id),
            "survival": _survival(steps, agent_id, dt),
            "cooperation": _cooperation(steps, agent_id, allies, teams, opts),
            "movement": _movement(steps, agent_id, dt, opts),
        }
    return out


# --------------------------------------------------------------------------------------
# groups


def _combat(steps: Sequence[Mapping[str, Any]], agent_id: str) -> dict[str, float]:
    dealt = taken = friendly = 0.0
    kills = deaths = wasteful = 0
    shots = hits = 0
    for step in steps:
        teams = _snapshot(step).get("agent_team_map") or {}
        for event in _events(step):
            kind = event.get("event_type")
            actor = event.get("actor_id")
            target = event.get("target_id")
            value = float(event.get("value") or 0.0)
            if kind == "damage":
                if actor == agent_id and target != agent_id:
                    dealt += value
                    hits += 1
                    if teams.get(actor) is not None and teams.get(actor) == teams.get(target):
                        friendly += value
                if target == agent_id:
                    taken += value
            elif kind in ("death", "elimination"):
                if actor == agent_id and target != agent_id:
                    kills += 1
                if target == agent_id:
                    deaths += 1
            elif kind == "fire" and actor == agent_id:
                shots += 1
        # a fire command from an armed agent that produced no shot: out of ammo, on cooldown,
        # reloading. Unarmed "fire" is a melee swing, which is not a wasted shot.
        action = _action_body(step, agent_id)
        state = _agent_state(step, agent_id)
        armed = bool(state.get("armed")) if state is not None else True
        if armed and action is not None and float(action.get("fire") or 0.0) > 0.0:
            if not any(e.get("event_type") == "fire" and e.get("actor_id") == agent_id for e in _events(step)):
                wasteful += 1
    return {
        "damageDealt": dealt,
        "damageTaken": taken,
        "kills": float(kills),
        "deaths": float(deaths),
        "friendlyFireDamage": friendly,
        "wastefulFireCount": float(wasteful),
        "shots": float(shots),
        "hitsGiven": float(hits),
    }


def _survival(steps: Sequence[Mapping[str, Any]], agent_id: str, dt: float) -> dict[str, Any]:
    alive_steps = 0
    hp_sum = 0.0
    last_alive = False
    last_hp = 0.0
    for step in steps:
        state = _agent_state(step, agent_id)
        if state is None:
            continue
        alive = bool(state.get("alive")) and float(state.get("hp") or 0.0) > 0.0
        last_alive = alive
        last_hp = float(state.get("hp") or 0.0)
        if alive:
            alive_steps += 1
            hp_sum += last_hp
    return {
        "aliveSteps": alive_steps,
        "aliveTime": alive_steps * dt,
        "aliveAtEnd": last_alive,
        "died": not last_alive,
        "hpMean": hp_sum / alive_steps if alive_steps else 0.0,
        "hpEnd": last_hp if last_alive else 0.0,
    }


def _cooperation(
    steps: Sequence[Mapping[str, Any]],
    agent_id: str,
    allies: Sequence[str],
    teams: Mapping[str, str],
    opts: MetricOptions,
) -> dict[str, Any]:
    """Solo mode has no allies, so cooperation is reported as not applicable (schema rule)."""
    if not allies:
        return {"applicable": False}

    ally_distances: list[float] = []
    isolated = formation_good = 0
    pressure_steps: list[int] = []
    for index, step in enumerate(steps):
        me = _agent_state(step, agent_id)
        if me is None or not me.get("alive"):
            continue
        nearest = _nearest_distance(step, me, allies, alive_only=True)
        if nearest is not None:
            ally_distances.append(nearest)
            if nearest > opts.isolation_distance:
                isolated += 1
            if nearest <= opts.support_distance:
                formation_good += 1
        if _ally_under_pressure(step, allies, teams, opts):
            pressure_steps.append(index)

    responses = sum(1 for i in pressure_steps if _responded(steps, i, agent_id, allies, teams, opts))
    counted = len(ally_distances)
    return {
        "applicable": True,
        "teammateUnderPressureEvents": len(pressure_steps),
        "teammateUnderPressureResponses": responses,
        "teammateResponseRate": responses / len(pressure_steps) if pressure_steps else 0.0,
        "isolatedSteps": isolated,
        "isolationRate": isolated / counted if counted else 0.0,
        "avgAllyDistance": sum(ally_distances) / counted if counted else None,
        "formationGoodSteps": formation_good,
        # no trajectory currently carries a dedicated abandonment signal (see evaluation-metrics.md)
        "allyAbandonmentEvents": 0,
    }


def _movement(
    steps: Sequence[Mapping[str, Any]],
    agent_id: str,
    dt: float,
    opts: MetricOptions,
) -> dict[str, Any]:
    stuck = 0
    moving_steps = 0
    distance = 0.0
    previous: Mapping[str, Any] | None = None
    for step in steps:
        state = _agent_state(step, agent_id)
        if state is None or not state.get("alive"):
            previous = None
            continue
        if previous is not None:
            moved = _distance(previous.get("position"), state.get("position"))
            distance += moved
            moving_steps += 1
            if moved / dt < opts.stuck_speed:
                stuck += 1
        previous = state
    return {
        "stuckSteps": stuck,
        "stuckRate": stuck / moving_steps if moving_steps else 0.0,
        "distanceTravelled": distance,
        # no trajectory currently carries a dedicated movement-penalty signal
        "movementPenaltyEvents": 0,
    }


# --------------------------------------------------------------------------------------
# pressure / response


def _ally_under_pressure(
    step: Mapping[str, Any],
    allies: Sequence[str],
    teams: Mapping[str, str],
    opts: MetricOptions,
) -> bool:
    """A downed ally, or a hurt ally with an enemy inside the shooting range."""
    for ally in allies:
        state = _agent_state(step, ally)
        if state is None or not state.get("alive"):
            continue
        if state.get("downed"):
            return True
        if float(state.get("hp") or 0.0) >= opts.low_hp:
            continue
        ally_team = teams.get(ally)
        enemies = [a for a, t in teams.items() if t != ally_team]
        nearest = _nearest_distance(step, state, enemies, alive_only=True)
        if nearest is not None and nearest <= opts.threat_distance:
            return True
    return False


def _responded(
    steps: Sequence[Mapping[str, Any]],
    index: int,
    agent_id: str,
    allies: Sequence[str],
    teams: Mapping[str, str],
    opts: MetricOptions,
) -> bool:
    """Acted on the pressure within the window: shot, revived, or closed a real gap.

    Closing the gap only counts when the agent was *outside* support range to begin with.
    Duo partners travel a few units apart, so drifting a hair closer would otherwise score as
    a response on almost every step and the rate would say nothing about behaviour.
    """
    start = _agent_state(steps[index], agent_id)
    if start is None or not start.get("alive"):
        return False
    start_distance = _nearest_distance(steps[index], start, allies, alive_only=True)
    approach_counts = start_distance is not None and start_distance > opts.support_distance
    window = steps[index + 1 : index + 1 + max(1, opts.response_window_steps)]
    for step in window:
        for event in _events(step):
            if event.get("actor_id") != agent_id:
                continue
            if event.get("event_type") in ("fire", "damage", "revive"):
                return True
        if not approach_counts:
            continue
        state = _agent_state(step, agent_id)
        if state is None:
            continue
        now = _nearest_distance(step, state, allies, alive_only=True)
        if now is not None and now <= opts.support_distance:
            return True
    return False


# --------------------------------------------------------------------------------------
# snapshot helpers


def _snapshot(step: Mapping[str, Any]) -> Mapping[str, Any]:
    info = step.get("info")
    snapshot = info.get("snapshot") if isinstance(info, Mapping) else None
    return snapshot if isinstance(snapshot, Mapping) else {}


def _events(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    snapshot = _snapshot(step)
    events = snapshot.get("events")
    if not isinstance(events, list):
        info = step.get("info")
        events = info.get("events") if isinstance(info, Mapping) else None
    return [e for e in (events or []) if isinstance(e, Mapping)]


def _agent_state(step: Mapping[str, Any], agent_id: str) -> Mapping[str, Any] | None:
    agents = _snapshot(step).get("agents")
    if not isinstance(agents, Mapping):
        return None
    state = agents.get(agent_id)
    return state if isinstance(state, Mapping) else None


def _action_body(step: Mapping[str, Any], agent_id: str) -> Mapping[str, Any] | None:
    actions = step.get("actions")
    if not isinstance(actions, Mapping):
        return None
    action = actions.get(agent_id)
    body = action.get("action") if isinstance(action, Mapping) else None
    return body if isinstance(body, Mapping) else None


def _agent_ids(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = _snapshot(steps[0]).get("agent_ids")
    return [a for a in (ids or []) if isinstance(a, str)]


def _agent_team_map(steps: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    mapping = _snapshot(steps[0]).get("agent_team_map")
    return dict(mapping) if isinstance(mapping, Mapping) else {}


def _step_seconds(episode: Mapping[str, Any]) -> float:
    """Seconds of game time per step, from the config when the producer records it."""
    config = episode.get("config")
    if isinstance(config, Mapping):
        dt = config.get("step_seconds")
        if isinstance(dt, (int, float)) and dt > 0:
            return float(dt)
    return 1.0


def _distance(a: Any, b: Any) -> float:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return 0.0
    return math.hypot(float(b.get("x", 0.0)) - float(a.get("x", 0.0)),
                      float(b.get("y", 0.0)) - float(a.get("y", 0.0)))


def _nearest_distance(
    step: Mapping[str, Any],
    origin: Mapping[str, Any],
    others: Iterable[str],
    alive_only: bool = False,
) -> float | None:
    best: float | None = None
    for other in others:
        state = _agent_state(step, other)
        if state is None:
            continue
        if alive_only and not state.get("alive"):
            continue
        d = _distance(origin.get("position"), state.get("position"))
        if best is None or d < best:
            best = d
    return best
