"""Reward shaping for controlled agents (per step, per agent).

Defaults are the user's decision for v0 (one step = 0.1 s at ``ticks=10``):

* ``alive_per_step`` +0.01 while not dead (downed counts as alive, like the metrics),
* ``hp_delta`` x0.01 of the own HP change (damage taken is therefore negative),
* ``damage_dealt`` x0.02 per HP of damage dealt to enemies,
* ``team_win`` +1.0 at episode end for the winning team,
* ``death`` -1.0 when the agent's ``kill`` event fires,
* everything else 0 (``damage_taken``, ``kill``, cooperation terms, cover, time penalty).

All weights are *signed* and multiplied by a non-negative quantity, so a penalty is a
negative weight (e.g. ``damage_taken=-0.01``). ``team_mix`` in [0, 1] blends each agent's
own reward with the mean reward of its controlled teammates (0 = individual, 1 = shared).

HP bookkeeping uses *effective HP* = standing HP, 0 while downed or dead. The engine resets
HP to 100 on a down (bleed pool) and to 24 on a revive; with effective HP a down counts as
losing the HP that was left, bleeding counts 0 (the death penalty covers it) and a revive
counts as +24 (so ``partner_hp_delta`` rewards the reviver). Evaluation never uses these
rewards: the metric vector in ``info.metrics`` stays separate (harness convention).

Extensibility hooks: ``partner_hp_delta`` / ``partner_alive_per_step`` (cooperation),
``cover_bonus`` (placeholder term, 0 until obstacle/LOS features exist) and
``time_penalty_after_s`` / ``time_penalty_per_step`` (anti-stalling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .protocol import AgentObservation, Event, ObsInfo

COMPONENT_KEYS: tuple[str, ...] = (
    "alive",
    "hp",
    "damage_dealt",
    "damage_taken",
    "kill",
    "death",
    "team_win",
    "partner_hp",
    "partner_alive",
    "cover",
    "time",
)


@dataclass
class RewardConfig:
    alive_per_step: float = 0.01
    hp_delta: float = 0.01
    damage_dealt: float = 0.02
    damage_taken: float = 0.0
    kill: float = 0.0
    death: float = -1.0
    team_win: float = 1.0
    partner_hp_delta: float = 0.0
    partner_alive_per_step: float = 0.0
    cover_bonus: float = 0.0
    time_penalty_after_s: float = 0.0  # 0 = disabled
    time_penalty_per_step: float = 0.0  # applied per step once t > time_penalty_after_s
    team_mix: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.team_mix <= 1.0:
            raise ValueError("team_mix must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewardConfig":
        return cls(**{k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RewardBreakdown:
    """Per-agent reward components (pre ``team_mix``) plus the mixed totals."""

    components: dict[str, dict[str, float]] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)


def effective_hp(state: Mapping[str, Any]) -> float:
    """Standing HP; 0 while downed or dead (the engine's downed 100-HP bleed pool is ignored)."""
    if state.get("dead") or state.get("downed"):
        return 0.0
    return float(state.get("hp", 0.0))


def hp_delta_corrected(prev: Mapping[str, Any], cur: Mapping[str, Any]) -> float:
    """HP change with the engine's down/revive resets removed.

    standing -> downed/dead counts as losing the HP that was left, bleeding while downed
    counts 0 (the death penalty covers it), a revive counts as gaining ``reviveHealth``.
    """
    return effective_hp(cur) - effective_hp(prev)


def _cover_term(obs: AgentObservation) -> float:
    """Placeholder for cover/LOS shaping; returns 0 until obstacle features are real."""
    return 0.0


def _partner_entry(obs: AgentObservation) -> Mapping[str, Any] | None:
    teammates = obs.get("teammates") or []
    return teammates[0] if teammates else None


def compute_reward_components(
    prev_obs: Mapping[str, AgentObservation],
    obs: Mapping[str, AgentObservation],
    events: Iterable[Event],
    info: ObsInfo | Mapping[str, Any] | None,
    config: RewardConfig,
    controlled: Sequence[str] | None = None,
    t: float | None = None,
) -> dict[str, dict[str, float]]:
    """Signed reward components per controlled agent (before ``team_mix``)."""
    if isinstance(info, ObsInfo):
        winner, reason = info.winner_team, info.reason
    else:
        info = info or {}
        winner, reason = info.get("winner_team"), info.get("reason")
    done = reason is not None
    agents = list(controlled) if controlled is not None else list(obs.keys())
    events = list(events)
    team_of = {aid: str(obs[aid]["self"].get("team", "")) for aid in obs}
    out: dict[str, dict[str, float]] = {}
    for aid in agents:
        cur = obs[aid]
        prev = prev_obs.get(aid, cur)
        me, me_prev = cur["self"], prev["self"]
        team = team_of.get(aid, "")
        comp = {k: 0.0 for k in COMPONENT_KEYS}
        if not me.get("dead"):
            comp["alive"] = config.alive_per_step
        comp["hp"] = config.hp_delta * hp_delta_corrected(me_prev, me)
        dealt = taken = 0.0
        kills = deaths = 0
        for e in events:
            kind = e.get("type")
            if kind == "damage":
                amount = float(e.get("amount", 0.0))
                victim = str(e.get("agent", ""))
                if e.get("source") == aid and team_of.get(victim, "") != team:
                    dealt += amount
                if victim == aid:
                    taken += amount
            elif kind == "kill":
                victim = str(e.get("agent", ""))
                if victim == aid:
                    deaths += 1
                elif e.get("source") == aid and team_of.get(victim, "") != team:
                    kills += 1
        comp["damage_dealt"] = config.damage_dealt * dealt
        comp["damage_taken"] = config.damage_taken * taken
        comp["kill"] = config.kill * kills
        comp["death"] = config.death * deaths
        if done and winner is not None and winner == team:
            comp["team_win"] = config.team_win
        partner, partner_prev = _partner_entry(cur), _partner_entry(prev)
        if partner is not None:
            if partner_prev is not None:
                comp["partner_hp"] = config.partner_hp_delta * hp_delta_corrected(partner_prev, partner)
            if not partner.get("dead"):
                comp["partner_alive"] = config.partner_alive_per_step
        comp["cover"] = config.cover_bonus * _cover_term(cur)
        if (
            config.time_penalty_after_s > 0.0
            and t is not None
            and t > config.time_penalty_after_s
            and not me.get("dead")
        ):
            comp["time"] = config.time_penalty_per_step
        out[aid] = comp
    return out


def mix_team_rewards(
    totals: Mapping[str, float], teams: Mapping[str, str], team_mix: float
) -> dict[str, float]:
    if team_mix <= 0.0:
        return dict(totals)
    by_team: dict[str, list[float]] = {}
    for aid, r in totals.items():
        by_team.setdefault(teams.get(aid, ""), []).append(r)
    means = {team: sum(v) / len(v) for team, v in by_team.items()}
    return {
        aid: (1.0 - team_mix) * r + team_mix * means[teams.get(aid, "")] for aid, r in totals.items()
    }


def compute_rewards(
    prev_obs: Mapping[str, AgentObservation],
    obs: Mapping[str, AgentObservation],
    events: Iterable[Event],
    info: ObsInfo | Mapping[str, Any] | None,
    config: RewardConfig,
    controlled: Sequence[str] | None = None,
    t: float | None = None,
) -> dict[str, float]:
    """Scalar reward per controlled agent for the transition ``prev_obs -> obs``."""
    return compute_reward_breakdown(prev_obs, obs, events, info, config, controlled, t).totals


def compute_reward_breakdown(
    prev_obs: Mapping[str, AgentObservation],
    obs: Mapping[str, AgentObservation],
    events: Iterable[Event],
    info: ObsInfo | Mapping[str, Any] | None,
    config: RewardConfig,
    controlled: Sequence[str] | None = None,
    t: float | None = None,
) -> RewardBreakdown:
    components = compute_reward_components(prev_obs, obs, events, info, config, controlled, t)
    totals = {aid: float(sum(c.values())) for aid, c in components.items()}
    teams = {aid: str(obs[aid]["self"].get("team", "")) for aid in components}
    return RewardBreakdown(components=components, totals=mix_team_rewards(totals, teams, config.team_mix))
