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

``gun_pickup`` pays once, on the step the agent's weapon slots go from no gun to a gun (computed
from consecutive observations, no engine event needed) — the "arm yourself first" heuristic every
human follows, which nothing else in the reward expresses before a gun is in hand.

Race terms (``capture`` events from the bridge's race objective): ``capture`` per point the agent's
*team* took — both members are credited, so teammates do not compete for the same point — and
``enemy_capture`` per point the other team took (negative weight = the loss of a contested point).

Waypoint terms (v1 "global point" objective, all 0 by default; ``goal`` = world (x, y) passed by
the env): ``goal_progress`` x (distance to the point removed this step, in u, while standing),
``goal_hold`` per step while standing within ``goal_radius`` of the point, and ``enemy_at_goal`` x
sum over living enemies of max(0, 1 - d_enemy_to_point / enemy_goal_radius) — a penalty that only
goes away when enemies are kept (or killed) away from the point. Enemy positions come from the
bridge's per-agent observations of the *enemies* (privileged, reward-side only; the policy still
sees only its own observation).
"""

from __future__ import annotations

import math
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
    "intent",
    "time",
    "goal_progress",
    "goal_hold",
    "enemy_goal",
    "capture",
    "enemy_capture",
    "gun_pickup",
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
    #: how close an armed enemy must be for `retreat` to count as still in contact (world units)
    retreat_contact_dist: float = 25.0
    #: how far from the play area's centre `retreat` may go before it starts costing (world units);
    #: 0 disables the leash
    retreat_leash_radius: float = 45.0
    #: weight of the per-intent shaping term (see ``_intent_term``); 0 keeps every intent identical,
    #: which is what v1 did — and why "retreat" produced the fastest wipes in the intent evaluation
    intent_bonus: float = 0.0
    time_penalty_after_s: float = 0.0  # 0 = disabled
    time_penalty_per_step: float = 0.0  # applied per step once t > time_penalty_after_s
    goal_progress: float = 0.0  # per world unit of distance-to-goal removed (standing agents only)
    goal_hold: float = 0.0  # per step while standing within goal_radius of the goal
    goal_radius: float = 6.0
    enemy_at_goal: float = 0.0  # per step x sum_enemies max(0, 1 - d / enemy_goal_radius); use a negative weight
    enemy_goal_radius: float = 30.0
    capture: float = 0.0  # per race point taken by the agent's team (team credit: both members get it)
    enemy_capture: float = 0.0  # per race point taken by the other team (use a negative weight)
    gun_pickup: float = 0.0  # once per episode, the step the agent first holds a gun (empty slots -> a gun)
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
    """Exposure while you cannot shoot back, as a signed shaping term in [-1, 1].

    -1 while an armed enemy has a clear line and the gun in hand is empty: reloading in the open is
    what the first playtest died of. +1 while a fight is on and every armed enemy's line is broken.
    0 when no armed enemy is in view, so walking an empty map pays nothing.

    Line of sight is the server's, computed with the engine's own bullet rule, so this cannot reward
    hiding behind something a bullet goes through. Pair it with ``damage_dealt``: on its own a policy
    learns that sitting behind a wall is free.
    """
    me = obs.get("self") or {}
    armed_enemies = [
        p for p in (obs.get("players") or [])
        if not p.get("dead") and not p.get("downed")
        and str(p.get("weapon") or "fists") not in ("", "fists")
    ]
    if not armed_enemies:
        return 0.0
    if all(p.get("los_blocked") for p in armed_enemies):
        return 1.0
    return -1.0 if int(me.get("clip", 0) or 0) == 0 else 0.0


#: What each reward term is worth under each intent.
#:
#: Two intents, because two is what the controller can actually express. `push`, `hold_angle` and
#: `trade` were tried for three seeds each, priced first through an added shaping term and then by
#: scaling these same weights up to six-fold apart: the three came out within 0.2 damage of each
#: other every time. Fights here last five to seven seconds — both sides hit from the first tick —
#: so there is no room for a stance to matter. `retreat` separates because it is the one intent that
#: changes how long the fight lasts at all.
#:
#: `engage` is absent on purpose: every term keeps its weight, so engaging is simply the normal
#: reward. Under `retreat` nothing about winning pays, and the per-step term for a broken line of
#: sight (``_intent_term``) is the only score.
#: What each combat term is worth under each intent. Naming an intent does not give it meaning --
#: the reward does, and the reward has to make the intent's optimum *different*, not merely cheaper.
#:
#: v7 zeroed `kill` for `retreat` and the two intents still collapsed into one behaviour (seed 1 was
#: identical: 0.82 win / 13.0 s / 0.82 kills either way). Zero is not enough, because killing is
#: still the surest way to stop `damage_taken`: the agent was paid for withdrawing and wiped the
#: enemy anyway, since that is what the shared network's engage solution does and nothing opposed it.
#:
#: So the two scales below have to be read together. `kill` is negative -- a withdrawal that ends in
#: a wipe was not a withdrawal -- and `damage_taken` is doubled, because being hit is how withdrawing
#: fails. Either one alone backfires: doubling `damage_taken` on its own makes killing *more*
#: attractive (it is the fastest way to stop the bleeding), and a negative `kill` on its own leaves
#: standing in the open costless. Together the only route left is to break contact and stay
#: un-shootable, which is what `_intent_term` pays for.
INTENT_SCALE: dict[str, dict[str, float]] = {
    "retreat": {
        "kill": -1.0,
        "damage_dealt": 0.0,
        "team_win": 0.0,
        "time": 0.0,
        "damage_taken": 2.0,
    },
}


def _scale(intent: str | None, term: str) -> float:
    """Weight multiplier for one term under one intent; 1.0 when the intent leaves it alone."""
    return INTENT_SCALE.get(intent or "", {}).get(term, 1.0)

#: the band `hold_angle` is asked to keep (world units)
HOLD_BAND = (12.0, 25.0)


def _armed_enemies(obs: AgentObservation) -> list[Mapping[str, Any]]:
    return [
        p for p in (obs.get("players") or [])
        if not p.get("dead") and not p.get("downed")
        and str(p.get("weapon") or "fists") not in ("", "fists")
    ]


def _nearest_armed_dist(obs: AgentObservation) -> float | None:
    enemies = _armed_enemies(obs)
    return min(float(p.get("dist", 0.0)) for p in enemies) if enemies else None


def _leash_penalty(obs: AgentObservation, radius: float) -> float:
    """How far outside the play area the agent has wandered, as a penalty in [0, 1].

    The centre comes from ``gas.pos``, which the server already sends. In these scenarios the circle
    has not started closing (``mode`` 0, ``rad`` ~196), so its radius says nothing useful — but its
    centre is the play area's, which is all the leash needs and is one less constant to hardcode.
    An observation without a gas field (unit fixtures) simply has no leash.
    """
    if radius <= 0.0:
        return 0.0
    centre = (obs.get("gas") or {}).get("pos")
    me = (obs.get("self") or {}).get("pos")
    if not centre or not me:
        return 0.0
    d = math.hypot(
        float(me.get("x", 0.0)) - float(centre.get("x", 0.0)),
        float(me.get("y", 0.0)) - float(centre.get("y", 0.0)),
    )
    return min(1.0, max(0.0, (d - radius) / radius))


def _intent_term(
    intent: str | None,
    prev: AgentObservation,
    cur: AgentObservation,
    dealt: float,
    taken: float,
    *,
    contact_dist: float = 25.0,
    leash_radius: float = 45.0,
) -> float:
    """What `retreat` asks for this step, in [-1, 1]. Only `retreat` has a shaping term.

    **Withdrawing means becoming un-shootable, not becoming invisible.** The term pays +1 while an
    armed enemy is still within ``contact_dist`` *and* every one of them has its line broken — the
    server's own bullet rule, so nothing here credits hiding behind something a shot goes through.
    That predicate is exactly the one the game's heal mask uses to decide whether healing is
    suicide, so scoring it and "reaching a state where you can heal or reload" are the same thing.

    Leaving pays nothing, and wandering out of the play area costs (``_leash_penalty``).

    This replaces the v5/v6 term, which paid +1 whenever no armed enemy was in view. 64-85% of
    retreat steps satisfied that, so essentially the whole retreat reward came from the enemy not
    being rendered, and breaking a line while in contact earned 0 in six training runs out of six.
    The cheapest way to collect it was to run until the enemy stopped being drawn: all three seeds
    of both generations ended up against the map edge (median x = 1 of a 128-wide map) with no
    obstacle in view and an armed enemy still 22 m away. Cornered, not escaped.
    """
    if intent != "retreat":
        return 0.0
    enemies = _armed_enemies(cur)
    near = _nearest_armed_dist(cur)
    in_contact = near is not None and near <= contact_dist
    # Two ways to withdraw, worth the same, so the policy may pick whichever the ground allows:
    #   * still in the fight but un-shootable -- every armed enemy's line broken;
    #   * out of the fight -- an armed enemy still in view, but none of them inside contact_dist.
    # The second one only became reachable when the scripted bots stopped being omniscient; paying
    # for it while it was impossible is why v7/v8 could only learn the first.
    safe = in_contact and all(p.get("los_blocked") for p in enemies)
    broke_off = bool(enemies) and not in_contact
    # `not enemies` deliberately pays nothing: v5/v6 paid +1 whenever no armed enemy was in view,
    # 64-85% of retreat steps met it, and running until the enemy stopped being drawn was the result.
    earned = 1.0 if (safe or broke_off) else 0.0
    return max(-1.0, min(1.0, earned - _leash_penalty(cur, leash_radius)))


def has_gun(state: Mapping[str, Any]) -> bool:
    """True when a gun sits in either gun slot (0 or 1), equipped or not."""
    for w in state.get("weapons") or []:
        if int(w.get("slot", -1)) in (0, 1) and str(w.get("type") or ""):
            return True
    return False


def _partner_entry(obs: AgentObservation) -> Mapping[str, Any] | None:
    teammates = obs.get("teammates") or []
    return teammates[0] if teammates else None


def _pos(state: Mapping[str, Any]) -> tuple[float, float]:
    p = state.get("pos") or {}
    return float(p.get("x", 0.0)), float(p.get("y", 0.0))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def enemy_goal_pressure(
    obs: Mapping[str, AgentObservation], team: str, goal: tuple[float, float], radius: float
) -> float:
    """sum over living enemies of max(0, 1 - d(enemy, goal) / radius) (0 when nobody is near)."""
    if radius <= 0.0:
        return 0.0
    total = 0.0
    for other in obs.values():
        st = other["self"]
        if str(st.get("team", "")) == team or st.get("dead"):
            continue
        total += max(0.0, 1.0 - _dist(_pos(st), goal) / radius)
    return total


def compute_reward_components(
    prev_obs: Mapping[str, AgentObservation],
    obs: Mapping[str, AgentObservation],
    events: Iterable[Event],
    info: ObsInfo | Mapping[str, Any] | None,
    config: RewardConfig,
    controlled: Sequence[str] | None = None,
    t: float | None = None,
    goal: tuple[float, float] | None = None,
    intent: str | None = None,
) -> dict[str, dict[str, float]]:
    """Signed reward components per controlled agent (before ``team_mix``).

    ``intent`` is the commitment the episode is playing under: it pays for what its name claims and
    scales the combat terms by ``INTENT_COMBAT_SCALE``.
    """
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
        captures = enemy_captures = 0
        for e in events:
            kind = e.get("type")
            if kind == "capture":
                if str(e.get("team", team_of.get(str(e.get("agent", "")), ""))) == team:
                    captures += 1
                else:
                    enemy_captures += 1
            elif kind == "damage":
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
        comp["damage_dealt"] = config.damage_dealt * dealt * _scale(intent, "damage_dealt")
        comp["damage_taken"] = config.damage_taken * taken * _scale(intent, "damage_taken")
        comp["kill"] = config.kill * kills * _scale(intent, "kill")
        comp["death"] = config.death * deaths
        comp["capture"] = config.capture * captures
        comp["enemy_capture"] = config.enemy_capture * enemy_captures
        if config.gun_pickup != 0.0 and has_gun(me) and not has_gun(me_prev) and not me.get("dead"):
            comp["gun_pickup"] = config.gun_pickup
        if done and winner is not None and winner == team:
            comp["team_win"] = config.team_win * _scale(intent, "team_win")
        partner, partner_prev = _partner_entry(cur), _partner_entry(prev)
        if partner is not None:
            if partner_prev is not None:
                comp["partner_hp"] = config.partner_hp_delta * hp_delta_corrected(partner_prev, partner)
            if not partner.get("dead"):
                comp["partner_alive"] = config.partner_alive_per_step
        comp["cover"] = config.cover_bonus * _cover_term(cur)
        if config.intent_bonus != 0.0 and not me.get("dead"):
            comp["intent"] = config.intent_bonus * _intent_term(
                intent,
                prev,
                cur,
                dealt,
                taken,
                contact_dist=config.retreat_contact_dist,
                leash_radius=config.retreat_leash_radius,
            )
        if (
            config.time_penalty_after_s > 0.0
            and t is not None
            and t > config.time_penalty_after_s
            and not me.get("dead")
        ):
            comp["time"] = config.time_penalty_per_step * _scale(intent, "time")
        if goal is not None:
            standing = not me.get("dead") and not me.get("downed")
            if standing and not me_prev.get("dead") and not me_prev.get("downed"):
                comp["goal_progress"] = config.goal_progress * (_dist(_pos(me_prev), goal) - _dist(_pos(me), goal))
            if standing and _dist(_pos(me), goal) <= config.goal_radius:
                comp["goal_hold"] = config.goal_hold
            if not me.get("dead") and config.enemy_at_goal != 0.0:
                comp["enemy_goal"] = config.enemy_at_goal * enemy_goal_pressure(obs, team, goal, config.enemy_goal_radius)
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
    goal: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Scalar reward per controlled agent for the transition ``prev_obs -> obs``."""
    return compute_reward_breakdown(prev_obs, obs, events, info, config, controlled, t, goal).totals


def compute_reward_breakdown(
    prev_obs: Mapping[str, AgentObservation],
    obs: Mapping[str, AgentObservation],
    events: Iterable[Event],
    info: ObsInfo | Mapping[str, Any] | None,
    config: RewardConfig,
    controlled: Sequence[str] | None = None,
    t: float | None = None,
    goal: tuple[float, float] | None = None,
    intent: str | None = None,
) -> RewardBreakdown:
    components = compute_reward_components(prev_obs, obs, events, info, config, controlled, t, goal, intent)
    totals = {aid: float(sum(c.values())) for aid, c in components.items()}
    teams = {aid: str(obs[aid]["self"].get("team", "")) for aid in components}
    return RewardBreakdown(components=components, totals=mix_team_rewards(totals, teams, config.team_mix))
