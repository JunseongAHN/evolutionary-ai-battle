"""Typed representation of the survev CPC bridge protocol v0.

This module is the Python-side contract for ``docs/survev-bridge-v0.md``. It contains

* ``CpcAction`` (what Python sends per controlled agent per step),
* TypedDict views of ``AgentObservation`` and its sub-objects (the JSON stays a dict),
* ``Event``, ``ObsInfo``, ``Metrics`` and ``ObsMessage`` dataclasses for the responses,
* the ``Input`` name <-> number table of the engine (``shared/gameConfig.ts``),
* JSON encode/decode helpers and request builders,
* an allowlist validator for observation keys (M6 in the plan: unknown keys raise).

No numpy / torch here so the module can be reused by loggers and tooling.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence, TypedDict

PROTOCOL_VERSION = "survev-bridge-v0"
DEFAULT_SCENARIO = "duo2v2_field"
DEFAULT_BRIDGE_URL = "ws://127.0.0.1:8765"
TICK_DT = 0.01  # 1 tick = 1/100 s
NET_SYNC_TICKS = 3

# Agent / team ids of the duo2v2_field scenario, in the order the bridge reports them.
DUO2V2_AGENT_IDS: tuple[str, ...] = ("team-a-0", "team-a-1", "team-b-0", "team-b-1")
DUO2V2_TEAMS: dict[str, str] = {
    "team-a-0": "team-a",
    "team-a-1": "team-a",
    "team-b-0": "team-b",
    "team-b-1": "team-b",
}

ScriptedPolicy = Literal["chaser", "idle"]


class ProtocolError(ValueError):
    """Raised for malformed messages or observations that violate the allowlist."""


# --------------------------------------------------------------------------------------
# Input enum (engine ``shared/gameConfig.ts`` ``Input``). The entries listed in the spec
# (Reload 5, Interact 7, Revive 8, EquipPrimary 11, EquipSecondary 12, EquipMelee 13,
# EquipNextWeap 17, UseBandage 23, UseHealthKit 24, UseSoda 25, UsePainkiller 26) are the
# contract; the remaining numbers follow the same enum and are provided for completeness.
# The client sends *names* by default so numbering differences cannot break anything.
# --------------------------------------------------------------------------------------
INPUT_NAMES: tuple[str, ...] = (
    "MoveLeft",  # 0
    "MoveRight",  # 1
    "MoveUp",  # 2
    "MoveDown",  # 3
    "Fire",  # 4
    "Reload",  # 5
    "Cancel",  # 6
    "Interact",  # 7
    "Revive",  # 8
    "Use",  # 9
    "Loot",  # 10
    "EquipPrimary",  # 11
    "EquipSecondary",  # 12
    "EquipMelee",  # 13
    "EquipThrowable",  # 14
    "EquipFragGrenade",  # 15
    "EquipSmokeGrenade",  # 16
    "EquipNextWeap",  # 17
    "EquipPrevWeap",  # 18
    "EquipLastWeap",  # 19
    "EquipOtherGun",  # 20
    "EquipPrevScope",  # 21
    "EquipNextScope",  # 22
    "UseBandage",  # 23
    "UseHealthKit",  # 24
    "UseSoda",  # 25
    "UsePainkiller",  # 26
    "StowWeapons",  # 27
    "SwapWeapSlots",  # 28
    "ToggleMap",  # 29
    "CycleUIMode",  # 30
    "EmoteMenu",  # 31
    "TeamPingMenu",  # 32
    "Fullscreen",  # 33
    "HideUI",  # 34
    "TeamPingSingle",  # 35
)
INPUT_TO_NUMBER: dict[str, int] = {name: i for i, name in enumerate(INPUT_NAMES)}
NUMBER_TO_INPUT: dict[int, str] = {i: name for i, name in enumerate(INPUT_NAMES)}

# Inputs the spec calls out as useful for agents.
SPEC_INPUTS: dict[str, int] = {
    "Reload": 5,
    "Interact": 7,
    "Revive": 8,
    "EquipPrimary": 11,
    "EquipSecondary": 12,
    "EquipMelee": 13,
    "EquipNextWeap": 17,
    "UseBandage": 23,
    "UseHealthKit": 24,
    "UseSoda": 25,
    "UsePainkiller": 26,
}
assert all(INPUT_TO_NUMBER[k] == v for k, v in SPEC_INPUTS.items())

USE_ITEMS: tuple[str, ...] = ("bandage", "healthkit", "soda", "painkiller")
SCOPES: tuple[str, ...] = ("1xscope", "2xscope", "4xscope", "8xscope", "15xscope")
AMMO_TYPES: tuple[str, ...] = ("762mm", "9mm", "12gauge", "556mm")
INVENTORY_KEYS: tuple[str, ...] = USE_ITEMS + AMMO_TYPES

# GameConfig.Action
ACTION_NONE, ACTION_RELOAD, ACTION_USE_ITEM, ACTION_REVIVE = 0, 1, 2, 3


def input_to_number(value: str | int) -> int:
    """Map an ``Input`` name (or number) to its enum number."""
    if isinstance(value, bool):
        raise ProtocolError(f"invalid input {value!r}")
    if isinstance(value, int):
        if value not in NUMBER_TO_INPUT:
            raise ProtocolError(f"unknown Input number {value}")
        return value
    try:
        return INPUT_TO_NUMBER[value]
    except KeyError as exc:
        raise ProtocolError(f"unknown Input name {value!r}") from exc


def input_to_name(value: str | int) -> str:
    """Map an ``Input`` number (or name) to its enum name."""
    return NUMBER_TO_INPUT[input_to_number(value)]


# --------------------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------------------
class Vec2(TypedDict):
    x: float
    y: float


def vec2(x: float, y: float) -> Vec2:
    return {"x": float(x), "y": float(y)}


def normalize(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-9:
        return 0.0, 0.0
    return x / length, y / length


# --------------------------------------------------------------------------------------
# CpcAction
# --------------------------------------------------------------------------------------
@dataclass
class CpcAction:
    """One controlled agent's action for one step (see spec ``step``).

    ``move`` / ``aim`` are world-space directions (length ignored by the server). ``None``
    or a zero move means stand still. ``fire_start`` is an edge-triggered single shot,
    ``fire_hold`` is level-triggered auto fire. ``inputs`` are ``Input`` names or numbers
    applied once when the step starts. ``use_item`` is an item id to consume (or a scope id).
    """

    move: tuple[float, float] | None = None
    aim: tuple[float, float] | None = None
    fire_start: bool = False
    fire_hold: bool = False
    inputs: list[str | int] = field(default_factory=list)
    use_item: str = ""

    @classmethod
    def release(cls) -> "CpcAction":
        """No movement, no fire, nothing pressed (encodes as ``{}``)."""
        return cls()

    def is_release(self) -> bool:
        move_zero = self.move is None or (abs(self.move[0]) < 1e-9 and abs(self.move[1]) < 1e-9)
        return (
            move_zero
            and self.aim is None
            and not self.fire_start
            and not self.fire_hold
            and not self.inputs
            and not self.use_item
        )

    def to_json(self) -> dict[str, Any]:
        """Wire form. A full release is sent as ``{}`` per the spec."""
        if self.is_release():
            return {}
        out: dict[str, Any] = {}
        if self.move is not None:
            out["move"] = vec2(*self.move)
        if self.aim is not None:
            out["aim"] = vec2(*self.aim)
        out["fire"] = {"start": bool(self.fire_start), "hold": bool(self.fire_hold)}
        if self.inputs:
            out["inputs"] = [input_to_name(i) if isinstance(i, int) else i for i in self.inputs]
        out["useItem"] = self.use_item or ""
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "CpcAction":
        move = data.get("move")
        aim = data.get("aim")
        fire = data.get("fire") or {}
        inputs = list(data.get("inputs") or [])
        for value in inputs:
            input_to_number(value)  # validate
        return cls(
            move=(float(move["x"]), float(move["y"])) if move else None,
            aim=(float(aim["x"]), float(aim["y"])) if aim else None,
            fire_start=bool(fire.get("start", False)),
            fire_hold=bool(fire.get("hold", False)),
            inputs=inputs,
            use_item=str(data.get("useItem") or ""),
        )


# --------------------------------------------------------------------------------------
# AgentObservation (TypedDict views over the JSON; the dicts are passed around as-is)
# --------------------------------------------------------------------------------------
class WeaponSlot(TypedDict):
    slot: int
    type: str
    ammo: int


class SelfState(TypedDict):
    id: str
    team: str
    pos: Vec2
    dir: Vec2
    hp: float
    boost: float
    downed: bool
    dead: bool
    weapon: str
    clip: int
    reserve: int
    weapons: list[WeaponSlot]
    inventory: dict[str, int]
    scope: str
    zoom: int
    action: int
    cur_weap_idx: int


class TeammateView(TypedDict):
    id: str
    pos: Vec2
    dist: float
    hp: float
    downed: bool
    dead: bool


class PlayerView(TypedDict):
    id: str
    team: str
    pos: Vec2
    dist: float
    dir: Vec2
    downed: bool
    dead: bool
    weapon: str


class LootView(TypedDict):
    id: int
    type: str
    pos: Vec2
    dist: float
    count: int


class ObstacleView(TypedDict):
    id: int
    type: str
    pos: Vec2
    dist: float
    collidable: bool
    height: float
    scale: float


class BulletView(TypedDict):
    pos: Vec2
    dir: Vec2
    player_id: int


class DeadBodyView(TypedDict):
    pos: Vec2
    dist: float


class GasView(TypedDict):
    mode: int
    rad: float
    pos: Vec2
    rad_new: float
    pos_new: Vec2


class AgentObservation(TypedDict):
    self: SelfState
    teammates: list[TeammateView]
    players: list[PlayerView]
    loot: list[LootView]
    obstacles: list[ObstacleView]
    bullets: list[BulletView]
    dead_bodies: list[DeadBodyView]
    gas: GasView
    alive_count: int
    alive_teams: int


# Allowlist (M6). Keys not listed here make ``validate_agent_observation`` raise.
VEC2_KEYS = frozenset({"x", "y"})
ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "root": frozenset(AgentObservation.__annotations__),
    "self": frozenset(SelfState.__annotations__),
    "self.weapons[]": frozenset(WeaponSlot.__annotations__),
    "self.inventory": frozenset(INVENTORY_KEYS),
    "teammates[]": frozenset(TeammateView.__annotations__),
    "players[]": frozenset(PlayerView.__annotations__),
    "loot[]": frozenset(LootView.__annotations__),
    "obstacles[]": frozenset(ObstacleView.__annotations__),
    "bullets[]": frozenset(BulletView.__annotations__),
    "dead_bodies[]": frozenset(DeadBodyView.__annotations__),
    "gas": frozenset(GasView.__annotations__),
}
_VEC2_FIELDS: dict[str, tuple[str, ...]] = {
    "self": ("pos", "dir"),
    "teammates[]": ("pos",),
    "players[]": ("pos", "dir"),
    "loot[]": ("pos",),
    "obstacles[]": ("pos",),
    "bullets[]": ("pos", "dir"),
    "dead_bodies[]": ("pos",),
    "gas": ("pos", "pos_new"),
}
_LIST_FIELDS = ("teammates", "players", "loot", "obstacles", "bullets", "dead_bodies")


def _check_keys(obj: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    if not isinstance(obj, Mapping):
        raise ProtocolError(f"{where}: expected an object, got {type(obj).__name__}")
    unknown = set(obj) - allowed
    if unknown:
        raise ProtocolError(f"{where}: unknown keys {sorted(unknown)} (allowlist violation)")


def _check_vec2(obj: Any, where: str) -> None:
    _check_keys(obj, VEC2_KEYS, where)


def validate_agent_observation(obs: Mapping[str, Any], where: str = "obs") -> None:
    """Raise ``ProtocolError`` if ``obs`` contains any key outside the allowlist.

    Missing keys are tolerated (forward compatibility of the mock), extra keys are not:
    an extra key means the server leaks information the client would not have.
    """
    _check_keys(obs, ALLOWED_KEYS["root"], where)
    if "self" in obs:
        me = obs["self"]
        _check_keys(me, ALLOWED_KEYS["self"], f"{where}.self")
        for name in _VEC2_FIELDS["self"]:
            if name in me:
                _check_vec2(me[name], f"{where}.self.{name}")
        for i, slot in enumerate(me.get("weapons") or []):
            _check_keys(slot, ALLOWED_KEYS["self.weapons[]"], f"{where}.self.weapons[{i}]")
        if "inventory" in me:
            _check_keys(me["inventory"], ALLOWED_KEYS["self.inventory"], f"{where}.self.inventory")
    for list_name in _LIST_FIELDS:
        key = f"{list_name}[]"
        for i, entry in enumerate(obs.get(list_name) or []):
            _check_keys(entry, ALLOWED_KEYS[key], f"{where}.{list_name}[{i}]")
            for name in _VEC2_FIELDS[key]:
                if name in entry:
                    _check_vec2(entry[name], f"{where}.{list_name}[{i}].{name}")
    if "gas" in obs:
        _check_keys(obs["gas"], ALLOWED_KEYS["gas"], f"{where}.gas")
        for name in _VEC2_FIELDS["gas"]:
            if name in obs["gas"]:
                _check_vec2(obs["gas"][name], f"{where}.gas.{name}")


# --------------------------------------------------------------------------------------
# Events, info, metrics, obs message
# --------------------------------------------------------------------------------------
EventType = Literal["fire", "damage", "down", "kill", "loot", "heal", "revive", "shots_heard"]


class Event(TypedDict, total=False):
    """fire / damage / down / kill (later: loot, heal, revive, shots_heard)."""

    type: str
    t: float
    agent: str
    source: str
    weapon: str
    pos: Vec2
    dir: Vec2
    amount: float
    hp_before: float
    hp_after: float
    downed: bool
    dead: bool


METRIC_KEYS: tuple[str, ...] = (
    "survival_time",
    "alive_at_end",
    "downed_time",
    "hp_mean",
    "hp_end",
    "damage_dealt",
    "damage_taken",
    "kills",
    "shots",
    "hits_given",
    "team_win",
    "partner_survival_time",
    "partner_hp_end",
)


@dataclass
class Metrics:
    """Per-agent end-of-episode metrics (``info.metrics[agent_id]`` when ``done``)."""

    survival_time: float = 0.0
    alive_at_end: bool = False
    downed_time: float = 0.0
    hp_mean: float = 0.0
    hp_end: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    kills: int = 0
    shots: int = 0
    hits_given: int = 0
    team_win: bool = False
    partner_survival_time: float = 0.0
    partner_hp_end: float = 0.0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Metrics":
        unknown = set(data) - set(METRIC_KEYS)
        if unknown:
            raise ProtocolError(f"metrics: unknown keys {sorted(unknown)}")
        return cls(**{k: data[k] for k in METRIC_KEYS if k in data})

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def as_float_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class ObsInfo:
    alive_teams: int = 2
    winner_team: str | None = None
    reason: str | None = None  # "elimination" | "time_limit" | None
    metrics: dict[str, Metrics] | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "ObsInfo":
        data = data or {}
        raw_metrics = data.get("metrics")
        metrics = (
            {aid: Metrics.from_json(m) for aid, m in raw_metrics.items()} if raw_metrics else None
        )
        return cls(
            alive_teams=int(data.get("alive_teams", 2)),
            winner_team=data.get("winner_team"),
            reason=data.get("reason"),
            metrics=metrics,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "alive_teams": self.alive_teams,
            "winner_team": self.winner_team,
            "reason": self.reason,
            "metrics": {k: v.to_json() for k, v in self.metrics.items()} if self.metrics else None,
        }


@dataclass
class ObsMessage:
    """Decoded ``obs`` message (response to ``reset`` and ``step``)."""

    env_id: int
    t: float
    tick: int
    done: bool
    agent_ids: list[str]
    teams: dict[str, str]
    obs: dict[str, AgentObservation]
    events: list[Event]
    info: ObsInfo

    @classmethod
    def from_json(cls, data: Mapping[str, Any], validate: bool = True) -> "ObsMessage":
        if data.get("type") != "obs":
            raise ProtocolError(f"expected an obs message, got {data.get('type')!r}")
        obs = data.get("obs") or {}
        if validate:
            for aid, agent_obs in obs.items():
                validate_agent_observation(agent_obs, where=f"obs[{aid}]")
        return cls(
            env_id=int(data["env_id"]),
            t=float(data.get("t", 0.0)),
            tick=int(data.get("tick", 0)),
            done=bool(data.get("done", False)),
            agent_ids=list(data.get("agent_ids") or obs.keys()),
            teams=dict(data.get("teams") or {}),
            obs=dict(obs),
            events=list(data.get("events") or []),
            info=ObsInfo.from_json(data.get("info")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "obs",
            "env_id": self.env_id,
            "t": self.t,
            "tick": self.tick,
            "done": self.done,
            "agent_ids": list(self.agent_ids),
            "teams": dict(self.teams),
            "obs": self.obs,
            "events": list(self.events),
            "info": self.info.to_json(),
        }

    def team_of(self, agent_id: str) -> str:
        return self.teams.get(agent_id) or self.obs[agent_id]["self"]["team"]

    def teammates_of(self, agent_id: str) -> list[str]:
        team = self.team_of(agent_id)
        return [a for a in self.agent_ids if a != agent_id and self.team_of(a) == team]


# --------------------------------------------------------------------------------------
# Request builders + JSON helpers
# --------------------------------------------------------------------------------------
def make_reset(
    env_id: int,
    scenario: str = DEFAULT_SCENARIO,
    seed: str | int = "cpc-duo2v2-seed-0",
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "reset",
        "env_id": int(env_id),
        "scenario": scenario,
        "seed": seed,
        "options": dict(options or {}),
    }


def make_reset_options(
    controlled: Sequence[str] = ("team-a-0", "team-a-1"),
    scripted: str = "chaser",
    map_size: int = 128,
    time_limit: float = 60.0,
) -> dict[str, Any]:
    return {
        "mapSize": int(map_size),
        "timeLimit": float(time_limit),
        "controlled": list(controlled),
        "scripted": scripted,
    }


def actions_to_json(actions: Mapping[str, CpcAction | Mapping[str, Any]]) -> dict[str, Any]:
    return {
        aid: (a.to_json() if isinstance(a, CpcAction) else dict(a)) for aid, a in actions.items()
    }


def make_step(
    env_id: int, actions: Mapping[str, CpcAction | Mapping[str, Any]], ticks: int = 10
) -> dict[str, Any]:
    if int(ticks) <= 0:
        raise ProtocolError("ticks must be a positive integer")
    return {
        "type": "step",
        "env_id": int(env_id),
        "ticks": int(ticks),
        "actions": actions_to_json(actions),
    }


def make_step_batch(
    envs: Mapping[int, tuple[Mapping[str, CpcAction | Mapping[str, Any]], int]],
) -> dict[str, Any]:
    """Batched step: ``{env_id: (actions, ticks)}`` -> one ``step`` message with ``envs``."""
    body: dict[str, Any] = {}
    for env_id, (actions, ticks) in envs.items():
        if int(ticks) <= 0:
            raise ProtocolError("ticks must be a positive integer")
        body[str(int(env_id))] = {"ticks": int(ticks), "actions": actions_to_json(actions)}
    return {"type": "step", "envs": body}


def make_close(env_id: int) -> dict[str, Any]:
    return {"type": "close", "env_id": int(env_id)}


def encode_message(message: Mapping[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


def decode_message(raw: str | bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or "type" not in data:
        raise ProtocolError("message must be a JSON object with a 'type' field")
    return data


def parse_response(data: Mapping[str, Any], validate: bool = True) -> ObsMessage:
    """Turn a decoded response into an ``ObsMessage``; ``error`` responses raise."""
    kind = data.get("type")
    if kind == "error":
        raise ProtocolError(
            f"bridge error (env {data.get('env_id')}): {data.get('message', 'unknown error')}"
        )
    if kind != "obs":
        raise ProtocolError(f"unexpected response type {kind!r}")
    return ObsMessage.from_json(data, validate=validate)


def parse_batch_response(data: Mapping[str, Any], validate: bool = True) -> dict[int, ObsMessage]:
    kind = data.get("type")
    if kind == "error":
        raise ProtocolError(
            f"bridge error (env {data.get('env_id')}): {data.get('message', 'unknown error')}"
        )
    if kind != "obs_batch":
        raise ProtocolError(f"unexpected response type {kind!r} (expected obs_batch)")
    out: dict[int, ObsMessage] = {}
    for key, msg in (data.get("envs") or {}).items():
        if msg.get("type") == "error":
            raise ProtocolError(f"bridge error (env {key}): {msg.get('message')}")
        out[int(key)] = ObsMessage.from_json(msg, validate=validate)
    return out


def iter_events(events: Iterable[Event], kind: str) -> Iterable[Event]:
    return (e for e in events if e.get("type") == kind)
