"""Mock survev CPC bridge (protocol v0) with a kinematic ``duo2v2_field`` simulation.

The mock exists so the Python side can be developed and tested before the real bridge
(``pnpm cpc:bridge``) is available. It speaks the exact wire protocol of
``docs/survev-bridge-v0.md`` and reproduces the engine semantics that matter for RL:

* 100 Hz ticks, ``step(ticks)`` holds the last action like a held key, ``{}`` releases;
* 4 agents, fixed spawns, seeded loot layout (same item list as the spec);
* movement 12 u/s (+1 with fists), 8-direction quantization, water penalty, map clamp;
* ``Interact`` picks up the nearest loot within 2 u (guns -> weapon slots with a full 30
  round clip, ammo/heals -> inventory with level-0 bag caps, armor, scopes);
* firing (``fire.start`` edge / ``fire.hold`` level) with fireDelay spacing 0.10 s (ak47)
  / 0.11 s (mp5), 13 / 9 damage per hit, projectile bullets with angular spread so the hit
  probability decreases with distance, armor reduction, auto-reload on empty;
* duo down/kill semantics: a lethal hit downs a player whose teammate is still standing;
  the last standing teammate's death kills the downed ones; downed players bleed;
* revive (``Revive``/``Interact`` within 5 u, 8 s), healing items, boost regen;
* fire / damage / down / kill events, ``done`` + ``info`` + per-agent metrics as the spec;
* per-agent observation culling by a (zoom + 4) x 16:9 rectangle (32 x 18 u half extents
  at 1x scope) so observations are genuinely partial;
* scripted ``chaser`` opponents (loot nearest gun/ammo -> approach to 22 u -> strafe ->
  hold fire within 30 u -> revive a downed teammate when no enemy is within 25 u) and
  ``idle``.

Numbers not in the spec (bullet speed/range, spread, reload/switch times, bleed, boost
regen, revive range/hp, bag caps, armor) approximate the engine's ``GameConfig`` /
weapon definitions and are documented next to the constants below.

Run standalone::

    python -m experiment.survev_rl.mock_bridge --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .protocol import (
    ACTION_NONE,
    ACTION_RELOAD,
    ACTION_REVIVE,
    ACTION_USE_ITEM,
    DEFAULT_SCENARIO,
    INVENTORY_KEYS,
    TICK_DT,
    ProtocolError,
    decode_message,
    encode_message,
    input_to_name,
    normalize,
    vec2,
)

# --------------------------------------------------------------------------------------
# Engine-ish constants (approximations of survev GameConfig / defs where not in the spec)
# --------------------------------------------------------------------------------------
MAP_SIZE = 264.0
SHORE = 48.0  # water outside [SHORE, MAP_SIZE - SHORE]
CENTER = (132.0, 132.0)
PLAYER_RADIUS = 1.0
BASE_MOVE_SPEED = 12.0  # spec
FISTS_SPEED_BONUS = 1.0  # spec: +1 with fists equipped
DOWNED_MOVE_SPEED = 4.0  # GameConfig.player.downedMoveSpeed (approx)
WATER_SPEED_PENALTY = 3.0  # GameConfig.player.waterSpeedPenalty
PICKUP_RANGE = 2.0  # task: Interact picks up nearest loot within 2 u
REVIVE_RANGE = 5.0  # GameConfig.player.reviveRange
REVIVE_DURATION = 8.0  # GameConfig.player.reviveDuration (M4)
REVIVE_HP = 24.0  # GameConfig.player.reviveHealth
DOWNED_HP = 100.0  # spec: HP is reset to 100 when downed, then bleeds
BLEED_INTERVAL = 1.0  # bleedTickRate
BLEED_DAMAGE = 2.0  # bleedDamage
BLEED_MULT = 1.25  # bleedDamageMult per additional down
BOOST_DECAY = 0.375  # per second
MAX_HP = 100.0
DEFAULT_ZOOM = 28
CULL_MARGIN = 4.0  # culling half-width = zoom + 4 (spec), 16:9
GAS_RADIUS = 196.0

ZOOM_BY_SCOPE = {"1xscope": 28, "2xscope": 36, "4xscope": 48, "8xscope": 68, "15xscope": 104}
SCOPE_LEVEL = {"1xscope": 1, "2xscope": 2, "4xscope": 4, "8xscope": 8, "15xscope": 15}
ARMOR_REDUCTION = {"helmet01": 0.075, "chest01": 0.25}
BAG_CAPS_L0 = {
    "bandage": 5,
    "healthkit": 1,
    "soda": 2,
    "painkiller": 1,
    "762mm": 90,
    "9mm": 120,
    "12gauge": 15,
    "556mm": 90,
}
# item -> (use time s, hp heal, boost add)
HEAL_ITEMS = {
    "bandage": (3.0, 15.0, 0.0),
    "healthkit": (6.0, 100.0, 0.0),
    "soda": (3.0, 0.0, 25.0),
    "painkiller": (5.0, 0.0, 50.0),
}
USE_INPUT_TO_ITEM = {
    "UseBandage": "bandage",
    "UseHealthKit": "healthkit",
    "UseSoda": "soda",
    "UsePainkiller": "painkiller",
}


@dataclass(frozen=True)
class GunDef:
    name: str
    ammo: str
    max_clip: int
    damage: float
    fire_delay: float
    reload_time: float
    switch_delay: float
    bullet_speed: float
    bullet_distance: float
    spread: float  # degrees, standing
    move_spread: float  # degrees, added while moving

    @property
    def fire_delay_ticks(self) -> int:
        return max(1, int(round(self.fire_delay / TICK_DT)))


GUNS: dict[str, GunDef] = {
    "ak47": GunDef("ak47", "762mm", 30, 13.0, 0.10, 2.3, 0.75, 110.0, 120.0, 2.5, 7.5),
    "mp5": GunDef("mp5", "9mm", 30, 9.0, 0.11, 1.9, 0.75, 85.0, 90.0, 3.0, 7.0),
}
FISTS = "fists"
MELEE_SLOT = 2

# --------------------------------------------------------------------------------------
# Scenario layout (spec): spawns, starter kits 10 u toward the center, contested center kit
# --------------------------------------------------------------------------------------
SPAWNS: dict[str, tuple[float, float]] = {
    "team-a-0": (100.0, 125.6),
    "team-a-1": (100.0, 138.4),
    "team-b-0": (164.0, 125.6),
    "team-b-1": (164.0, 138.4),
}
TEAMS: dict[str, str] = {
    "team-a-0": "team-a",
    "team-a-1": "team-a",
    "team-b-0": "team-b",
    "team-b-1": "team-b",
}
AGENT_IDS: tuple[str, ...] = tuple(SPAWNS)
SPAWN_DIR = {"team-a": (1.0, 0.0), "team-b": (-1.0, 0.0)}
STARTER_KIT: tuple[tuple[str, int], ...] = (
    ("ak47", 1),
    ("mp5", 1),
    ("bandage", 4),
    ("soda", 2),
    ("helmet01", 1),
    ("chest01", 1),
    ("2xscope", 1),
)
CENTER_KIT: tuple[tuple[str, int], ...] = (("healthkit", 1), ("painkiller", 1), ("4xscope", 1))
AMMO_PILE = 30
KIT_RING_RADIUS = 1.5
KIT_CENTERS = {"team-a": (110.0, 132.0), "team-b": (154.0, 132.0), "center": CENTER}


def item_class(item_type: str) -> str:
    if item_type in GUNS:
        return "gun"
    if item_type in ("762mm", "9mm", "12gauge", "556mm"):
        return "ammo"
    if item_type in HEAL_ITEMS:
        return "heal"
    if item_type in ARMOR_REDUCTION:
        return "armor"
    if item_type in ZOOM_BY_SCOPE:
        return "scope"
    return "other"


# --------------------------------------------------------------------------------------
# Simulation state
# --------------------------------------------------------------------------------------
@dataclass
class Loot:
    id: int
    type: str
    pos: tuple[float, float]
    count: int


@dataclass
class Bullet:
    owner: "SimPlayer"
    gun: GunDef
    pos: tuple[float, float]
    dir: tuple[float, float]
    travelled: float = 0.0


@dataclass
class HeldInput:
    move: tuple[float, float] = (0.0, 0.0)
    aim: tuple[float, float] | None = None
    fire_hold: bool = False


@dataclass
class SimPlayer:
    id: str
    team: str
    numeric_id: int
    pos: tuple[float, float]
    dir: tuple[float, float]
    hp: float = MAX_HP
    boost: float = 0.0
    downed: bool = False
    dead: bool = False
    weapons: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"slot": 0, "type": "", "ammo": 0},
            {"slot": 1, "type": "", "ammo": 0},
            {"slot": 2, "type": FISTS, "ammo": 0},
            {"slot": 3, "type": "", "ammo": 0},
        ]
    )
    cur_weap_idx: int = MELEE_SLOT
    inventory: dict[str, int] = field(default_factory=lambda: {k: 0 for k in INVENTORY_KEYS})
    scope: str = "1xscope"
    scopes: set[str] = field(default_factory=lambda: {"1xscope"})
    helmet: str = ""
    chest: str = ""
    # action state (GameConfig.Action)
    action_type: int = ACTION_NONE
    action_end_tick: int = 0
    action_item: str = ""
    action_target: str = ""
    fire_ready_tick: int = 0
    # held input / edge inputs
    held: HeldInput = field(default_factory=HeldInput)
    fire_start_pending: bool = False
    # bookkeeping
    downed_since_tick: int = 0
    down_count: int = 0
    next_bleed_tick: int = 0
    death_time: float | None = None
    alive_ticks: int = 0
    hp_sum: float = 0.0
    downed_ticks: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    kills: int = 0
    shots: int = 0
    hits_given: int = 0
    captures: int = 0
    # scripted controller memory
    script: dict[str, Any] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------------------
    @property
    def zoom(self) -> int:
        return ZOOM_BY_SCOPE.get(self.scope, DEFAULT_ZOOM)

    @property
    def cur_weapon(self) -> str:
        return self.weapons[self.cur_weap_idx]["type"] or FISTS

    @property
    def cur_gun(self) -> GunDef | None:
        return GUNS.get(self.cur_weapon)

    @property
    def standing(self) -> bool:
        return not self.dead and not self.downed

    def has_gun(self) -> bool:
        return any(self.weapons[i]["type"] in GUNS for i in (0, 1))

    def gun_slot(self) -> int | None:
        for i in (0, 1):
            if self.weapons[i]["type"] in GUNS:
                return i
        return None

    def clip(self) -> int:
        return int(self.weapons[self.cur_weap_idx]["ammo"]) if self.cur_gun else 0

    def reserve(self) -> int:
        gun = self.cur_gun
        return int(self.inventory.get(gun.ammo, 0)) if gun else 0


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def quantize_move(x: float, y: float) -> tuple[float, float]:
    """Quantize a world-space direction to the 8 keyboard directions (unit length)."""
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return (0.0, 0.0)
    theta = math.atan2(y, x)
    k = int(round(theta / (math.pi / 4.0))) % 8
    dx = float(round(math.cos(k * math.pi / 4.0)))
    dy = float(round(math.sin(k * math.pi / 4.0)))
    return normalize(dx, dy)


def in_rect(center: tuple[float, float], pos: tuple[float, float], hw: float, hh: float) -> bool:
    return abs(pos[0] - center[0]) <= hw and abs(pos[1] - center[1]) <= hh


class FieldSim:
    """One ``duo2v2_field`` episode. Pure Python, deterministic for a given seed."""

    def __init__(
        self,
        env_id: int,
        seed: str | int = "cpc-duo2v2-seed-0",
        options: Mapping[str, Any] | None = None,
        scenario: str = DEFAULT_SCENARIO,
    ) -> None:
        if scenario != DEFAULT_SCENARIO:
            raise ProtocolError(f"unknown scenario {scenario!r} (mock supports {DEFAULT_SCENARIO})")
        options = dict(options or {})
        self.env_id = int(env_id)
        self.seed = seed
        self.time_limit = float(options.get("timeLimit", 60.0))
        self.map_size_option = int(options.get("mapSize", 128))
        controlled = options.get("controlled", ["team-a-0", "team-a-1"])
        unknown = [a for a in controlled if a not in TEAMS]
        if unknown:
            raise ProtocolError(f"unknown controlled agent ids {unknown}")
        self.controlled: list[str] = list(controlled)
        self.scripted: str = str(options.get("scripted", "chaser"))
        if self.scripted not in ("chaser", "idle", "racer"):
            raise ProtocolError(f"unknown scripted policy {self.scripted!r}")
        self.loadout = str(options.get("loadout", "fists"))  # extension (not in spec v0): "armed"
        if self.loadout not in ("fists", "armed"):
            raise ProtocolError(f"unknown loadout {self.loadout!r}")
        self.layout = str(options.get("layout", "fixed"))  # accepted for parity; the mock keeps its fixed geometry
        if self.layout not in ("fixed", "random"):
            raise ProtocolError(f"unknown layout {self.layout!r}")
        objective = dict(options.get("objective") or {"mode": "none"})
        if objective.get("mode", "none") not in ("none", "race"):
            raise ProtocolError(f"unknown objective mode {objective.get('mode')!r}")
        self.race = objective.get("mode") == "race"
        self.objective_radius = float(objective.get("radius", 4.0))
        self.objective_min = float(objective.get("minDist", 30.0))
        self.objective_max = float(objective.get("maxDist", 70.0))
        self.objective_margin = float(objective.get("margin", 12.0))
        self.end_on_elimination = bool(options.get("endOnElimination", True))
        # opponent strength (bridge scriptedOptions): aim noise, reaction delay, racer engage distance
        scripted_options = dict(options.get("scriptedOptions") or {})
        unknown = set(scripted_options) - {"aimNoiseDeg", "reactionDelay", "engageDist"}
        if unknown:
            raise ProtocolError(f"unknown scriptedOptions {sorted(unknown)}")
        self.aim_noise_deg = float(scripted_options.get("aimNoiseDeg", 0.0))
        self.reaction_delay = float(scripted_options.get("reactionDelay", 0.0))
        self.engage_dist = float(scripted_options.get("engageDist", RACER_ENGAGE_DIST))
        self.rng = random.Random(f"{seed}")
        self.objective_rng = random.Random(f"{seed}/objective")
        self.objective_index = 0
        self.objective_spawned_at = 0.0
        self.captures: dict[str, int] = {"team-a": 0, "team-b": 0}
        self.objective_pos: tuple[float, float] | None = None
        self.tick = 0
        self.done = False
        self.reason: str | None = None
        self.winner_team: str | None = None
        self.metrics: dict[str, dict[str, Any]] | None = None
        self.events: list[dict[str, Any]] = []
        self.bullets: list[Bullet] = []
        self._next_loot_id = 512
        self.players: dict[str, SimPlayer] = {
            aid: SimPlayer(aid, TEAMS[aid], 1024 + i, SPAWNS[aid], SPAWN_DIR[TEAMS[aid]])
            for i, aid in enumerate(AGENT_IDS)
        }
        self.loot: dict[int, Loot] = {}
        self._spawn_loot()
        if self.race:
            self.objective_pos = self._sample_objective((132.0, 132.0))
        if self.loadout == "armed":
            for p in self.players.values():
                p.weapons[0] = {"slot": 0, "type": "ak47", "ammo": GUNS["ak47"].max_clip}
                p.inventory["762mm"] = BAG_CAPS_L0["762mm"]
                p.cur_weap_idx = 0

    # -- race objective ----------------------------------------------------------------------
    def _sample_objective(self, frm: tuple[float, float]) -> tuple[float, float]:
        lo, hi = 68.0 + self.objective_margin, 196.0 - self.objective_margin  # 128 region centered at 132
        pos = (self.objective_rng.uniform(lo, hi), self.objective_rng.uniform(lo, hi))
        for _ in range(64):
            d = dist(pos, frm)
            if self.objective_min <= d <= self.objective_max:
                break
            pos = (self.objective_rng.uniform(lo, hi), self.objective_rng.uniform(lo, hi))
        return pos

    def _check_capture(self) -> None:
        if not self.race or self.objective_pos is None:
            return
        best: SimPlayer | None = None
        best_d = self.objective_radius + 1.0
        for p in self.players.values():
            if not p.standing:
                continue
            d = dist(p.pos, self.objective_pos)
            if d <= self.objective_radius and d < best_d:
                best, best_d = p, d
        if best is None:
            return
        self.captures[best.team] += 1
        best.captures += 1
        self.events.append({
            "type": "capture", "t": round(self.t, 3), "agent": best.id, "team": best.team,
            "index": self.objective_index, "pos": vec2(round(self.objective_pos[0], 3), round(self.objective_pos[1], 3)),
            "time_to_capture": round(self.t - self.objective_spawned_at, 3),
        })
        self.objective_index += 1
        self.objective_spawned_at = self.t
        self.objective_pos = self._sample_objective(self.objective_pos)

    def _objective_info(self) -> dict[str, Any] | None:
        if not self.race or self.objective_pos is None:
            return None
        return {
            "index": self.objective_index,
            "pos": vec2(round(self.objective_pos[0], 3), round(self.objective_pos[1], 3)),
            "radius": self.objective_radius,
            "captures": dict(self.captures),
        }

    # -- properties ------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return round(self.tick * TICK_DT, 4)

    def teammates(self, p: SimPlayer) -> list[SimPlayer]:
        return [q for q in self.players.values() if q.team == p.team and q.id != p.id]

    def enemies(self, p: SimPlayer) -> list[SimPlayer]:
        return [q for q in self.players.values() if q.team != p.team]

    def team_alive(self, team: str) -> bool:
        return any(not q.dead for q in self.players.values() if q.team == team)

    def alive_teams(self) -> list[str]:
        return [t for t in ("team-a", "team-b") if self.team_alive(t)]

    # -- loot layout ------------------------------------------------------------------------
    def _add_loot(self, item: str, pos: tuple[float, float], count: int = 1) -> Loot:
        loot = Loot(self._next_loot_id, item, (round(pos[0], 3), round(pos[1], 3)), count)
        self._next_loot_id += 1
        self.loot[loot.id] = loot
        return loot

    def _spawn_kit(self, center: tuple[float, float], items: tuple[tuple[str, int], ...]) -> None:
        phase = self.rng.uniform(0.0, 2.0 * math.pi)
        n = len(items)
        for i, (item, count) in enumerate(items):
            angle = phase + 2.0 * math.pi * i / n
            r = KIT_RING_RADIUS + self.rng.uniform(-0.2, 0.2)
            pos = (center[0] + r * math.cos(angle), center[1] + r * math.sin(angle))
            self._add_loot(item, pos, count)
            if item in GUNS:  # the engine drops two ammo piles next to a gun
                ammo = GUNS[item].ammo
                px, py = -math.sin(angle), math.cos(angle)
                self._add_loot(ammo, (pos[0] + 1.0 * px, pos[1] + 1.0 * py), AMMO_PILE)
                self._add_loot(ammo, (pos[0] - 1.0 * px, pos[1] - 1.0 * py), AMMO_PILE)

    def _spawn_loot(self) -> None:
        self._spawn_kit(KIT_CENTERS["team-a"], STARTER_KIT)
        self._spawn_kit(KIT_CENTERS["team-b"], STARTER_KIT)
        self._spawn_kit(KIT_CENTERS["center"], CENTER_KIT)

    # -- actions -------------------------------------------------------------------------
    def apply_action(self, agent_id: str, action: Mapping[str, Any] | None) -> None:
        """Apply a CpcAction JSON object at step start (``None`` = keep the held input)."""
        if agent_id not in self.players:
            raise ProtocolError(f"unknown agent {agent_id!r}")
        if agent_id not in self.controlled:
            raise ProtocolError(f"agent {agent_id!r} is not controlled")
        if action is None:
            return
        p = self.players[agent_id]
        if not action:  # {} releases everything
            p.held = HeldInput()
            p.fire_start_pending = False
            return
        move = action.get("move") or {}
        mx, my = float(move.get("x", 0.0)), float(move.get("y", 0.0))
        aim = action.get("aim")
        aim_vec = None
        if aim:
            ax, ay = normalize(float(aim.get("x", 0.0)), float(aim.get("y", 0.0)))
            if ax or ay:
                aim_vec = (ax, ay)
        fire = action.get("fire") or {}
        p.held = HeldInput(move=quantize_move(mx, my), aim=aim_vec, fire_hold=bool(fire.get("hold")))
        p.fire_start_pending = bool(fire.get("start"))
        for raw in action.get("inputs") or []:
            self._apply_input(p, input_to_name(raw))
        use_item = action.get("useItem") or ""
        if use_item:
            self._use_item(p, str(use_item))

    def _apply_input(self, p: SimPlayer, name: str) -> None:
        if p.dead:
            return
        if name == "Interact":
            self._interact(p)
        elif name == "Revive":
            self._try_revive(p)
        elif name == "Reload":
            self._try_reload(p)
        elif name == "Cancel":
            self._cancel_action(p)
        elif name == "EquipPrimary":
            self._equip(p, 0)
        elif name == "EquipSecondary":
            self._equip(p, 1)
        elif name == "EquipMelee":
            self._equip(p, MELEE_SLOT)
        elif name in ("EquipNextWeap", "EquipPrevWeap"):
            step = 1 if name == "EquipNextWeap" else -1
            for k in range(1, 4):
                idx = (p.cur_weap_idx + step * k) % 4
                if p.weapons[idx]["type"]:
                    self._equip(p, idx)
                    break
        elif name in USE_INPUT_TO_ITEM:
            self._use_item(p, USE_INPUT_TO_ITEM[name])
        # other inputs (map, emotes, ...) are no-ops in the mock

    def _cancel_action(self, p: SimPlayer) -> None:
        if p.action_type == ACTION_REVIVE and p.action_target in self.players:
            target = self.players[p.action_target]
            if target.action_type == ACTION_REVIVE and target.action_target == p.id:
                target.action_type, target.action_target = ACTION_NONE, ""
        p.action_type, p.action_item, p.action_target = ACTION_NONE, "", ""

    def _equip(self, p: SimPlayer, idx: int) -> None:
        if p.downed or idx == p.cur_weap_idx or not p.weapons[idx]["type"]:
            return
        if p.action_type == ACTION_RELOAD:
            self._cancel_action(p)
        p.cur_weap_idx = idx
        gun = p.cur_gun
        delay = gun.switch_delay if gun else 0.25
        p.fire_ready_tick = max(p.fire_ready_tick, self.tick + int(round(delay / TICK_DT)))

    def _try_reload(self, p: SimPlayer) -> None:
        gun = p.cur_gun
        if p.downed or gun is None or p.action_type != ACTION_NONE:
            return
        if p.clip() >= gun.max_clip or p.reserve() <= 0:
            return
        p.action_type = ACTION_RELOAD
        p.action_end_tick = self.tick + int(round(gun.reload_time / TICK_DT))

    def _use_item(self, p: SimPlayer, item: str) -> None:
        if p.downed or p.dead:
            return
        if item in ZOOM_BY_SCOPE:
            if item in p.scopes:  # switch between owned scopes
                p.scope = item
            return
        if item not in HEAL_ITEMS or p.inventory.get(item, 0) <= 0 or p.action_type != ACTION_NONE:
            return
        use_time, hp_heal, boost_add = HEAL_ITEMS[item]
        if hp_heal > 0 and p.hp >= MAX_HP:
            return
        if boost_add > 0 and p.boost >= 100.0:
            return
        p.action_type = ACTION_USE_ITEM
        p.action_item = item
        p.action_end_tick = self.tick + int(round(use_time / TICK_DT))

    def _try_revive(self, p: SimPlayer) -> bool:
        if not p.standing or p.action_type != ACTION_NONE:
            return False
        targets = [
            q for q in self.teammates(p) if q.downed and not q.dead and dist(p.pos, q.pos) <= REVIVE_RANGE
        ]
        if not targets:
            return False
        target = min(targets, key=lambda q: dist(p.pos, q.pos))
        p.action_type, p.action_target = ACTION_REVIVE, target.id
        p.action_end_tick = self.tick + int(round(REVIVE_DURATION / TICK_DT))
        target.action_type, target.action_target = ACTION_REVIVE, p.id
        return True

    def _interact(self, p: SimPlayer) -> None:
        if p.downed:
            return
        candidates: list[tuple[float, str, Any]] = []
        for loot in self.loot.values():
            d = dist(p.pos, loot.pos)
            if d <= PICKUP_RANGE:
                candidates.append((d, "loot", loot))
        for q in self.teammates(p):
            if q.downed and not q.dead:
                d = dist(p.pos, q.pos)
                if d <= REVIVE_RANGE:
                    candidates.append((d, "revive", q))
        if not candidates:
            return
        candidates.sort(key=lambda c: c[0])
        _, kind, obj = candidates[0]
        if kind == "loot":
            self._pickup(p, obj)
        else:
            self._try_revive(p)

    def _pickup(self, p: SimPlayer, loot: Loot) -> None:
        item, cls = loot.type, item_class(loot.type)
        if cls == "gun":
            slot = 0 if not p.weapons[0]["type"] else (1 if not p.weapons[1]["type"] else None)
            if slot is None:
                slot = p.cur_weap_idx if p.cur_weap_idx in (0, 1) else 0
                old = p.weapons[slot]
                self._add_loot(old["type"], p.pos, 1)  # drop the replaced gun
            p.weapons[slot] = {"slot": slot, "type": item, "ammo": GUNS[item].max_clip}
            del self.loot[loot.id]
            if p.cur_weapon == FISTS:  # engine auto-equips a gun picked up while holding fists
                self._equip(p, slot)
        elif cls in ("ammo", "heal"):
            cap = BAG_CAPS_L0.get(item, 0)
            have = p.inventory.get(item, 0)
            take = min(loot.count, max(0, cap - have))
            if take <= 0:
                return
            p.inventory[item] = have + take
            loot.count -= take
            if loot.count <= 0:
                del self.loot[loot.id]
        elif cls == "armor":
            if item.startswith("helmet"):
                if p.helmet == item:
                    return
                if p.helmet:
                    self._add_loot(p.helmet, p.pos, 1)
                p.helmet = item
            else:
                if p.chest == item:
                    return
                if p.chest:
                    self._add_loot(p.chest, p.pos, 1)
                p.chest = item
            del self.loot[loot.id]
        elif cls == "scope":
            if item in p.scopes:
                return
            p.scopes.add(item)
            if SCOPE_LEVEL[item] > SCOPE_LEVEL.get(p.scope, 1):
                p.scope = item
            del self.loot[loot.id]

    # -- stepping ----------------------------------------------------------------------------
    def step(self, ticks: int) -> dict[str, Any]:
        if self.done:
            raise ProtocolError("episode is done; call reset")
        if int(ticks) <= 0:
            raise ProtocolError("ticks must be a positive integer")
        self.events = []
        for _ in range(int(ticks)):
            self._tick()
            if self._check_done():
                break
        return self.observe()

    def _tick(self) -> None:
        # scripted agents decide at the human input cadence (every netSync = 3 ticks)
        if self.scripted in ("chaser", "racer") and self.tick % 3 == 0:
            for p in self.players.values():
                if p.id not in self.controlled and not p.dead:
                    chaser_decide(self, p, race=self.scripted == "racer")
        for p in self.players.values():
            if p.dead:
                continue
            self._update_action(p)
            self._move(p)
            self._fire(p)
        self._update_bullets()
        for p in self.players.values():
            if p.dead:
                continue
            self._bleed_and_boost(p)
            p.alive_ticks += 1
            p.hp_sum += p.hp
            if p.downed:
                p.downed_ticks += 1
        self.tick += 1
        self._check_capture()

    def _update_action(self, p: SimPlayer) -> None:
        if p.action_type == ACTION_NONE:
            return
        if p.action_type == ACTION_REVIVE:
            target = self.players.get(p.action_target)
            if p.downed:
                # the downed side of a revive: cancelled when the reviver stops
                if target is None or target.action_type != ACTION_REVIVE or target.action_target != p.id:
                    p.action_type, p.action_target = ACTION_NONE, ""
                return
            if target is None or not target.downed or target.dead or dist(p.pos, target.pos) > REVIVE_RANGE:
                self._cancel_action(p)
                return
            if self.tick >= p.action_end_tick:
                target.downed = False
                target.hp = REVIVE_HP
                target.action_type, target.action_target = ACTION_NONE, ""
                p.action_type, p.action_target = ACTION_NONE, ""
                self.events.append({"type": "revive", "t": self.t, "agent": target.id, "source": p.id})
            return
        if self.tick < p.action_end_tick:
            return
        if p.action_type == ACTION_RELOAD:
            gun = p.cur_gun
            if gun is not None:
                slot = p.weapons[p.cur_weap_idx]
                need = gun.max_clip - int(slot["ammo"])
                take = min(need, p.inventory.get(gun.ammo, 0))
                slot["ammo"] = int(slot["ammo"]) + take
                p.inventory[gun.ammo] -= take
        elif p.action_type == ACTION_USE_ITEM:
            item = p.action_item
            if p.inventory.get(item, 0) > 0:
                _, hp_heal, boost_add = HEAL_ITEMS[item]
                p.inventory[item] -= 1
                hp_before = p.hp
                p.hp = min(MAX_HP, p.hp + hp_heal)
                p.boost = min(100.0, p.boost + boost_add)
                self.events.append(
                    {"type": "heal", "t": self.t, "agent": p.id, "weapon": item,
                     "amount": round(p.hp - hp_before, 3)}
                )
        p.action_type, p.action_item, p.action_target = ACTION_NONE, "", ""

    def _move(self, p: SimPlayer) -> None:
        if p.held.aim is not None:
            p.dir = p.held.aim
        mx, my = p.held.move
        if p.action_type == ACTION_REVIVE and not p.downed:
            mx, my = 0.0, 0.0  # reviver kneels
        if mx == 0.0 and my == 0.0:
            return
        if p.downed:
            speed = DOWNED_MOVE_SPEED
        else:
            speed = BASE_MOVE_SPEED + (FISTS_SPEED_BONUS if p.cur_weapon == FISTS else 0.0)
        x, y = p.pos
        if not (SHORE <= x <= MAP_SIZE - SHORE and SHORE <= y <= MAP_SIZE - SHORE):
            speed = max(1.0, speed - WATER_SPEED_PENALTY)
        nx = min(MAP_SIZE - PLAYER_RADIUS, max(PLAYER_RADIUS, x + mx * speed * TICK_DT))
        ny = min(MAP_SIZE - PLAYER_RADIUS, max(PLAYER_RADIUS, y + my * speed * TICK_DT))
        p.pos = (nx, ny)

    def _fire(self, p: SimPlayer) -> None:
        want = p.fire_start_pending or p.held.fire_hold
        p.fire_start_pending = False
        if not want or p.downed:
            return
        gun = p.cur_gun
        if gun is None:
            return  # fists: no melee in the mock
        if p.action_type == ACTION_USE_ITEM:
            self._cancel_action(p)  # shooting interrupts healing
        if p.action_type != ACTION_NONE or self.tick < p.fire_ready_tick:
            return
        slot = p.weapons[p.cur_weap_idx]
        if int(slot["ammo"]) <= 0:
            self._try_reload(p)  # engine auto-reloads when firing on empty
            return
        slot["ammo"] = int(slot["ammo"]) - 1
        p.fire_ready_tick = self.tick + gun.fire_delay_ticks
        p.shots += 1
        moving = p.held.move != (0.0, 0.0)
        spread = gun.spread + (gun.move_spread if moving else 0.0)
        angle = math.atan2(p.dir[1], p.dir[0]) + math.radians(self.rng.uniform(-spread, spread))
        bdir = (math.cos(angle), math.sin(angle))
        start = (p.pos[0] + p.dir[0] * (PLAYER_RADIUS + 0.5), p.pos[1] + p.dir[1] * (PLAYER_RADIUS + 0.5))
        self.bullets.append(Bullet(p, gun, start, bdir))
        self.events.append(
            {"type": "fire", "t": self.t, "agent": p.id, "weapon": gun.name,
             "pos": vec2(round(start[0], 3), round(start[1], 3)),
             "dir": vec2(round(bdir[0], 4), round(bdir[1], 4))}
        )

    def _update_bullets(self) -> None:
        if not self.bullets:
            return
        survivors: list[Bullet] = []
        for b in self.bullets:
            step_len = b.gun.bullet_speed * TICK_DT
            ax, ay = b.pos
            bx, by = ax + b.dir[0] * step_len, ay + b.dir[1] * step_len
            hit: SimPlayer | None = None
            hit_t = 2.0
            for q in self.players.values():
                if q.dead or q.team == b.owner.team:
                    continue
                vx, vy = bx - ax, by - ay
                seg = vx * vx + vy * vy
                tt = 0.0 if seg <= 1e-12 else max(0.0, min(1.0, ((q.pos[0] - ax) * vx + (q.pos[1] - ay) * vy) / seg))
                cx, cy = ax + tt * vx, ay + tt * vy
                if math.hypot(q.pos[0] - cx, q.pos[1] - cy) <= PLAYER_RADIUS + 0.25 and tt < hit_t:
                    hit, hit_t = q, tt
            if hit is not None:
                self._apply_damage(hit, b.gun.damage, b.owner, b.gun.name, armor=True)
                continue
            b.pos = (bx, by)
            b.travelled += step_len
            if b.travelled < b.gun.bullet_distance and 0.0 <= bx <= MAP_SIZE and 0.0 <= by <= MAP_SIZE:
                survivors.append(b)
        self.bullets = survivors

    def _bleed_and_boost(self, p: SimPlayer) -> None:
        if p.downed:
            if self.tick >= p.next_bleed_tick:
                p.next_bleed_tick = self.tick + int(round(BLEED_INTERVAL / TICK_DT))
                amount = BLEED_DAMAGE * (BLEED_MULT ** max(0, p.down_count - 1))
                self._apply_damage(p, amount, None, "", armor=False)
            return
        if p.boost > 0.0:
            regen = 1.0 if p.boost <= 25 else 3.75 if p.boost <= 50 else 4.75 if p.boost <= 87.5 else 5.0
            p.hp = min(MAX_HP, p.hp + regen * TICK_DT)
            p.boost = max(0.0, p.boost - BOOST_DECAY * TICK_DT)

    def _apply_damage(
        self, target: SimPlayer, raw: float, source: SimPlayer | None, weapon: str, armor: bool
    ) -> None:
        if target.dead:
            return
        amount = raw
        if armor:
            if target.helmet in ARMOR_REDUCTION:
                amount -= amount * ARMOR_REDUCTION[target.helmet]
            if target.chest in ARMOR_REDUCTION:
                amount -= amount * ARMOR_REDUCTION[target.chest]
        hp_before = target.hp
        source_id = source.id if source is not None else "bleed"
        pos = vec2(round(target.pos[0], 3), round(target.pos[1], 3))
        if hp_before - amount > 0.0:
            target.hp = hp_before - amount
            self._record_damage(target, source, amount)
            self.events.append(
                {"type": "damage", "t": self.t, "agent": target.id, "source": source_id, "weapon": weapon,
                 "damage_type": "player",  # the sim only models player fire
                 "amount": round(amount, 3), "hp_before": round(hp_before, 3),
                 "hp_after": round(target.hp, 3), "downed": target.downed, "dead": False, "pos": pos}
            )
            return
        # lethal hit: everything that was left is removed
        amount = hp_before
        self._record_damage(target, source, amount)
        teammate_standing = any(q.standing for q in self.teammates(target))
        if not target.downed and teammate_standing:
            target.downed = True
            target.hp = DOWNED_HP
            target.down_count += 1
            target.downed_since_tick = self.tick
            target.next_bleed_tick = self.tick + int(round(BLEED_INTERVAL / TICK_DT))
            self._cancel_action(target)
            target.held.fire_hold = False
            self.events.append(
                {"type": "damage", "t": self.t, "agent": target.id, "source": source_id, "weapon": weapon,
                 "damage_type": "player",  # the sim only models player fire
                 "amount": round(amount, 3), "hp_before": round(hp_before, 3), "hp_after": 0.0,
                 "downed": True, "dead": False, "pos": pos}
            )
            self.events.append({"type": "down", "t": self.t, "agent": target.id, "source": source_id})
            return
        self.events.append(
            {"type": "damage", "t": self.t, "agent": target.id, "source": source_id, "weapon": weapon,
                 "damage_type": "player",  # the sim only models player fire
             "amount": round(amount, 3), "hp_before": round(hp_before, 3), "hp_after": 0.0,
             "downed": target.downed, "dead": True, "pos": pos}
        )
        self._kill(target, source, source_id)
        # last standing teammate died -> downed teammates die too (engine semantics)
        if not any(q.standing for q in self.teammates(target)):
            for q in self.teammates(target):
                if q.downed and not q.dead:
                    self._kill(q, source, source_id)

    def _record_damage(self, target: SimPlayer, source: SimPlayer | None, amount: float) -> None:
        target.damage_taken += amount
        if source is not None and source.team != target.team:
            source.damage_dealt += amount
            source.hits_given += 1

    def _kill(self, target: SimPlayer, source: SimPlayer | None, source_id: str) -> None:
        self._cancel_action(target)
        target.dead = True
        target.hp = 0.0
        target.death_time = self.t
        target.held = HeldInput()
        if source is not None and source.team != target.team:
            source.kills += 1
        self.events.append({"type": "kill", "t": self.t, "agent": target.id, "source": source_id})

    def _race_winner(self) -> str | None:
        a, b = self.captures["team-a"], self.captures["team-b"]
        return None if a == b else ("team-a" if a > b else "team-b")

    def _check_done(self) -> bool:
        alive = self.alive_teams()
        controlled_dead = bool(self.controlled) and all(self.players[a].dead for a in self.controlled)
        if self.end_on_elimination and len(alive) <= 1:
            self.done, self.reason = True, "elimination"
            self.winner_team = self._race_winner() if self.race else (alive[0] if alive else None)
        elif not self.end_on_elimination and (controlled_dead or not alive):
            self.done, self.reason = True, "controlled_dead"
            self.winner_team = self._race_winner() if self.race else (alive[0] if len(alive) == 1 else None)
        elif self.t >= self.time_limit - 1e-9:
            self.done, self.reason = True, "time_limit"
            self.winner_team = self._race_winner() if self.race else None
        if self.done:
            self.metrics = {aid: self._metrics_for(p) for aid, p in self.players.items()}
        return self.done

    def _metrics_for(self, p: SimPlayer) -> dict[str, Any]:
        partner = self.teammates(p)[0] if self.teammates(p) else None
        survival = p.death_time if p.death_time is not None else self.t
        return {
            "survival_time": round(survival, 3),
            "alive_at_end": not p.dead,
            "downed_time": round(p.downed_ticks * TICK_DT, 3),
            "hp_mean": round(p.hp_sum / p.alive_ticks, 3) if p.alive_ticks else 0.0,
            "hp_end": round(p.hp, 3),
            "damage_dealt": round(p.damage_dealt, 3),
            "damage_taken": round(p.damage_taken, 3),
            "kills": p.kills,
            "shots": p.shots,
            "hits_given": p.hits_given,
            "team_win": self.winner_team == p.team,
            "partner_survival_time": round(
                partner.death_time if partner and partner.death_time is not None else self.t, 3
            ),
            "partner_hp_end": round(partner.hp, 3) if partner else 0.0,
            "captures": p.captures,
            "team_captures": p.captures + (partner.captures if partner else 0),
        }

    # -- observations -------------------------------------------------------------------------
    def culling_rect(self, p: SimPlayer) -> tuple[float, float]:
        hw = p.zoom + CULL_MARGIN
        return hw, hw * 9.0 / 16.0

    def observe_agent(self, p: SimPlayer) -> dict[str, Any]:
        hw, hh = self.culling_rect(p)
        me = p.pos
        gun = p.cur_gun
        obs: dict[str, Any] = {
            "self": {
                "id": p.id,
                "team": p.team,
                "pos": vec2(round(me[0], 3), round(me[1], 3)),
                "dir": vec2(round(p.dir[0], 4), round(p.dir[1], 4)),
                "hp": round(p.hp, 3),
                "boost": round(p.boost, 3),
                "downed": p.downed,
                "dead": p.dead,
                "weapon": p.cur_weapon,
                "clip": p.clip(),
                "reserve": p.reserve() if gun else 0,
                "weapons": [dict(w) for w in p.weapons],
                "inventory": dict(p.inventory),
                "scope": p.scope,
                "zoom": p.zoom,
                "action": p.action_type,
                "cur_weap_idx": p.cur_weap_idx,
            },
            "teammates": [
                {
                    "id": q.id,
                    "pos": vec2(round(q.pos[0], 3), round(q.pos[1], 3)),
                    "dist": round(dist(me, q.pos), 3),
                    "hp": round(q.hp, 3),
                    "downed": q.downed,
                    "dead": q.dead,
                }
                for q in self.teammates(p)
            ],
            "players": [],
            "loot": [],
            "obstacles": [],
            "bullets": [],
            "dead_bodies": [],
            "gas": {
                "mode": 0,
                "rad": GAS_RADIUS,
                "pos": vec2(*CENTER),
                "rad_new": GAS_RADIUS,
                "pos_new": vec2(*CENTER),
            },
            "alive_count": sum(1 for q in self.players.values() if not q.dead),
            "alive_teams": len(self.alive_teams()),
        }
        for q in self.enemies(p):
            if not in_rect(me, q.pos, hw, hh):
                continue
            if q.dead:
                obs["dead_bodies"].append(
                    {"pos": vec2(round(q.pos[0], 3), round(q.pos[1], 3)), "dist": round(dist(me, q.pos), 3)}
                )
                continue
            obs["players"].append(
                {
                    "id": q.id,
                    "team": q.team,
                    "pos": vec2(round(q.pos[0], 3), round(q.pos[1], 3)),
                    "dist": round(dist(me, q.pos), 3),
                    "dir": vec2(round(q.dir[0], 4), round(q.dir[1], 4)),
                    "downed": q.downed,
                    "dead": False,
                    "weapon": q.cur_weapon,
                }
            )
        for loot in self.loot.values():
            if in_rect(me, loot.pos, hw, hh):
                obs["loot"].append(
                    {"id": loot.id, "type": loot.type, "pos": vec2(*loot.pos),
                     "dist": round(dist(me, loot.pos), 3), "count": loot.count}
                )
        for b in self.bullets:
            if in_rect(me, b.pos, hw, hh):
                obs["bullets"].append(
                    {"pos": vec2(round(b.pos[0], 3), round(b.pos[1], 3)),
                     "dir": vec2(round(b.dir[0], 4), round(b.dir[1], 4)), "player_id": b.owner.numeric_id}
                )
        obs["players"].sort(key=lambda e: e["dist"])
        obs["loot"].sort(key=lambda e: e["dist"])
        if self.race and self.objective_pos is not None:
            obs["objective"] = {
                "index": self.objective_index,
                "pos": vec2(round(self.objective_pos[0], 3), round(self.objective_pos[1], 3)),
                "radius": self.objective_radius,
                "dist": round(dist(me, self.objective_pos), 3),
            }
        else:
            obs["objective"] = None
        return obs

    def observe(self) -> dict[str, Any]:
        return {
            "type": "obs",
            "env_id": self.env_id,
            "t": self.t,
            "tick": self.tick,
            "done": self.done,
            "agent_ids": list(AGENT_IDS),
            "teams": dict(TEAMS),
            "obs": {aid: self.observe_agent(p) for aid, p in self.players.items()},
            "events": list(self.events),
            "info": {
                "alive_teams": len(self.alive_teams()),
                "winner_team": self.winner_team,
                "reason": self.reason,
                "metrics": self.metrics,
                "objective": self._objective_info(),
            },
        }


# --------------------------------------------------------------------------------------
# Scripted chaser (mimics the bridge's built-in policy, information-set parity respected)
# --------------------------------------------------------------------------------------
CHASER_APPROACH_DIST = 22.0
CHASER_FIRE_DIST = 30.0
CHASER_SAFE_REVIVE_DIST = 25.0
CHASER_MEMORY_S = 10.0


def _toward(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return normalize(b[0] - a[0], b[1] - a[1])


RACER_ENGAGE_DIST = 25.0


def chaser_decide(sim: FieldSim, p: SimPlayer, race: bool = False) -> None:
    """Scripted opponent. ``race=True`` (the bridge's ``racer``): same loot/heal/combat, but it only
    engages enemies inside ``RACER_ENGAGE_DIST`` and otherwise runs to the shared race point."""
    """Write ``p.held`` (and fire edge inputs) for one decision of the scripted chaser."""
    hw, hh = sim.culling_rect(p)
    visible = [q for q in sim.enemies(p) if not q.dead and in_rect(p.pos, q.pos, hw, hh)]
    mem: dict[str, Any] = p.script
    for q in visible:
        mem[f"seen:{q.id}"] = (q.pos, sim.t)
    nearest = min(visible, key=lambda q: (q.downed, dist(p.pos, q.pos))) if visible else None
    held = HeldInput()
    inputs: list[str] = []

    if p.downed:
        if nearest is not None:
            held.aim = _toward(p.pos, nearest.pos)
        p.held = held
        return

    def apply() -> None:
        p.held = held
        for name in inputs:
            sim._apply_input(p, name)

    # 1) revive a downed teammate when no enemy is close
    for tm in sim.teammates(p):
        if tm.downed and not tm.dead:
            threat = nearest is not None and dist(p.pos, nearest.pos) <= CHASER_SAFE_REVIVE_DIST
            if not threat:
                d = dist(p.pos, tm.pos)
                held.aim = _toward(p.pos, tm.pos)
                if p.action_type == ACTION_REVIVE:
                    apply()
                    return
                if d > 2.5:
                    held.move = quantize_move(*_toward(p.pos, tm.pos))
                else:
                    inputs.append("Revive")
                apply()
                return

    # 2) loot: nearest visible gun when unarmed; matching ammo when safe
    if not p.has_gun():
        guns = [l for l in sim.loot.values() if l.type in GUNS and in_rect(p.pos, l.pos, hw, hh)]
        if guns:
            target = min(guns, key=lambda l: dist(p.pos, l.pos))
            d = dist(p.pos, target.pos)
            if d > 1.0:
                held.move = quantize_move(*_toward(p.pos, target.pos))
            else:
                inputs.append("Interact")
            held.aim = _toward(p.pos, target.pos) if nearest is None else _toward(p.pos, nearest.pos)
            apply()
            return
    elif nearest is None:
        gun = p.cur_gun or GUNS[p.weapons[p.gun_slot()]["type"]]  # type: ignore[index]
        if p.inventory.get(gun.ammo, 0) < 60:
            piles = [
                l for l in sim.loot.values()
                if l.type == gun.ammo and dist(p.pos, l.pos) <= 12.0 and in_rect(p.pos, l.pos, hw, hh)
            ]
            if piles:
                target = min(piles, key=lambda l: dist(p.pos, l.pos))
                if dist(p.pos, target.pos) > 1.0:
                    held.move = quantize_move(*_toward(p.pos, target.pos))
                else:
                    inputs.append("Interact")
                held.aim = _toward(p.pos, target.pos)
                apply()
                return

    # equip the gun if still holding fists (the engine normally auto-equips on pickup)
    if p.cur_gun is None and p.has_gun() and p.action_type == ACTION_NONE:
        slot = p.gun_slot()
        inputs.append("EquipPrimary" if slot == 0 else "EquipSecondary")

    # 3) heal when safe
    if nearest is None and p.hp < 45.0 and p.action_type == ACTION_NONE and p.inventory.get("bandage", 0) > 0:
        inputs.append("UseBandage")
        apply()
        return

    # 4) engage (the racer only when the enemy is close enough to matter)
    racing = race and sim.race and sim.objective_pos is not None
    # reaction clock: game time since an enemy first came inside the fire range (reset when none is)
    if nearest is not None and dist(p.pos, nearest.pos) <= CHASER_FIRE_DIST:
        mem.setdefault("contact_since", sim.t)
    else:
        mem.pop("contact_since", None)
    reacted = sim.reaction_delay <= 0 or sim.t - mem.get("contact_since", sim.t) >= sim.reaction_delay
    if nearest is not None and racing and dist(p.pos, nearest.pos) > sim.engage_dist:
        held.aim = _toward(p.pos, nearest.pos)
        held.move = quantize_move(*_toward(p.pos, sim.objective_pos))
        apply()
        return
    if nearest is not None:
        d = dist(p.pos, nearest.pos)
        to_enemy = _toward(p.pos, nearest.pos)
        held.aim = to_enemy
        if sim.aim_noise_deg > 0:
            angle = math.atan2(to_enemy[1], to_enemy[0]) + math.radians(sim.rng.gauss(0.0, sim.aim_noise_deg))
            held.aim = (math.cos(angle), math.sin(angle))
        if p.cur_gun is not None and d <= CHASER_FIRE_DIST and reacted and (p.clip() > 0 or p.reserve() > 0):
            held.fire_hold = True
        if d > CHASER_APPROACH_DIST:
            held.move = quantize_move(*to_enemy)
        else:
            if sim.tick >= mem.get("strafe_until", -1):
                mem["strafe_sign"] = sim.rng.choice((-1.0, 1.0))
                mem["strafe_until"] = sim.tick + sim.rng.randint(60, 120)
            s = mem.get("strafe_sign", 1.0)
            back = 1.0 if d < 8.0 else 0.0  # give ground when the enemy pushes in
            held.move = quantize_move(-to_enemy[1] * s - to_enemy[0] * back, to_enemy[0] * s - to_enemy[1] * back)
        apply()
        return

    # 5) nothing visible: the racer runs to the point; the chaser searches
    if racing:
        to_point = _toward(p.pos, sim.objective_pos)
        if dist(p.pos, sim.objective_pos) > sim.objective_radius * 0.5:
            held.move = quantize_move(*to_point)
        held.aim = to_point
        apply()
        return
    goal = _search_goal(sim, p, mem)
    if goal is not None:
        held.move = quantize_move(*_toward(p.pos, goal))
        held.aim = _toward(p.pos, goal)
    apply()


def _search_goal(sim: FieldSim, p: SimPlayer, mem: dict[str, Any]) -> tuple[float, float] | None:
    goal: tuple[float, float] | None = None
    best_age = CHASER_MEMORY_S
    for key, value in list(mem.items()):
        if not str(key).startswith("seen:"):
            continue
        pos, seen_t = value
        age = sim.t - seen_t
        if age < best_age and dist(p.pos, pos) > 3.0:
            goal, best_age = pos, age
    if goal is not None:
        return goal
    for tm in sim.teammates(p):
        if tm.standing and dist(p.pos, tm.pos) > 15.0:
            return tm.pos  # regroup (team HUD position is always known)
    enemy_kit = KIT_CENTERS["team-b" if p.team == "team-a" else "team-a"]
    if not mem.get("visited_enemy_kit"):
        if dist(p.pos, enemy_kit) > 4.0:
            return enemy_kit
        mem["visited_enemy_kit"] = True
    waypoint = mem.get("waypoint")
    if waypoint is None or dist(p.pos, waypoint) < 3.0:
        # sweep the scenario region most of the time, the whole map (shore included) sometimes,
        # so hiding at the map edge is not perfectly safe
        half = sim.map_size_option / 2.0 - 8.0 if sim.rng.random() < 0.7 else MAP_SIZE / 2.0 - 4.0
        waypoint = (CENTER[0] + sim.rng.uniform(-half, half), CENTER[1] + sim.rng.uniform(-half, half))
        mem["waypoint"] = waypoint
    return waypoint


# --------------------------------------------------------------------------------------
# WebSocket server
# --------------------------------------------------------------------------------------
class BridgeSession:
    """Per-connection env table + message dispatch (protocol v0)."""

    def __init__(self, sim_factory: Callable[..., FieldSim] = FieldSim) -> None:
        self.envs: dict[int, FieldSim] = {}
        self._sim_factory = sim_factory

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any]:
        kind = message.get("type")
        env_id = message.get("env_id")
        try:
            if kind == "reset":
                return self._reset(message)
            if kind == "step":
                if "envs" in message:
                    return self._step_batch(message)
                return self._step_one(message)
            if kind == "close":
                self.envs.pop(int(env_id), None)
                return {"type": "closed", "env_id": int(env_id)}
            raise ProtocolError(f"unknown message type {kind!r}")
        except ProtocolError as exc:
            return {"type": "error", "env_id": env_id, "message": str(exc)}
        except (KeyError, TypeError, ValueError) as exc:
            return {"type": "error", "env_id": env_id, "message": f"malformed message: {exc!r}"}

    def _reset(self, message: Mapping[str, Any]) -> dict[str, Any]:
        env_id = int(message["env_id"])
        sim = self._sim_factory(
            env_id,
            message.get("seed", "cpc-duo2v2-seed-0"),
            message.get("options") or {},
            message.get("scenario", DEFAULT_SCENARIO),
        )
        self.envs[env_id] = sim
        return sim.observe()

    def _step_env(self, env_id: int, actions: Mapping[str, Any], ticks: int) -> dict[str, Any]:
        sim = self.envs.get(env_id)
        if sim is None:
            raise ProtocolError(f"unknown env {env_id}; send reset first")
        for aid, action in (actions or {}).items():
            sim.apply_action(aid, action)
        return sim.step(int(ticks))

    def _step_one(self, message: Mapping[str, Any]) -> dict[str, Any]:
        return self._step_env(int(message["env_id"]), message.get("actions") or {}, message.get("ticks", 10))

    def _step_batch(self, message: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, body in (message.get("envs") or {}).items():
            env_id = int(key)
            try:
                out[str(env_id)] = self._step_env(env_id, body.get("actions") or {}, body.get("ticks", 10))
            except ProtocolError as exc:
                out[str(env_id)] = {"type": "error", "env_id": env_id, "message": str(exc)}
        return {"type": "obs_batch", "envs": out}


def _connection_handler(websocket: Any) -> None:
    session = BridgeSession()
    try:
        for raw in websocket:
            try:
                message = decode_message(raw)
            except ProtocolError as exc:
                websocket.send(encode_message({"type": "error", "env_id": None, "message": str(exc)}))
                continue
            websocket.send(encode_message(session.handle(message)))
    finally:
        session.envs.clear()  # closing a connection closes all its envs


def make_server(host: str = "127.0.0.1", port: int = 8765) -> Any:
    from websockets.sync.server import serve as ws_serve

    return ws_serve(_connection_handler, host, port, max_size=None, compression=None)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Blocking: run the mock bridge until interrupted."""
    with make_server(host, port) as server:
        print(f"mock bridge listening on ws://{host}:{server.socket.getsockname()[1]}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover
            pass


class MockBridgeThread:
    """Run the mock bridge in a daemon thread (tests, ``--mock`` mode).

    ``port=0`` picks a free port; the resolved URL is in ``.url`` after ``start()``.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.url = ""
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> "MockBridgeThread":
        self._server = make_server(self.host, self.port)
        self.port = int(self._server.socket.getsockname()[1])
        self.url = f"ws://{self.host}:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, name="mock-bridge", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "MockBridgeThread":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the mock survev CPC bridge (protocol v0).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
