"""AgentObservation -> fixed-size float32 vector (with ``vector_keys``).

Layout (default config, 216 floats; ``Featurizer.describe()`` prints the exact table):

======  ==================  =====================================================================
block   size                features
======  ==================  =====================================================================
self    31                  hp/100, boost/100, downed, dead, dir(x,y), abs pos (2, /far_scale from
                            the map center), weapon class one-hot (fists/ak47/mp5/other_gun),
                            has_primary, has_secondary, gun_equipped, clip/30, reserve/90,
                            action one-hot (none/reload/use_item/revive), inventory counts (8,
                            normalized by level-0 bag caps), scope (zoom-28)/40, time fraction
tm      9 x K_tm (1)        present, dx, dy, dist, unit(x,y), hp/100, downed, dead
en      10 x K_en (3)       present, dx, dy, dist, unit(x,y), facing dir(x,y), downed, armed
loot    11 x K_loot (6)     present, dx, dy, dist, class one-hot (gun/ammo/heal/armor/scope/
                            other), count/30
ob      8 x K_ob (4)        present, dx, dy, dist, collidable, scale, los_blocked*, cover_score*
bl      6 x K_bl (4)        present, dx, dy, dir(x,y), approaching
gas     7                   active, inside, shrinking, edge distance, new-center unit(x,y),
                            new-edge distance
counts  5                   alive_count/4, alive_teams/2, enemies/K_en, loot/K_loot, bullets/K_bl
mem     4 x K_en (3)        seen, last dx, last dy, recency = exp(-age / memory_decay_s)
goal    6 (if goal=True)    present, dx, dy, dist, unit(x,y) of the waypoint handed to featurize()
                            (or of obs["objective"], the shared race point, when none is handed in)
======  ==================  =====================================================================

``*`` = placeholders (always 0 today) for line-of-sight / cover features once the map has
obstacles; the slots exist so the vector layout does not change when they land.

Relative positions are egocentric (target - self) divided by ``pos_scale`` (32 u, the
culling half-width at 1x) and clipped to ``+-clip_value``. With ``rotate_to_facing=True``
every relative vector / direction is rotated into the agent's facing frame (facing = +x).
The default is *off* because the action space uses absolute aim bins; keep both consistent.

Nearest-K entries are sorted by distance; absent slots are all-zero with ``present = 0``.
The per-agent ``AgentMemory`` (optional) remembers the last seen position of every enemy
across steps, keyed by roster order so the slots are stable across an episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import AgentObservation, INVENTORY_KEYS

WEAPON_CLASSES: tuple[str, ...] = ("fists", "ak47", "mp5", "other_gun")
LOOT_CLASSES: tuple[str, ...] = ("gun", "ammo", "heal", "armor", "scope", "other")
GUN_TYPES: frozenset[str] = frozenset({"ak47", "mp5", "m416", "scar", "mosin", "m870", "spas12", "mp220", "saiga", "m9", "ot38", "m1911", "deagle", "hk416", "m4a1", "mk12", "l86", "an94", "groza", "grozas", "famas", "vector", "ump9", "scorpion", "dp28", "bar", "m249", "qbb97", "pkp", "sv98", "awc", "blr", "garand", "m1a1", "model94", "svd", "vss", "scout"})
AMMO_ITEMS: frozenset[str] = frozenset({"762mm", "9mm", "12gauge", "556mm", "50AE", "308sub", "flare", "45acp"})
HEAL_ITEMS: frozenset[str] = frozenset({"bandage", "healthkit", "soda", "painkiller"})
ARMOR_PREFIXES: tuple[str, ...] = ("helmet", "chest", "backpack")
INVENTORY_CAPS: dict[str, float] = {
    "bandage": 5, "healthkit": 1, "soda": 2, "painkiller": 1,
    "762mm": 90, "9mm": 120, "12gauge": 15, "556mm": 90,
}
ACTION_CLASSES: tuple[str, ...] = ("none", "reload", "use_item", "revive")
MAP_CENTER = (132.0, 132.0)


def loot_class(item_type: str) -> str:
    if item_type in GUN_TYPES:
        return "gun"
    if item_type in AMMO_ITEMS:
        return "ammo"
    if item_type in HEAL_ITEMS:
        return "heal"
    if item_type.startswith(ARMOR_PREFIXES):
        return "armor"
    if item_type.endswith("scope"):
        return "scope"
    return "other"


def weapon_class(weapon: str) -> str:
    if weapon in ("", "fists"):
        return "fists"
    if weapon in ("ak47", "mp5"):
        return weapon
    return "other_gun"


@dataclass
class FeaturizerConfig:
    max_teammates: int = 1
    max_enemies: int = 3
    max_loot: int = 6
    max_obstacles: int = 4
    max_bullets: int = 4
    rotate_to_facing: bool = False
    pos_scale: float = 32.0  # culling half-width at 1x scope
    far_scale: float = 128.0  # scenario region (absolute position, gas)
    clip_value: float = 3.0
    memory: bool = True
    memory_decay_s: float = 10.0
    time_limit: float = 60.0
    goal: bool = False  # append the waypoint block (goal-conditioned policies); off keeps old checkpoints valid

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeaturizerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentMemory:
    """Last-seen enemy positions for one controlled agent (reset every episode)."""

    enemy_order: list[str] = field(default_factory=list)
    last_seen: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # id -> (x, y, t)

    def observe(self, enemy_id: str, x: float, y: float, t: float) -> None:
        if enemy_id not in self.enemy_order:
            self.enemy_order.append(enemy_id)
        self.last_seen[enemy_id] = (x, y, t)


class Featurizer:
    """Deterministic AgentObservation -> np.float32[size] mapping."""

    def __init__(self, config: FeaturizerConfig | None = None) -> None:
        self.config = config or FeaturizerConfig()
        self._keys, self._layout = self._build_keys()

    # -- layout -----------------------------------------------------------------------------
    def _build_keys(self) -> tuple[list[str], dict[str, tuple[int, int]]]:
        c = self.config
        keys: list[str] = []
        layout: dict[str, tuple[int, int]] = {}

        def block(name: str, names: Sequence[str]) -> None:
            start = len(keys)
            keys.extend(names)
            layout[name] = (start, len(keys))

        block(
            "self",
            ["self_hp", "self_boost", "self_downed", "self_dead", "self_dir_x", "self_dir_y",
             "self_pos_x", "self_pos_y"]
            + [f"self_weapon_{w}" for w in WEAPON_CLASSES]
            + ["self_has_primary", "self_has_secondary", "self_gun_equipped", "self_clip", "self_reserve"]
            + [f"self_action_{a}" for a in ACTION_CLASSES]
            + [f"self_inv_{k}" for k in INVENTORY_KEYS]
            + ["self_scope", "self_time_frac"],
        )
        for i in range(c.max_teammates):
            block(f"tm{i}", [f"tm{i}_{n}" for n in ("present", "dx", "dy", "dist", "ux", "uy", "hp", "downed", "dead")])
        for i in range(c.max_enemies):
            block(f"en{i}", [f"en{i}_{n}" for n in ("present", "dx", "dy", "dist", "ux", "uy", "dir_x", "dir_y", "downed", "armed")])
        for i in range(c.max_loot):
            block(f"loot{i}", [f"loot{i}_{n}" for n in ("present", "dx", "dy", "dist")]
                  + [f"loot{i}_cls_{k}" for k in LOOT_CLASSES] + [f"loot{i}_count"])
        for i in range(c.max_obstacles):
            block(f"ob{i}", [f"ob{i}_{n}" for n in ("present", "dx", "dy", "dist", "collidable", "scale", "los_blocked", "cover_score")])
        for i in range(c.max_bullets):
            block(f"bl{i}", [f"bl{i}_{n}" for n in ("present", "dx", "dy", "dir_x", "dir_y", "approaching")])
        block("gas", ["gas_active", "gas_inside", "gas_shrinking", "gas_edge_dist", "gas_new_ux", "gas_new_uy", "gas_new_edge_dist"])
        block("counts", ["alive_count", "alive_teams", "n_enemies_visible", "n_loot_visible", "n_bullets_visible"])
        if c.memory:
            for i in range(c.max_enemies):
                block(f"mem{i}", [f"mem{i}_{n}" for n in ("seen", "dx", "dy", "recency")])
        if c.goal:
            block("goal", [f"goal_{n}" for n in ("present", "dx", "dy", "dist", "ux", "uy")])
        return keys, layout

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def vector_keys(self) -> list[str]:
        return list(self._keys)

    @property
    def layout(self) -> dict[str, tuple[int, int]]:
        return dict(self._layout)

    def describe(self) -> str:
        lines = [f"obs_dim = {self.size}", "block  start  end  size"]
        for name, (s, e) in self._layout.items():
            lines.append(f"{name:<6} {s:>5} {e:>4} {e - s:>5}")
        return "\n".join(lines)

    def new_memory(self, roster_enemies: Sequence[str] = ()) -> AgentMemory:
        return AgentMemory(enemy_order=list(roster_enemies))

    # -- featurization ---------------------------------------------------------------------
    def featurize(
        self,
        obs: AgentObservation,
        t: float = 0.0,
        memory: AgentMemory | None = None,
        goal: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """``goal`` is the waypoint (world x, y) for the ``goal`` block; ignored unless ``config.goal``.
        When no explicit goal is given, the shared race point in ``obs["objective"]`` (if any) is used."""
        c = self.config
        if goal is None and c.goal:
            objective = obs.get("objective")
            if objective:
                goal = (float(objective["pos"]["x"]), float(objective["pos"]["y"]))
        me = obs["self"]
        sx, sy = float(me["pos"]["x"]), float(me["pos"]["y"])
        dx_dir, dy_dir = float(me["dir"]["x"]), float(me["dir"]["y"])
        norm = math.hypot(dx_dir, dy_dir)
        if norm < 1e-6:
            dx_dir, dy_dir = 1.0, 0.0
        else:
            dx_dir, dy_dir = dx_dir / norm, dy_dir / norm
        if c.rotate_to_facing:
            cos_t, sin_t = dx_dir, dy_dir  # rotate by -theta: (x,y) -> (x cos + y sin, -x sin + y cos)
        else:
            cos_t, sin_t = 1.0, 0.0
        clip = c.clip_value

        def rot(x: float, y: float) -> tuple[float, float]:
            return x * cos_t + y * sin_t, -x * sin_t + y * cos_t

        def rel(px: float, py: float, scale: float) -> tuple[float, float, float, float, float]:
            """Returns (dx, dy, dist, ux, uy) normalized/rotated."""
            rx, ry = rot(px - sx, py - sy)
            d = math.hypot(rx, ry)
            ux, uy = (rx / d, ry / d) if d > 1e-6 else (0.0, 0.0)
            return (
                max(-clip, min(clip, rx / scale)),
                max(-clip, min(clip, ry / scale)),
                min(clip, d / scale),
                ux,
                uy,
            )

        out: list[float] = []
        # -- self
        weapons = me.get("weapons") or []
        slot_type = {int(w["slot"]): str(w.get("type") or "") for w in weapons}
        wclass = weapon_class(str(me.get("weapon") or ""))
        gun_equipped = 1.0 if wclass != "fists" else 0.0
        inv = me.get("inventory") or {}
        fdx, fdy = rot(dx_dir, dy_dir)
        out += [
            float(me.get("hp", 0.0)) / 100.0,
            float(me.get("boost", 0.0)) / 100.0,
            1.0 if me.get("downed") else 0.0,
            1.0 if me.get("dead") else 0.0,
            fdx,
            fdy,
            (sx - MAP_CENTER[0]) / c.far_scale,
            (sy - MAP_CENTER[1]) / c.far_scale,
        ]
        out += [1.0 if wclass == w else 0.0 for w in WEAPON_CLASSES]
        out += [
            1.0 if slot_type.get(0, "") else 0.0,
            1.0 if slot_type.get(1, "") else 0.0,
            gun_equipped,
            min(clip, float(me.get("clip", 0)) / 30.0),
            min(clip, float(me.get("reserve", 0)) / 90.0),
        ]
        action = int(me.get("action", 0))
        out += [1.0 if action == i else 0.0 for i in range(len(ACTION_CLASSES))]
        out += [min(clip, float(inv.get(k, 0)) / INVENTORY_CAPS.get(k, 1.0)) for k in INVENTORY_KEYS]
        out += [
            (float(me.get("zoom", 28)) - 28.0) / 40.0,
            min(1.0, float(t) / c.time_limit) if c.time_limit > 0 else 0.0,
        ]
        # -- teammates (always reported by the server, nearest first)
        tms = sorted(obs.get("teammates") or [], key=lambda e: float(e.get("dist", 0.0)))
        for i in range(c.max_teammates):
            if i < len(tms):
                e = tms[i]
                dx, dy, d, ux, uy = rel(float(e["pos"]["x"]), float(e["pos"]["y"]), c.pos_scale)
                out += [1.0, dx, dy, d, ux, uy, float(e.get("hp", 0.0)) / 100.0,
                        1.0 if e.get("downed") else 0.0, 1.0 if e.get("dead") else 0.0]
            else:
                out += [0.0] * 9
        # -- enemies (culled by the server; only visible ones arrive)
        ens = sorted(obs.get("players") or [], key=lambda e: float(e.get("dist", 0.0)))
        if memory is not None:
            for e in ens:
                memory.observe(str(e["id"]), float(e["pos"]["x"]), float(e["pos"]["y"]), float(t))
        for i in range(c.max_enemies):
            if i < len(ens):
                e = ens[i]
                dx, dy, d, ux, uy = rel(float(e["pos"]["x"]), float(e["pos"]["y"]), c.pos_scale)
                edir = e.get("dir") or {"x": 0.0, "y": 0.0}
                ddx, ddy = rot(float(edir["x"]), float(edir["y"]))
                armed = 1.0 if str(e.get("weapon") or "fists") not in ("", "fists") else 0.0
                out += [1.0, dx, dy, d, ux, uy, ddx, ddy, 1.0 if e.get("downed") else 0.0, armed]
            else:
                out += [0.0] * 10
        # -- loot
        loots = sorted(obs.get("loot") or [], key=lambda e: float(e.get("dist", 0.0)))
        for i in range(c.max_loot):
            if i < len(loots):
                e = loots[i]
                dx, dy, d, _, _ = rel(float(e["pos"]["x"]), float(e["pos"]["y"]), c.pos_scale)
                cls = loot_class(str(e.get("type") or ""))
                out += [1.0, dx, dy, d] + [1.0 if cls == k else 0.0 for k in LOOT_CLASSES]
                out += [min(clip, float(e.get("count", 1)) / 30.0)]
            else:
                out += [0.0] * 11
        # -- obstacles (LOS / cover placeholders stay 0 until the map has cover)
        obs_list = sorted(obs.get("obstacles") or [], key=lambda e: float(e.get("dist", 0.0)))
        for i in range(c.max_obstacles):
            if i < len(obs_list):
                e = obs_list[i]
                dx, dy, d, _, _ = rel(float(e["pos"]["x"]), float(e["pos"]["y"]), c.pos_scale)
                out += [1.0, dx, dy, d, 1.0 if e.get("collidable") else 0.0,
                        min(clip, float(e.get("scale", 1.0))), 0.0, 0.0]
            else:
                out += [0.0] * 8
        # -- bullets
        bullets = obs.get("bullets") or []
        bl = sorted(
            bullets,
            key=lambda e: math.hypot(float(e["pos"]["x"]) - sx, float(e["pos"]["y"]) - sy),
        )
        for i in range(c.max_bullets):
            if i < len(bl):
                e = bl[i]
                dx, dy, _, ux, uy = rel(float(e["pos"]["x"]), float(e["pos"]["y"]), c.pos_scale)
                bdir = e.get("dir") or {"x": 0.0, "y": 0.0}
                bdx, bdy = rot(float(bdir["x"]), float(bdir["y"]))
                approaching = 1.0 if (bdx * -ux + bdy * -uy) > 0.0 else 0.0
                out += [1.0, dx, dy, bdx, bdy, approaching]
            else:
                out += [0.0] * 6
        # -- gas
        gas = obs.get("gas") or {}
        gpos = gas.get("pos") or {"x": MAP_CENTER[0], "y": MAP_CENTER[1]}
        gnew = gas.get("pos_new") or gpos
        rad = float(gas.get("rad", 0.0))
        rad_new = float(gas.get("rad_new", rad))
        d_center = math.hypot(sx - float(gpos["x"]), sy - float(gpos["y"]))
        d_new = math.hypot(sx - float(gnew["x"]), sy - float(gnew["y"]))
        _, _, _, gux, guy = rel(float(gnew["x"]), float(gnew["y"]), c.far_scale)
        out += [
            1.0 if int(gas.get("mode", 0)) > 0 else 0.0,
            1.0 if d_center <= rad else 0.0,
            1.0 if rad_new < rad - 1e-6 else 0.0,
            max(-clip, min(clip, (rad - d_center) / c.far_scale)),
            gux,
            guy,
            max(-clip, min(clip, (rad_new - d_new) / c.far_scale)),
        ]
        # -- counts
        out += [
            float(obs.get("alive_count", 0)) / 4.0,
            float(obs.get("alive_teams", 0)) / 2.0,
            min(1.0, len(ens) / max(1, c.max_enemies)),
            min(1.0, len(loots) / max(1, c.max_loot)),
            min(1.0, len(bullets) / max(1, c.max_bullets)),
        ]
        # -- memory
        if c.memory:
            if memory is not None:
                order, last_seen = memory.enemy_order, memory.last_seen
            else:  # stateless call: the block degrades to "currently visible"
                order = [str(e["id"]) for e in ens]
                last_seen = {
                    str(e["id"]): (float(e["pos"]["x"]), float(e["pos"]["y"]), float(t)) for e in ens
                }
            for i in range(c.max_enemies):
                if i < len(order) and order[i] in last_seen:
                    lx, ly, lt = last_seen[order[i]]
                    dx, dy, _, _, _ = rel(lx, ly, c.pos_scale)
                    age = max(0.0, float(t) - lt)
                    out += [1.0, dx, dy, math.exp(-age / c.memory_decay_s) if c.memory_decay_s > 0 else 1.0]
                else:
                    out += [0.0] * 4
        # -- goal (waypoint)
        if c.goal:
            if goal is not None:
                gdx, gdy, gd, gux, guy = rel(float(goal[0]), float(goal[1]), c.pos_scale)
                out += [1.0, gdx, gdy, gd, gux, guy]
            else:
                out += [0.0] * 6
        vec = np.asarray(out, dtype=np.float32)
        if vec.shape[0] != self.size:  # pragma: no cover - layout guard
            raise RuntimeError(f"featurizer produced {vec.shape[0]} values, expected {self.size}")
        return vec
