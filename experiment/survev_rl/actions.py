"""Action spaces: MultiDiscrete primitives (harness bins) and a high-level skill mode.

Primitive mode (default) is ``MultiDiscrete([9, 16, 2, 2])``:

* ``move``  in 0..8   -> harness ``MOVE_VECTORS`` (0 = stand still, then the 8 directions)
                       used directly as the world-space ``move`` vector;
* ``aim``   in 0..15  -> harness ``aim_bin_to_vec`` (absolute world angle, 22.5 deg bins);
* ``fire``  in {0,1}  -> ``fire.hold`` (level-triggered auto fire);
* ``interact`` {0,1}  -> ``Interact`` input (pick up loot / revive) applied at step start.

The *assist layer* (``assist=True``) adds housekeeping inputs the policy would otherwise
have to discover by exploration: ``EquipPrimary`` / ``EquipSecondary`` when a gun sits in a
slot but fists are equipped, and ``Reload`` when the equipped gun's clip is empty (and
reserve ammo exists). It only ever *adds* inputs; the four policy dimensions are untouched.
``auto_pickup=True`` (off by default, curriculum aid) additionally presses ``Interact``
whenever loot is within pickup range, making the ``interact`` dimension redundant.

Skill mode is ``Discrete(len(SKILLS))`` over named skills (``move_to_partner``,
``loot_nearest``, ``engage``, ``retreat``, ``take_cover``, ``revive``, ``hold``) producing
``{"skill": name, "params": {...}}``. The bridge does **not** execute skills yet (System 1
skills land in the TS server later); this module only implements the mapping + tests so
the PPO code is already shaped for it. ``ActionSpace(mode="skill").to_cpc_action`` raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from experiment.core.cpc_actions import AIM_BINS, MOVE_BINS, MOVE_LABELS, MOVE_VECTORS, aim_bin_to_vec

from .protocol import ACTION_NONE, ACTION_RELOAD, AgentObservation, CpcAction

ActionMode = Literal["primitive", "skill"]
PRIMITIVE_DIMS: tuple[str, ...] = ("move", "aim", "fire", "interact")
PRIMITIVE_NVEC: tuple[int, ...] = (MOVE_BINS, AIM_BINS, 2, 2)
SKILLS: tuple[str, ...] = (
    "move_to_partner",
    "loot_nearest",
    "engage",
    "retreat",
    "take_cover",
    "revive",
    "hold",
)
GUN_SLOTS = (0, 1)


@dataclass(frozen=True)
class ActionSpace:
    """MultiDiscrete (primitive) or Discrete (skill) action space -> bridge actions."""

    mode: ActionMode = "primitive"
    assist: bool = True
    auto_pickup: bool = False  # assist extra: press Interact whenever loot is within reach

    # -- shape --------------------------------------------------------------------------
    @property
    def nvec(self) -> tuple[int, ...]:
        return PRIMITIVE_NVEC if self.mode == "primitive" else (len(SKILLS),)

    @property
    def dims(self) -> tuple[str, ...]:
        return PRIMITIVE_DIMS if self.mode == "primitive" else ("skill",)

    @property
    def n_dims(self) -> int:
        return len(self.nvec)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.integers(n) for n in self.nvec], dtype=np.int64)

    def validate(self, action: Sequence[int]) -> tuple[int, ...]:
        if len(action) != self.n_dims:
            raise ValueError(f"expected {self.n_dims} action dims, got {len(action)}")
        out = []
        for value, n in zip(action, self.nvec):
            v = int(value)
            if not 0 <= v < n:
                raise ValueError(f"action component {v} out of range [0, {n})")
            out.append(v)
        return tuple(out)

    # -- primitive mode -------------------------------------------------------------------
    def to_cpc_action(self, action: Sequence[int], obs: AgentObservation | None = None) -> CpcAction:
        """Map one policy action row to a ``CpcAction`` (assist inputs need ``obs``)."""
        if self.mode != "primitive":
            raise NotImplementedError(
                "skill mode is not executable through the bridge yet; use to_skill_action()"
            )
        move_bin, aim_bin, fire, interact = self.validate(action)
        mx, my = MOVE_VECTORS[move_bin]
        aim = aim_bin_to_vec(aim_bin, AIM_BINS)
        inputs: list[str | int] = []
        if interact:
            inputs.append("Interact")
        if self.assist and obs is not None:
            inputs.extend(assist_inputs(obs, auto_pickup=self.auto_pickup, already=inputs))
        return CpcAction(
            move=(float(mx), float(my)) if (mx or my) else None,
            aim=(float(aim["x"]), float(aim["y"])),
            fire_start=False,
            fire_hold=bool(fire),
            inputs=inputs,
            use_item="",
        )

    def describe(self, action: Sequence[int]) -> dict[str, Any]:
        if self.mode == "skill":
            return {"skill": SKILLS[int(action[0])]}
        move_bin, aim_bin, fire, interact = self.validate(action)
        return {
            "move": MOVE_LABELS[move_bin],
            "aim_deg": 360.0 * aim_bin / AIM_BINS,
            "fire": bool(fire),
            "interact": bool(interact),
        }

    # -- skill mode -------------------------------------------------------------------
    def to_skill_action(self, action: Sequence[int] | int, obs: AgentObservation | None = None) -> dict[str, Any]:
        """Map a skill index to ``{"skill": name, "params": {...}}`` (params resolved from obs)."""
        idx = int(action if isinstance(action, (int, np.integer)) else action[0])
        if not 0 <= idx < len(SKILLS):
            raise ValueError(f"skill index {idx} out of range")
        name = SKILLS[idx]
        return {"skill": name, "params": skill_params(name, obs)}


# --------------------------------------------------------------------------------------
# assist layer
# --------------------------------------------------------------------------------------
PICKUP_RANGE = 2.0


def assist_inputs(
    obs: AgentObservation, auto_pickup: bool = False, already: Sequence[str | int] = ()
) -> list[str]:
    """Housekeeping inputs derived from the agent's own state (never touches move/aim/fire)."""
    me = obs["self"]
    if me.get("dead") or me.get("downed"):
        return []
    inputs: list[str] = []
    if auto_pickup and "Interact" not in already:
        if any(float(l.get("dist", math.inf)) <= PICKUP_RANGE for l in (obs.get("loot") or [])):
            inputs.append("Interact")
    weapon = str(me.get("weapon") or "fists")
    slots = {int(w["slot"]): str(w.get("type") or "") for w in (me.get("weapons") or [])}
    if weapon in ("", "fists"):
        if slots.get(0):
            inputs.append("EquipPrimary")
        elif slots.get(1):
            inputs.append("EquipSecondary")
        return inputs
    clip = int(me.get("clip", 0))
    reserve = int(me.get("reserve", 0))
    action = int(me.get("action", ACTION_NONE))
    if clip <= 0 and reserve > 0 and action != ACTION_RELOAD:
        inputs.append("Reload")
    return inputs


# --------------------------------------------------------------------------------------
# skill parameter resolution
# --------------------------------------------------------------------------------------
def _nearest(entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return min(entries, key=lambda e: float(e.get("dist", math.inf)))


def skill_params(name: str, obs: AgentObservation | None) -> dict[str, Any]:
    """Fill skill parameters from the observation (ids / positions), empty when unknown."""
    if obs is None:
        return {}
    me = obs["self"]
    teammates = [t for t in (obs.get("teammates") or []) if not t.get("dead")]
    partner = _nearest(teammates)
    enemy = _nearest([p for p in (obs.get("players") or []) if not p.get("dead")])
    if name == "move_to_partner":
        return {"target": partner["id"], "distance": 6.0} if partner else {}
    if name == "loot_nearest":
        loot = _nearest(obs.get("loot") or [])
        return {"target": loot["id"], "type": loot["type"]} if loot else {}
    if name == "engage":
        return {"target": enemy["id"], "style": "hold_angle"} if enemy else {}
    if name == "retreat":
        if enemy:
            return {"away_from": enemy["id"], "distance": 30.0}
        return {"away_from": None, "distance": 30.0}
    if name == "take_cover":
        obstacles = [o for o in (obs.get("obstacles") or []) if o.get("collidable")]
        cover = _nearest(obstacles)
        return {"cover": cover["id"] if cover else None, "face": enemy["id"] if enemy else None}
    if name == "revive":
        downed = [t for t in teammates if t.get("downed")]
        target = _nearest(downed)
        return {"target": target["id"]} if target else {}
    if name == "hold":
        return {"face": enemy["id"] if enemy else None, "pos": dict(me["pos"])}
    raise ValueError(f"unknown skill {name!r}")
