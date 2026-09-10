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
``aim_assist=True`` (off by default) is the one exception to "dimensions untouched": while
``fire`` is held and an enemy is visible, the aim is replaced by the direction to the nearest
standing enemy — the same exact aim the scripted opponents have, so the policy's job becomes
*when* to shoot / loot / run rather than hitting a 22.5-degree bin at range.

Skill mode is ``Discrete(len(SKILLS))`` over the System 1 skills the TS server executes —
``move_to``, ``follow``, ``loot``, ``heal``, ``engage``, ``retreat``, ``revive`` — producing
``{"skill": name, "params": {...}}``, which the bridge runs every tick until another action
replaces it (see `docs/survev-bridge-v0.md`). Params name agents by agent id and points by
``{x, y}``; the server resolves them and reports completion in ``info.skills``.
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
#: The week-2 subset the server implements. `hold` / `peek` / `rotate_zone` / `idle_look` arrive
#: with cover and the gas schedule; `take_cover` needs obstacles, which the open field has none of.
SKILLS: tuple[str, ...] = (
    "move_to",
    "follow",
    "loot",
    "heal",
    "engage",
    "retreat",
    "revive",
)
GUN_SLOTS = (0, 1)


@dataclass(frozen=True)
class ActionSpace:
    """MultiDiscrete (primitive) or Discrete (skill) action space -> bridge actions."""

    mode: ActionMode = "primitive"
    assist: bool = True
    auto_pickup: bool = False  # assist extra: press Interact whenever loot is within reach
    aim_assist: bool = False  # while fire is held, aim at the nearest visible standing enemy

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
            raise NotImplementedError("skill mode produces a skill request; use to_wire_action()")
        move_bin, aim_bin, fire, interact = self.validate(action)
        mx, my = MOVE_VECTORS[move_bin]
        aim = aim_bin_to_vec(aim_bin, AIM_BINS)
        aim_xy = (float(aim["x"]), float(aim["y"]))
        if self.aim_assist and fire and obs is not None:
            aim_xy = aim_at_enemy(obs) or aim_xy
        inputs: list[str | int] = []
        if interact:
            inputs.append("Interact")
        if self.assist and obs is not None:
            inputs.extend(assist_inputs(obs, auto_pickup=self.auto_pickup, already=inputs))
        return CpcAction(
            move=(float(mx), float(my)) if (mx or my) else None,
            aim=aim_xy,
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

    # -- either mode ----------------------------------------------------------------------
    def to_wire_action(self, action: Sequence[int] | int, obs: AgentObservation | None = None) -> dict[str, Any]:
        """What goes into a ``step`` message: raw inputs in primitive mode, a skill request in skill mode."""
        if self.mode == "skill":
            return self.to_skill_action(action, obs)
        return self.to_cpc_action(action, obs).to_json()


# --------------------------------------------------------------------------------------
# assist layer
# --------------------------------------------------------------------------------------
PICKUP_RANGE = 2.0


def aim_at_enemy(obs: AgentObservation) -> tuple[float, float] | None:
    """Unit vector to the nearest visible enemy (standing before downed); ``None`` when none is visible."""
    me = obs["self"]
    team = me.get("team")
    best: tuple[tuple[int, float], Any] | None = None
    for p in obs.get("players") or []:
        if p.get("dead") or (team is not None and p.get("team") == team):
            continue
        key = (1 if p.get("downed") else 0, float(p.get("dist", math.inf)))
        if best is None or key < best[0]:
            best = (key, p)
    if best is None:
        return None
    my_pos, pos = me.get("pos") or {}, best[1].get("pos") or {}
    dx = float(pos.get("x", 0.0)) - float(my_pos.get("x", 0.0))
    dy = float(pos.get("y", 0.0)) - float(my_pos.get("y", 0.0))
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return (dx / norm, dy / norm)


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
    if name == "move_to":
        # with no objective to run for, the fallback destination is the nearest loot, then the enemy
        loot = _nearest(obs.get("loot") or [])
        goal = (loot or enemy or {}).get("pos")
        return {"pos": dict(goal)} if goal else {"pos": dict(me["pos"])}
    if name == "follow":
        return {"target": partner["id"], "distance": 6.0} if partner else {}
    if name == "loot":
        # the skill picks what it needs next (gun, then ammo) when no type is named
        return {}
    if name == "heal":
        return {}
    if name == "engage":
        return {"target": enemy["id"], "style": "hold_angle"} if enemy else {}
    if name == "retreat":
        if enemy:
            return {"away_from": enemy["id"], "distance": 30.0}
        return {"distance": 30.0}
    if name == "revive":
        downed = [t for t in teammates if t.get("downed")]
        target = _nearest(downed)
        return {"target": target["id"]} if target else {}
    raise ValueError(f"unknown skill {name!r}")
