from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .cpc_actions import MOVE_VECTORS, normalize_move
from .schema import CPCAction, CPCObservation, PrimitiveAction


def build_cpc_observation(env_state: Any, actor_id: str) -> CPCObservation:
    """Snapshot raw state without coupling the contract types to an env implementation."""
    source = env_state
    state = getattr(env_state, "state", env_state)
    if not isinstance(state, Mapping):
        raise TypeError("env_state must be a mapping or expose a mapping 'state'")
    context = {"env": source} if source is not state else {}
    return CPCObservation(actor_id=str(actor_id), state=deepcopy(dict(state)), context=context)


def move_bin_for_vector(move: tuple[float, float]) -> int | None:
    normalized = normalize_move(*move)
    for move_bin, vector in MOVE_VECTORS.items():
        if normalize_move(*vector) == normalized:
            return move_bin
    return None


def apply_cpc_action(env: Any, actor_id: str, action: CPCAction) -> PrimitiveAction:
    """Translate a contract action to the existing primitive action representation."""
    move_bin = move_bin_for_vector(action.move)
    primitive = {
        "move": move_bin if move_bin is not None else action.move,
        "move_bin": move_bin,
        "aim_dx": action.aim[0],
        "aim_dy": action.aim[1],
        "fire": int(action.fire),
    }
    apply_action = getattr(env, "apply_cpc_action", None)
    result = apply_action(actor_id, primitive) if callable(apply_action) else None
    fire_applied = bool(result.get("fire_applied", action.fire)) if isinstance(result, Mapping) else bool(action.fire)
    return PrimitiveAction(move_bin, action.move, action.aim, bool(action.fire), fire_applied)
