from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from experiment.baselines.aim_oracle import TacticalAimOracleBot
    from experiment.baselines.move_score import TacticalMoveScorer
    from experiment.core.local_occupancy_grid import CHANNEL_ENEMY, build_local_occupancy_grid
    from experiment.training.cpc_actions import AIM_BINS, MOVE_BINS, decode_action
except ModuleNotFoundError:
    from baselines.aim_oracle import TacticalAimOracleBot
    from baselines.move_score import TacticalMoveScorer
    from core.local_occupancy_grid import CHANNEL_ENEMY, build_local_occupancy_grid
    from training.cpc_actions import AIM_BINS, MOVE_BINS, decode_action

from .fire_rule import FireRule


class TacticalBaselineBot:
    """Compose aim oracle, move scorer, and fire rule into one env action."""

    def __init__(
        self,
        aim_oracle: Any,
        move_scorer: Any,
        fire_rule: FireRule,
        *,
        default_move_bin: int = 0,
        default_aim_bin: int = 0,
    ) -> None:
        self.aim_oracle = aim_oracle
        self.move_scorer = move_scorer
        self.fire_rule = fire_rule
        self.default_move_bin = int(default_move_bin)
        self.default_aim_bin = int(default_aim_bin)

    def act(self, obs: Any, state_snapshot: Any | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        snapshot = _normalize_snapshot(state_snapshot)
        observation = _mapping_copy(obs)
        obs_with_grid = _with_local_grid(observation, snapshot)

        aim_bin, aim_debug = self._choose_aim(obs_with_grid)
        move_bin, move_debug = self._choose_move(observation, snapshot)
        fire_obs = {
            **observation,
            "selected_aim_bin": int(aim_bin),
            "aim_bin": int(aim_bin),
            "move_bin": int(move_bin),
        }
        fire, fire_debug = self._choose_fire(fire_obs, snapshot)
        action = _action(move_bin=move_bin, aim_bin=aim_bin, fire=fire)

        debug = {
            "action": dict(action),
            "aim": aim_debug,
            "move": move_debug,
            "fire": fire_debug,
            "reason": _combined_reason(aim_debug, move_debug, fire_debug),
        }
        return action, debug

    def _choose_aim(self, obs: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            aim_action, debug = self.aim_oracle.act(obs)
            aim_bin = int(aim_action.get("aim", aim_action.get("aim_bin", self.default_aim_bin))) % AIM_BINS
            return aim_bin, dict(debug)
        except Exception as exc:  # Defensive baseline fallback: keep autoplay running.
            aim_bin = int(self.default_aim_bin) % AIM_BINS
            return aim_bin, {
                "aim_bin": aim_bin,
                "reason": "aim_oracle_failed_using_default",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _choose_move(self, obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            if hasattr(self.move_scorer, "choose_move"):
                move_bin, debug = self.move_scorer.choose_move(obs, state_snapshot=snapshot)
                return _valid_move_bin(move_bin), dict(debug)
            action, debug = self.move_scorer.act(obs, state_snapshot=snapshot)
            return _valid_move_bin(action.get("move", action.get("move_bin", self.default_move_bin))), dict(debug)
        except Exception as exc:  # Defensive baseline fallback: stay still.
            move_bin = _valid_move_bin(self.default_move_bin)
            return move_bin, {
                "selected_move_bin": move_bin,
                "reason": "move_scorer_failed_using_stay",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _choose_fire(self, obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            fire, debug = self.fire_rule.decide_fire(obs, state_snapshot=snapshot)
            return _valid_fire(fire), dict(debug)
        except Exception as exc:  # Defensive baseline fallback: do not fire.
            return 0, {
                "fire": 0,
                "reason": "fire_rule_failed_using_no_fire",
                "error": f"{type(exc).__name__}: {exc}",
            }


def build_tactical_baseline_bot(state_snapshot: Any | None = None) -> TacticalBaselineBot:
    snapshot = _normalize_snapshot(state_snapshot)
    grid = build_local_occupancy_grid(snapshot, agent_id="self")
    enemy_channel = grid.channel_index(CHANNEL_ENEMY)
    aim_oracle = TacticalAimOracleBot(
        num_aim_bins=AIM_BINS,
        enemy_channel_index=enemy_channel,
        cell_size=grid.cell_size,
        stay_move_bin=0,
        default_aim_bin=0,
    )
    return TacticalBaselineBot(
        aim_oracle=aim_oracle,
        move_scorer=TacticalMoveScorer(),
        fire_rule=FireRule(num_aim_bins=AIM_BINS),
    )


def _normalize_snapshot(state_snapshot: Any | None) -> dict[str, Any]:
    if state_snapshot is None:
        return {}
    if hasattr(state_snapshot, "get_debug_state"):
        return dict(state_snapshot.get_debug_state())
    if isinstance(state_snapshot, Mapping):
        return dict(state_snapshot)
    return {}


def _mapping_copy(obs: Any) -> dict[str, Any]:
    if isinstance(obs, Mapping):
        return dict(obs)
    return {}


def _with_local_grid(obs: dict[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if any(key in obs for key in ("local_occupancy_grid", "local_grid", "occupancy_grid", "grid")):
        return obs
    if not snapshot:
        return obs
    try:
        return {**obs, "local_occupancy_grid": build_local_occupancy_grid(snapshot, agent_id="self")}
    except Exception:
        return obs


def _action(*, move_bin: int, aim_bin: int, fire: int) -> dict[str, int]:
    move = _valid_move_bin(move_bin)
    aim = int(aim_bin) % AIM_BINS
    selected_fire = _valid_fire(fire)
    action = {
        "move": move,
        "aim": aim,
        "fire": selected_fire,
        "move_bin": move,
        "aim_bin": aim,
    }
    decode_action(action)
    return action


def _valid_move_bin(value: Any) -> int:
    move = int(value)
    if not 0 <= move < MOVE_BINS:
        return 0
    return move


def _valid_fire(value: Any) -> int:
    return 1 if int(value) == 1 else 0


def _combined_reason(
    aim_debug: Mapping[str, Any],
    move_debug: Mapping[str, Any],
    fire_debug: Mapping[str, Any],
) -> str:
    return (
        f"aim={aim_debug.get('reason', 'ok')}; "
        f"move={move_debug.get('reason', 'ok')}; "
        f"fire={fire_debug.get('reason', 'ok')}"
    )


__all__ = ["TacticalBaselineBot", "build_tactical_baseline_bot"]
