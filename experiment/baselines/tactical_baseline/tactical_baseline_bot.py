from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from experiment.baselines.aim_oracle import TacticalAimOracleBot
    from experiment.core.cpc_actions import MOVE_BINS, decode_action
    from experiment.core.local_occupancy_grid import CHANNEL_ENEMY, CHANNEL_OBSTACLE, build_local_occupancy_grid
except ModuleNotFoundError:
    from baselines.aim_oracle import TacticalAimOracleBot
    from core.cpc_actions import MOVE_BINS, decode_action
    from core.local_occupancy_grid import CHANNEL_ENEMY, CHANNEL_OBSTACLE, build_local_occupancy_grid

from .fire_rule import FireRule
from .mode_conditioned_bfs_planner import ModeConditionedBFSPlanner
from .tactical_mode_selector import RuleBasedTacticalModeSelector


class TacticalBaselineBot:
    """Compose mode selection, BFS movement, aim oracle, and fire rule."""

    def __init__(
        self,
        aim_oracle: Any,
        move_scorer: Any,
        fire_rule: FireRule,
        *,
        tactical_mode_selector: Any | None = None,
        default_move_bin: int = 0,
    ) -> None:
        self.aim_oracle = aim_oracle
        self.move_scorer = move_scorer
        self.move_planner = move_scorer
        self.fire_rule = fire_rule
        self.tactical_mode_selector = tactical_mode_selector or RuleBasedTacticalModeSelector()
        self.default_move_bin = int(default_move_bin)

    def act(self, obs: Any, state_snapshot: Any | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        snapshot = _normalize_snapshot(state_snapshot)
        observation = _mapping_copy(obs)
        obs_with_grid = _with_local_grid(observation, snapshot)

        tactical_mode, mode_debug = self._choose_mode(obs_with_grid, snapshot)
        aim_action, aim_debug = self._choose_aim(obs_with_grid)
        move_obs = {
            **obs_with_grid,
            **aim_action,
        }
        move_bin, move_debug = self._choose_move(move_obs, tactical_mode, snapshot)
        fire_obs = {
            **obs_with_grid,
            **aim_action,
            "move_bin": int(move_bin),
        }
        fire, fire_debug = self._choose_fire(fire_obs, snapshot)
        action = _action(move_bin=move_bin, aim_action=aim_action, fire=fire)

        debug = {
            "action": dict(action),
            "mode": mode_debug,
            "aim": aim_debug,
            "move": move_debug,
            "fire": fire_debug,
            "reason": _combined_reason(aim_debug, move_debug, fire_debug),
        }
        return action, debug

    def _choose_mode(self, obs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        try:
            mode, debug = self.tactical_mode_selector.select_mode(obs, state_snapshot=snapshot)
            return str(mode), dict(debug)
        except Exception as exc:  # Defensive baseline fallback: seek a safe local position.
            return "reposition", {
                "mode": "reposition",
                "reason": "mode_selector_failed_using_reposition",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _choose_aim(self, obs: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        try:
            aim_action, debug = self.aim_oracle.act(obs)
            return {
                "aim_dx": float(aim_action.get("aim_dx", 0.0)),
                "aim_dy": float(aim_action.get("aim_dy", 0.0)),
            }, dict(debug)
        except Exception as exc:  # Defensive baseline fallback: keep autoplay running.
            return {"aim_dx": 1.0, "aim_dy": 0.0}, {
                "reason": "aim_oracle_failed_using_default",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _choose_move(
        self,
        obs: Mapping[str, Any],
        tactical_mode: str,
        snapshot: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        try:
            if hasattr(self.move_planner, "choose_move"):
                move_bin, debug = self.move_planner.choose_move(
                    obs,
                    tactical_mode=tactical_mode,
                    state_snapshot=snapshot,
                )
                return _valid_move_bin(move_bin), dict(debug)
            action, debug = self.move_planner.act(obs, state_snapshot=snapshot)
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
    obstacle_channel = grid.channel_index(CHANNEL_OBSTACLE)
    weapon_range = float(snapshot.get("combat", {}).get("fire_range", 280.0))
    aim_oracle = TacticalAimOracleBot(
        enemy_channel_index=enemy_channel,
        cell_size=grid.cell_size,
        stay_move_bin=0,
    )
    return TacticalBaselineBot(
        aim_oracle=aim_oracle,
        move_scorer=ModeConditionedBFSPlanner(
            obstacle_channel_index=obstacle_channel,
            enemy_channel_index=enemy_channel,
            cell_size=grid.cell_size,
            weapon_range=weapon_range,
        ),
        fire_rule=FireRule(),
        tactical_mode_selector=RuleBasedTacticalModeSelector(weapon_range=weapon_range),
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


def _action(*, move_bin: int, aim_action: Mapping[str, Any], fire: int) -> dict[str, int | float]:
    move = _valid_move_bin(move_bin)
    selected_fire = _valid_fire(fire)
    action = {
        "move": move,
        "aim_dx": float(aim_action.get("aim_dx", 1.0)),
        "aim_dy": float(aim_action.get("aim_dy", 0.0)),
        "fire": selected_fire,
        "move_bin": move,
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
