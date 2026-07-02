from __future__ import annotations

import json
import math
import uuid
from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from experiment.baselines.hierarchical_baseline import (
        BaselineConfig,
        ExecutionDirective,
        HierarchicalBaselineAgent,
    )
    from experiment.core.cpc_actions import decode_action
    from experiment.core.cpc_env import CPCEnv
    from experiment.core.env_config import EnvConfig
except ModuleNotFoundError:
    from baselines.hierarchical_baseline import BaselineConfig, ExecutionDirective, HierarchicalBaselineAgent
    from core.cpc_actions import decode_action
    from core.cpc_env import CPCEnv
    from core.env_config import EnvConfig

from .cpc_intent import (
    AppliedAction,
    CombatAction,
    CpcIntentArbiter,
    CpcIntentInputs,
    CpcTargetResolver,
    DecisionTrace,
    Layer1Output,
    Layer2Output,
    derive_decision_trace,
    format_decision_trace,
)


POLICY_IDS = ("baseline_selfish", "cpc_support", "current_best")
SCHEMA_VERSION = "cpc-evaluation-v0"
ISOLATION_DISTANCE = 220.0
DEFAULT_SELFISH_LEVELS = {
    "baseline_selfish": 1.0,
    "cpc_support": 0.0,
    "current_best": 1.0,
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"cpc-{timestamp}-{uuid.uuid4().hex[:10]}"


@dataclass(frozen=True)
class QuestionnaireAnswers:
    helpful: int
    predictable: int
    interfered: int
    trusted: int
    play_again: int
    feedback: str = ""

    def __post_init__(self) -> None:
        for field_name in ("helpful", "predictable", "interfered", "trusted", "play_again"):
            value = int(getattr(self, field_name))
            if not 1 <= value <= 5:
                raise ValueError(f"{field_name} must be between 1 and 5")


@dataclass(frozen=True)
class SessionMetadata:
    schema_version: str
    session_id: str
    episode_id: str
    policy_id: str
    scenario_id: str
    seed: int
    selfish_level: float
    started_at: str


@dataclass(frozen=True)
class BotDecision:
    layer1: Layer1Output
    layer2: Layer2Output
    combat_action: CombatAction
    decision_trace: DecisionTrace
    reason: str
    debug: dict[str, Any]

    @property
    def requested_action(self) -> dict[str, int | float]:
        return {
            "move": self.combat_action.move_bin,
            "move_bin": self.combat_action.move_bin,
            "aim_dx": self.combat_action.aim[0],
            "aim_dy": self.combat_action.aim[1],
            "fire": self.combat_action.fire_requested,
        }


class CpcBotPolicy:
    def __init__(self, policy_id: str, selfish_level: float | None = None):
        if policy_id not in POLICY_IDS:
            raise ValueError(f"unknown policy_id {policy_id!r}; expected one of {POLICY_IDS}")
        resolved_selfish = DEFAULT_SELFISH_LEVELS[policy_id] if selfish_level is None else float(selfish_level)
        if not 0.0 <= resolved_selfish <= 1.0:
            raise ValueError("selfish_level must be between 0.0 and 1.0")
        self.policy_id = policy_id
        self.selfish_level = resolved_selfish
        self.agent = HierarchicalBaselineAgent(BaselineConfig(combat_movement_profile="poke_out"))
        self.arbiter = CpcIntentArbiter()
        self.target_resolver = CpcTargetResolver()
        self._last_human_hp: float | None = None
        self._human_damage_history: deque[float] = deque(maxlen=5)

    def decide(self, env: CPCEnv) -> BotDecision:
        inputs = self._intent_inputs(env)
        enemy_alive = float(env.state["enemy_hp"]) > 0.0
        enemy_position = env.state["enemy_pos"] if enemy_alive else None
        layer1 = self.arbiter.select(inputs)
        layer2 = self.target_resolver.resolve(
            layer1,
            bot_position=env.state["ally_pos"],
            human_position=env.state["self_pos"],
            enemy_position=enemy_position,
            goal_position=env.goal_position,
            enemy_id=str(env.enemy_id) if enemy_alive else None,
            weapon_range=float(env.fire_range),
            map_width=float(env.width),
            map_height=float(env.height),
        )
        bot_obs, bot_snapshot = _ally_view(env)
        requested, debug = self.agent.act(
            bot_obs,
            bot_snapshot,
            ExecutionDirective(
                target_ref=layer2.target_ref.as_dict(),
                anchor_position=layer2.anchor_position,
            ),
        )
        combat_action = CombatAction(
            move_bin=int(requested.get("move", requested.get("move_bin", 0))),
            aim=(float(requested.get("aim_dx", 1.0)), float(requested.get("aim_dy", 0.0))),
            fire_requested=int(requested.get("fire", 0)),
        )
        decision_trace = derive_decision_trace(layer1, layer2, combat_action)
        reason = _bot_reason(debug)
        return BotDecision(layer1, layer2, combat_action, decision_trace, reason, dict(debug))

    def _intent_inputs(self, env: CPCEnv) -> CpcIntentInputs:
        human_hp = float(env.state["self_hp"])
        damage = (
            max(0.0, self._last_human_hp - human_hp)
            if self._last_human_hp is not None
            else 0.0
        )
        self._last_human_hp = human_hp
        self._human_damage_history.append(damage)
        teammate_distance = _distance(env.state["self_pos"], env.state["ally_pos"])
        enemy_alive = float(env.state["enemy_hp"]) > 0.0
        human_enemy_distance = _distance(env.state["self_pos"], env.state["enemy_pos"])
        bot_enemy_distance = (
            _distance(env.state["ally_pos"], env.state["enemy_pos"])
            if enemy_alive
            else None
        )
        bot_goal_distance = (
            math.dist(
                (float(env.state["ally_pos"]["x"]), float(env.state["ally_pos"]["y"])),
                env.goal_position,
            )
            if env.goal_position is not None
            else None
        )
        return CpcIntentInputs(
            selfish_level=self.selfish_level,
            teammate_distance=teammate_distance,
            human_recent_damage=sum(self._human_damage_history),
            human_hp=human_hp,
            human_isolated=teammate_distance > ISOLATION_DISTANCE,
            enemy_threatening_human=bool(
                enemy_alive and human_enemy_distance <= float(env.fire_range)
            ),
            bot_goal_distance=bot_goal_distance,
            bot_enemy_distance=bot_enemy_distance,
        )


class EvaluationRecorder:
    def __init__(self, metadata: SessionMetadata, output_root: str | Path):
        self.metadata = metadata
        self.session_dir = Path(output_root) / metadata.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.trajectory_path = self.session_dir / "trajectory.jsonl"
        self.questionnaire_path = self.session_dir / "questionnaire.json"
        self.summary_path = self.session_dir / "summary.json"
        self._trajectory_file = self.trajectory_path.open("w", encoding="utf-8")
        self.steps = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.bot_damage_taken = 0.0
        self.bot_damage_dealt = 0.0
        self.teammate_distance_sum = 0.0
        self.isolation_steps = 0
        self.support_opportunities = 0
        self.support_responses = 0
        self.event_counts: Counter[str] = Counter()
        self.intent_counts: Counter[str] = Counter()

    def record_step(
        self,
        *,
        timestamp: str,
        snapshot_after: Mapping[str, Any],
        human_action: Mapping[str, Any],
        bot_decision: BotDecision,
        applied_action: AppliedAction,
        reward: float,
        done: bool,
        info: Mapping[str, Any],
        teammate_distance_before: float,
    ) -> None:
        player = _mapping(snapshot_after.get("player"))
        enemy = _first_mapping(snapshot_after.get("enemies"))
        bot = _mapping(_mapping(info.get("evaluation")).get("bot"))
        teammate_distance = float(bot.get("teammate_distance", 0.0))
        isolated = teammate_distance > ISOLATION_DISTANCE
        support_delta = teammate_distance_before - teammate_distance
        support_response = bool(teammate_distance_before > ISOLATION_DISTANCE and support_delta > 1e-6)
        damage_delta = _mapping(info.get("damage_delta"))
        damage_dealt = float(damage_delta.get("enemy_hp", 0.0))
        damage_taken = float(damage_delta.get("self_hp", 0.0))
        bot_damage_taken = float(bot.get("damage_taken", 0.0))
        layers = _mapping(bot.get("layers"))
        layer1 = _mapping(layers.get("layer1"))
        events = [dict(event) for event in info.get("events", []) if isinstance(event, Mapping)]
        bot_damage_dealt = sum(
            float(event.get("damage", 0.0))
            for event in events
            if event.get("type") == "bullet_hit"
            and event.get("owner_id") == "ally"
            and event.get("target_id") == "enemy"
        )

        row = {
            **asdict(self.metadata),
            "timestamp": timestamp,
            "step": int(snapshot_after.get("step", self.steps + 1)),
            "player_hp": float(player.get("hp", 0.0)),
            "bot_hp": float(bot.get("hp", 0.0)),
            "enemy_hp": float(enemy.get("hp", 0.0)),
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "bot_damage_taken": bot_damage_taken,
            "bot_damage_dealt": bot_damage_dealt,
            "human_action": dict(human_action),
            "bot_action": bot_decision.combat_action.as_dict(),
            "bot_applied_action": applied_action.as_dict(),
            "bot_action_reason": bot_decision.reason,
            "bot_micro_intent": bot_decision.debug.get("micro_intent"),
            "poke_state": bot_decision.debug.get("poke_state"),
            "layers": dict(layers),
            "decision_trace": bot_decision.decision_trace.as_dict(),
            "decision_trace_line": format_decision_trace(bot_decision.decision_trace),
            "teammate_distance": teammate_distance,
            "isolated": isolated,
            "support_opportunity": teammate_distance_before > ISOLATION_DISTANCE,
            "support_response": support_response,
            "support_distance_delta": support_delta,
            "reward": float(reward),
            "done": bool(done),
            "events": events,
        }
        self._trajectory_file.write(json.dumps(row, sort_keys=True) + "\n")
        self._trajectory_file.flush()

        self.steps += 1
        self.damage_dealt += damage_dealt
        self.damage_taken += damage_taken
        self.bot_damage_taken += bot_damage_taken
        self.bot_damage_dealt += bot_damage_dealt
        self.teammate_distance_sum += teammate_distance
        self.isolation_steps += int(isolated)
        self.support_opportunities += int(teammate_distance_before > ISOLATION_DISTANCE)
        self.support_responses += int(support_response)
        self.event_counts.update(str(event.get("type", "unknown")) for event in events)
        self.intent_counts.update([str(layer1.get("cpc_intent", "unknown"))])

    def finish(
        self,
        *,
        questionnaire: QuestionnaireAnswers,
        final_snapshot: Mapping[str, Any],
        end_reason: str,
    ) -> dict[str, Any]:
        self._trajectory_file.close()
        ended_at = utc_timestamp()
        questionnaire_record = {
            **asdict(self.metadata),
            "submitted_at": ended_at,
            "answers": asdict(questionnaire),
            "questions": {
                "The bot was helpful.": questionnaire.helpful,
                "The bot was predictable.": questionnaire.predictable,
                "The bot interfered with me.": questionnaire.interfered,
                "I trusted the bot.": questionnaire.trusted,
                "I would play with this bot again.": questionnaire.play_again,
                "Free text feedback.": questionnaire.feedback,
            },
        }
        _write_json(self.questionnaire_path, questionnaire_record)

        player = _mapping(final_snapshot.get("player"))
        enemy = _first_mapping(final_snapshot.get("enemies"))
        bot_hp = float(_mapping(final_snapshot.get("evaluation_bot")).get("hp", 0.0))
        steps = max(1, self.steps)
        summary = {
            **asdict(self.metadata),
            "ended_at": ended_at,
            "end_reason": end_reason,
            "steps": self.steps,
            "final_player_hp": float(player.get("hp", 0.0)),
            "final_bot_hp": bot_hp,
            "final_enemy_hp": float(enemy.get("hp", 0.0)),
            "cpc_metrics": {
                "damage_dealt": self.damage_dealt,
                "damage_taken": self.damage_taken,
                "bot_damage_taken": self.bot_damage_taken,
                "bot_damage_dealt": self.bot_damage_dealt,
                "avg_teammate_distance": self.teammate_distance_sum / steps,
                "isolation_steps": self.isolation_steps,
                "isolation_rate": self.isolation_steps / steps,
                "support_opportunities": self.support_opportunities,
                "support_responses": self.support_responses,
                "support_response_rate": self.support_responses / max(1, self.support_opportunities),
                "intent_counts": dict(sorted(self.intent_counts.items())),
                "event_counts": dict(sorted(self.event_counts.items())),
            },
            "outputs": {
                "trajectory": str(self.trajectory_path),
                "questionnaire": str(self.questionnaire_path),
                "summary": str(self.summary_path),
            },
        }
        _write_json(self.summary_path, summary)
        return summary

    def close(self) -> None:
        if not self._trajectory_file.closed:
            self._trajectory_file.close()


class EvaluationEpisode:
    def __init__(
        self,
        config: EnvConfig,
        *,
        policy_id: str,
        scenario_id: str,
        seed: int,
        output_root: str | Path,
        session_id: str | None = None,
        selfish_level: float | None = None,
    ):
        resolved_session_id = session_id or create_session_id()
        self.policy = CpcBotPolicy(policy_id, selfish_level)
        self.metadata = SessionMetadata(
            schema_version=SCHEMA_VERSION,
            session_id=resolved_session_id,
            episode_id=resolved_session_id,
            policy_id=policy_id,
            scenario_id=scenario_id,
            seed=int(seed),
            selfish_level=self.policy.selfish_level,
            started_at=utc_timestamp(),
        )
        self.env = CPCEnv.from_config(config)
        self.obs = self.env.reset(seed=seed)
        self.recorder = EvaluationRecorder(self.metadata, output_root)
        self.done = False

    def step(self, human_action: Mapping[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode is already complete")
        teammate_distance_before = _distance(self.env.state["self_pos"], self.env.state["ally_pos"])
        bot_hp_before = float(self.env.state["ally_hp"])
        bot_position_before = deepcopy(self.env.state["ally_pos"])
        bot_decision = self.policy.decide(self.env)
        move_bin_applied, movement_blocked_reason = _apply_bot_movement(
            self.env,
            bot_decision.requested_action,
        )
        bot_position_after = deepcopy(self.env.state["ally_pos"])
        self.obs, reward, self.done, info = self.env.step(
            human_action,
            ally_action=bot_decision.requested_action,
        )
        info = dict(info)
        ally_fire = _mapping(info.get("ally_fire"))
        blocked_reasons = [reason for reason in (movement_blocked_reason,) if reason]
        if bot_decision.combat_action.fire_requested and not ally_fire.get("shot_fired"):
            blocked_reasons.append(str(ally_fire.get("fire_blocked_reason") or "fire_blocked"))
        applied_action = AppliedAction(
            move_bin_applied=move_bin_applied,
            fire_applied=int(bool(ally_fire.get("shot_fired", False))),
            blocked_reason=";".join(blocked_reasons) or None,
        )
        bot_hp_after = float(self.env.state["ally_hp"])
        teammate_distance = _distance(self.env.state["self_pos"], self.env.state["ally_pos"])
        info["evaluation"] = {
            "bot": {
                "hp": bot_hp_after,
                "damage_taken": max(0.0, bot_hp_before - bot_hp_after),
                "teammate_distance": teammate_distance,
                "position_before": bot_position_before,
                "position_after": bot_position_after,
                "requested_action": dict(bot_decision.requested_action),
                "applied_action": applied_action.as_dict(),
                "action_reason": bot_decision.reason,
                "fire_effect_enabled": True,
                "layers": {
                    "layer1": {"cpc_intent": bot_decision.layer1.cpc_intent},
                    "layer2": {
                        "target_ref": bot_decision.layer2.target_ref.as_dict(),
                        "anchor_position": list(bot_decision.layer2.anchor_position),
                    },
                    "layer3": {"combat_action": bot_decision.combat_action.as_dict()},
                    "layer4": {"applied_action": applied_action.as_dict()},
                },
                "decision_trace": bot_decision.decision_trace.as_dict(),
                "decision_trace_line": format_decision_trace(bot_decision.decision_trace),
                "debug": {
                    "intent": bot_decision.debug.get("intent"),
                    "combat_profile": bot_decision.debug.get("combat_profile"),
                    "micro_intent": bot_decision.debug.get("micro_intent"),
                    "poke_state": bot_decision.debug.get("poke_state"),
                    "poke_state_age": bot_decision.debug.get("poke_state_age"),
                    "movement_policy_reason": bot_decision.debug.get("movement_policy_reason"),
                    "fire_reason": bot_decision.debug.get("fire_reason"),
                    "fire_ready": bot_decision.debug.get("fire_ready"),
                    "target_in_range": bot_decision.debug.get("target_in_range"),
                    "can_fire_now": bot_decision.debug.get("can_fire_now"),
                    "dist_ratio": bot_decision.debug.get("dist_ratio"),
                },
            }
        }
        snapshot_after = self.env.get_snapshot()
        self.recorder.record_step(
            timestamp=utc_timestamp(),
            snapshot_after=snapshot_after,
            human_action=human_action,
            bot_decision=bot_decision,
            applied_action=applied_action,
            reward=reward,
            done=self.done,
            info=info,
            teammate_distance_before=teammate_distance_before,
        )
        return self.obs, reward, self.done, info

    def finish(self, questionnaire: QuestionnaireAnswers, *, end_reason: str) -> dict[str, Any]:
        snapshot = self.env.get_snapshot()
        snapshot["evaluation_bot"] = {
            "hp": float(self.env.state["ally_hp"]),
            "position": deepcopy(self.env.state["ally_pos"]),
        }
        return self.recorder.finish(
            questionnaire=questionnaire,
            final_snapshot=snapshot,
            end_reason=end_reason,
        )

    def close(self) -> None:
        self.recorder.close()


def prompt_questionnaire() -> QuestionnaireAnswers:
    print("\nCPC episode questionnaire (1=strongly disagree, 5=strongly agree)")
    return QuestionnaireAnswers(
        helpful=_prompt_rating("The bot was helpful."),
        predictable=_prompt_rating("The bot was predictable."),
        interfered=_prompt_rating("The bot interfered with me."),
        trusted=_prompt_rating("I trusted the bot."),
        play_again=_prompt_rating("I would play with this bot again."),
        feedback=input("Free text feedback: ").strip(),
    )


def _prompt_rating(label: str) -> int:
    while True:
        raw = input(f"{label} [1-5]: ").strip()
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if 1 <= value <= 5:
            return value
        print("Please enter an integer from 1 to 5.")


def _ally_view(env: CPCEnv) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = dict(env.observation())
    ally_pos = deepcopy(env.state["ally_pos"])
    ally_hp = float(env.state["ally_hp"])
    obs.update(
        {
            "self_pos": ally_pos,
            "self_hp": ally_hp,
            "distance_to_enemy": _distance(ally_pos, env.state["enemy_pos"]),
            "distance_to_ally": _distance(ally_pos, env.state["self_pos"]),
            "can_fire": int(env.ally_weapon.get("cooldown_remaining_steps", 0)) <= 0,
            "cooldown_ready": int(env.ally_weapon.get("cooldown_remaining_steps", 0)) <= 0,
        }
    )
    snapshot = env.get_snapshot()
    snapshot["player"] = {
        "position": [float(ally_pos["x"]), float(ally_pos["y"])],
        "hp": ally_hp,
        "alive": ally_hp > 0.0,
    }
    return obs, snapshot


def _apply_bot_movement(env: CPCEnv, action: Mapping[str, Any]) -> tuple[int, str | None]:
    if float(env.state.get("ally_hp", 0.0)) <= 0.0:
        return 0, "bot_dead"
    decoded = decode_action(action)
    requested_move = int(action.get("move", action.get("move_bin", 0)))
    start = dict(env.state["ally_pos"])
    target = {
        "x": env._clamp(start["x"] + float(decoded["moveX"]) * env.move_speed, 0.0, env.width),
        "y": env._clamp(start["y"] + float(decoded["moveY"]) * env.move_speed, 0.0, env.height),
    }
    resolved = env._resolve_obstacle_blocked_move(start, target, env.ally_radius)
    env.state["ally_pos"] = resolved
    moved = _distance(start, resolved) > 1e-6
    if requested_move != 0 and not moved:
        return 0, "movement_blocked"
    return requested_move, None


def _bot_reason(debug: Mapping[str, Any]) -> str:
    return str(
        debug.get("movement_policy_reason")
        or debug.get("kiting_policy_reason")
        or debug.get("global_plan_reason")
        or debug.get("micro_intent")
        or "hierarchical_policy"
    )


def _distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return value[0]
    return {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
