from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


SAVE_RESULT_MODES = ("compact", "full", "summary", "jsonl")
POLICY_DEBUG_MODES = ("none", "topk", "full")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    torch_jsonable = _torch_to_jsonable(value)
    if torch_jsonable is not _UNHANDLED:
        return torch_jsonable

    numpy_jsonable = _numpy_to_jsonable(value)
    if numpy_jsonable is not _UNHANDLED:
        return numpy_jsonable

    return str(value)


def serialize_gameplay_result(
    result: dict[str, Any],
    *,
    mode: str = "compact",
    policy_debug_mode: str = "none",
    include_policy_debug: bool = False,
    include_state_before: bool = False,
    include_observations: bool = False,
    include_full_state: bool = False,
) -> dict[str, Any]:
    _validate_mode(mode)
    if mode == "full":
        return serialize_step_full(result)
    if mode == "summary":
        return serialize_result_summary(result)
    if mode == "jsonl":
        return serialize_result_compact(
            result,
            policy_debug_mode=policy_debug_mode,
            include_policy_debug=include_policy_debug,
            include_state_before=include_state_before,
            include_observations=include_observations,
            include_full_state=include_full_state,
        )
    return serialize_result_compact(
        result,
        policy_debug_mode=policy_debug_mode,
        include_policy_debug=include_policy_debug,
        include_state_before=include_state_before,
        include_observations=include_observations,
        include_full_state=include_full_state,
    )


def save_gameplay_result(
    result: dict[str, Any],
    path: str | Path,
    *,
    mode: str = "compact",
    policy_debug_mode: str = "none",
    include_policy_debug: bool = False,
    include_state_before: bool = False,
    include_observations: bool = False,
    include_full_state: bool = False,
) -> Path:
    _validate_mode(mode)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "jsonl":
        lines = serialize_result_jsonl(
            result,
            policy_debug_mode=policy_debug_mode,
            include_policy_debug=include_policy_debug,
            include_state_before=include_state_before,
            include_observations=include_observations,
            include_full_state=include_full_state,
        )
        output_path.write_text("\n".join(json.dumps(line, sort_keys=True) for line in lines) + "\n", encoding="utf-8")
        return output_path

    output_path.write_text(
        json.dumps(
            serialize_gameplay_result(
                result,
                mode=mode,
                policy_debug_mode=policy_debug_mode,
                include_policy_debug=include_policy_debug,
                include_state_before=include_state_before,
                include_observations=include_observations,
                include_full_state=include_full_state,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def serialize_step_full(result: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(result)


def serialize_result_compact(
    result: dict[str, Any],
    *,
    policy_debug_mode: str = "none",
    include_policy_debug: bool = False,
    include_state_before: bool = False,
    include_observations: bool = False,
    include_full_state: bool = False,
) -> dict[str, Any]:
    _validate_policy_mode(policy_debug_mode)
    episodes = result.get("episodes", [])
    first_state = _first_env_state(result)
    return to_jsonable(
        {
            "schema_version": result.get("schema_version", "cpc-common-v0"),
            "source": result.get("source", "run_model_gameplay"),
            "format_version": "model-gameplay-compact-v1",
            "checkpoints": result.get("checkpoints", {}),
            "config": result.get("config", {}),
            "stopped_early": result.get("stopped_early", False),
            "static_env": make_static_env_snapshot(first_state),
            "episodes": [
                serialize_episode_compact(
                    episode,
                    policy_debug_mode=policy_debug_mode,
                    include_policy_debug=include_policy_debug,
                    include_state_before=include_state_before,
                    include_observations=include_observations,
                    include_full_state=include_full_state,
                )
                for episode in episodes
            ],
        }
    )


def serialize_episode_compact(
    episode: dict[str, Any],
    *,
    policy_debug_mode: str = "none",
    include_policy_debug: bool = False,
    include_state_before: bool = False,
    include_observations: bool = False,
    include_full_state: bool = False,
) -> dict[str, Any]:
    steps = episode.get("steps", [])
    initial_state = _initial_state_from_episode(episode)
    compact_steps = [
        serialize_step_compact(
            step,
            policy_debug_mode=policy_debug_mode,
            include_policy_debug=include_policy_debug,
            include_state_before=include_state_before,
            include_observations=include_observations,
            include_full_state=include_full_state,
        )
        for step in steps
    ]
    return {
        "episode_index": episode.get("episode_index", 0),
        "initial_state": initial_state,
        "steps": compact_steps,
        "episode_return": episode.get("episode_return", {}),
        "episode_length": episode.get("episode_length", len(compact_steps)),
        "final_metrics": episode.get("final_metrics", {}),
        "stopped_early": episode.get("stopped_early", False),
    }


def serialize_step_compact(
    step_record: dict[str, Any],
    *,
    policy_debug_mode: str = "none",
    include_policy_debug: bool = False,
    include_state_before: bool = False,
    include_observations: bool = False,
    include_full_state: bool = False,
) -> dict[str, Any]:
    env = step_record.get("env", {})
    info = env.get("info", {})
    agent = (step_record.get("agents", {}) or {}).get("agent", {})
    env_state = env.get("state", {})
    compact = {
        "t": step_record.get("step", env_state.get("step", env_state.get("step_count", 0))),
        "obs": compact_observation(agent.get("observation", {})),
        "action": compact_action(agent.get("raw_action", {}), agent.get("decoded_action", {})),
        "fire": compact_fire_info(info),
        "aim": compact_aim_debug(info),
        "range": compact_range_debug(info),
        "zone": compact_zone_debug(info),
        "events": compact_events(info),
        "bullets": compact_bullets(info or env_state),
        "reward": float((env.get("rewards", {}) or {}).get("agent", 0.0)),
        "reward_components": compact_reward_components(info.get("reward_components", {})),
        "state_after": compact_state(env_state),
        "metrics_delta": compact_metrics_delta(info),
    }
    policy_debug = compact_policy_debug(agent.get("policy_debug", {}), policy_debug_mode)
    if include_policy_debug or policy_debug_mode != "none":
        compact["policy_debug"] = policy_debug
    if include_state_before:
        compact["state_before"] = compact_state(env.get("state_before_step", {}))
    if include_observations:
        compact["observations"] = {
            "before": agent.get("observation", {}),
            "after": env.get("observation_after_step", {}),
        }
    if include_full_state:
        compact["full_state"] = env_state
    return compact


def serialize_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    episodes = []
    for episode in result.get("episodes", []):
        metrics = episode.get("final_metrics", {})
        episodes.append(
            {
                "episode_index": episode.get("episode_index", 0),
                "initial_state": _initial_state_from_episode(episode),
                "episode_return": episode.get("episode_return", {}),
                "episode_length": episode.get("episode_length", 0),
                "final_metrics": metrics,
                "stopped_early": episode.get("stopped_early", False),
                "key_counts": {
                    key: metrics.get(key, 0.0)
                    for key in (
                        "bullet_hit_count",
                        "shot_fired_count",
                        "hit_ratio",
                        "missed_shot_rate",
                        "damage_dealt",
                        "damage_taken",
                        "damage_trade_ratio",
                    )
                },
            }
        )
    return to_jsonable(
        {
            "schema_version": result.get("schema_version", "cpc-common-v0"),
            "source": result.get("source", "run_model_gameplay"),
            "format_version": "model-gameplay-summary-v1",
            "checkpoints": result.get("checkpoints", {}),
            "config": result.get("config", {}),
            "stopped_early": result.get("stopped_early", False),
            "episodes": episodes,
        }
    )


def serialize_result_jsonl(
    result: dict[str, Any],
    *,
    policy_debug_mode: str = "none",
    include_policy_debug: bool = False,
    include_state_before: bool = False,
    include_observations: bool = False,
    include_full_state: bool = False,
) -> list[dict[str, Any]]:
    static_env = make_static_env_snapshot(_first_env_state(result))
    lines: list[dict[str, Any]] = []
    for episode in result.get("episodes", []):
        episode_index = int(episode.get("episode_index", 0))
        lines.append(
            to_jsonable(
                {
                    "type": "episode_start",
                    "episode_index": episode_index,
                    "config": result.get("config", {}),
                    "static_env": static_env,
                    "initial_state": _initial_state_from_episode(episode),
                }
            )
        )
        for step in episode.get("steps", []):
            compact_step = serialize_step_compact(
                step,
                policy_debug_mode=policy_debug_mode,
                include_policy_debug=include_policy_debug,
                include_state_before=include_state_before,
                include_observations=include_observations,
                include_full_state=include_full_state,
            )
            lines.append(to_jsonable({"type": "step", "episode_index": episode_index, "step": compact_step.get("t"), **compact_step}))
        lines.append(
            to_jsonable(
                {
                    "type": "episode_end",
                    "episode_index": episode_index,
                    "final_metrics": episode.get("final_metrics", {}),
                    "episode_return": episode.get("episode_return", {}),
                    "episode_length": episode.get("episode_length", 0),
                    "stopped_early": episode.get("stopped_early", False),
                }
            )
        )
    return lines


def make_static_env_snapshot(env_state: dict[str, Any]) -> dict[str, Any]:
    map_info = env_state.get("map", {})
    combat = env_state.get("combat", {})
    return {
        "map": {
            "width": map_info.get("width", 1000.0),
            "height": map_info.get("height", 1000.0),
            "center": map_info.get("center", {"x": 500.0, "y": 500.0}),
        },
        "combat": {
            "fire_range": combat.get("fire_range", 260.0),
            "projectile_speed": combat.get("projectile_speed", 140.0),
            "projectile_radius": combat.get("projectile_radius", 8.0),
            "fire_alignment": combat.get("fire_alignment", 0.65),
        },
    }


def compact_observation(obs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "self_hp",
        "enemy_hp",
        "ally_hp",
        "distance_to_enemy",
        "distance_to_ally",
        "aim_alignment",
        "can_fire",
        "weapon_cooldown_fraction",
        "outside_safe_zone",
    )
    return {key: obs[key] for key in keys if key in obs}


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    raw_state = state.get("state", state)
    safe_zone = state.get("safe_zone", {})
    distances = state.get("distances", {})
    weapon = state.get("weapon", {})
    return {
        "step": state.get("step", state.get("step_count")),
        "self": _compact_agent(raw_state, "self"),
        "ally": _compact_agent(raw_state, "ally"),
        "enemy": _compact_agent(raw_state, "enemy"),
        "dist": {
            "enemy": distances.get("self_to_enemy"),
            "ally": distances.get("self_to_ally"),
            "center": distances.get("self_to_center", safe_zone.get("distance")),
        },
        "safe_zone": {
            "radius": safe_zone.get("radius", state.get("map", {}).get("safe_radius")),
            "outside": safe_zone.get("outside"),
        },
        "weapon": {
            "cooldown_remaining_steps": weapon.get("cooldown_remaining_steps"),
            "fire_interval_steps": weapon.get("fire_interval_steps"),
        },
    }


def compact_action(raw_action: dict[str, Any], decoded_action: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": dict(raw_action),
        "decoded": {
            "move": {"x": decoded_action.get("move_x", decoded_action.get("moveX", 0.0)), "y": decoded_action.get("move_y", decoded_action.get("moveY", 0.0))},
            "aim": {"x": decoded_action.get("aim_x", decoded_action.get("aimX", 0.0)), "y": decoded_action.get("aim_y", decoded_action.get("aimY", 0.0))},
            "fire": decoded_action.get("fire", 0.0),
        },
    }


def compact_fire_info(info: dict[str, Any]) -> dict[str, Any]:
    fire = info.get("fire", {})
    return {
        "requested": fire.get("fire_requested", info.get("fire_selected", False)),
        "shot_fired": fire.get("shot_fired", info.get("shot_fired", False)),
        "blocked_reason": fire.get("fire_blocked_reason"),
        "cooldown_before": fire.get("cooldown_remaining_steps_before"),
        "cooldown_after": fire.get("cooldown_remaining_steps_after"),
    }


def compact_aim_debug(info_or_state: dict[str, Any]) -> dict[str, Any]:
    aim = info_or_state.get("aim_debug", {})
    return {
        "alignment": aim.get("aim_alignment", 0.0),
        "angle_error_deg": aim.get("angle_error_deg", 0.0),
        "aim_bin": aim.get("aim_bin"),
        "ideal_aim_bin": aim.get("ideal_aim_bin"),
        "aim_bin_error": aim.get("aim_bin_error"),
        "aligned": aim.get("is_aim_aligned", False),
        "exact": aim.get("is_exact_aim", False),
        "near": aim.get("is_near_aim", False),
    }


def compact_zone_debug(info_or_state: dict[str, Any]) -> dict[str, Any]:
    zone = info_or_state.get("zone_debug", {})
    return {
        "distance_to_center": zone.get("distance_to_center"),
        "safe_radius": zone.get("safe_radius"),
        "outside": zone.get("outside_safe_zone", False),
        "move_toward_center": zone.get("move_toward_center"),
    }


def compact_range_debug(info_or_state: dict[str, Any]) -> dict[str, Any]:
    range_debug = info_or_state.get("range_debug", {})
    return {
        "distance_to_enemy": range_debug.get("distance_to_enemy"),
        "in_good_range": range_debug.get("in_good_range", False),
        "too_close": range_debug.get("too_close", False),
        "too_far": range_debug.get("too_far", False),
    }


def compact_events(info: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for event in info.get("bullet_events", []):
        compact = {"type": event.get("type")}
        for key in ("bullet_id", "owner_id", "target_id", "damage", "reason", "pos", "from", "traveled_distance"):
            if key in event:
                compact[key] = event[key]
        events.append(compact)
    return events


def compact_bullets(state_or_info: dict[str, Any]) -> list[dict[str, Any]]:
    bullets = state_or_info.get("bullets", state_or_info.get("projectiles", []))
    compact = []
    for bullet in bullets:
        compact.append(
            {
                "id": bullet.get("bullet_id"),
                "owner": bullet.get("owner_id"),
                "pos": bullet.get("pos"),
                "dir": bullet.get("direction"),
                "traveled": bullet.get("traveled_distance"),
                "alive": bullet.get("alive", True),
            }
        )
    return compact


def compact_reward_components(reward_components: dict[str, Any]) -> dict[str, Any]:
    selected = (
        "damage_dealt_ratio",
        "damage_taken_ratio",
        "bullet_hit",
        "missed_shot",
        "aim_bin_exact",
        "aim_bin_wrong",
        "good_range",
        "too_close",
        "too_far",
        "kill",
        "death",
        "timeout_hp_lead",
        "accuracy_bonus",
    )
    compact = {key: reward_components[key] for key in selected if key in reward_components}
    compact.update({key: value for key, value in reward_components.items() if float(value or 0.0) != 0.0})
    return compact


def compact_policy_debug(policy_debug: dict[str, Any], mode: str) -> dict[str, Any] | None:
    _validate_policy_mode(mode)
    if mode == "none":
        return None
    if mode == "full":
        return dict(policy_debug)
    return {
        "value": policy_debug.get("value"),
        "move_topk": _topk(policy_debug.get("move_logits", [])),
        "aim_topk": _topk(policy_debug.get("aim_logits", [])),
        "fire_topk": _topk(policy_debug.get("fire_logits", [])),
    }


def compact_metrics_delta(info: dict[str, Any]) -> dict[str, Any]:
    damage = info.get("damage_delta", {})
    events = info.get("bullet_events", [])
    return {
        "damage_dealt_delta": damage.get("enemy_hp", 0.0),
        "damage_taken_delta": damage.get("self_hp", 0.0),
        "bullet_hit_delta": float(sum(1 for event in events if event.get("type") == "bullet_hit")),
    }


def save_report(result: dict[str, Any], path: str | Path, mode: str) -> dict[str, Any]:
    path = Path(path)
    episodes = result.get("episodes", [])
    metrics = episodes[-1].get("final_metrics", {}) if episodes else {}
    final_return = episodes[-1].get("episode_return", {}).get("agent", 0.0) if episodes else 0.0
    return {
        "path": str(path),
        "mode": mode,
        "episodes": len(episodes),
        "steps": sum(int(ep.get("episode_length", len(ep.get("steps", [])))) for ep in episodes),
        "size_kb": round(path.stat().st_size / 1024.0, 2) if path.exists() else 0.0,
        "final_return": final_return,
        "damage_dealt": metrics.get("damage_dealt", 0.0),
        "damage_taken": metrics.get("damage_taken", 0.0),
        "damage_trade_ratio": metrics.get("damage_trade_ratio", 0.0),
        "hit_ratio": metrics.get("hit_ratio", 0.0),
        "bullet_hit_count": metrics.get("bullet_hit_count", 0.0),
        "shot_fired_count": metrics.get("shot_fired_count", 0.0),
    }


def _initial_state_from_episode(episode: dict[str, Any]) -> dict[str, Any]:
    steps = episode.get("steps", [])
    if steps:
        return compact_state(steps[0].get("env", {}).get("state_before_step", {}))
    obs = episode.get("initial_observation", {})
    return {
        "step": obs.get("step_count", 0),
        "self": {"hp": obs.get("self_hp"), "pos": obs.get("self_pos")},
        "ally": {"hp": obs.get("ally_hp"), "pos": obs.get("ally_pos")},
        "enemy": {"hp": obs.get("enemy_hp"), "pos": obs.get("enemy_pos")},
        "safe_zone": {"radius": obs.get("safe_radius"), "outside": obs.get("outside_safe_zone")},
        "weapon": {"cooldown_remaining_steps": 0, "fire_interval_steps": None},
    }


def _first_env_state(result: dict[str, Any]) -> dict[str, Any]:
    for episode in result.get("episodes", []):
        steps = episode.get("steps", [])
        if steps:
            return steps[0].get("env", {}).get("state", {})
    return {}


def _compact_agent(raw_state: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "hp": raw_state.get(f"{name}_hp"),
        "pos": raw_state.get(f"{name}_pos"),
    }


def _topk(logits: list[Any], k: int = 3) -> list[dict[str, float]]:
    values = [(idx, float(value)) for idx, value in enumerate(logits)]
    values.sort(key=lambda item: item[1], reverse=True)
    return [{"idx": idx, "logit": value} for idx, value in values[:k]]


def _validate_mode(mode: str) -> None:
    if mode not in SAVE_RESULT_MODES:
        raise ValueError(f"save result mode must be one of {SAVE_RESULT_MODES}, got {mode!r}")


def _validate_policy_mode(mode: str) -> None:
    if mode not in POLICY_DEBUG_MODES:
        raise ValueError(f"policy debug mode must be one of {POLICY_DEBUG_MODES}, got {mode!r}")


class _Unhandled:
    pass


_UNHANDLED = _Unhandled()


def _torch_to_jsonable(value: Any) -> Any:
    try:
        import torch
    except Exception:
        return _UNHANDLED

    if not isinstance(value, torch.Tensor):
        return _UNHANDLED

    tensor = value.detach().to("cpu")
    if tensor.numel() == 1:
        return tensor.reshape(-1)[0].item()
    return tensor.tolist()


def _numpy_to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        return _UNHANDLED

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return _UNHANDLED
