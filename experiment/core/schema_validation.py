from typing import Any, Dict, List

SCHEMA_VERSION = "cpc-common-v0"


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _schema_version_errors(value: Any, path: str) -> List[str]:
    if not _is_dict(value) or value.get("schema_version") != SCHEMA_VERSION:
        return [f"{path}.schema_version must be {SCHEMA_VERSION}"]
    return []


def _has_duplicates(values: List[str]) -> bool:
    return len(set(values)) != len(values)


def _validate_config(config: Dict[str, Any]) -> List[str]:
    errors = _schema_version_errors(config, "config")
    mode = config.get("mode")
    players_per_team = config.get("players_per_team")
    if mode == "solo" and players_per_team != 1:
        errors.append("config.players_per_team must be 1 in solo mode")
    if mode == "duo" and players_per_team != 2:
        errors.append("config.players_per_team must be 2 in duo mode")
    if mode not in ("solo", "duo"):
        errors.append("config.mode must be solo or duo")
    if not isinstance(config.get("team_count"), int) or config.get("team_count") < 1:
        errors.append("config.team_count must be a positive integer")
    if not isinstance(config.get("max_steps"), int) or config.get("max_steps") < 1:
        errors.append("config.max_steps must be a positive integer")
    return errors


def _validate_observation(
    observation: Dict[str, Any],
    agent_id: str,
    step_path: str,
    config: Dict[str, Any],
) -> List[str]:
    path = f"{step_path}.observations.{agent_id}"
    errors = _schema_version_errors(observation, path)
    if observation.get("agent_id") != agent_id:
        errors.append(f"{path}.agent_id must match observation key")

    vector = observation.get("vector", [])
    vector_keys = observation.get("vector_keys", [])
    if len(vector) != len(vector_keys):
        errors.append(f"{path}.vector length must match vector_keys length")

    spec = config.get("observation_spec", {})
    checks = [
        ("visible_enemies", "visible_enemies_mask", spec.get("max_visible_enemies", 0)),
        ("visible_allies", "visible_allies_mask", spec.get("max_visible_allies", 0)),
        ("visible_obstacles", "visible_obstacles_mask", spec.get("max_visible_obstacles", 0)),
        ("recent_events", "recent_events_mask", spec.get("max_recent_events", 0)),
    ]
    for array_key, mask_key, max_count in checks:
        entities = observation.get(array_key, [])
        mask = observation.get(mask_key, [])
        if len(entities) > max_count:
            errors.append(f"{path}.{array_key} length exceeds {max_count}")
        if len(mask) not in (len(entities), max_count):
            errors.append(f"{path}.{mask_key} length must match {array_key} length or padded max count")
    return errors


def _validate_step(step: Dict[str, Any], index: int, config: Dict[str, Any]) -> List[str]:
    path = f"steps[{index}]"
    errors = _schema_version_errors(step, path)
    info = step.get("info")
    snapshot = info.get("snapshot") if isinstance(info, dict) else None
    if not isinstance(snapshot, dict):
        return errors + [f"{path}.info.snapshot is required"]

    errors.extend(_schema_version_errors(snapshot, f"{path}.info.snapshot"))
    agent_ids = [agent_id for agent_id in snapshot.get("agent_ids", []) if isinstance(agent_id, str)]
    team_ids = [team_id for team_id in snapshot.get("team_ids", []) if isinstance(team_id, str)]

    if not agent_ids:
        errors.append(f"{path}.info.snapshot.agent_ids must be non-empty")
    if _has_duplicates(agent_ids):
        errors.append(f"{path}.info.snapshot.agent_ids must be unique")
    if _has_duplicates(team_ids):
        errors.append(f"{path}.info.snapshot.team_ids must be unique")

    agent_team_map = snapshot.get("agent_team_map")
    if not isinstance(agent_team_map, dict):
        errors.append(f"{path}.info.snapshot.agent_team_map is required")
    else:
        for agent_id in agent_ids:
            if not isinstance(agent_team_map.get(agent_id), str):
                errors.append(f"{path}.info.snapshot.agent_team_map.{agent_id} is required")

    observations = step.get("observations") if isinstance(step.get("observations"), dict) else {}
    actions = step.get("actions") if isinstance(step.get("actions"), dict) else {}
    for agent_id in agent_ids:
        if agent_id not in observations:
            errors.append(f"{path}.observations missing {agent_id}")
        else:
            errors.extend(_validate_observation(observations[agent_id], agent_id, path, config))
        if agent_id not in actions:
            errors.append(f"{path}.actions missing {agent_id}")
        else:
            errors.extend(_schema_version_errors(actions[agent_id], f"{path}.actions.{agent_id}"))
    return errors


def _validate_solo_cooperation(episode: Dict[str, Any]) -> List[str]:
    config = episode.get("config", {})
    if config.get("mode") != "solo":
        return []
    errors: List[str] = []
    for agent_id, metrics in episode.get("final_metrics", {}).items():
        if metrics.get("cooperation", {}).get("applicable") is not False:
            errors.append(f"final_metrics.{agent_id}.cooperation.applicable must be false in solo mode")
    for step_index, step in enumerate(episode.get("steps", [])):
        metrics_by_agent = step.get("info", {}).get("metrics", {})
        for agent_id, metrics in metrics_by_agent.items():
            if metrics.get("cooperation", {}).get("applicable") is not False:
                errors.append(f"steps[{step_index}].info.metrics.{agent_id}.cooperation.applicable must be false in solo mode")
    return errors


def validate_episode(episode: Dict[str, Any]) -> List[str]:
    errors = _schema_version_errors(episode, "episode")
    config = episode.get("config") if isinstance(episode, dict) else None
    if not isinstance(config, dict):
        return errors + ["config is required"]
    errors.extend(_validate_config(config))
    steps = episode.get("steps")
    if not isinstance(steps, list):
        return errors + ["steps must be an array"]
    for index, step in enumerate(steps):
        errors.extend(_validate_step(step, index, config))
    errors.extend(_validate_solo_cooperation(episode))
    return errors


def assert_valid_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    errors = validate_episode(episode)
    if errors:
        raise ValueError("Invalid CPC common episode: " + "; ".join(errors))
    return episode

