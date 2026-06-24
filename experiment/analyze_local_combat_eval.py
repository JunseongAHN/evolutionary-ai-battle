from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EPSILON = 1e-9
SHAPING_COMPONENTS = {
    "aim_bin_exact",
    "aim_bin_wrong",
    "good_range",
    "too_close",
    "too_far",
    "bullet_hit",
    "missed_shot",
    "accuracy_bonus",
}
OUTCOME_COMPONENTS = {
    "damage_dealt_ratio",
    "damage_taken_ratio",
    "kill",
    "death",
    "timeout_hp_lead",
}


def analyze_result(result: dict[str, Any]) -> dict[str, Any]:
    episodes = [analyze_episode(episode, result.get("config", {})) for episode in result.get("episodes", [])]
    return {
        "episode_count": len(episodes),
        "episodes": episodes,
        "aggregate": aggregate_episodes(episodes),
    }


def analyze_episode(episode: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = episode.get("steps", [])
    total_steps = max(1, len(steps))
    final_metrics = episode.get("final_metrics", {})
    total_reward = float(episode.get("episode_return", {}).get("agent", sum(float(step.get("reward", 0.0)) for step in steps)))

    component_stats = defaultdict(_component_stats)
    damage_dealt = 0.0
    damage_taken = 0.0
    fire_requested_count = 0
    shot_fired_count = 0
    aim_bins = Counter()
    aim_errors = Counter()
    shot_aim_errors = Counter()
    hit_aim_errors = Counter()
    missed_aim_errors = Counter()
    range_counts = Counter()
    shot_good_range_count = 0
    hit_good_range_count = 0
    miss_too_far_count = 0
    distance_sum = 0.0
    distance_count = 0
    damage_taken_too_close_count = 0

    bullets: dict[str, dict[str, Any]] = {}

    for step in steps:
        reward_components = step.get("reward_components", {})
        for name, value in reward_components.items():
            _update_component_stats(component_stats[name], float(value or 0.0))

        metrics_delta = step.get("metrics_delta", {})
        damage_dealt += float(metrics_delta.get("damage_dealt_delta", 0.0) or 0.0)
        damage_taken_delta = float(metrics_delta.get("damage_taken_delta", 0.0) or 0.0)
        damage_taken += damage_taken_delta

        fire = step.get("fire", {})
        if bool(fire.get("requested", False)):
            fire_requested_count += 1
        shot_fired = bool(fire.get("shot_fired", False))
        if shot_fired:
            shot_fired_count += 1

        aim = step.get("aim", {})
        aim_bin = _safe_int(aim.get("aim_bin"))
        aim_error = _safe_int(aim.get("aim_bin_error"))
        if aim_bin is not None:
            aim_bins[aim_bin] += 1
        if aim_error is not None:
            aim_errors[aim_error] += 1

        range_info = step.get("range", {})
        distance = _safe_float(range_info.get("distance_to_enemy"))
        if distance is not None:
            distance_sum += distance
            distance_count += 1
        if range_info.get("in_good_range"):
            range_counts["good"] += 1
        elif range_info.get("too_close"):
            range_counts["too_close"] += 1
        elif range_info.get("too_far"):
            range_counts["too_far"] += 1
        if damage_taken_delta > 0.0 and range_info.get("too_close"):
            damage_taken_too_close_count += 1

        for event in step.get("events", []):
            event_type = event.get("type")
            bullet_id = event.get("bullet_id")
            owner_id = event.get("owner_id", "self")
            if bullet_id is None:
                continue
            if owner_id != "self":
                continue
            if event_type == "bullet_spawned":
                bullets[bullet_id] = {
                    "bullet_id": bullet_id,
                    "spawn_step": int(step.get("t", 0)),
                    "aim_bin_at_spawn": aim_bin,
                    "aim_bin_error_at_spawn": aim_error,
                    "range_at_spawn": distance,
                    "hit_step": None,
                    "expire_step": None,
                    "hit_or_miss": "pending",
                    "travel_steps": 0,
                    "target_id": None,
                    "good_range_at_spawn": bool(range_info.get("in_good_range", False)),
                    "too_far_at_spawn": bool(range_info.get("too_far", False)),
                }
                if aim_error is not None:
                    shot_aim_errors[aim_error] += 1
                if range_info.get("in_good_range"):
                    shot_good_range_count += 1
            elif event_type == "bullet_moved" and bullet_id in bullets:
                bullets[bullet_id]["travel_steps"] += 1
            elif event_type == "bullet_hit" and bullet_id in bullets:
                bullets[bullet_id]["hit_step"] = int(step.get("t", 0))
                bullets[bullet_id]["hit_or_miss"] = "hit"
                bullets[bullet_id]["target_id"] = event.get("target_id")
                spawn_error = bullets[bullet_id].get("aim_bin_error_at_spawn")
                if spawn_error is not None:
                    hit_aim_errors[int(spawn_error)] += 1
                if bullets[bullet_id].get("good_range_at_spawn"):
                    hit_good_range_count += 1
            elif event_type == "bullet_expired" and bullet_id in bullets and bullets[bullet_id].get("hit_step") is None:
                bullets[bullet_id]["expire_step"] = int(step.get("t", 0))
                bullets[bullet_id]["hit_or_miss"] = "miss"
                spawn_error = bullets[bullet_id].get("aim_bin_error_at_spawn")
                if spawn_error is not None:
                    missed_aim_errors[int(spawn_error)] += 1
                if bullets[bullet_id].get("too_far_at_spawn"):
                    miss_too_far_count += 1

    self_bullet_hit_count = sum(1 for bullet in bullets.values() if bullet.get("hit_or_miss") == "hit")
    self_missed_shot_count = sum(1 for bullet in bullets.values() if bullet.get("hit_or_miss") == "miss")
    shot_fired_count = max(shot_fired_count, len(bullets))

    enemy_max_hp = float(final_metrics.get("enemy_max_hp", 100.0) or 100.0)
    self_max_hp = float(final_metrics.get("self_max_hp", 100.0) or 100.0)
    damage_dealt = float(final_metrics.get("damage_dealt", damage_dealt) or damage_dealt)
    damage_taken = float(final_metrics.get("damage_taken", damage_taken) or damage_taken)
    damage_dealt_ratio = damage_dealt / max(enemy_max_hp, 1.0)
    damage_taken_ratio = damage_taken / max(self_max_hp, 1.0)
    hit_ratio = self_bullet_hit_count / max(shot_fired_count, 1)
    missed_shot_rate = self_missed_shot_count / max(shot_fired_count, 1)

    max_aim_count = max(aim_bins.values(), default=0)
    exact_steps = sum(count for error, count in aim_errors.items() if error == 0)
    within_1_steps = sum(count for error, count in aim_errors.items() if error <= 1)
    bad_steps = sum(count for error, count in aim_errors.items() if error >= 3)
    shot_exact = sum(count for error, count in shot_aim_errors.items() if error == 0)
    shot_within_1 = sum(count for error, count in shot_aim_errors.items() if error <= 1)
    shot_bad = sum(count for error, count in shot_aim_errors.items() if error >= 2)
    hit_exact = sum(count for error, count in hit_aim_errors.items() if error == 0)

    metrics = {
        "total_reward": total_reward,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
        "enemy_max_hp": enemy_max_hp,
        "self_max_hp": self_max_hp,
        "damage_dealt_ratio": damage_dealt_ratio,
        "damage_taken_ratio": damage_taken_ratio,
        "damage_trade_ratio": damage_dealt_ratio - damage_taken_ratio,
        "self_dead": bool(float(final_metrics.get("self_dead", 0.0) or 0.0)),
        "enemy_dead": bool(float(final_metrics.get("enemy_dead", 0.0) or 0.0)),
        "timeout": not bool(float(final_metrics.get("self_dead", 0.0) or 0.0))
        and not bool(float(final_metrics.get("enemy_dead", 0.0) or 0.0)),
        "fire_requested_count": fire_requested_count,
        "shot_fired_count": shot_fired_count,
        "self_bullet_hit_count": self_bullet_hit_count,
        "self_missed_shot_count": self_missed_shot_count,
        "hit_ratio": hit_ratio,
        "missed_shot_rate": missed_shot_rate,
        "bullet_hit_per_shot": hit_ratio,
        "shot_efficiency": hit_ratio * min(damage_dealt_ratio, 1.0),
        "aim_bin_distribution": dict(sorted(aim_bins.items())),
        "aim_bin_0_rate": aim_bins.get(0, 0) / total_steps,
        "aim_error_distribution": dict(sorted(aim_errors.items())),
        "exact_aim_match_rate": exact_steps / total_steps,
        "within_1_bin_aim_rate": within_1_steps / total_steps,
        "bad_aim_rate": bad_steps / total_steps,
        "shot_exact_aim_rate": shot_exact / max(shot_fired_count, 1),
        "shot_within_1_bin_rate": shot_within_1 / max(shot_fired_count, 1),
        "shot_bad_aim_rate": shot_bad / max(shot_fired_count, 1),
        "hit_exact_aim_rate": hit_exact / max(self_bullet_hit_count, 1),
        "missed_shot_aim_error_distribution": dict(sorted(missed_aim_errors.items())),
        "good_range_rate": range_counts["good"] / total_steps,
        "too_close_rate": range_counts["too_close"] / total_steps,
        "too_far_rate": range_counts["too_far"] / total_steps,
        "avg_distance_to_enemy": distance_sum / max(distance_count, 1),
        "shot_good_range_rate": shot_good_range_count / max(shot_fired_count, 1),
        "hit_good_range_rate": hit_good_range_count / max(self_bullet_hit_count, 1),
        "miss_too_far_rate": miss_too_far_count / max(self_missed_shot_count, 1),
        "damage_taken_too_close_rate": damage_taken_too_close_count / max(total_steps, 1),
    }
    warnings = detect_warnings(metrics, component_stats, total_steps, max_aim_count)
    metrics["reward_hacking_warning_count"] = len(warnings)

    return {
        "episode_index": episode.get("episode_index", 0),
        "total_steps": len(steps),
        "metrics": metrics,
        "reward_components": _finalize_component_stats(component_stats, total_steps),
        "bullets": list(bullets.values()),
        "warnings": warnings,
    }


def detect_warnings(
    metrics: dict[str, Any],
    component_stats: dict[str, dict[str, float]],
    total_steps: int,
    max_aim_count: int,
) -> list[str]:
    warnings = []
    if metrics["total_reward"] > 0.0 and metrics["damage_dealt_ratio"] < 0.2:
        warnings.append("high_return_low_damage")
    if not metrics["self_dead"] and metrics["damage_dealt_ratio"] < 0.1 and metrics["shot_fired_count"] < 3:
        warnings.append("survival_without_combat")
    if metrics["exact_aim_match_rate"] > 0.7 and metrics["bullet_hit_per_shot"] < 0.2:
        warnings.append("aim_reward_without_hits")
    if metrics["fire_requested_count"] / max(total_steps, 1) > 0.8 and metrics["shot_fired_count"] * 2 < metrics["fire_requested_count"]:
        warnings.append("fire_spam")
    if max_aim_count / max(total_steps, 1) > 0.75:
        warnings.append("aim_bin_collapse")
    if metrics["hit_ratio"] > 0.8 and metrics["shot_fired_count"] <= 2:
        warnings.append("high_accuracy_low_volume")
    shaping_abs = sum(component_stats.get(name, {}).get("abs_sum", 0.0) for name in SHAPING_COMPONENTS)
    outcome_abs = sum(component_stats.get(name, {}).get("abs_sum", 0.0) for name in OUTCOME_COMPONENTS)
    if shaping_abs > outcome_abs * 1.5:
        warnings.append("reward_dominated_by_shaping")
    return warnings


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {}
    keys = (
        "total_reward",
        "damage_dealt_ratio",
        "damage_taken_ratio",
        "damage_trade_ratio",
        "hit_ratio",
        "missed_shot_rate",
        "bullet_hit_per_shot",
        "aim_bin_0_rate",
        "exact_aim_match_rate",
        "good_range_rate",
        "reward_hacking_warning_count",
    )
    aggregate = {}
    for key in keys:
        aggregate[key] = sum(float(ep["metrics"].get(key, 0.0)) for ep in episodes) / len(episodes)
    warning_counts = Counter(warning for ep in episodes for warning in ep.get("warnings", []))
    aggregate["warnings"] = dict(sorted(warning_counts.items()))
    return aggregate


def write_markdown(analysis: dict[str, Any], path: str | Path) -> Path:
    output = render_markdown(analysis)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    return path


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = ["# Local Combat Eval Analysis", ""]
    aggregate = analysis.get("aggregate", {})
    if aggregate:
        lines.extend(["## Aggregate", ""])
        for key, value in aggregate.items():
            if key == "warnings":
                continue
            lines.append(f"- `{key}`: {value:.4f}")
        lines.append(f"- `warnings`: {aggregate.get('warnings', {})}")
        lines.append("")
    for episode in analysis.get("episodes", []):
        metrics = episode["metrics"]
        lines.extend([f"## Episode {episode['episode_index']}", ""])
        for key in (
            "total_reward",
            "damage_dealt_ratio",
            "damage_taken_ratio",
            "damage_trade_ratio",
            "shot_fired_count",
            "self_bullet_hit_count",
            "hit_ratio",
            "missed_shot_rate",
            "aim_bin_0_rate",
            "exact_aim_match_rate",
            "shot_exact_aim_rate",
            "good_range_rate",
        ):
            lines.append(f"- `{key}`: {metrics.get(key)}")
        lines.append(f"- `warnings`: {', '.join(episode['warnings']) if episode['warnings'] else 'none'}")
        lines.extend(["", "### Reward Component Contribution", ""])
        lines.append("| component | sum | abs_sum | avg_step | avg_nonzero | nonzero_count |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, stats in sorted(episode["reward_components"].items()):
            lines.append(
                f"| {name} | {stats['sum']:.4f} | {stats['abs_sum']:.4f} | "
                f"{stats['average_per_step']:.4f} | {stats['average_when_nonzero']:.4f} | {stats['nonzero_count']} |"
            )
        lines.append("")
    return "\n".join(lines)


def print_summary(analysis: dict[str, Any]) -> None:
    for episode in analysis.get("episodes", []):
        metrics = episode["metrics"]
        print(f"Episode {episode['episode_index']}:")
        for key in (
            "total_reward",
            "damage_dealt_ratio",
            "damage_taken_ratio",
            "damage_trade_ratio",
            "shot_fired_count",
            "self_bullet_hit_count",
            "hit_ratio",
            "missed_shot_rate",
            "aim_bin_0_rate",
            "exact_aim_match_rate",
            "shot_exact_aim_rate",
            "good_range_rate",
        ):
            print(f"  {key}: {metrics.get(key)}")
        print(f"  warnings: {episode['warnings'] or ['none']}")
        print("  Reward component contribution:")
        print("  component | sum | abs_sum | avg_step | avg_nonzero | nonzero_count")
        for name, stats in sorted(episode["reward_components"].items()):
            print(
                f"  {name} | {stats['sum']:.4f} | {stats['abs_sum']:.4f} | "
                f"{stats['average_per_step']:.4f} | {stats['average_when_nonzero']:.4f} | {stats['nonzero_count']}"
            )
        print()


def _component_stats() -> dict[str, float]:
    return {
        "sum": 0.0,
        "abs_sum": 0.0,
        "positive_sum": 0.0,
        "negative_sum": 0.0,
        "nonzero_sum": 0.0,
        "nonzero_count": 0,
    }


def _update_component_stats(stats: dict[str, float], value: float) -> None:
    stats["sum"] += value
    stats["abs_sum"] += abs(value)
    if value > 0.0:
        stats["positive_sum"] += value
    if value < 0.0:
        stats["negative_sum"] += value
    if abs(value) > EPSILON:
        stats["nonzero_sum"] += value
        stats["nonzero_count"] += 1


def _finalize_component_stats(component_stats: dict[str, dict[str, float]], total_steps: int) -> dict[str, dict[str, float]]:
    finalized = {}
    for name, stats in component_stats.items():
        nonzero_count = int(stats["nonzero_count"])
        finalized[name] = {
            "sum": stats["sum"],
            "abs_sum": stats["abs_sum"],
            "positive_sum": stats["positive_sum"],
            "negative_sum": stats["negative_sum"],
            "average_per_step": stats["sum"] / max(total_steps, 1),
            "average_when_nonzero": stats["nonzero_sum"] / max(nonzero_count, 1),
            "nonzero_count": nonzero_count,
        }
    return finalized


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze compact Stage 1 local-combat gameplay results.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    analysis = analyze_result(result)
    print_summary(analysis)
    if args.output_md:
        write_markdown(analysis, args.output_md)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
