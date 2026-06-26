from __future__ import annotations

from typing import Dict, Iterable, List

from features import DEFAULT_THRESHOLDS, FEATURE_NAMES, extract_model_features, feature_vector_from_features


def _scenario_constraint_status(sample: Dict[str, object], features: Dict[str, int | float]) -> List[str]:
    scenario_id = sample.get("scenarioId")
    predicate_debug = sample.get("predicateDebug", {})
    issues: List[str] = []
    checks = {
        "direct_enemy_contact": predicate_debug.get("enemyNearby") == 1 and predicate_debug.get("allyUnderPressure") == 0 and predicate_debug.get("isIsolated") == 0 and predicate_debug.get("selfLowHp") == 0,
        "teammate_under_pressure": predicate_debug.get("allyUnderPressure") == 1 and predicate_debug.get("allyLowHp") == 1 and predicate_debug.get("enemyNearAlly") == 1 and predicate_debug.get("selfLowHp") == 0,
        "isolated_teammate": predicate_debug.get("isIsolated") == 1 and predicate_debug.get("allyUnderPressure") == 0 and predicate_debug.get("enemyNearby") == 0 and predicate_debug.get("selfLowHp") == 0,
        "self_low_hp": predicate_debug.get("selfLowHp") == 1 and predicate_debug.get("enemyNearby") == 1 and predicate_debug.get("allyUnderPressure") == 0,
    }
    if scenario_id in checks and not checks[scenario_id]:
        issues.append(f"scenario constraint failed for {scenario_id}")
    return issues


def inspect_sample_status(sample: Dict[str, object], thresholds: Dict[str, int] | None = None) -> Dict[str, object]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    features = extract_model_features(sample["state"], thresholds)
    stored_features = sample.get("features", {})
    feature_vector = sample.get("featureVector", [])
    recomputed_vector = feature_vector_from_features(features)
    checks = {
        "featureVectorLength": len(feature_vector) == len(FEATURE_NAMES),
        "featuresMatch": stored_features == features,
        "featureVectorMatchesOrder": feature_vector == recomputed_vector,
        "scenarioConstraint": not _scenario_constraint_status(sample, features),
    }

    current_status = "ok" if all(checks.values()) else "needs_attention"
    return {
        "sampleId": sample.get("sampleId"),
        "scenarioId": sample.get("scenarioId"),
        "split": sample.get("split"),
        "label": sample.get("label"),
        "labelIndex": sample.get("labelIndex"),
        "currentStatus": current_status,
        "checks": checks,
        "featureSnapshot": features,
        "predicateDebug": sample.get("predicateDebug", {}),
        "state": sample.get("state"),
        "issues": _scenario_constraint_status(sample, features),
    }


def format_sample_status(sample: Dict[str, object], thresholds: Dict[str, int] | None = None) -> str:
    status = inspect_sample_status(sample, thresholds)
    players = status["state"]["players"]
    self_player = next((player for player in players if player.get("role") == "self"), None)
    ally_player = next((player for player in players if player.get("role") == "ally"), None)
    lines = [
        f"sampleId: {status['sampleId']}",
        f"scenarioId: {status['scenarioId']}",
        f"label: {status['label']} ({status['labelIndex']})",
        f"currentStatus: {status['currentStatus']}",
        f"self: hp={self_player.get('hp') if self_player else 'n/a'} cooldown={self_player.get('weaponCooldownSteps') if self_player else 'n/a'}",
        f"ally: hp={ally_player.get('hp') if ally_player else 'n/a'}",
        f"featureVector: {status['featureSnapshot']}",
        f"predicateDebug: {status['predicateDebug']}",
    ]
    if status["issues"]:
        lines.append(f"issues: {', '.join(status['issues'])}")
    return "\n".join(lines)


def plot_sample_state(sample: Dict[str, object], thresholds: Dict[str, int] | None = None, ax=None, annotate: bool = True):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError as error:
        raise ImportError("matplotlib is required for plot_sample_state") from error

    status = inspect_sample_status(sample, thresholds)
    state = status["state"]
    players = state["players"]
    scenario_id = status["scenarioId"] or "unknown"
    label = status["label"] or "unknown"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    grid_width = thresholds["grid_width"]
    grid_height = thresholds["grid_height"]
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(0, grid_width, 2))
    ax.set_yticks(range(0, grid_height, 2))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("#fafafa")

    title = f"{scenario_id} | {label} | {status['currentStatus']}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    colors = {
        "self": "#1f77b4",
        "ally": "#2ca02c",
        "enemy": "#d62728",
    }
    markers = {
        "self": "o",
        "ally": "s",
        "enemy": "^",
    }

    for player in players:
        role = str(player.get("role", "enemy"))
        x = player.get("x", 0)
        y = player.get("y", 0)
        hp = player.get("hp", 0)
        alive = player.get("alive", False)
        color = colors.get(role, "#7f7f7f")
        marker = markers.get(role, "o")
        size = 280 if role == "self" else 220
        alpha = 1.0 if alive else 0.35

        ax.scatter([x], [y], s=size, c=color, marker=marker, alpha=alpha, edgecolors="black", linewidths=1.0, zorder=3)
        if annotate:
            ax.text(x + 0.18, y + 0.18, f"{player.get('id')}\nhp={hp}", fontsize=8, ha="left", va="bottom")

    self_player = next((player for player in players if player.get("role") == "self"), None)
    if self_player:
        cooldown = self_player.get("weaponCooldownSteps", 0)
        ax.text(
            0.02,
            0.98,
            f"self hp={self_player.get('hp', 'n/a')} cooldown={cooldown}\n"
            f"features: enemyNearby={status['featureSnapshot']['enemyNearby']} allyUnderPressure={status['featureSnapshot']['allyUnderPressure']} "
            f"isIsolated={status['featureSnapshot']['isIsolated']} canFire={status['featureSnapshot']['canFire']}",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )

    legend_handles = [
        Circle((0, 0), radius=0.1, facecolor=colors["self"], edgecolor="black", label="self"),
        Circle((0, 0), radius=0.1, facecolor=colors["ally"], edgecolor="black", label="ally"),
        Circle((0, 0), radius=0.1, facecolor=colors["enemy"], edgecolor="black", label="enemy"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    return fig, ax
