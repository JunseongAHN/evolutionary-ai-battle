from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from analyze_local_combat_eval import analyze_result, write_markdown


def _step(
    t: int,
    *,
    requested: bool = False,
    shot_fired: bool = False,
    aim_bin: int = 0,
    ideal_aim_bin: int = 0,
    aim_error: int = 0,
    blocked_reason: str | None = None,
    events: list[dict] | None = None,
    reward_components: dict | None = None,
    damage_dealt: float = 0.0,
    damage_taken: float = 0.0,
    in_good_range: bool = True,
    too_far: bool = False,
) -> dict:
    if blocked_reason is None and requested and not shot_fired:
        blocked_reason = "cooldown"
    return {
        "t": t,
        "fire": {"requested": requested, "shot_fired": shot_fired, "blocked_reason": blocked_reason},
        "aim": {"aim_bin": aim_bin, "ideal_aim_bin": ideal_aim_bin, "aim_bin_error": aim_error, "alignment": 1.0, "angle_error_deg": 0.0},
        "range": {"distance_to_enemy": 120.0, "in_good_range": in_good_range, "too_close": False, "too_far": too_far},
        "events": events or [],
        "reward": sum((reward_components or {}).values()),
        "reward_components": reward_components or {},
        "metrics_delta": {"damage_dealt_delta": damage_dealt, "damage_taken_delta": damage_taken},
    }


def _result(steps: list[dict], final_metrics: dict | None = None) -> dict:
    return {
        "episodes": [
            {
                "episode_index": 0,
                "steps": steps,
                "episode_return": {"agent": sum(float(step.get("reward", 0.0)) for step in steps)},
                "final_metrics": final_metrics or {},
            }
        ]
    }


def test_eval_analysis_counts_self_bullets_only():
    analysis = analyze_result(
        _result(
            [
                _step(
                    0,
                    requested=True,
                    shot_fired=True,
                    events=[
                        {"type": "bullet_spawned", "bullet_id": "self-1", "owner_id": "self"},
                        {"type": "bullet_spawned", "bullet_id": "enemy-1", "owner_id": "enemy"},
                    ],
                ),
                _step(
                    1,
                    events=[
                        {"type": "bullet_hit", "bullet_id": "self-1", "owner_id": "self", "target_id": "enemy"},
                        {"type": "bullet_hit", "bullet_id": "enemy-1", "owner_id": "enemy", "target_id": "self"},
                    ],
                    damage_dealt=10.0,
                    damage_taken=10.0,
                ),
            ],
            {"damage_dealt": 10.0, "damage_taken": 10.0},
        )
    )

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["self_bullet_hit_count"] == 1
    assert metrics["hit_ratio"] == pytest.approx(1.0)


def test_bullet_lifecycle_hit_not_counted_as_miss():
    analysis = analyze_result(
        _result(
            [
                _step(0, requested=True, shot_fired=True, events=[{"type": "bullet_spawned", "bullet_id": "b1", "owner_id": "self"}]),
                _step(1, events=[{"type": "bullet_hit", "bullet_id": "b1", "owner_id": "self", "target_id": "enemy"}], damage_dealt=10.0),
            ],
            {"damage_dealt": 10.0},
        )
    )

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["self_bullet_hit_count"] == 1
    assert metrics["self_missed_shot_count"] == 0


def test_bullet_expired_without_hit_counts_as_miss():
    analysis = analyze_result(
        _result(
            [
                _step(0, requested=True, shot_fired=True, events=[{"type": "bullet_spawned", "bullet_id": "b1", "owner_id": "self"}]),
                _step(1, events=[{"type": "bullet_expired", "bullet_id": "b1", "owner_id": "self"}]),
            ]
        )
    )

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["self_missed_shot_count"] == 1
    assert metrics["missed_shot_rate"] == pytest.approx(1.0)


def test_fire_requested_during_cooldown_not_counted_as_shot():
    analysis = analyze_result(_result([_step(0, requested=True, shot_fired=False)]))

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["fire_requested_count"] == 1
    assert metrics["shot_fired_count"] == 0
    assert metrics["fire_blocked_cooldown_count"] == 1


def test_eval_analysis_includes_fire_counts():
    metrics = analyze_result(_result([_step(0, requested=True, shot_fired=False)]))["episodes"][0]["metrics"]

    assert "fire_requested_count" in metrics
    assert "shot_fired_count" in metrics
    assert "fire_blocked_cooldown_count" in metrics


def test_self_bullet_hit_count_excludes_enemy_bullets():
    analysis = analyze_result(
        _result(
            [
                _step(0, events=[{"type": "bullet_spawned", "bullet_id": "enemy-1", "owner_id": "enemy"}]),
                _step(1, events=[{"type": "bullet_hit", "bullet_id": "enemy-1", "owner_id": "enemy", "target_id": "self"}]),
            ]
        )
    )

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["self_bullet_hit_count"] == 0


def test_enemy_bullet_hit_self_count():
    analysis = analyze_result(
        _result(
            [
                _step(0, events=[{"type": "bullet_spawned", "bullet_id": "enemy-1", "owner_id": "enemy"}]),
                _step(1, events=[{"type": "bullet_hit", "bullet_id": "enemy-1", "owner_id": "enemy", "target_id": "self"}]),
            ]
        )
    )

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["enemy_bullet_hit_self_count"] == 1


def test_self_bullet_expired_without_hit_counts_as_miss():
    analysis = analyze_result(
        _result(
            [
                _step(0, requested=True, shot_fired=True, events=[{"type": "bullet_spawned", "bullet_id": "b1", "owner_id": "self"}]),
                _step(1, events=[{"type": "bullet_expired", "bullet_id": "b1", "owner_id": "self"}]),
            ]
        )
    )

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["self_bullet_missed_count"] == 1


def test_bullet_hit_not_counted_as_miss():
    analysis = analyze_result(
        _result(
            [
                _step(0, requested=True, shot_fired=True, events=[{"type": "bullet_spawned", "bullet_id": "b1", "owner_id": "self"}]),
                _step(1, events=[{"type": "bullet_hit", "bullet_id": "b1", "owner_id": "self", "target_id": "enemy"}]),
            ]
        )
    )

    metrics = analysis["episodes"][0]["metrics"]
    assert metrics["self_bullet_hit_count"] == 1
    assert metrics["self_bullet_missed_count"] == 0


def test_alive_bullet_at_episode_end():
    analysis = analyze_result(
        _result([_step(0, requested=True, shot_fired=True, events=[{"type": "bullet_spawned", "bullet_id": "b1", "owner_id": "self"}])])
    )

    assert analysis["episodes"][0]["metrics"]["self_bullet_alive_at_episode_end"] == 1


def test_aim_distribution_fields_exist():
    metrics = analyze_result(_result([_step(0, aim_bin=2, ideal_aim_bin=3, aim_error=1)]))["episodes"][0]["metrics"]

    assert metrics["aim_bin_distribution"] == {2: 1}
    assert metrics["ideal_aim_bin_distribution"] == {3: 1}
    assert metrics["aim_error_distribution"] == {1: 1}


def test_shot_time_aim_distribution():
    metrics = analyze_result(
        _result(
            [
                _step(0, requested=True, shot_fired=True, aim_error=2),
                _step(1, requested=True, shot_fired=False, aim_error=0),
            ]
        )
    )["episodes"][0]["metrics"]

    assert metrics["shot_aim_error_distribution"] == {2: 1}


def test_warning_no_actual_shots():
    warnings = analyze_result(_result([_step(0, requested=True, shot_fired=False)]))["episodes"][0]["warnings"]

    assert "fire_requested_but_no_actual_shots" in warnings


def test_warning_no_damage_dealt():
    warnings = analyze_result(_result([_step(0)]))["episodes"][0]["warnings"]

    assert "no_damage_dealt" in warnings


def test_warning_full_damage_taken():
    warnings = analyze_result(_result([_step(0)], {"damage_taken": 100.0}))["episodes"][0]["warnings"]

    assert "agent_lost_all_hp" in warnings


def test_training_log_eval_analysis_has_new_keys():
    analysis = analyze_result(
        _result(
            [
                _step(0, requested=True, shot_fired=True, events=[{"type": "bullet_spawned", "bullet_id": "b1", "owner_id": "self"}]),
                _step(1, events=[{"type": "bullet_spawned", "bullet_id": "e1", "owner_id": "enemy"}]),
                _step(2, events=[{"type": "bullet_hit", "bullet_id": "e1", "owner_id": "enemy", "target_id": "self"}]),
            ]
        )
    )

    aggregate = analysis["aggregate"]
    for key in ("fire_requested_count", "shot_fired_count", "self_bullet_spawn_count", "enemy_bullet_hit_self_count"):
        assert key in aggregate


def test_damage_trade_ratio():
    analysis = analyze_result(_result([_step(0)], {"damage_dealt": 30.0, "damage_taken": 10.0}))

    assert analysis["episodes"][0]["metrics"]["damage_trade_ratio"] == pytest.approx(0.2)


def test_high_hit_ratio_low_volume_warning():
    analysis = analyze_result(
        _result(
            [
                _step(0, requested=True, shot_fired=True, events=[{"type": "bullet_spawned", "bullet_id": "b1", "owner_id": "self"}]),
                _step(1, events=[{"type": "bullet_hit", "bullet_id": "b1", "owner_id": "self"}], damage_dealt=10.0),
            ],
            {"damage_dealt": 10.0},
        )
    )

    assert "high_accuracy_low_volume" in analysis["episodes"][0]["warnings"]


def test_aim_bin_collapse_warning():
    analysis = analyze_result(_result([_step(i, aim_bin=0) for i in range(8)] + [_step(8, aim_bin=1)]))

    assert "aim_bin_collapse" in analysis["episodes"][0]["warnings"]


def test_reward_dominated_by_shaping_warning():
    analysis = analyze_result(
        _result(
            [
                _step(i, reward_components={"aim_bin_exact": 0.04, "good_range": 0.01})
                for i in range(10)
            ]
        )
    )

    assert "reward_dominated_by_shaping" in analysis["episodes"][0]["warnings"]


def test_analyze_local_combat_eval_outputs_markdown(tmp_path):
    output = tmp_path / "analysis.md"
    analysis = analyze_result(_result([_step(0, reward_components={"aim_bin_exact": 0.04})]))

    write_markdown(analysis, output)

    text = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "warnings" in text


def test_baseline_eval_outputs_expected_columns():
    pytest.importorskip("torch")
    from eval_local_combat_baselines import render_baseline_markdown

    markdown = render_baseline_markdown(
        {
            "rows": [
                {
                    "policy": "random",
                    "return": 0.0,
                    "damage_dealt_ratio": 0.0,
                    "damage_taken_ratio": 0.0,
                    "damage_trade_ratio": 0.0,
                    "hit_ratio": 0.0,
                    "missed_shot_rate": 0.0,
                    "aim_bin_0_rate": 0.0,
                    "exact_aim_match_rate": 0.0,
                    "self_dead": 0.0,
                    "enemy_dead": 0.0,
                    "warnings": {},
                }
            ]
        }
    )

    assert "policy | return | damage_dealt_ratio" in markdown
    assert "damage_trade_ratio" in markdown
