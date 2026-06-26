from __future__ import annotations

import csv
import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")
pytest.importorskip("tensordict")

from check_pr3_acceptance import (
    check_decode_bounds,
    check_forced_move_fire_action,
    check_same_seed_reproducibility,
    validate_checkpoint_load,
    validate_eval_report,
    validate_metrics_csv,
)
from training.eval_ppo import eval_checkpoint
from training.train_ppo import PPOConfig, train_ppo


def tiny_cfg(tmp_path: pathlib.Path) -> PPOConfig:
    return PPOConfig(
        seed=123,
        device="cpu",
        total_steps=16,
        rollout_steps=8,
        num_epochs=1,
        minibatch_size=4,
        max_episode_steps=4,
        hidden_dim=16,
        run_dir=str(tmp_path),
    )


def test_forced_move_fire_action_decodes_and_steps():
    status, detail = check_forced_move_fire_action(seed=123)

    assert status == "PASS"
    assert detail["raw_action"]["move"] != 0
    assert detail["raw_action"]["fire"] == 1
    assert detail["decoded_action"]["fire"] == 1
    assert abs(detail["decoded_action"]["move_x"]) + abs(detail["decoded_action"]["move_y"]) > 0.0


def test_decoded_action_bounds():
    status, detail = check_decode_bounds()

    assert status == "PASS"
    assert detail["checked_actions"] == 9 * 16 * 2


def test_policy_same_seed_first_action_cpu():
    status, detail = check_same_seed_reproducibility(seed=123)

    assert status == "PASS"
    assert set(detail["first_action"]) == {"move", "aim", "fire"}


def test_metrics_csv_validation(tmp_path):
    path = tmp_path / "metrics.csv"
    columns = [
        "update",
        "step",
        "episodic_return_mean",
        "episode_length_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "avg_ally_distance",
        "isolation_rate",
        "damage_dealt",
        "damage_taken",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: 0 for key in columns})
        writer.writerow({key: 1 for key in columns})

    status, detail = validate_metrics_csv(path, min_rows=2)

    assert status == "PASS"
    assert detail["rows"] == 2


def test_checkpoint_load_validation(tmp_path):
    result = train_ppo(tiny_cfg(tmp_path))

    status, detail = validate_checkpoint_load(result["checkpoint"], eval_episodes=2)

    assert status == "PASS"
    assert detail["checksum"]
    assert detail["eval_report"]["episodes"] == 2


def test_eval_10_episodes_smoke(tmp_path):
    result = train_ppo(tiny_cfg(tmp_path))
    report = eval_checkpoint(result["checkpoint"], episodes=10)
    status, detail = validate_eval_report(report, episodes=10)

    assert status == "PASS"
    assert detail["episodes"] == 10
    assert "mean_episode_return" in detail
    assert "mean_episode_length" in detail
    assert "mean_metrics" in detail
