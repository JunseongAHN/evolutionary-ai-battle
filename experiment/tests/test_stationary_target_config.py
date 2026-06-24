from __future__ import annotations

import csv
import json
import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")
pytest.importorskip("tensordict")

from training.torchrl_env import TorchRLCPCEnv
from training.train_ppo import PPOConfig, load_config, train_ppo


def test_stationary_target_config_sets_enemy_flags():
    cfg = load_config("experiment/configs/local_combat_stationary_target.yaml")

    assert cfg.stage == "local_combat"
    assert cfg.randomize_enemy_spawn_direction is True
    assert cfg.enemy_move is False
    assert cfg.enemy_fire is False
    assert cfg.stationary_target_mode is True


def test_stage1_config_missing_required_fields_raises(tmp_path):
    path = tmp_path / "local_combat_broken.yaml"
    path.write_text(
        "\n".join(
            [
                "stage: local_combat",
                "max_episode_steps: 100",
                "shrink_safe_zone: false",
                "use_zone_reward: false",
                "enemy_move: false",
                "stationary_target_mode: true",
                "enemy_spawn_distance_min: 180",
                "enemy_spawn_distance_max: 240",
                "bullet_range: 280.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field"):
        load_config(path)


def test_stage1_config_resolves_from_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cfg = load_config("experiment/configs/local_combat_in_range.yaml")

    assert cfg.enemy_move is False
    assert cfg.enemy_fire is False
    assert cfg.stationary_target_mode is True
    assert cfg.enemy_spawn_distance_min == pytest.approx(180.0)
    assert cfg.enemy_spawn_distance_max == pytest.approx(240.0)
    assert cfg.bullet_range == pytest.approx(280.0)


def test_in_range_config_spawns_enemy_within_bullet_range():
    cfg = load_config("experiment/configs/local_combat_in_range.yaml")
    env = TorchRLCPCEnv(
        seed=7,
        max_steps=8,
        device="cpu",
        randomize_enemy_spawn_direction=cfg.randomize_enemy_spawn_direction,
        enemy_spawn_directions=cfg.enemy_spawn_directions,
        enemy_spawn_distance_min=cfg.enemy_spawn_distance_min,
        enemy_spawn_distance_max=cfg.enemy_spawn_distance_max,
        enemy_move=cfg.enemy_move,
        enemy_fire=cfg.enemy_fire,
        stationary_target_mode=cfg.stationary_target_mode,
        bullet_range=cfg.bullet_range,
        fire_interval_steps=cfg.fire_interval_steps,
        bullet_speed=cfg.bullet_speed,
        bullet_damage=cfg.bullet_damage,
        bullet_hit_radius=cfg.bullet_hit_radius,
    )

    for _ in range(5):
        td = env.reset()
        distance = float(td["distance_to_enemy"].reshape(-1)[0].item())
        assert distance < float(cfg.bullet_range)
        assert distance >= float(cfg.enemy_spawn_distance_min)
        assert distance <= float(cfg.enemy_spawn_distance_max)


def test_in_range_config_debug_prints_stationary_flags_and_ranges(capsys):
    cfg = load_config("experiment/configs/local_combat_in_range.yaml")

    from training.train_ppo import debug_print_reset_samples

    debug_print_reset_samples(cfg, samples=10, config_path="experiment/configs/local_combat_in_range.yaml")
    captured = capsys.readouterr().out.strip().splitlines()

    assert captured
    first = json.loads(captured[0])
    assert first["config_path"].endswith("local_combat_in_range.yaml")
    assert float(first["bullet_range"]) == pytest.approx(float(cfg.bullet_range))
    assert first["enemy_move"] is False
    assert first["enemy_fire"] is False
    assert first["stationary_target_mode"] is True
    for line in captured[1:]:
        sample = json.loads(line)
        assert sample["distance_to_enemy"] < sample["bullet_range"]
        assert sample["within_bullet_range"] is True
        assert sample["enemy_move"] is False
        assert sample["enemy_fire"] is False
        assert sample["stationary_target_mode"] is True


def test_stationary_target_env_keeps_enemy_still_and_damage_taken_zero():
    env = TorchRLCPCEnv(
        seed=7,
        max_steps=8,
        device="cpu",
        enemy_move=False,
        enemy_fire=False,
        stationary_target_mode=True,
        randomize_enemy_spawn_direction=True,
    )
    obs = env.reset()
    initial_enemy_pos = dict(env.cpc_env.state["enemy_pos"])

    step_td = obs.clone()
    step_td["move"] = torch.tensor(0, dtype=torch.int64)
    step_td["aim"] = torch.tensor(0, dtype=torch.int64)
    step_td["fire"] = torch.tensor(0, dtype=torch.int64)
    next_td = env.step(step_td)["next"]

    assert env.cpc_env.state["enemy_pos"] == initial_enemy_pos
    assert float(next_td["metrics", "damage_taken"].reshape(-1)[0].item()) == 0.0
    assert float(next_td["metrics", "damage_taken_ratio"].reshape(-1)[0].item()) == 0.0


def test_stationary_target_training_writes_stationary_fields(tmp_path):
    cfg = PPOConfig(
        seed=11,
        device="cpu",
        total_steps=8,
        rollout_steps=4,
        num_epochs=1,
        minibatch_size=2,
        max_episode_steps=4,
        hidden_dim=16,
        run_dir=str(tmp_path),
        randomize_enemy_spawn_direction=True,
        enemy_move=False,
        enemy_fire=False,
        stationary_target_mode=True,
        eval_analysis_interval_steps=4,
        eval_analysis_episodes=1,
        selection_eval_episodes=1,
    )

    result = train_ppo(cfg)

    assert result["last_metrics"]["enemy_move"] == 0.0
    assert result["last_metrics"]["enemy_fire"] == 0.0
    assert result["last_metrics"]["stationary_target_mode"] == 1.0
    assert result["last_metrics"]["eval_analysis_enemy_move"] == 0.0
    assert result["last_metrics"]["eval_analysis_enemy_fire"] == 0.0
    assert result["last_metrics"]["eval_analysis_stationary_target_mode"] == 1.0
    assert result["last_metrics"]["eval_analysis_damage_taken"] == 0.0

    with pathlib.Path(result["metrics_csv"]).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    for column in (
        "enemy_move",
        "enemy_fire",
        "stationary_target_mode",
        "eval_analysis_damage_taken",
        "eval_analysis_enemy_move",
        "eval_analysis_enemy_fire",
        "eval_analysis_stationary_target_mode",
    ):
        assert column in columns

    assert rows
    assert float(rows[-1]["enemy_move"]) == 0.0
    assert float(rows[-1]["enemy_fire"]) == 0.0
    assert float(rows[-1]["stationary_target_mode"]) == 1.0
