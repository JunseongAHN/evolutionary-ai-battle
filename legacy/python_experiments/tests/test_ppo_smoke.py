from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")
pytest.importorskip("tensordict")

from training.ppo_policy import AIM_BINS, FIRE_BINS, MOVE_BINS, MultiDiscreteActorCritic, flatten_observation
from training.torchrl_env import TorchRLCPCEnv
from training.train_ppo import PPOConfig, collect_rollout, compute_gae, ppo_update, train_ppo


def test_policy_forward_on_env_reset_observation():
    env = TorchRLCPCEnv(seed=1, max_steps=8)
    obs = env.reset()
    policy = MultiDiscreteActorCritic(hidden_dim=16)

    move_logits, aim_logits, fire_logits, value = policy(obs)

    assert flatten_observation(obs).shape[-1] == 22
    assert move_logits.shape[-1] == MOVE_BINS
    assert aim_logits.shape[-1] == AIM_BINS
    assert fire_logits.shape[-1] == FIRE_BINS
    assert value.shape[-1] == 1


def test_policy_samples_valid_actions():
    env = TorchRLCPCEnv(seed=1, max_steps=8)
    policy = MultiDiscreteActorCritic(hidden_dim=16)

    output = policy.sample_action(env.reset())

    assert 0 <= int(output.action["move"]) < MOVE_BINS
    assert 0 <= int(output.action["aim"]) < AIM_BINS
    assert 0 <= int(output.action["fire"]) < FIRE_BINS


def test_fire_while_moving_can_be_represented():
    env = TorchRLCPCEnv(seed=1, max_steps=8)
    td = env.reset()
    td["move"] = torch.tensor(6, dtype=torch.int64)
    td["aim"] = torch.tensor(0, dtype=torch.int64)
    td["fire"] = torch.tensor(1, dtype=torch.int64)

    stepped = env.step(td)
    decoded = stepped["next", "decoded_action"]

    assert decoded[0].item() > 0.0
    assert decoded[1].item() < 0.0
    assert decoded[4].item() == 1.0


def test_short_rollout_and_gae_shapes():
    cfg = PPOConfig(rollout_steps=8, max_episode_steps=8)
    env = TorchRLCPCEnv(seed=1, max_steps=8)
    policy = MultiDiscreteActorCritic(hidden_dim=16)

    rollout = collect_rollout(env, policy, cfg)
    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["dones"],
        rollout["values"],
        rollout["next_value"],
        cfg.gamma,
        cfg.gae_lambda,
    )

    assert rollout["observations"].shape == torch.Size([8, 22])
    assert advantages.shape == torch.Size([8])
    assert returns.shape == torch.Size([8])


def test_one_ppo_update_runs_without_error():
    cfg = PPOConfig(rollout_steps=8, minibatch_size=4, num_epochs=1, max_episode_steps=8)
    env = TorchRLCPCEnv(seed=1, max_steps=8)
    policy = MultiDiscreteActorCritic(hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    rollout = collect_rollout(env, policy, cfg)
    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["dones"],
        rollout["values"],
        rollout["next_value"],
        cfg.gamma,
        cfg.gae_lambda,
    )

    stats = ppo_update(policy, optimizer, rollout, advantages, returns, cfg)

    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "entropy" in stats


def test_smoke_training_writes_checkpoint_and_metrics(tmp_path):
    cfg = PPOConfig(
        total_steps=16,
        rollout_steps=8,
        num_epochs=1,
        minibatch_size=4,
        max_episode_steps=8,
        hidden_dim=16,
        run_dir=str(tmp_path),
    )

    result = train_ppo(cfg)

    assert pathlib.Path(result["checkpoint"]).exists()
    assert pathlib.Path(result["metrics_csv"]).exists()


def test_eval_checkpoint_exposes_fire_diagnostics(tmp_path):
    cfg = PPOConfig(
        total_steps=16,
        rollout_steps=8,
        num_epochs=1,
        minibatch_size=4,
        max_episode_steps=8,
        hidden_dim=16,
        run_dir=str(tmp_path),
    )

    result = train_ppo(cfg)
    deterministic_report = eval_checkpoint(result["checkpoint"], episodes=2, deterministic=True)
    sampled_report = eval_checkpoint(result["checkpoint"], episodes=2, sampled=True)

    assert "fire_diagnostics" in deterministic_report
    assert deterministic_report["fire_diagnostics"]["sampled_fire_rate"] is None
    assert len(deterministic_report["fire_diagnostics"]["mean_logits"]) == 2
    assert len(deterministic_report["fire_diagnostics"]["mean_probs"]) == 2
    assert deterministic_report["fire_diagnostics"]["deterministic_fire_action"] in (0, 1)

    assert sampled_report["fire_diagnostics"]["sampled_fire_rate"] is not None
    assert 0.0 <= sampled_report["fire_diagnostics"]["sampled_fire_rate"] <= 1.0
