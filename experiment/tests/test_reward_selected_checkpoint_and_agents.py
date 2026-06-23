from __future__ import annotations

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

from checkpointing import save_selected_checkpoint_if_needed
from policy_agent import PPOPolicyAgent
from run_model_agents import run_two_agent_eval
from training.cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS
from training.cpc_env import CPCEnv
from training.train_ppo import PPOConfig, train_ppo


def tiny_cfg(tmp_path: pathlib.Path) -> PPOConfig:
    return PPOConfig(
        seed=321,
        device="cpu",
        total_steps=16,
        rollout_steps=8,
        num_epochs=1,
        minibatch_size=4,
        max_episode_steps=4,
        hidden_dim=16,
        run_dir=str(tmp_path),
        selection_eval_episodes=1,
    )


def test_checkpoint_max_reward_is_saved(tmp_path):
    result = train_ppo(tiny_cfg(tmp_path))

    assert pathlib.Path(result["checkpoint_latest"]).exists()
    assert pathlib.Path(result["checkpoint_max_reward"]).exists()
    assert pathlib.Path(result["checkpoint_selected"]).exists()
    assert pathlib.Path(result["selected_reward_checkpoint"]).exists()

    metadata = json.loads(pathlib.Path(result["selected_reward_checkpoint"]).read_text(encoding="utf-8"))
    assert metadata["selection_metric"] == "eval_mean_episode_reward"
    assert metadata["selection_mode"] == "max"
    assert metadata["selection_value"] is not None
    assert metadata["selected_update"] is not None


def test_selected_checkpoint_uses_max_reward(tmp_path):
    latest = tmp_path / "checkpoint_latest.pt"
    selected = tmp_path / "checkpoint_max_reward.pt"
    metadata = tmp_path / "selected_reward_checkpoint.json"
    latest.write_bytes(b"checkpoint-a")

    first_selected, first_value = save_selected_checkpoint_if_needed(
        run_dir=tmp_path,
        latest_checkpoint=latest,
        selected_checkpoint=selected,
        metadata_path=metadata,
        update=1,
        global_step=8,
        metrics={"selection_value": 2.0},
        selection_metric="episodic_return_mean",
        selection_mode="max",
        selection_value=2.0,
    )
    latest.write_bytes(b"checkpoint-b")
    second_selected, second_value = save_selected_checkpoint_if_needed(
        run_dir=tmp_path,
        latest_checkpoint=latest,
        selected_checkpoint=selected,
        metadata_path=metadata,
        update=2,
        global_step=16,
        metrics={"selection_value": -1.0},
        selection_metric="episodic_return_mean",
        selection_mode="max",
        selection_value=3.0,
    )

    assert first_selected is True
    assert first_value == 2.0
    assert second_selected is True
    assert second_value == 3.0
    assert selected.read_bytes() == b"checkpoint-b"


def test_checkpoint_load_to_agent(tmp_path):
    result = train_ppo(tiny_cfg(tmp_path))
    agent = PPOPolicyAgent.from_checkpoint(result["checkpoint_selected"], device="cpu")
    observation = CPCEnv(seed=1, max_steps=4).reset()

    action = agent.act(observation)

    assert set(action) == {"move", "aim", "fire"}
    assert 0 <= action["move"] < MOVE_BINS
    assert 0 <= action["aim"] < AIM_BINS
    assert 0 <= action["fire"] < FIRE_BINS


def test_deterministic_agent_action_is_stable(tmp_path):
    result = train_ppo(tiny_cfg(tmp_path))
    agent = PPOPolicyAgent.from_checkpoint(result["checkpoint_selected"], device="cpu")
    observation = CPCEnv(seed=1, max_steps=4).reset()

    assert agent.act(observation, deterministic=True) == agent.act(observation, deterministic=True)


def test_sampled_agent_action_valid(tmp_path):
    result = train_ppo(tiny_cfg(tmp_path))
    agent = PPOPolicyAgent.from_checkpoint(result["checkpoint_selected"], device="cpu")
    observation = CPCEnv(seed=1, max_steps=4).reset()

    action = agent.act(observation, deterministic=False)

    assert 0 <= action["move"] < MOVE_BINS
    assert 0 <= action["aim"] < AIM_BINS
    assert 0 <= action["fire"] < FIRE_BINS


def test_env_does_not_import_policy_agent():
    source = (EXPERIMENT_ROOT / "training" / "cpc_env.py").read_text(encoding="utf-8")

    forbidden = ("PPOPolicyAgent", "ppo_policy", "checkpointing", "load_checkpoint", "torch")
    assert all(token not in source for token in forbidden)


def test_two_agent_runner_fails_clearly_when_second_checkpoint_requested(tmp_path):
    result = train_ppo(tiny_cfg(tmp_path))

    with pytest.raises(NotImplementedError, match="Two-agent model gameplay requested"):
        run_two_agent_eval(
            checkpoint_a=result["checkpoint_selected"],
            checkpoint_b=result["checkpoint_selected"],
            episodes=1,
            device="cpu",
            deterministic=True,
        )
