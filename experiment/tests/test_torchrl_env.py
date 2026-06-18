from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")
pytest.importorskip("tensordict")

from torchrl_env import TorchRLCPCEnv
from torchrl_specs import import_check_env_specs


def test_torchrl_cpc_env_constructs_and_resets():
    env = TorchRLCPCEnv(seed=3, max_steps=4)

    td = env.reset()

    assert "self_hp" in td.keys()
    assert "ally_hp" in td.keys()
    assert "enemy_hp" in td.keys()
    assert "done" in td.keys()
    assert td["self_pos"].shape == torch.Size([2])
    assert td["self_hp"].shape == torch.Size([1])
    assert td["done"].shape == torch.Size([1])


def test_torchrl_cpc_env_action_spec_samples_and_steps():
    env = TorchRLCPCEnv(seed=3, max_steps=4)
    td = env.reset()
    action = env.action_spec.rand()
    td.update(action)

    stepped = env.step(td)

    assert ("next", "reward") in stepped.keys(True)
    assert ("next", "done") in stepped.keys(True)
    assert ("next", "terminated") in stepped.keys(True)
    assert ("next", "truncated") in stepped.keys(True)
    assert stepped["next", "reward"].shape == torch.Size([1])


def test_torchrl_cpc_env_allows_move_and_fire_together():
    env = TorchRLCPCEnv(seed=3, max_steps=4)
    td = env.reset()
    td["move"] = torch.tensor(6, dtype=torch.int64)
    td["aim"] = torch.tensor(0, dtype=torch.int64)
    td["fire"] = torch.tensor(1, dtype=torch.int64)

    stepped = env.step(td)
    decoded = stepped["next", "decoded_action"]

    assert decoded[0].item() > 0.0
    assert decoded[1].item() < 0.0
    assert decoded[4].item() == 1.0


def test_torchrl_cpc_env_check_env_specs_if_available():
    check_env_specs = import_check_env_specs()
    if check_env_specs is None:
        pytest.skip("TorchRL check_env_specs is not available in this version")

    env = TorchRLCPCEnv(seed=3, max_steps=4)
    check_env_specs(env)
