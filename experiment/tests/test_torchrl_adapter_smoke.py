from __future__ import annotations

import pathlib
import sys

import pytest

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

pytest.importorskip("torchrl")

from adapters.torchrl_env import CpcTorchRLEnv


def test_torchrl_adapter_reset_and_step_smoke():
    env = CpcTorchRLEnv()
    td = env.reset()
    action = env.action_spec.rand()
    td["action"] = action
    next_td = env.step(td)

    assert "observation" in td.keys()
    assert ("next", "observation") in next_td.keys(True)
    assert ("next", "reward") in next_td.keys(True)
    assert ("next", "done") in next_td.keys(True)
    assert ("next", "terminated") in next_td.keys(True)
    assert ("next", "truncated") in next_td.keys(True)

