from __future__ import annotations

import copy
import json
import threading

import pytest

torch = pytest.importorskip("torch")
from websockets.sync.client import connect  # noqa: E402
from websockets.sync.server import serve  # noqa: E402

from experiment.survev_rl.actions import ActionSpace  # noqa: E402
from experiment.survev_rl.featurizer import Featurizer, FeaturizerConfig  # noqa: E402
from experiment.survev_rl.policy_server import CheckpointPolicy  # noqa: E402
from experiment.survev_rl.ppo import ActorCritic, PPOConfig, save_checkpoint  # noqa: E402
from experiment.survev_rl.protocol import CpcAction  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path):
    feat = FeaturizerConfig(time_limit=60.0)
    obs_dim = Featurizer(feat).size + 2  # two controlled agents -> agent one-hot
    agent = ActorCritic(obs_dim, ActionSpace().nvec, hidden_dim=32)
    meta = {
        "env": {"controlled": ["team-a-0", "team-a-1"], "time_limit": 60.0},
        "featurizer": feat.to_dict(),
        "action_space": {"mode": "primitive", "assist": True, "auto_pickup": False},
    }
    return save_checkpoint(tmp_path / "ckpt.pt", agent, None, PPOConfig(hidden_dim=32), meta=meta)


def _two_agent_obs(spec_obs):
    a0 = copy.deepcopy(spec_obs)
    a1 = copy.deepcopy(spec_obs)
    a1["self"]["id"] = "team-a-1"
    a1["teammates"][0]["id"] = "team-a-0"
    return {"team-a-0": a0, "team-a-1": a1}


def test_checkpoint_policy_returns_wire_actions(checkpoint, spec_obs):
    torch.manual_seed(0)
    policy = CheckpointPolicy(checkpoint, deterministic=True)
    assert policy.controlled == ("team-a-0", "team-a-1")
    policy.reset(["team-a-0", "team-a-1", "team-b-0", "team-b-1"],
                 {"team-a-0": "team-a", "team-a-1": "team-a", "team-b-0": "team-b", "team-b-1": "team-b"})
    actions, debug = policy.act(1.0, _two_agent_obs(spec_obs))
    assert set(actions) == {"team-a-0", "team-a-1"}
    for aid, wire in actions.items():
        parsed = CpcAction.from_json(wire)  # valid bridge wire form (names validated)
        assert parsed.aim is not None
        assert "move" in debug[aid] and "value" in debug[aid]
    # deterministic policy + identical observations -> identical actions apart from the agent one-hot
    again, _ = policy.act(1.1, _two_agent_obs(spec_obs))
    assert again == actions


def test_checkpoint_policy_skips_dead_agents(checkpoint, spec_obs):
    policy = CheckpointPolicy(checkpoint, deterministic=True)
    obs = _two_agent_obs(spec_obs)
    obs["team-a-1"]["self"]["dead"] = True
    actions, debug = policy.act(2.0, obs)
    assert actions["team-a-1"] == {} and debug["team-a-1"] == {"inactive": True}
    assert actions["team-a-0"] != {}


def test_policy_server_round_trip(checkpoint, spec_obs):
    """The WebSocket layer used by main(): reset -> ready, act -> actions, unknown -> error."""
    from experiment.survev_rl.policy_server import make_handler

    policy = CheckpointPolicy(checkpoint, deterministic=True)
    with serve(make_handler(policy, deterministic=True), "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with connect(f"ws://127.0.0.1:{port}") as ws:
            ws.send(json.dumps({"type": "reset", "agent_ids": ["team-a-0", "team-a-1"],
                                "teams": {"team-a-0": "team-a", "team-a-1": "team-a"}}))
            ready = json.loads(ws.recv())
            assert ready["type"] == "ready" and ready["controlled"] == ["team-a-0", "team-a-1"] and ready["deterministic"]
            ws.send(json.dumps({"type": "act", "t": 0.5, "obs": _two_agent_obs(spec_obs)}))
            reply = json.loads(ws.recv())
            assert reply["type"] == "actions" and set(reply["actions"]) == {"team-a-0", "team-a-1"}
            ws.send(json.dumps({"type": "nope"}))
            assert json.loads(ws.recv())["type"] == "error"
        server.shutdown()
