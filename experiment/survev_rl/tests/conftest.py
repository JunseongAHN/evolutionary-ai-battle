from __future__ import annotations

import copy
from typing import Any

import pytest

from experiment.survev_rl.bridge_client import BridgeClient
from experiment.survev_rl.mock_bridge import MockBridgeThread

# The AgentObservation example from docs/survev-bridge-v0.md (verbatim shape).
SPEC_AGENT_OBS: dict[str, Any] = {
    "self": {
        "id": "team-a-0", "team": "team-a", "pos": {"x": 106.7, "y": 131.1},
        "dir": {"x": 0.93, "y": -0.37}, "hp": 87.5, "boost": 0, "downed": False, "dead": False,
        "weapon": "ak47", "clip": 21, "reserve": 45,
        "weapons": [{"slot": 0, "type": "ak47", "ammo": 21}, {"slot": 1, "type": "", "ammo": 0},
                    {"slot": 2, "type": "fists", "ammo": 0}, {"slot": 3, "type": "", "ammo": 0}],
        "inventory": {"bandage": 4, "healthkit": 0, "soda": 2, "painkiller": 0,
                      "762mm": 45, "9mm": 0, "12gauge": 0, "556mm": 0},
        "scope": "1xscope", "zoom": 28, "action": 0, "cur_weap_idx": 0,
    },
    "teammates": [{"id": "team-a-1", "pos": {"x": 108.2, "y": 126.3}, "dist": 5.0, "hp": 100,
                   "downed": False, "dead": False}],
    "players": [{"id": "team-b-0", "team": "team-b", "pos": {"x": 128.9, "y": 130.2}, "dist": 22.2,
                 "dir": {"x": -1, "y": 0}, "downed": False, "dead": False, "weapon": "ak47"}],
    "loot": [{"id": 512, "type": "bandage", "pos": {"x": 110.4, "y": 128.8}, "dist": 4.4, "count": 4}],
    "obstacles": [{"id": 77, "type": "tree_01", "pos": {"x": 120.0, "y": 140.0}, "dist": 16.0,
                   "collidable": True, "height": 1, "scale": 1}],
    "bullets": [{"pos": {"x": 118.0, "y": 131.0}, "dir": {"x": -1, "y": 0}, "player_id": 1027}],
    "dead_bodies": [{"pos": {"x": 130.0, "y": 129.0}, "dist": 23.3}],
    "gas": {"mode": 0, "rad": 196, "pos": {"x": 132, "y": 132}, "rad_new": 196,
            "pos_new": {"x": 132, "y": 132}},
    "alive_count": 4, "alive_teams": 2,
}


@pytest.fixture
def spec_obs() -> dict[str, Any]:
    return copy.deepcopy(SPEC_AGENT_OBS)


@pytest.fixture(scope="session")
def mock_server():
    with MockBridgeThread() as server:
        yield server


@pytest.fixture
def client(mock_server):
    with BridgeClient(mock_server.url, timeout=30.0) as c:
        yield c
