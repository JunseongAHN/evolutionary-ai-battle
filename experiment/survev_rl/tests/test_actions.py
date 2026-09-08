from __future__ import annotations

import numpy as np
import pytest

from experiment.core.cpc_actions import MOVE_VECTORS, aim_bin_to_vec
from experiment.survev_rl.actions import SKILLS, ActionSpace, assist_inputs


def test_primitive_mapping_uses_harness_bins(spec_obs):
    space = ActionSpace(assist=False)
    assert space.nvec == (9, 16, 2, 2) and space.n_dims == 4
    a = space.to_cpc_action([4, 4, 1, 1], spec_obs)
    assert a.move == MOVE_VECTORS[4] == (1.0, 0.0)
    aim = aim_bin_to_vec(4)
    assert a.aim == (pytest.approx(aim["x"]), pytest.approx(aim["y"]))  # bin 4 = 90 degrees
    assert a.fire_hold and not a.fire_start and a.inputs == ["Interact"]
    j = a.to_json()
    assert j["fire"] == {"start": False, "hold": True} and j["inputs"] == ["Interact"]
    still = space.to_cpc_action([0, 0, 0, 0], spec_obs)
    assert still.move is None and still.aim == (1.0, 0.0) and not still.fire_hold and still.inputs == []
    assert "move" not in still.to_json()  # no movement -> key omitted, aim still sent
    for move_bin, vec in MOVE_VECTORS.items():
        out = space.to_cpc_action([move_bin, 0, 0, 0]).move
        assert out == (vec if any(vec) else None)
    with pytest.raises(ValueError):
        space.to_cpc_action([9, 0, 0, 0])
    with pytest.raises(ValueError):
        space.to_cpc_action([0, 0, 0])
    assert space.describe([4, 4, 1, 0]) == {"move": "right", "aim_deg": 90.0, "fire": True, "interact": False}
    rng = np.random.default_rng(0)
    for _ in range(50):
        sample = space.sample(rng)
        assert all(0 <= s < n for s, n in zip(sample, space.nvec))


def test_assist_layer_equips_and_reloads(spec_obs):
    space = ActionSpace(assist=True)
    # gun equipped, clip 21 -> nothing to assist
    assert space.to_cpc_action([0, 0, 0, 0], spec_obs).inputs == []
    # fists equipped while a gun is in slot 0 -> EquipPrimary (+ Interact from the policy bit)
    spec_obs["self"]["weapon"], spec_obs["self"]["cur_weap_idx"] = "fists", 2
    assert space.to_cpc_action([0, 0, 0, 1], spec_obs).inputs == ["Interact", "EquipPrimary"]
    spec_obs["self"]["weapons"][0]["type"], spec_obs["self"]["weapons"][1]["type"] = "", "mp5"
    assert assist_inputs(spec_obs) == ["EquipSecondary"]
    spec_obs["self"]["weapons"][1]["type"] = ""
    assert assist_inputs(spec_obs) == []  # no gun anywhere
    # empty clip with reserve -> Reload, unless already reloading or no reserve
    spec_obs["self"].update({"weapon": "ak47", "cur_weap_idx": 0, "clip": 0, "reserve": 45, "action": 0})
    spec_obs["self"]["weapons"][0]["type"] = "ak47"
    assert assist_inputs(spec_obs) == ["Reload"]
    spec_obs["self"]["action"] = 1
    assert assist_inputs(spec_obs) == []
    spec_obs["self"]["action"], spec_obs["self"]["reserve"] = 0, 0
    assert assist_inputs(spec_obs) == []
    # downed / dead agents get no assist inputs; assist=False disables everything
    spec_obs["self"].update({"reserve": 45, "downed": True})
    assert assist_inputs(spec_obs) == []
    spec_obs["self"]["downed"] = False
    assert ActionSpace(assist=False).to_cpc_action([0, 0, 0, 0], spec_obs).inputs == []
    # without an observation the assist layer is skipped silently
    assert ActionSpace(assist=True).to_cpc_action([0, 0, 0, 0], None).inputs == []


def test_auto_pickup_assist(spec_obs):
    space = ActionSpace(assist=True, auto_pickup=True)
    spec_obs["loot"][0]["dist"] = 1.5
    assert space.to_cpc_action([0, 0, 0, 0], spec_obs).inputs == ["Interact"]
    assert space.to_cpc_action([0, 0, 0, 1], spec_obs).inputs == ["Interact"]  # not duplicated
    spec_obs["loot"][0]["dist"] = 4.4
    assert space.to_cpc_action([0, 0, 0, 0], spec_obs).inputs == []


def test_skill_mode_mapping(spec_obs):
    space = ActionSpace(mode="skill")
    assert space.nvec == (len(SKILLS),) == (7,) and space.dims == ("skill",)
    with pytest.raises(NotImplementedError):
        space.to_cpc_action([0], spec_obs)
    assert space.to_skill_action(SKILLS.index("engage"), spec_obs) == {
        "skill": "engage", "params": {"target": "team-b-0", "style": "hold_angle"}
    }
    assert space.to_skill_action([SKILLS.index("move_to_partner")], spec_obs)["params"] == {"target": "team-a-1", "distance": 6.0}
    assert space.to_skill_action(SKILLS.index("loot_nearest"), spec_obs)["params"] == {"target": 512, "type": "bandage"}
    assert space.to_skill_action(SKILLS.index("take_cover"), spec_obs)["params"] == {"cover": 77, "face": "team-b-0"}
    assert space.to_skill_action(SKILLS.index("retreat"), spec_obs)["params"] == {"away_from": "team-b-0", "distance": 30.0}
    assert space.to_skill_action(SKILLS.index("revive"), spec_obs)["params"] == {}  # nobody downed
    spec_obs["teammates"][0]["downed"] = True
    assert space.to_skill_action(SKILLS.index("revive"), spec_obs)["params"] == {"target": "team-a-1"}
    hold = space.to_skill_action(SKILLS.index("hold"), spec_obs)
    assert hold["skill"] == "hold" and hold["params"]["face"] == "team-b-0"
    assert space.to_skill_action(0, None) == {"skill": "move_to_partner", "params": {}}
    assert space.describe([2]) == {"skill": "engage"}
    with pytest.raises(ValueError):
        space.to_skill_action(7, spec_obs)
