from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from experiment.survev_rl.featurizer import Featurizer, FeaturizerConfig
from experiment.survev_rl.mock_bridge import FieldSim


def _idx(f: Featurizer, key: str) -> int:
    return f.vector_keys.index(key)


def test_layout_size_and_keys_are_consistent():
    f = Featurizer()
    assert f.size == 216 and len(f.vector_keys) == 216 and len(set(f.vector_keys)) == 216
    assert f.layout["self"] == (0, 31) and f.layout["mem2"] == (212, 216)
    small = Featurizer(FeaturizerConfig(max_enemies=1, max_loot=2, max_obstacles=0, max_bullets=0, memory=False))
    assert small.size == 31 + 9 + 10 + 22 + 0 + 0 + 7 + 5
    assert "obs_dim = 216" in f.describe()


def test_featurize_spec_example_values(spec_obs):
    f = Featurizer()
    v = f.featurize(spec_obs, t=6.0)
    assert v.shape == (216,) and v.dtype == np.float32 and np.isfinite(v).all()
    k = lambda name: float(v[_idx(f, name)])
    assert k("self_hp") == pytest.approx(0.875)
    assert k("self_weapon_ak47") == 1.0 and k("self_weapon_fists") == 0.0
    assert k("self_has_primary") == 1.0 and k("self_has_secondary") == 0.0 and k("self_gun_equipped") == 1.0
    assert k("self_clip") == pytest.approx(21 / 30) and k("self_reserve") == pytest.approx(45 / 90)
    assert k("self_inv_bandage") == pytest.approx(4 / 5) and k("self_inv_soda") == pytest.approx(1.0)
    assert k("self_time_frac") == pytest.approx(0.1)
    assert k("tm0_present") == 1.0 and k("tm0_hp") == 1.0
    assert k("tm0_dx") == pytest.approx((108.2 - 106.7) / 32.0) and k("tm0_dy") == pytest.approx((126.3 - 131.1) / 32.0)
    assert k("en0_present") == 1.0 and k("en1_present") == 0.0 and k("en2_present") == 0.0
    assert k("en0_dx") == pytest.approx((128.9 - 106.7) / 32.0) and k("en0_armed") == 1.0
    ux, uy = k("en0_ux"), k("en0_uy")
    assert math.hypot(ux, uy) == pytest.approx(1.0) and ux > 0.99
    assert k("loot0_present") == 1.0 and k("loot0_cls_heal") == 1.0 and k("loot0_count") == pytest.approx(4 / 30)
    assert k("loot1_present") == 0.0
    assert k("ob0_present") == 1.0 and k("ob0_collidable") == 1.0 and k("ob0_los_blocked") == 0.0
    assert k("bl0_present") == 1.0 and k("bl0_approaching") == 1.0  # bullet at +x moving -x
    assert k("gas_inside") == 1.0 and k("gas_active") == 0.0 and k("gas_shrinking") == 0.0
    assert k("alive_count") == 1.0 and k("alive_teams") == 1.0 and k("n_enemies_visible") == pytest.approx(1 / 3)
    # without a memory object the memory block reflects the currently visible enemies
    assert k("mem0_seen") == 1.0 and k("mem0_recency") == 1.0


def test_absent_entities_are_zero_masked(spec_obs):
    f = Featurizer()
    spec_obs["players"], spec_obs["loot"], spec_obs["bullets"], spec_obs["obstacles"] = [], [], [], []
    v = f.featurize(spec_obs, t=0.0)
    for name, (s, e) in f.layout.items():
        if name.startswith(("en", "loot", "bl", "ob", "mem")):
            assert not v[s:e].any(), name
    assert v[_idx(f, "tm0_present")] == 1.0


def test_determinism_and_no_mutation(spec_obs):
    f = Featurizer()
    before = copy.deepcopy(spec_obs)
    a, b = f.featurize(spec_obs, t=1.0), f.featurize(spec_obs, t=1.0)
    assert np.array_equal(a, b) and spec_obs == before


def test_egocentric_rotation_into_facing_frame(spec_obs):
    # facing +y with an enemy straight ahead (+y): rotated frame puts it at +x, dy ~ 0
    spec_obs["self"]["dir"] = {"x": 0.0, "y": 1.0}
    spec_obs["players"][0]["pos"] = {"x": 106.7, "y": 131.1 + 16.0}
    spec_obs["players"][0]["dir"] = {"x": 0.0, "y": -1.0}  # enemy facing us
    plain = Featurizer(FeaturizerConfig(rotate_to_facing=False))
    rot = Featurizer(FeaturizerConfig(rotate_to_facing=True))
    vp, vr = plain.featurize(spec_obs), rot.featurize(spec_obs)
    assert vp[_idx(plain, "en0_dx")] == pytest.approx(0.0, abs=1e-6) and vp[_idx(plain, "en0_dy")] == pytest.approx(0.5)
    assert vr[_idx(rot, "en0_dx")] == pytest.approx(0.5) and vr[_idx(rot, "en0_dy")] == pytest.approx(0.0, abs=1e-6)
    assert vr[_idx(rot, "en0_dir_x")] == pytest.approx(-1.0) and vr[_idx(rot, "en0_dir_y")] == pytest.approx(0.0, abs=1e-6)
    assert vr[_idx(rot, "self_dir_x")] == pytest.approx(1.0) and vr[_idx(rot, "self_dir_y")] == pytest.approx(0.0, abs=1e-6)
    assert vr[_idx(rot, "en0_dist")] == pytest.approx(vp[_idx(plain, "en0_dist")])  # distances are invariant
    # a 90 degree turn to face -x maps "+y ahead" to the agent's left (+y in the facing frame)
    spec_obs["self"]["dir"] = {"x": -1.0, "y": 0.0}
    vr2 = rot.featurize(spec_obs)
    assert vr2[_idx(rot, "en0_dx")] == pytest.approx(0.0, abs=1e-6) and vr2[_idx(rot, "en0_dy")] == pytest.approx(-0.5)


def test_memory_tracks_last_seen_enemy_with_decay(spec_obs):
    f = Featurizer(FeaturizerConfig(memory_decay_s=10.0))
    mem = f.new_memory(["team-b-0", "team-b-1"])
    v = f.featurize(spec_obs, t=1.0, memory=mem)
    assert v[_idx(f, "mem0_seen")] == 1.0 and v[_idx(f, "mem0_recency")] == 1.0
    assert v[_idx(f, "mem1_seen")] == 0.0  # team-b-1 never seen
    spec_obs["players"] = []  # enemy left the view; we moved 8 u towards where it was
    spec_obs["self"]["pos"] = {"x": 114.7, "y": 131.1}
    v = f.featurize(spec_obs, t=11.0, memory=mem)
    assert v[_idx(f, "en0_present")] == 0.0
    assert v[_idx(f, "mem0_seen")] == 1.0 and v[_idx(f, "mem0_recency")] == pytest.approx(math.exp(-1.0))
    assert v[_idx(f, "mem0_dx")] == pytest.approx((128.9 - 114.7) / 32.0)
    fresh = f.new_memory(["team-b-0", "team-b-1"])
    assert f.featurize(spec_obs, t=11.0, memory=fresh)[_idx(f, "mem0_seen")] == 0.0


def test_featurizes_mock_observations_for_all_agents():
    f = Featurizer()
    sim = FieldSim(0, "cpc-duo2v2-seed-0", {"controlled": [], "scripted": "chaser"})
    mems = {aid: f.new_memory([a for a in sim.players if sim.players[a].team != p.team]) for aid, p in sim.players.items()}
    seen_enemy = False
    while not sim.done:
        out = sim.step(10)
        for aid, obs in out["obs"].items():
            v = f.featurize(obs, out["t"], mems[aid])
            assert v.shape == (216,) and np.isfinite(v).all()
            assert np.abs(v).max() <= 3.0 + 1e-6
            seen_enemy = seen_enemy or v[_idx(f, "en0_present")] == 1.0
    assert seen_enemy
