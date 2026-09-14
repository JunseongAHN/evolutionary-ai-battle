"""The cover features: off by default, and when on they carry what the server measured.

`los` and `rays` are the first features that need a map with obstacles, so they are opt-in — an old
checkpoint keeps its exact 216-value layout. When they are on, the vector has to match what
``describe()`` promises, or a policy silently reads the wrong slot.
"""

from __future__ import annotations

import copy

import numpy as np

from experiment.survev_rl.featurizer import Featurizer, FeaturizerConfig


def _idx(f: Featurizer, key: str) -> int:
    return f.vector_keys.index(key)


def test_cover_features_are_off_by_default():
    f = Featurizer()
    assert f.size == 216
    assert not any(k.startswith(("los_en", "ray")) for k in f.vector_keys)


def test_layout_grows_by_exactly_the_new_blocks():
    f = Featurizer(FeaturizerConfig(los=True, rays=16))
    assert f.size == 216 + 3 + 16  # one flag per enemy slot, one reading per direction
    assert len(set(f.vector_keys)) == f.size
    assert f.layout["los"] == (216, 219) and f.layout["rays"] == (219, 235)


def test_values_come_from_the_observation(spec_obs):
    obs = copy.deepcopy(spec_obs)
    obs["players"][0]["los_blocked"] = True
    obs["obstacles"][0]["blocks_los"] = True
    obs["obstacles"][0]["cover_score"] = 1.0
    obs["rays"] = [48.0] * 16
    obs["rays"][0] = 8.0

    f = Featurizer(FeaturizerConfig(los=True, rays=16))
    v = f.featurize(obs, t=6.0)
    assert v.shape == (f.size,) and v.dtype == np.float32 and np.isfinite(v).all()
    assert v[_idx(f, "ob0_los_blocked")] == 1.0
    assert v[_idx(f, "ob0_cover_score")] == 1.0
    assert v[_idx(f, "los_en0")] == 1.0
    assert v[_idx(f, "los_en1")] == 0.0  # no second enemy in view
    assert v[_idx(f, "ray0")] == 8.0 / 32.0
    assert v[_idx(f, "ray1")] == 48.0 / 32.0


def test_an_observation_without_the_fields_reads_as_clear(spec_obs):
    f = Featurizer(FeaturizerConfig(los=True, rays=16))
    v = f.featurize(spec_obs, t=6.0)
    assert v[_idx(f, "los_en0")] == 0.0
    assert all(v[_idx(f, f"ray{i}")] == 0.0 for i in range(16))
