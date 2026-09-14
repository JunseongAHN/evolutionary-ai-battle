"""The intent the planner commits to, as an input to the controller.

System 2 decides *what the fight is* (push, hold an angle, trade, break off); the controller decides
how to stand, when to peek and where to reload. So the intent rides in the observation as a one-hot
and the action space is untouched — one controller, told what it is supposed to be doing.

During training no planner is called: an intent is drawn per episode and held, which is both fast and
the only way the controller sees every intent often enough to learn all of them.
"""

from __future__ import annotations

import numpy as np

from experiment.survev_rl.env import EnvConfig, make_vec_env
from experiment.survev_rl.featurizer import Featurizer, FeaturizerConfig

INTENTS = ("push", "hold_angle", "trade", "retreat")


def _idx(f: Featurizer, key: str) -> int:
    return f.vector_keys.index(key)


def test_intents_are_off_by_default():
    f = Featurizer()
    assert f.size == 216
    assert not any(k.startswith("intent_") for k in f.vector_keys)


def test_the_vector_ends_with_the_intent(spec_obs):
    f = Featurizer(FeaturizerConfig(intents=INTENTS))
    assert f.size == 216 + len(INTENTS)
    assert f.layout["intent"] == (216, 216 + len(INTENTS))

    v = f.featurize(spec_obs, t=6.0, intent="trade")
    assert v.shape == (f.size,) and np.isfinite(v).all()
    assert v[_idx(f, "intent_trade")] == 1.0
    assert v[_idx(f, "intent_push")] == 0.0
    # the first 216 values are what they were: an intent must not move an existing slot
    assert v[:216].tolist() == f.featurize(spec_obs, t=6.0, intent="push")[:216].tolist()


def test_no_intent_and_an_unknown_one_read_as_all_zero(spec_obs):
    f = Featurizer(FeaturizerConfig(intents=INTENTS))
    for intent in (None, "sprint"):
        v = f.featurize(spec_obs, t=6.0, intent=intent)
        assert v[216:].tolist() == [0.0] * len(INTENTS)


def test_the_env_draws_one_intent_per_episode(mock_server):
    cfg = EnvConfig(bridge_url=mock_server.url, n_envs=2, controlled=("team-a-0",), intents=INTENTS,
                    base_seed=5)
    env = make_vec_env(cfg, FeaturizerConfig(intents=INTENTS, time_limit=cfg.time_limit))
    try:
        obs = env.reset()
        assert all(i in INTENTS for i in env.env_intents)
        # the drawn intent is what the rows of that env are told
        for row, (env_index, _agent) in enumerate(env.rows):
            one_hot = obs[row, 216:216 + len(INTENTS)]
            assert one_hot.sum() == 1.0
            assert INTENTS[int(np.argmax(one_hot))] == env.env_intents[env_index]
    finally:
        env.close()


def test_the_draw_is_reproducible_from_the_base_seed(mock_server):
    def run() -> list[str | None]:
        cfg = EnvConfig(bridge_url=mock_server.url, n_envs=3, controlled=("team-a-0",),
                        intents=INTENTS, base_seed=11)
        env = make_vec_env(cfg, FeaturizerConfig(intents=INTENTS, time_limit=cfg.time_limit))
        try:
            env.reset()
            return list(env.env_intents)
        finally:
            env.close()

    assert run() == run()
