from __future__ import annotations

import numpy as np
import pytest

from experiment.survev_rl.actions import ActionSpace
from experiment.survev_rl.env import EnvConfig, SurvevSingleEnv, SurvevVecEnv, make_vec_env
from experiment.survev_rl.featurizer import FeaturizerConfig
from experiment.survev_rl.protocol import METRIC_KEYS
from experiment.survev_rl.rewards import RewardConfig


def test_vec_env_end_to_end_with_auto_reset(mock_server):
    env = SurvevVecEnv(mock_server.url, n_envs=2, controlled=("team-a-0", "team-a-1"), ticks=10, base_seed=7)
    try:
        assert env.n_rows == 4 and env.obs_dim == 216 + 2
        assert env.vector_keys[-2:] == ["agent_is_team-a-0", "agent_is_team-a-1"]
        obs = env.reset()
        assert obs.shape == (4, 218) and obs.dtype == np.float32
        # agent one-hot appended per row
        assert obs[0, 216:].tolist() == [1.0, 0.0] and obs[1, 216:].tolist() == [0.0, 1.0]
        assert obs[2, 216:].tolist() == [1.0, 0.0] and obs[3, 216:].tolist() == [0.0, 1.0]
        assert env.episode_index == [1, 1]
        seeds_seen = {env._last_seed[0], env._last_seed[1]}
        assert seeds_seen == {"cpc-duo2v2-seed-7", "cpc-duo2v2-seed-8"}
        rng = np.random.default_rng(0)
        finished = 0
        steps = 0
        while finished < 2 and steps < 2000:
            actions = np.stack([env.action_space.sample(rng) for _ in range(env.n_rows)])
            obs, rewards, dones, infos = env.step(actions)
            steps += 1
            assert obs.shape == (4, 218) and rewards.shape == (4,) and dones.shape == (4,) and len(infos) == 4
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
            for row, info in enumerate(infos):
                assert info["agent_id"] == env.rows[row][1] and info["env_index"] == env.rows[row][0]
                assert "active" in info and "reward_components" in info
            for i in range(2):
                r0, r1 = 2 * i, 2 * i + 1
                assert dones[r0] == dones[r1]  # both rows of an env end together
                if dones[r0]:
                    finished += 1
                    for row in (r0, r1):
                        info = infos[row]
                        assert set(info["episode_metrics"]) == set(METRIC_KEYS)
                        assert info["reason"] in ("elimination", "time_limit")
                        assert info["episode_length"] >= 1 and isinstance(info["episode_return"], float)
                        assert info["terminal_observation"].shape == (218,)
                    # auto-reset: the returned rows already belong to a fresh episode (t = 0)
                    time_frac = env.featurizer.vector_keys.index("self_time_frac")
                    assert obs[r0, time_frac] == 0.0 and obs[r1, time_frac] == 0.0
                    assert env.last_messages[i].t == 0.0 and env.step_messages[i].done
        assert finished >= 2
        assert sum(env.episode_index) >= 4
    finally:
        env.close()


def test_single_env_wrapper_and_rewards(mock_server):
    env = SurvevSingleEnv(mock_server.url, controlled="team-a-0", scripted="idle", time_limit=1.0, ticks=10)
    try:
        obs = env.reset()
        assert obs.shape == (216,)
        total, steps, done = 0.0, 0, False
        while not done:
            obs, r, done, info = env.step([4, 0, 0, 0])  # walk right
            total += r
            steps += 1
        assert steps == 10 and total == pytest.approx(0.10)  # alive bonus only, nothing else happens
        assert info["reason"] == "time_limit" and info["episode_metrics"]["survival_time"] == pytest.approx(1.0)
        assert env.last_message is not None and env.last_message.t == 0.0  # already reset
    finally:
        env.close()


def test_make_vec_env_from_configs_and_action_validation(mock_server):
    env = make_vec_env(
        EnvConfig(bridge_url=mock_server.url, n_envs=1, controlled=("team-a-1",), scripted="idle", time_limit=2.0),
        FeaturizerConfig(memory=False, max_loot=2),
        ActionSpace(assist=False),
        RewardConfig(team_mix=0.0),
    )
    try:
        assert env.obs_dim == env.featurizer.size and env.agent_onehot_dim == 0
        env.reset()
        with pytest.raises(ValueError):
            env.step(np.zeros((2, 4), dtype=np.int64))
        obs, r, d, infos = env.step(np.array([[0, 0, 0, 0]]))
        assert obs.shape == (1, env.obs_dim) and not d[0]
    finally:
        env.close()
    with SurvevVecEnv(mock_server.url, n_envs=1) as env2:
        with pytest.raises(RuntimeError, match="reset"):
            env2.step(np.zeros((2, 4), dtype=np.int64))
