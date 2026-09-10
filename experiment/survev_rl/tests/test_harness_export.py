"""S6 / M8: bridge episode -> common-schema EpisodeTrajectory -> metric vector.

Runs a real episode against the in-process mock bridge, so the whole path is exercised without
a game server. The load-bearing assertion is the cross-check: the metric vector computed from
the exported episode has to agree with the metrics the bridge itself reported, because those
come from the engine's own bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiment.core.harness_metrics import MetricOptions, compute_metrics
from experiment.core.schema_validation import validate_episode
from experiment.survev_rl.env import EnvConfig, make_vec_env
from experiment.survev_rl.eval import run_episodes
from experiment.survev_rl.harness_export import (
    MAX_VISIBLE_ALLIES,
    MAX_VISIBLE_ENEMIES,
    VECTOR_KEYS,
    add_final_metrics,
    to_episode_trajectory,
)


@pytest.fixture(scope="module")
def records(mock_server):
    """Two full episodes with observations for every agent (what the export needs)."""
    env = make_vec_env(
        EnvConfig(bridge_url=mock_server.url, n_envs=1, ticks=10, time_limit=8.0, scripted="chaser"),
    )
    rng = np.random.default_rng(7)

    def policy(obs):
        return rng.integers(0, env.action_space.nvec, size=(obs.shape[0], env.action_space.n_dims))

    try:
        return run_episodes(env, policy, 2, full_obs=True)
    finally:
        env.close()


@pytest.fixture(scope="module")
def episodes(records):
    return [add_final_metrics(e, compute_metrics(e)) for e in (to_episode_trajectory(r) for r in records)]


def test_export_passes_validate_episode(episodes):
    for episode in episodes:
        assert validate_episode(episode) == []


def test_every_agent_has_an_observation_and_an_action(episodes):
    for episode in episodes:
        for step in episode["steps"]:
            agent_ids = step["info"]["snapshot"]["agent_ids"]
            assert set(step["observations"]) == set(agent_ids)
            assert set(step["actions"]) == set(agent_ids)
            for agent_id in agent_ids:
                obs = step["observations"][agent_id]
                assert len(obs["vector"]) == len(obs["vector_keys"]) == len(VECTOR_KEYS)
                assert len(obs["visible_enemies"]) <= MAX_VISIBLE_ENEMIES
                assert len(obs["visible_allies"]) <= MAX_VISIBLE_ALLIES


def test_enemy_hp_is_absent_and_ally_hp_is_present(episodes):
    """Information-set parity: a client is never told an enemy's HP, but group status carries a
    teammate's. Omission, not a placeholder, is how the export says "unknown"."""
    seen_ally_hp = False
    for episode in episodes:
        for step in episode["steps"]:
            for obs in step["observations"].values():
                for enemy in obs["visible_enemies"]:
                    assert "hp" not in enemy
                for ally in obs["visible_allies"]:
                    assert isinstance(ally["hp"], float)
                    seen_ally_hp = True
    assert seen_ally_hp, "the duo scenario should have produced at least one visible ally"


def test_metrics_agree_with_the_bridge(records, episodes):
    """combat / survival against the engine's own numbers (M8)."""
    for record, episode in zip(records, episodes):
        for agent_id, bridge in record["metrics"].items():
            metrics = episode["final_metrics"][agent_id]
            combat, survival = metrics["combat"], metrics["survival"]
            assert combat["damageDealt"] == pytest.approx(bridge["damage_dealt"], abs=0.6)
            assert combat["damageTaken"] == pytest.approx(bridge["damage_taken"], abs=0.6)
            assert combat["kills"] == pytest.approx(bridge["kills"], abs=0.5)
            # a step of slack: survival_time is counted in ticks, aliveTime in policy steps
            assert survival["aliveTime"] == pytest.approx(bridge["survival_time"], abs=0.2)
            assert survival["hpEnd"] == pytest.approx(bridge["hp_end"], abs=0.6)
            assert survival["aliveAtEnd"] == bridge["alive_at_end"]


def test_duo_cooperation_is_applicable_and_bounded(episodes):
    for episode in episodes:
        for metrics in episode["final_metrics"].values():
            coop = metrics["cooperation"]
            assert coop["applicable"] is True
            assert 0.0 <= coop["teammateResponseRate"] <= 1.0
            assert 0.0 <= coop["isolationRate"] <= 1.0
            assert coop["teammateUnderPressureResponses"] <= coop["teammateUnderPressureEvents"]
            assert coop["avgAllyDistance"] is None or coop["avgAllyDistance"] >= 0.0


def test_scripted_actions_are_reconstructed_not_blank(episodes):
    """The server decides for the scripted duo, so their actions are read back off the world."""
    scripted = [a for a in episodes[0]["steps"][0]["actions"] if a.startswith("team-b")]
    assert scripted, "the scenario should have scripted agents"
    aims = [
        step["actions"][agent]["action"]
        for step in episodes[0]["steps"]
        for agent in scripted
    ]
    # an aim direction is a unit vector (or zero before the bot has faced anywhere)
    for body in aims:
        length = (body["aim_x"] ** 2 + body["aim_y"] ** 2) ** 0.5
        assert length == pytest.approx(0.0, abs=1e-6) or length == pytest.approx(1.0, abs=1e-6)
    assert any(abs(b["aim_x"]) + abs(b["aim_y"]) > 0 for b in aims), "chasers should aim somewhere"
    assert any(abs(b["move_x"]) + abs(b["move_y"]) > 0 for b in aims), "chasers should move"


def test_snapshot_is_the_post_step_state(episodes):
    """The observation is what the action was chosen from; the snapshot is what it led to."""
    episode = episodes[0]
    for previous, following in zip(episode["steps"], episode["steps"][1:]):
        for agent_id, obs in following["observations"].items():
            snapshot = previous["info"]["snapshot"]["agents"][agent_id]
            assert snapshot["hp"] == pytest.approx(obs["self"]["hp"], abs=1e-6)
            assert snapshot["position"]["x"] == pytest.approx(obs["self"]["position"]["x"], abs=1e-6)


def test_solo_marks_cooperation_not_applicable():
    """Schema rule: with no allies the cooperation group must say applicable = false."""
    episode = {
        "schema_version": "cpc-common-v0",
        "config": {"schema_version": "cpc-common-v0", "mode": "solo", "team_count": 2,
                   "players_per_team": 1, "max_steps": 1},
        "steps": [{
            "schema_version": "cpc-common-v0",
            "observations": {}, "actions": {},
            "info": {"snapshot": {
                "schema_version": "cpc-common-v0",
                "agent_ids": ["a", "b"], "team_ids": ["t0", "t1"],
                "agent_team_map": {"a": "t0", "b": "t1"},
                "agents": {"a": {"position": {"x": 0.0, "y": 0.0}, "hp": 100.0, "alive": True},
                           "b": {"position": {"x": 5.0, "y": 0.0}, "hp": 100.0, "alive": True}},
                "events": [],
            }},
        }],
    }
    metrics = compute_metrics(episode)
    assert metrics["a"]["cooperation"] == {"applicable": False}
    assert metrics["b"]["cooperation"] == {"applicable": False}


def test_isolation_and_pressure_thresholds_are_options():
    """The two cooperation dials, checked on a hand-built two-step duo episode."""
    def step(ally_hp: float, ally_distance: float, enemy_distance: float) -> dict:
        return {
            "schema_version": "cpc-common-v0",
            "observations": {}, "actions": {},
            "info": {"snapshot": {
                "schema_version": "cpc-common-v0",
                "agent_ids": ["me", "mate", "foe"], "team_ids": ["us", "them"],
                "agent_team_map": {"me": "us", "mate": "us", "foe": "them"},
                "agents": {
                    "me": {"position": {"x": 0.0, "y": 0.0}, "hp": 100.0, "alive": True},
                    "mate": {"position": {"x": ally_distance, "y": 0.0}, "hp": ally_hp, "alive": True},
                    "foe": {"position": {"x": ally_distance + enemy_distance, "y": 0.0},
                            "hp": 100.0, "alive": True},
                },
                "events": [],
            }},
        }

    episode = {
        "schema_version": "cpc-common-v0",
        "config": {"schema_version": "cpc-common-v0", "mode": "duo", "team_count": 2,
                   "players_per_team": 2, "max_steps": 2},
        # step 0: partner hurt with an enemy on top of it, and far enough away to be isolated
        # step 1: partner healthy and close
        "steps": [step(20.0, 60.0, 5.0), step(100.0, 5.0, 100.0)],
    }
    opts = MetricOptions(isolation_distance=48.0, support_distance=18.0, threat_distance=30.0)
    coop = compute_metrics(episode, opts)["me"]["cooperation"]
    assert coop["teammateUnderPressureEvents"] == 1
    assert coop["isolatedSteps"] == 1 and coop["isolationRate"] == pytest.approx(0.5)
    assert coop["formationGoodSteps"] == 1
    assert coop["avgAllyDistance"] == pytest.approx(32.5)
    # the agent closed from 60 u to 5 u, i.e. into support range, which counts as a response
    assert coop["teammateUnderPressureResponses"] == 1
