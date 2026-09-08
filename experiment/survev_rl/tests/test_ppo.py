from __future__ import annotations

import csv
import json
import math

import pytest

torch = pytest.importorskip("torch")

from experiment.survev_rl.env import SurvevVecEnv  # noqa: E402
from experiment.survev_rl.ppo import ActorCritic, PPOConfig, load_checkpoint, save_checkpoint, train  # noqa: E402


def test_actor_critic_shapes_and_logprobs():
    torch.manual_seed(0)
    agent = ActorCritic(obs_dim=12, nvec=(9, 16, 2, 2), hidden_dim=32)
    obs = torch.randn(5, 12)
    action, logp, ent, value = agent.get_action_and_value(obs)
    assert action.shape == (5, 4) and logp.shape == (5,) and ent.shape == (5,) and value.shape == (5,)
    assert (action[:, 0] < 9).all() and (action[:, 1] < 16).all() and (action[:, 2] < 2).all()
    _, logp2, _, _ = agent.get_action_and_value(obs, action)
    assert torch.allclose(logp, logp2)
    logits, v = agent(obs)
    assert logits.shape == (5, 29) and v.shape == (5, 1)
    assert [l.shape[-1] for l in agent.split_logits(logits)] == [9, 16, 2, 2]
    det, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
    det2, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
    assert torch.equal(det, det2)
    # a fresh policy is near-uniform: entropy close to the maximum
    assert ent.mean().item() == pytest.approx(math.log(9) + math.log(16) + 2 * math.log(2), abs=0.05)


def test_ppo_smoke_on_mock(mock_server, tmp_path):
    env = SurvevVecEnv(mock_server.url, n_envs=2, controlled=("team-a-0", "team-a-1"), ticks=10, base_seed=100)
    cfg = PPOConfig(total_steps=4 * 16 * 3, rollout_steps=16, num_minibatches=2, update_epochs=2,
                    hidden_dim=64, seed=1, device="cpu", checkpoint_every=1)
    try:
        row = train(cfg, env, tmp_path / "run", meta={"note": "smoke"}, progress=False)
    finally:
        env.close()
    assert row["update"] == 3 and row["global_step"] == 4 * 16 * 3
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        assert math.isfinite(row[key]), key
    assert row["entropy"] > 0
    assert (tmp_path / "run" / "config.json").exists()
    with (tmp_path / "run" / "progress.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and float(rows[-1]["global_step"]) == 192
    lines = (tmp_path / "run" / "log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["update"] == 1
    # checkpoint round trip reproduces the policy exactly
    agent, ckpt = load_checkpoint(tmp_path / "run" / "checkpoint_final.pt", "cpu")
    assert ckpt["obs_dim"] == 218 and ckpt["nvec"] == [9, 16, 2, 2] and ckpt["meta"]["note"] == "smoke"
    assert ckpt["ppo_config"]["hidden_dim"] == 64 and ckpt["global_step"] == 192
    latest, _ = load_checkpoint(tmp_path / "run" / "checkpoint_latest.pt", "cpu")
    x = torch.randn(3, 218)
    with torch.no_grad():
        assert torch.allclose(agent(x)[0], latest(x)[0])
    # save/load of a freshly built agent
    other = ActorCritic(218, (9, 16, 2, 2), 64)
    path = save_checkpoint(tmp_path / "again.pt", other, None, cfg, global_step=5, update=1)
    reloaded, meta = load_checkpoint(path)
    with torch.no_grad():
        assert torch.allclose(other(x)[1], reloaded(x)[1]) and meta["global_step"] == 5


def test_resume_continues_step_count(mock_server, tmp_path):
    env = SurvevVecEnv(mock_server.url, n_envs=1, controlled=("team-a-0",), ticks=10, scripted="idle", time_limit=2.0)
    cfg = PPOConfig(total_steps=2 * 8, rollout_steps=8, num_minibatches=1, update_epochs=1, hidden_dim=16,
                    seed=0, device="cpu", checkpoint_every=1)
    try:
        row = train(cfg, env, tmp_path / "a", progress=False)
        assert row["global_step"] == 16
        cfg2 = PPOConfig(**{**cfg.to_dict(), "total_steps": 4 * 8})
        row2 = train(cfg2, env, tmp_path / "b", resume=tmp_path / "a" / "checkpoint_final.pt", progress=False)
        assert row2["global_step"] == 32 and row2["update"] == 4
    finally:
        env.close()
