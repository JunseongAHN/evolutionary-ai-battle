from __future__ import annotations

import copy

import pytest

from experiment.survev_rl.protocol import ObsInfo
from experiment.survev_rl.rewards import (
    RewardConfig,
    compute_reward_breakdown,
    compute_rewards,
    hp_delta_corrected,
)


def _agent(aid: str, team: str, hp: float = 100.0, downed: bool = False, dead: bool = False, partner=None):
    teammates = [] if partner is None else [partner]
    return {
        "self": {"id": aid, "team": team, "hp": hp, "downed": downed, "dead": dead,
                 "pos": {"x": 0.0, "y": 0.0}, "dir": {"x": 1.0, "y": 0.0}},
        "teammates": teammates,
        "players": [], "loot": [], "obstacles": [], "bullets": [], "dead_bodies": [],
        "gas": {"mode": 0, "rad": 196, "pos": {"x": 132, "y": 132}, "rad_new": 196, "pos_new": {"x": 132, "y": 132}},
        "alive_count": 4, "alive_teams": 2,
    }


def _tm(aid: str, hp: float = 100.0, downed: bool = False, dead: bool = False):
    return {"id": aid, "pos": {"x": 1.0, "y": 0.0}, "dist": 1.0, "hp": hp, "downed": downed, "dead": dead}


def _obs(a0_hp=100.0, a1_hp=100.0, a0_downed=False, a0_dead=False, a1_downed=False, a1_dead=False):
    return {
        "team-a-0": _agent("team-a-0", "team-a", a0_hp, a0_downed, a0_dead, _tm("team-a-1", a1_hp, a1_downed, a1_dead)),
        "team-a-1": _agent("team-a-1", "team-a", a1_hp, a1_downed, a1_dead, _tm("team-a-0", a0_hp, a0_downed, a0_dead)),
        "team-b-0": _agent("team-b-0", "team-b"),
        "team-b-1": _agent("team-b-1", "team-b"),
    }


CONTROLLED = ("team-a-0", "team-a-1")
RUNNING = ObsInfo()


def test_defaults_match_user_decision():
    cfg = RewardConfig()
    assert cfg.alive_per_step == 0.01 and cfg.hp_delta == 0.01 and cfg.damage_dealt == 0.02
    assert cfg.team_win == 1.0 and cfg.death == -1.0
    assert cfg.damage_taken == cfg.kill == cfg.partner_hp_delta == cfg.partner_alive_per_step == cfg.cover_bonus == 0.0
    assert cfg.time_penalty_after_s == 0.0 and cfg.team_mix == 0.0
    with pytest.raises(ValueError):
        RewardConfig(team_mix=1.5)


def test_survival_and_hp_delta():
    cfg = RewardConfig()
    r = compute_rewards(_obs(), _obs(), [], RUNNING, cfg, CONTROLLED)
    assert r == {"team-a-0": pytest.approx(0.01), "team-a-1": pytest.approx(0.01)}
    r = compute_rewards(_obs(), _obs(a0_hp=87.0), [], RUNNING, cfg, CONTROLLED)
    assert r["team-a-0"] == pytest.approx(0.01 - 0.13) and r["team-a-1"] == pytest.approx(0.01)
    r = compute_rewards(_obs(a0_hp=50.0), _obs(a0_hp=65.0), [], RUNNING, cfg, CONTROLLED)  # bandage
    assert r["team-a-0"] == pytest.approx(0.01 + 0.15)


def test_damage_dealt_kill_and_death_events():
    cfg = RewardConfig(kill=0.5)
    events = [
        {"type": "damage", "t": 1.0, "agent": "team-b-0", "source": "team-a-0", "weapon": "ak47", "amount": 13.0},
        {"type": "damage", "t": 1.0, "agent": "team-b-0", "source": "team-a-0", "weapon": "ak47", "amount": 7.0},
        {"type": "damage", "t": 1.0, "agent": "team-a-1", "source": "team-b-1", "weapon": "mp5", "amount": 9.0},
        {"type": "kill", "t": 1.0, "agent": "team-b-0", "source": "team-a-0"},
    ]
    b = compute_reward_breakdown(_obs(), _obs(a1_hp=91.0), events, RUNNING, cfg, CONTROLLED)
    assert b.components["team-a-0"]["damage_dealt"] == pytest.approx(0.4)
    assert b.components["team-a-0"]["kill"] == pytest.approx(0.5)
    assert b.components["team-a-1"]["damage_dealt"] == 0.0 and b.components["team-a-1"]["hp"] == pytest.approx(-0.09)
    assert b.totals["team-a-0"] == pytest.approx(0.01 + 0.4 + 0.5)
    # damage_taken is a separate signed weight (default 0)
    b2 = compute_reward_breakdown(_obs(), _obs(a1_hp=91.0), events, RUNNING, RewardConfig(damage_taken=-0.02), CONTROLLED)
    assert b2.components["team-a-1"]["damage_taken"] == pytest.approx(-0.18)
    # death: the agent's own kill event
    death = [{"type": "kill", "t": 2.0, "agent": "team-a-0", "source": "team-b-0"}]
    r = compute_rewards(_obs(a0_hp=20.0), _obs(a0_hp=0.0, a0_dead=True), death, RUNNING, cfg, CONTROLLED)
    assert r["team-a-0"] == pytest.approx(-1.0 - 0.2)  # death + remaining HP lost, no alive bonus
    assert r["team-a-1"] == pytest.approx(0.01)


def test_down_and_revive_transitions_use_effective_hp():
    prev = {"hp": 40.0, "downed": False, "dead": False}
    assert hp_delta_corrected(prev, {"hp": 100.0, "downed": True, "dead": False}) == pytest.approx(-40.0)
    assert hp_delta_corrected({"hp": 100.0, "downed": True, "dead": False}, {"hp": 96.0, "downed": True, "dead": False}) == 0.0
    assert hp_delta_corrected({"hp": 96.0, "downed": True, "dead": False}, {"hp": 0.0, "downed": True, "dead": True}) == 0.0
    assert hp_delta_corrected({"hp": 80.0, "downed": True, "dead": False}, {"hp": 24.0, "downed": False, "dead": False}) == pytest.approx(24.0)
    assert hp_delta_corrected({"hp": 0.0, "downed": False, "dead": True}, {"hp": 0.0, "downed": False, "dead": True}) == 0.0
    cfg = RewardConfig(partner_hp_delta=0.01, partner_alive_per_step=0.005)
    # partner gets downed: the reviver-to-be loses partner hp, still gets partner_alive (downed counts alive)
    b = compute_reward_breakdown(_obs(a1_hp=30.0), _obs(a1_hp=100.0, a1_downed=True), [], RUNNING, cfg, CONTROLLED)
    assert b.components["team-a-0"]["partner_hp"] == pytest.approx(-0.30)
    assert b.components["team-a-0"]["partner_alive"] == pytest.approx(0.005)
    assert b.components["team-a-1"]["hp"] == pytest.approx(-0.30)
    # revive: partner effective HP 0 -> 24
    b = compute_reward_breakdown(_obs(a1_hp=90.0, a1_downed=True), _obs(a1_hp=24.0), [], RUNNING, cfg, CONTROLLED)
    assert b.components["team-a-0"]["partner_hp"] == pytest.approx(0.24)
    assert b.components["team-a-1"]["hp"] == pytest.approx(0.24)


def test_team_win_only_at_done_for_winner():
    cfg = RewardConfig()
    done_win = ObsInfo(alive_teams=1, winner_team="team-a", reason="elimination")
    done_loss = ObsInfo(alive_teams=1, winner_team="team-b", reason="elimination")
    timeout = ObsInfo(alive_teams=2, winner_team=None, reason="time_limit")
    r = compute_rewards(_obs(), _obs(), [], done_win, cfg, CONTROLLED)
    assert r["team-a-0"] == pytest.approx(1.01) and r["team-a-1"] == pytest.approx(1.01)
    r = compute_rewards(_obs(), _obs(), [], done_loss, cfg, CONTROLLED)
    assert r["team-a-0"] == pytest.approx(0.01)
    r = compute_rewards(_obs(), _obs(), [], timeout, cfg, CONTROLLED)
    assert r["team-a-0"] == pytest.approx(0.01)
    # plain-dict info works too (raw JSON)
    r = compute_rewards(_obs(), _obs(), [], {"winner_team": "team-a", "reason": "elimination"}, cfg, CONTROLLED)
    assert r["team-a-0"] == pytest.approx(1.01)


def test_team_mix_blends_toward_team_mean():
    events = [{"type": "damage", "t": 1.0, "agent": "team-b-0", "source": "team-a-0", "weapon": "ak47", "amount": 50.0}]
    solo = compute_rewards(_obs(), _obs(), events, RUNNING, RewardConfig(team_mix=0.0), CONTROLLED)
    assert solo["team-a-0"] == pytest.approx(1.01) and solo["team-a-1"] == pytest.approx(0.01)
    shared = compute_rewards(_obs(), _obs(), events, RUNNING, RewardConfig(team_mix=1.0), CONTROLLED)
    assert shared["team-a-0"] == pytest.approx(0.51) and shared["team-a-1"] == pytest.approx(0.51)
    half = compute_rewards(_obs(), _obs(), events, RUNNING, RewardConfig(team_mix=0.5), CONTROLLED)
    assert half["team-a-0"] == pytest.approx(0.76) and half["team-a-1"] == pytest.approx(0.26)
    # mixing never crosses teams: a controlled team-b agent keeps its own reward
    both = compute_rewards(_obs(), _obs(), events, RUNNING, RewardConfig(team_mix=1.0), ("team-a-0", "team-b-0"))
    assert both["team-b-0"] == pytest.approx(0.01)


def test_time_penalty_after_threshold():
    cfg = RewardConfig(time_penalty_after_s=30.0, time_penalty_per_step=-0.02)
    assert compute_rewards(_obs(), _obs(), [], RUNNING, cfg, CONTROLLED, t=10.0)["team-a-0"] == pytest.approx(0.01)
    assert compute_rewards(_obs(), _obs(), [], RUNNING, cfg, CONTROLLED, t=31.0)["team-a-0"] == pytest.approx(-0.01)
    obs = _obs(a0_hp=0.0, a0_dead=True)
    assert compute_rewards(obs, copy.deepcopy(obs), [], RUNNING, cfg, CONTROLLED, t=31.0)["team-a-0"] == 0.0


def _placed(obs, positions):
    """Copy of ``obs`` with agents moved to the given world positions."""
    out = copy.deepcopy(obs)
    for aid, (x, y) in positions.items():
        out[aid]["self"]["pos"] = {"x": float(x), "y": float(y)}
    return out


POINT = (132.0, 132.0)
POINT_CFG = RewardConfig(alive_per_step=0.0, hp_delta=0.0, damage_dealt=0.0, death=0.0, team_win=0.0,
                         goal_progress=0.05, goal_hold=0.01, goal_radius=6.0, enemy_at_goal=-0.02, enemy_goal_radius=30.0)


def test_goal_terms_are_inactive_without_a_goal_or_weights():
    prev = _placed(_obs(), {"team-a-0": (100, 132)})
    cur = _placed(_obs(), {"team-a-0": (110, 132)})
    # default config: no waypoint weights -> identical to before even when a goal is passed
    b = compute_reward_breakdown(prev, cur, [], RUNNING, RewardConfig(), CONTROLLED, goal=POINT)
    assert b.components["team-a-0"]["goal_progress"] == 0.0 and b.components["team-a-0"]["enemy_goal"] == 0.0
    # waypoint weights but no goal passed -> terms stay 0
    b = compute_reward_breakdown(prev, cur, [], RUNNING, POINT_CFG, CONTROLLED)
    assert b.totals["team-a-0"] == 0.0


def test_goal_progress_hold_and_enemy_pressure():
    far = {"team-b-0": (300, 300), "team-b-1": (300, 300)}
    prev = _placed(_obs(), {"team-a-0": (100, 132), "team-a-1": (100, 140), **far})
    cur = _placed(_obs(), {"team-a-0": (110, 132), "team-a-1": (100, 140), **far})
    b = compute_reward_breakdown(prev, cur, [], RUNNING, POINT_CFG, CONTROLLED, goal=POINT)
    a0, a1 = b.components["team-a-0"], b.components["team-a-1"]
    assert a0["goal_progress"] == pytest.approx(0.05 * 10.0)  # 32 u -> 22 u from the point
    assert a1["goal_progress"] == pytest.approx(0.0) and a0["goal_hold"] == 0.0 == a1["goal_hold"]
    assert a0["enemy_goal"] == 0.0  # enemies far from the point

    # standing inside goal_radius pays the hold bonus; moving away is negative progress
    on_point = _placed(_obs(), {"team-a-0": (135, 132), **far})
    b = compute_reward_breakdown(cur, on_point, [], RUNNING, POINT_CFG, CONTROLLED, goal=POINT)
    assert b.components["team-a-0"]["goal_hold"] == pytest.approx(0.01)
    assert b.components["team-a-0"]["goal_progress"] == pytest.approx(0.05 * (22.0 - 3.0))
    back = _placed(_obs(), {"team-a-0": (120, 132), **far})
    b = compute_reward_breakdown(on_point, back, [], RUNNING, POINT_CFG, CONTROLLED, goal=POINT)
    assert b.components["team-a-0"]["goal_progress"] == pytest.approx(-0.05 * 9.0)

    # one enemy 15 u from the point (pressure 0.5), one on it (1.0): both controlled agents pay -0.02 x 1.5
    near = _placed(_obs(), {"team-a-0": (135, 132), "team-b-0": (147, 132), "team-b-1": (132, 132)})
    b = compute_reward_breakdown(on_point, near, [], RUNNING, POINT_CFG, CONTROLLED, goal=POINT)
    assert b.components["team-a-0"]["enemy_goal"] == pytest.approx(-0.02 * 1.5)
    assert b.components["team-a-1"]["enemy_goal"] == pytest.approx(-0.02 * 1.5)
    # a dead enemy exerts no pressure; the penalty is what killing it removes
    near_dead = copy.deepcopy(near)
    near_dead["team-b-1"]["self"]["dead"] = True
    b = compute_reward_breakdown(on_point, near_dead, [], RUNNING, POINT_CFG, CONTROLLED, goal=POINT)
    assert b.components["team-a-0"]["enemy_goal"] == pytest.approx(-0.02 * 0.5)


def test_goal_terms_stop_for_downed_and_dead_agents():
    far = {"team-b-0": (300, 300), "team-b-1": (300, 300)}
    prev = _placed(_obs(), {"team-a-0": (120, 132), **far})
    downed = _placed(_obs(a0_downed=True), {"team-a-0": (130, 132), **far})
    b = compute_reward_breakdown(prev, downed, [], RUNNING, POINT_CFG, CONTROLLED, goal=POINT)
    assert b.components["team-a-0"]["goal_progress"] == 0.0 and b.components["team-a-0"]["goal_hold"] == 0.0
    dead = _placed(_obs(a0_dead=True), {"team-a-0": (132, 132), "team-b-0": (132, 132), "team-b-1": (300, 300)})
    b = compute_reward_breakdown(downed, dead, [], RUNNING, POINT_CFG, CONTROLLED, goal=POINT)
    assert b.totals["team-a-0"] == 0.0  # dead: no progress, no hold, no pressure penalty
    assert b.components["team-a-1"]["enemy_goal"] == pytest.approx(-0.02)
