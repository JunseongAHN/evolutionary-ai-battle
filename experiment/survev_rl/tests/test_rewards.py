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


def test_capture_is_team_credited():
    cfg = RewardConfig(alive_per_step=0.0, hp_delta=0.0, damage_dealt=0.0, death=0.0, team_win=0.0,
                       capture=1.0, enemy_capture=-0.5)
    ours = {"type": "capture", "t": 3.0, "agent": "team-a-1", "team": "team-a", "index": 0,
            "pos": {"x": 150.0, "y": 118.0}, "time_to_capture": 3.0}
    theirs = {"type": "capture", "t": 7.0, "agent": "team-b-0", "team": "team-b", "index": 1,
              "pos": {"x": 110.0, "y": 140.0}, "time_to_capture": 4.0}
    b = compute_reward_breakdown(_obs(), _obs(), [ours], RUNNING, cfg, CONTROLLED)
    # both members of team-a get the point, whoever touched it
    assert b.components["team-a-0"]["capture"] == 1.0 and b.components["team-a-1"]["capture"] == 1.0
    assert b.totals == {"team-a-0": 1.0, "team-a-1": 1.0}
    b = compute_reward_breakdown(_obs(), _obs(), [theirs], RUNNING, cfg, CONTROLLED)
    assert b.components["team-a-0"]["enemy_capture"] == -0.5 and b.components["team-a-0"]["capture"] == 0.0
    # default weights: capture events are ignored
    assert compute_rewards(_obs(), _obs(), [ours, theirs], RUNNING, RewardConfig(), CONTROLLED)["team-a-0"] == pytest.approx(0.01)


def test_gun_pickup_pays_once_on_the_transition():
    cfg = RewardConfig(alive_per_step=0.0, hp_delta=0.0, damage_dealt=0.0, death=0.0, team_win=0.0, gun_pickup=1.0)
    fists = _obs()
    armed = _obs()
    for o in (fists, armed):
        o["team-a-0"]["self"]["weapons"] = [{"slot": 0, "type": "", "ammo": 0}, {"slot": 1, "type": "", "ammo": 0},
                                            {"slot": 2, "type": "fists", "ammo": 0}, {"slot": 3, "type": "", "ammo": 0}]
    armed["team-a-0"]["self"]["weapons"][1] = {"slot": 1, "type": "mp5", "ammo": 30}  # picked up, not yet equipped
    b = compute_reward_breakdown(fists, armed, [], RUNNING, cfg, CONTROLLED)
    assert b.components["team-a-0"]["gun_pickup"] == 1.0 and b.components["team-a-1"]["gun_pickup"] == 0.0
    # holding the gun on later steps pays nothing more; dropping and re-picking would pay again by design
    assert compute_reward_breakdown(armed, armed, [], RUNNING, cfg, CONTROLLED).totals["team-a-0"] == 0.0
    # the "armed" loadout spawns with a gun already in the slot: no pickup reward on reset -> first step
    assert compute_reward_breakdown(armed, armed, [], RUNNING, cfg, CONTROLLED).components["team-a-0"]["gun_pickup"] == 0.0
    # default weights: term off
    assert compute_reward_breakdown(fists, armed, [], RUNNING, RewardConfig(), CONTROLLED).components["team-a-0"]["gun_pickup"] == 0.0


def _cover_obs(weapon: str = "ak47", clip: int = 30, los_blocked: bool = False, downed: bool = False):
    return {
        "self": {"hp": 80.0, "clip": clip, "reserve": 45, "weapon": "ak47", "pos": {"x": 0.0, "y": 0.0}},
        "players": [{"id": "team-b-0", "team": "team-b", "pos": {"x": 10.0, "y": 0.0}, "dist": 10.0,
                     "dir": {"x": -1.0, "y": 0.0}, "downed": downed, "dead": False, "weapon": weapon,
                     "los_blocked": los_blocked}],
    }


def test_cover_term_reads_exposure_the_way_a_bullet_does():
    from experiment.survev_rl.rewards import _cover_term

    # nobody who can shoot: nothing to say
    assert _cover_term({"self": {"clip": 0}, "players": []}) == 0.0
    assert _cover_term(_cover_obs(weapon="fists", clip=0)) == 0.0
    assert _cover_term(_cover_obs(clip=0, downed=True)) == 0.0

    # the playtest failure: empty gun, and the enemy has a clear line
    assert _cover_term(_cover_obs(clip=0)) == -1.0
    # the same moment behind a wall
    assert _cover_term(_cover_obs(clip=0, los_blocked=True)) == 1.0
    # exposed but able to shoot back is neither rewarded nor punished
    assert _cover_term(_cover_obs(clip=30)) == 0.0


def _intent_obs(
    dist: float,
    los_blocked: bool = False,
    clip: int = 30,
    cover_score: float = 0.0,
    *,
    pos: tuple[float, float] = (0.0, 0.0),
    gas_centre: tuple[float, float] | None = None,
    armed_enemy_in_view: bool = True,
):
    obs = {
        # the team matters: kills and dealt damage are only counted across teams, and the counter
        # reads the victim's team from the observations it was given, not from `players`
        "self": {"team": "team-a", "hp": 90.0, "clip": clip, "reserve": 45, "weapon": "ak47",
                 "pos": {"x": pos[0], "y": pos[1]}},
        "players": [{"id": "team-b-0", "team": "team-b", "pos": {"x": dist, "y": 0.0}, "dist": dist,
                     "dir": {"x": -1.0, "y": 0.0}, "downed": False, "dead": False, "weapon": "ak47",
                     "los_blocked": los_blocked}],
        "obstacles": [{"id": 1, "type": "stone_01", "pos": {"x": 3.0, "y": 0.0}, "dist": 3.0,
                       "collidable": True, "height": 0.5, "scale": 1.0,
                       "blocks_los": los_blocked, "cover_score": cover_score}],
    }
    if not armed_enemy_in_view:
        obs["players"] = []
    if gas_centre is not None:
        # the shape the server actually sends: the circle has not started closing, so only `pos`
        # carries information — it is the play area's centre
        centre = {"x": gas_centre[0], "y": gas_centre[1]}
        obs["gas"] = {"mode": 0, "rad": 196.0, "pos": centre, "rad_new": 196.0, "pos_new": centre}
    return obs


def test_retreat_pays_for_cover_at_contact_not_for_kills():
    from experiment.survev_rl.rewards import INTENT_SCALE as INTENT_COMBAT_SCALE, _intent_term

    behind_cover = _intent_term("retreat", _intent_obs(20.0), _intent_obs(25.0, los_blocked=True), 0.0, 0.0)
    in_the_open = _intent_term("retreat", _intent_obs(25.0), _intent_obs(20.0), 0.0, 0.0)
    assert behind_cover > in_the_open
    # the anchor: under retreat a wipe is a cost, not a shortcut. It was 0.0 for one generation and
    # that was not enough -- see test_retreat_prices_the_kill_and_the_hit_together
    assert INTENT_COMBAT_SCALE["retreat"]["kill"] < 0.0


def test_retreat_pays_nothing_for_simply_leaving():
    """The v5/v6 failure, pinned so it cannot come back.

    The old term returned 1.0 whenever no armed enemy was in view. 64-85% of retreat steps met that,
    so the whole reward was collectable by running until the enemy stopped being drawn — which is
    what three seeds of both generations learned to do, ending pinned against the map edge.
    """
    from experiment.survev_rl.rewards import _intent_term

    gone = _intent_obs(0.0, armed_enemy_in_view=False)
    assert _intent_term("retreat", gone, gone, 0.0, 0.0) == 0.0


def test_contact_range_decides_which_route_has_to_be_earned():
    """`contact_dist` is the line between the two ways of withdrawing, not a cut-off.

    Inside it the agent is in the fight and has to be un-shootable to score; outside it the fight is
    broken off, which is the thing being scored. An earlier version of this test asserted that being
    outside paid nothing -- true while the omniscient bots made breaking off impossible, wrong now.
    """
    from experiment.survev_rl.rewards import _intent_term

    for blocked in (True, False):
        out_of_contact = _intent_obs(40.0, los_blocked=blocked)
        assert _intent_term("retreat", out_of_contact, out_of_contact, 0.0, 0.0) == 1.0

    in_contact_covered = _intent_obs(20.0, los_blocked=True)
    in_contact_exposed = _intent_obs(20.0, los_blocked=False)
    assert _intent_term("retreat", in_contact_covered, in_contact_covered, 0.0, 0.0) == 1.0
    assert _intent_term("retreat", in_contact_exposed, in_contact_exposed, 0.0, 0.0) == 0.0


def test_retreat_pays_for_being_unshootable_not_for_distance():
    """Same distance, only the line differs — that is the whole term."""
    from experiment.survev_rl.rewards import _intent_term

    exposed = _intent_obs(20.0, los_blocked=False)
    covered = _intent_obs(20.0, los_blocked=True)
    assert _intent_term("retreat", exposed, exposed, 0.0, 0.0) == 0.0
    assert _intent_term("retreat", covered, covered, 0.0, 0.0) == 1.0


def test_retreat_is_leashed_to_the_play_area():
    """Being safe at the map edge is worth less than being safe in the fight."""
    from experiment.survev_rl.rewards import _intent_term

    centre = (132.0, 132.0)
    at_centre = _intent_obs(20.0, los_blocked=True, pos=centre, gas_centre=centre)
    # 67 u out with a 45 u leash: (67 - 45) / 45 of the term is eaten
    part_way = _intent_obs(20.0, los_blocked=True, pos=(199.0, 132.0), gas_centre=centre)
    # where the v6 policies actually ended up: x = 1 on a map whose centre is 132
    at_the_wall = _intent_obs(20.0, los_blocked=True, pos=(1.0, 121.0), gas_centre=centre)

    assert _intent_term("retreat", at_centre, at_centre, 0.0, 0.0) == 1.0
    assert 0.0 < _intent_term("retreat", part_way, part_way, 0.0, 0.0) < 1.0
    assert _intent_term("retreat", at_the_wall, at_the_wall, 0.0, 0.0) == 0.0


def test_the_leash_needs_a_gas_field():
    """Unit fixtures and any scenario without a gas circle simply have no leash."""
    from experiment.survev_rl.rewards import _intent_term

    nowhere = _intent_obs(20.0, los_blocked=True, pos=(9999.0, 9999.0))
    assert _intent_term("retreat", nowhere, nowhere, 0.0, 0.0) == 1.0


def test_only_retreat_has_a_shaping_term_now():
    """push/hold_angle/trade were steered by an added term for one round and came out
    indistinguishable across three seeds; they are priced through INTENT_SCALE instead."""
    from experiment.survev_rl.rewards import _intent_term

    for intent in ("engage", None):
        assert _intent_term(intent, _intent_obs(30.0), _intent_obs(20.0), 40.0, 0.0) == 0.0


def test_engaging_keeps_the_plain_weights_and_retreat_drops_the_win():
    """Two intents: `engage` is the ordinary reward, `retreat` is paid for nothing about winning."""
    from experiment.survev_rl.rewards import RewardConfig, compute_reward_components

    cfg = RewardConfig(damage_dealt=0.02, damage_taken=-0.02, kill=1.0, team_win=0.5)
    prev, cur = {"a": _intent_obs(20.0)}, {"a": _intent_obs(20.0)}
    events = [{"type": "damage", "agent": "a", "source": "team-b-0", "amount": 30.0},
              {"type": "damage", "agent": "team-b-0", "source": "a", "amount": 40.0},
              {"type": "kill", "agent": "team-b-0", "source": "a"}]

    engaging = compute_reward_components(prev, cur, events, None, cfg, ["a"], intent="engage")
    assert engaging["a"]["kill"] == 1.0
    assert engaging["a"]["damage_dealt"] > 0.0
    assert engaging["a"]["damage_taken"] < 0.0

    retreating = compute_reward_components(prev, cur, events, None, cfg, ["a"], intent="retreat")
    assert retreating["a"]["kill"] < 0.0        # a withdrawal that ends in a wipe was not one
    assert retreating["a"]["damage_dealt"] == 0.0
    # being hit is how withdrawing fails, so it costs more here than in an ordinary fight
    assert retreating["a"]["damage_taken"] < engaging["a"]["damage_taken"]


def test_retreat_does_not_pay_the_clock():
    from experiment.survev_rl.rewards import RewardConfig, compute_reward_components

    cfg = RewardConfig(time_penalty_after_s=3.0, time_penalty_per_step=-0.01)
    prev, cur = {"a": _intent_obs(20.0)}, {"a": _intent_obs(20.0)}
    at = lambda intent: compute_reward_components(prev, cur, [], None, cfg, ["a"], t=10.0, intent=intent)["a"]["time"]
    assert at("engage") == -0.01  # the clock applies to an ordinary fight if a preset turns it on
    assert at("retreat") == 0.0   # ...but withdrawing takes as long as it takes


def test_retreat_is_not_paid_for_a_win_its_partner_delivered():
    """The v2 leak: with `kill` and `damage_dealt` zeroed but `team_win` intact, withdrawing while the
    partner wiped the enemy still scored — and half the retreat episodes ended in elimination."""
    from experiment.survev_rl.rewards import RewardConfig, compute_reward_components

    obs = {"a": _intent_obs(25.0)}
    prev = {"a": _intent_obs(20.0)}
    cfg = RewardConfig(team_win=1.0, intent_bonus=0.0)

    # the fixture's agent is on team-b (its own "self" carries no team, so winner must match "")
    won = compute_reward_components(prev, obs, [], {"winner_team": "team-a", "reason": "elimination"}, cfg, ["a"])
    assert won["a"]["team_win"] == 1.0

    retreating = compute_reward_components(
        prev, obs, [], {"winner_team": "team-a", "reason": "elimination"}, cfg, ["a"], intent="retreat"
    )
    assert retreating["a"]["team_win"] == 0.0

    engaging = compute_reward_components(
        prev, obs, [], {"winner_team": "team-a", "reason": "elimination"}, cfg, ["a"], intent="engage"
    )
    assert engaging["a"]["team_win"] == 1.0


def test_retreat_prices_the_kill_and_the_hit_together():
    """v7's lesson: zeroing the kill is not enough to make retreat a different behaviour.

    Killing is the surest way to stop `damage_taken`, so a retreat that merely does not *pay* for
    kills still wipes the enemy -- v7's seed 1 came out identical to engage on every metric. The two
    scales have to oppose each other: a kill costs, and being hit costs double.
    """
    from experiment.survev_rl.rewards import RewardConfig, compute_reward_components

    cfg = RewardConfig(damage_dealt=0.02, damage_taken=-0.02, kill=1.0, team_win=0.5)
    prev, cur = {"a": _intent_obs(20.0)}, {"a": _intent_obs(20.0)}
    events = [{"type": "damage", "agent": "a", "source": "team-b-0", "amount": 30.0},
              {"type": "kill", "agent": "team-b-0", "source": "a"}]

    engaging = compute_reward_components(prev, cur, events, None, cfg, ["a"], intent="engage")
    retreating = compute_reward_components(prev, cur, events, None, cfg, ["a"], intent="retreat")

    # the same kill: worth a point when engaging, a cost when withdrawing
    assert engaging["a"]["kill"] == 1.0
    assert retreating["a"]["kill"] == -1.0
    # the same 30 damage: twice as expensive when withdrawing
    assert retreating["a"]["damage_taken"] == 2.0 * engaging["a"]["damage_taken"]
    assert retreating["a"]["damage_taken"] < 0.0


def test_neither_retreat_scale_works_alone():
    """Pinning the reasoning, so a later edit cannot quietly drop one half of the pair."""
    from experiment.survev_rl.rewards import INTENT_SCALE

    retreat = INTENT_SCALE["retreat"]
    # a wipe has to cost, or killing stays the cheapest way to stop the bleeding
    assert retreat["kill"] < 0.0
    # being hit has to cost more than it does when engaging, or standing in the open is free
    assert retreat["damage_taken"] > 1.0
    # and nothing about winning the fight may pay under retreat
    assert retreat["damage_dealt"] == 0.0 and retreat["team_win"] == 0.0


def test_retreat_pays_for_breaking_off_now_that_it_is_possible():
    """The second route, unreachable until the bots stopped being omniscient.

    While the scripted bots chased through walls, pushing the enemy past `contact_dist` could not be
    done, so the only way to score was to stand un-shootable inside the fight -- which looks like
    engaging. Paying for a genuine break-off is what lets the planner's two words mean two things.
    """
    from experiment.survev_rl.rewards import _intent_term

    covered = _intent_obs(20.0, los_blocked=True)   # in the fight, un-shootable
    broken_off = _intent_obs(40.0, los_blocked=False)  # out of the fight, still watching them
    assert _intent_term("retreat", covered, covered, 0.0, 0.0) == 1.0
    assert _intent_term("retreat", broken_off, broken_off, 0.0, 0.0) == 1.0

    # exposed inside the fight is still worth nothing, and vanishing still pays nothing at all
    exposed = _intent_obs(20.0, los_blocked=False)
    gone = _intent_obs(0.0, armed_enemy_in_view=False)
    assert _intent_term("retreat", exposed, exposed, 0.0, 0.0) == 0.0
    assert _intent_term("retreat", gone, gone, 0.0, 0.0) == 0.0


def test_breaking_off_outside_the_play_area_still_costs():
    """Otherwise 'break off' collapses back into 'leave the map', which is where v5 and v6 ended."""
    from experiment.survev_rl.rewards import _intent_term

    centre = (132.0, 132.0)
    near = _intent_obs(40.0, pos=centre, gas_centre=centre)
    far = _intent_obs(40.0, pos=(1.0, 121.0), gas_centre=centre)
    assert _intent_term("retreat", near, near, 0.0, 0.0) == 1.0
    assert _intent_term("retreat", far, far, 0.0, 0.0) == 0.0
