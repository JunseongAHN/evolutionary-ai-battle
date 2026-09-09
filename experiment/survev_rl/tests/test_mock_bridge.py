from __future__ import annotations

import math

import pytest

from experiment.survev_rl import mock_bridge as M
from experiment.survev_rl.protocol import METRIC_KEYS, CpcAction, ProtocolError, validate_agent_observation


def _walk_to(sim: M.FieldSim, agent: str, target: tuple[float, float], max_ticks: int = 400) -> None:
    """Drive a controlled agent toward ``target`` with 8-direction moves (test helper)."""
    p = sim.players[agent]
    for _ in range(max_ticks):
        dx, dy = target[0] - p.pos[0], target[1] - p.pos[1]
        if math.hypot(dx, dy) < 0.3:
            break
        sim.apply_action(agent, {"move": {"x": dx, "y": dy}, "aim": {"x": 1, "y": 0}})
        sim.step(1)
    sim.apply_action(agent, {})


def test_reset_layout_and_partial_observation(client):
    msg = client.reset(0, seed="cpc-duo2v2-seed-0")
    assert msg.t == 0.0 and msg.tick == 0 and not msg.done
    assert msg.agent_ids == ["team-a-0", "team-a-1", "team-b-0", "team-b-1"]
    assert msg.teams["team-b-1"] == "team-b"
    for aid, obs in msg.obs.items():
        validate_agent_observation(obs)
        assert obs["self"]["weapon"] == "fists" and obs["self"]["hp"] == 100
        assert obs["players"] == []  # enemies spawn 64 u apart: outside the 32 x 18 rectangle
        assert len(obs["teammates"]) == 1 and obs["teammates"][0]["hp"] == 100
    a0 = msg.obs["team-a-0"]
    assert a0["self"]["pos"] == {"x": 100.0, "y": 125.6}
    assert a0["self"]["zoom"] == 28 and a0["gas"]["rad"] == 196
    visible_types = sorted(l["type"] for l in a0["loot"])
    assert visible_types.count("ak47") == 1 and visible_types.count("mp5") == 1
    assert visible_types.count("762mm") == 2 and visible_types.count("9mm") == 2
    assert all(l["pos"]["x"] < 132 + 4 for l in a0["loot"])  # team-b kit at x=154 is culled
    # the seeded layout is complete and deterministic
    sim = M.FieldSim(0, "cpc-duo2v2-seed-0")
    assert len(sim.loot) == 25
    assert sorted({l.type for l in sim.loot.values()}) == sorted(
        {"ak47", "mp5", "bandage", "soda", "helmet01", "chest01", "2xscope", "762mm", "9mm", "healthkit",
         "painkiller", "4xscope"}
    )
    other = M.FieldSim(0, "cpc-duo2v2-seed-0")
    assert [(l.type, l.pos) for l in sim.loot.values()] == [(l.type, l.pos) for l in other.loot.values()]
    assert [(l.type, l.pos) for l in M.FieldSim(0, "other-seed").loot.values()] != [
        (l.type, l.pos) for l in sim.loot.values()
    ]


def test_held_inputs_ticks_and_release(client):
    client.reset(0, seed=1, scripted="idle")
    msg = client.step(0, {"team-a-0": CpcAction(move=(1.0, 0.0), aim=(1.0, 0.0))}, ticks=1)
    assert msg.tick == 1 and msg.t == pytest.approx(0.01)
    msg = client.step(0, {"team-a-0": CpcAction(move=(1.0, 0.0), aim=(1.0, 0.0))}, ticks=99)
    assert msg.tick == 100 and msg.t == pytest.approx(1.0)
    x1 = msg.obs["team-a-0"]["self"]["pos"]["x"]
    assert x1 == pytest.approx(113.0, abs=0.05)  # 12 u/s + 1 with fists
    msg = client.step(0, {}, ticks=100)  # agent omitted: input is held
    assert msg.obs["team-a-0"]["self"]["pos"]["x"] == pytest.approx(126.0, abs=0.05)
    msg = client.step(0, {"team-a-0": {}}, ticks=100)  # {} releases everything
    assert msg.obs["team-a-0"]["self"]["pos"]["x"] == pytest.approx(126.0, abs=0.05)
    assert msg.obs["team-a-0"]["self"]["dir"] == {"x": 1.0, "y": 0.0}
    # diagonal moves are quantized to the 8 keyboard directions and normalized
    msg = client.step(0, {"team-a-0": CpcAction(move=(1.0, 0.2))}, ticks=100)
    assert msg.obs["team-a-0"]["self"]["pos"]["x"] == pytest.approx(139.0, abs=0.05)
    assert msg.obs["team-a-0"]["self"]["pos"]["y"] == pytest.approx(125.6, abs=0.05)
    with pytest.raises(ProtocolError):
        client.step(0, {"team-b-0": CpcAction(move=(1, 0))}, ticks=1)  # not controlled


def test_pickup_via_interact_and_equip():
    sim = M.FieldSim(0, "cpc-duo2v2-seed-0", {"controlled": ["team-a-0", "team-a-1"], "scripted": "idle"})
    ak = next(l for l in sim.loot.values() if l.type == "ak47" and l.pos[0] < 132)
    _walk_to(sim, "team-a-0", ak.pos)
    p = sim.players["team-a-0"]
    assert math.hypot(p.pos[0] - ak.pos[0], p.pos[1] - ak.pos[1]) <= M.PICKUP_RANGE
    sim.apply_action("team-a-0", {"inputs": ["Interact"]})
    obs = sim.step(1)["obs"]["team-a-0"]["self"]
    assert obs["weapons"][0] == {"slot": 0, "type": "ak47", "ammo": 30}
    assert obs["weapon"] == "ak47" and obs["cur_weap_idx"] == 0 and obs["clip"] == 30  # auto-equipped
    assert ak.id not in sim.loot
    # ammo piles next to the gun -> inventory (reserve reported for the equipped gun)
    piles = [l for l in sim.loot.values() if l.type == "762mm" and math.hypot(l.pos[0] - ak.pos[0], l.pos[1] - ak.pos[1]) < 1.5]
    assert len(piles) == 2
    for pile in piles:
        for _ in range(5):  # Interact takes the *nearest* loot; other kit items may be closer
            _walk_to(sim, "team-a-0", pile.pos)
            sim.apply_action("team-a-0", {"inputs": [7]})  # numeric Interact
            sim.step(1)
            if pile.id not in sim.loot:
                break
        assert pile.id not in sim.loot
    obs = sim.observe()["obs"]["team-a-0"]["self"]
    assert obs["inventory"]["762mm"] == 60 and obs["reserve"] == 60
    # switching to fists and back goes through EquipMelee / EquipPrimary
    sim.apply_action("team-a-0", {"inputs": ["EquipMelee"]})
    assert sim.step(1)["obs"]["team-a-0"]["self"]["weapon"] == "fists"
    sim.apply_action("team-a-0", {"inputs": ["EquipPrimary"]})
    assert sim.step(1)["obs"]["team-a-0"]["self"]["weapon"] == "ak47"
    # with a gun equipped the move speed drops to 12 u/s
    sim.apply_action("team-a-0", {"move": {"x": 0, "y": 1}})
    y0 = sim.players["team-a-0"].pos[1]
    sim.step(100)
    assert sim.players["team-a-0"].pos[1] - y0 == pytest.approx(12.0, abs=0.05)


def test_fire_damage_down_and_kill_semantics():
    sim = M.FieldSim(0, 3, {"controlled": ["team-a-0", "team-a-1"], "scripted": "idle"})
    shooter, victim, partner = sim.players["team-a-0"], sim.players["team-b-0"], sim.players["team-b-1"]
    shooter.weapons[0] = {"slot": 0, "type": "ak47", "ammo": 30}
    shooter.cur_weap_idx = 0
    shooter.inventory["762mm"] = 90
    shooter.pos, victim.pos, partner.pos = (120.0, 132.0), (130.0, 132.0), (130.0, 150.0)
    sim.apply_action("team-a-0", {"aim": {"x": 1, "y": 0}, "fire": {"start": True, "hold": False}})
    out = sim.step(1)
    fires = [e for e in out["events"] if e["type"] == "fire"]
    assert len(fires) == 1 and fires[0]["agent"] == "team-a-0" and fires[0]["weapon"] == "ak47"
    out = sim.step(30)  # fire.start is edge-triggered: no further shots while only "start" was sent
    assert all(e["type"] != "fire" for e in out["events"])
    # hold fire for 0.5 s: fireDelay 0.1 s -> 5 shots, hits deal 13 (no armor) and hp drops
    sim.apply_action("team-a-0", {"aim": {"x": 1, "y": 0}, "fire": {"start": False, "hold": True}})
    out = sim.step(50)
    fires = [e for e in out["events"] if e["type"] == "fire"]
    assert len(fires) == 5 and not victim.downed
    dmg = [e for e in out["events"] if e["type"] == "damage" and e["agent"] == "team-b-0"]
    assert dmg and all(e["source"] == "team-a-0" and e["amount"] <= 13.0 for e in dmg)
    for e in dmg:
        assert e["hp_after"] == pytest.approx(e["hp_before"] - e["amount"], abs=1e-6) or e["downed"] or e["dead"]
    # keep shooting until the victim goes down: teammate is standing so it is a down, not a kill
    for _ in range(500):
        if victim.downed:
            break
        out = sim.step(1)
    assert victim.downed and not victim.dead and victim.hp == pytest.approx(100.0)  # engine reset
    downs = [e for e in out["events"] if e["type"] == "down"]
    assert downs and downs[0]["agent"] == "team-b-0" and downs[0]["source"] == "team-a-0"
    lethal = [e for e in out["events"] if e["type"] == "damage" and e["agent"] == "team-b-0" and e["downed"]]
    assert lethal and lethal[-1]["hp_after"] == 0.0
    obs = sim.observe()["obs"]["team-a-0"]
    assert any(p["id"] == "team-b-0" and p["downed"] for p in obs["players"])
    assert "hp" not in obs["players"][0]  # enemy HP never leaks
    # downed players bleed (source "bleed")
    sim.apply_action("team-a-0", {})
    out = sim.step(150)
    bleed = [e for e in out["events"] if e["type"] == "damage" and e["source"] == "bleed"]
    assert bleed and victim.hp < 100.0
    # killing the last standing teammate kills the downed one too (kill events for both)
    shooter.pos = (120.0, 150.0)
    sim.apply_action("team-a-0", {"aim": {"x": 1, "y": 0}, "fire": {"start": False, "hold": True}})
    for _ in range(80):
        out = sim.step(10)
        if sim.done:
            break
    assert sim.done and out["info"]["reason"] == "elimination" and out["info"]["winner_team"] == "team-a"
    kills = [e for e in out["events"] if e["type"] == "kill"]
    assert {e["agent"] for e in kills} == {"team-b-0", "team-b-1"}
    assert victim.dead and partner.dead
    metrics = out["info"]["metrics"]
    assert set(metrics) == set(sim.players) and all(set(m) == set(METRIC_KEYS) for m in metrics.values())
    assert metrics["team-a-0"]["kills"] == 2 and metrics["team-a-0"]["damage_dealt"] > 100
    assert metrics["team-a-0"]["team_win"] and not metrics["team-b-0"]["team_win"]
    assert metrics["team-b-0"]["downed_time"] > 1.0 and not metrics["team-b-0"]["alive_at_end"]
    assert metrics["team-b-1"]["partner_survival_time"] == metrics["team-b-0"]["survival_time"]
    with pytest.raises(ProtocolError):
        sim.step(1)


def test_revive_restores_downed_teammate():
    sim = M.FieldSim(0, 5, {"controlled": ["team-a-0", "team-a-1"], "scripted": "idle"})
    a0, a1 = sim.players["team-a-0"], sim.players["team-a-1"]
    a1.pos = (a0.pos[0] + 2.0, a0.pos[1])
    sim._apply_damage(a1, 500.0, None, "test", armor=False)
    assert a1.downed and not a1.dead and a1.hp == 100.0
    sim.apply_action("team-a-0", {"inputs": ["Revive"]})
    out = sim.step(10)
    assert out["obs"]["team-a-0"]["self"]["action"] == 3 and out["obs"]["team-a-1"]["self"]["action"] == 3
    sim.step(int(M.REVIVE_DURATION * 100))
    assert not a1.downed and a1.hp == pytest.approx(M.REVIVE_HP)
    assert any(e["type"] == "revive" and e["agent"] == "team-a-1" for e in sim.events)


def test_heal_items_and_use_item():
    sim = M.FieldSim(0, 5, {"controlled": ["team-a-0"], "scripted": "idle"})
    a0 = sim.players["team-a-0"]
    a0.hp = 50.0
    a0.inventory["bandage"] = 2
    sim.apply_action("team-a-0", {"useItem": "bandage"})
    assert sim.step(1)["obs"]["team-a-0"]["self"]["action"] == 2
    out = sim.step(300)
    assert a0.hp == pytest.approx(65.0) and a0.inventory["bandage"] == 1
    assert any(e["type"] == "heal" for e in out["events"])
    sim.apply_action("team-a-0", {"inputs": ["UseBandage"]})
    sim.step(301)
    assert a0.hp == pytest.approx(80.0) and a0.inventory["bandage"] == 0


def test_time_limit_and_done_info(client):
    msg = client.reset(0, seed=0, scripted="idle", time_limit=1.0, controlled=["team-a-0"])
    while not msg.done:
        msg = client.step(0, {}, ticks=10)
    assert msg.t == pytest.approx(1.0) and msg.info.reason == "time_limit" and msg.info.winner_team is None
    assert msg.info.alive_teams == 2 and msg.info.metrics is not None
    for aid, m in msg.info.metrics.items():
        assert m.alive_at_end and m.survival_time == pytest.approx(1.0) and m.hp_mean == 100.0
    with pytest.raises(ProtocolError, match="done"):
        client.step(0, {}, ticks=10)


def test_chaser_eliminates_idle_team_and_events_are_consistent(client):
    msg = client.reset(0, seed="cpc-duo2v2-seed-0", controlled=["team-a-0", "team-a-1"], scripted="chaser")
    damage_sum: dict[str, float] = {}
    order: list[tuple[str, str]] = []
    while not msg.done:
        msg = client.step(0, {}, ticks=10)
        for e in msg.events:
            if e["type"] == "damage":
                damage_sum[e["agent"]] = damage_sum.get(e["agent"], 0.0) + e["amount"]
            if e["type"] in ("down", "kill"):
                order.append((e["type"], e["agent"]))
    assert msg.info.reason == "elimination" and msg.info.winner_team == "team-b"
    assert msg.t < 60.0
    metrics = msg.info.metrics
    for aid in ("team-a-0", "team-a-1"):
        assert metrics[aid].damage_taken == pytest.approx(damage_sum[aid], abs=1e-6)
        assert not metrics[aid].alive_at_end and not metrics[aid].team_win
    assert metrics["team-b-0"].team_win and metrics["team-b-1"].team_win
    assert sum(m.shots for m in metrics.values()) > 0
    assert sum(m.kills for m in metrics.values()) == 2
    assert [k for k, _ in order].count("kill") == 2


def test_batched_step_and_errors(client):
    client.reset(0, seed=1, scripted="idle", controlled=["team-a-0"])
    client.reset(1, seed=2, scripted="idle", controlled=["team-a-0"])
    res = client.step_batch({0: ({"team-a-0": CpcAction(move=(1, 0))}, 10), 1: ({}, 3)})
    assert res[0].tick == 10 and res[1].tick == 3
    assert res[0].obs["team-a-0"]["self"]["pos"]["x"] > 100.0
    assert res[1].obs["team-a-0"]["self"]["pos"]["x"] == 100.0
    with pytest.raises(ProtocolError, match="unknown env"):
        client.step_batch({0: ({}, 10), 7: ({}, 10)})
    with pytest.raises(ProtocolError, match="unknown env"):
        client.step(42, {}, 10)
    raw = client.request({"type": "bogus"})
    assert raw["type"] == "error"
    client.close(1)
    assert client.env_ids == frozenset({0})
    with pytest.raises(ProtocolError):
        client.step(1, {}, 10)
    with pytest.raises(ProtocolError):
        client.reset(2, options={"controlled": ["nobody"]})
    with pytest.raises(ProtocolError):
        client.reset(2, options={"scripted": "genius"})


def test_same_seed_same_trajectory():
    def run(seed):
        sim = M.FieldSim(0, seed, {"controlled": [], "scripted": "chaser"})
        while not sim.done:
            out = sim.step(10)
        return out["t"], out["info"]["winner_team"], {a: m["damage_dealt"] for a, m in out["info"]["metrics"].items()}

    assert run("cpc-duo2v2-seed-3") == run("cpc-duo2v2-seed-3")


def test_culling_rectangle_hides_far_enemies_and_grows_with_scope():
    sim = M.FieldSim(0, 0, {"controlled": ["team-a-0"], "scripted": "idle"})
    a0, b0, b1 = sim.players["team-a-0"], sim.players["team-b-0"], sim.players["team-b-1"]
    a0.pos, b0.pos, b1.pos = (132.0, 132.0), (132.0 + 33.0, 132.0), (200.0, 200.0)
    assert sim.observe_agent(a0)["players"] == []
    b0.pos = (132.0 + 31.0, 132.0 + 17.0)
    assert [p["id"] for p in sim.observe_agent(a0)["players"]] == ["team-b-0"]
    b0.pos = (132.0 + 31.0, 132.0 + 19.0)  # inside x, outside the 18 u half-height
    assert sim.observe_agent(a0)["players"] == []
    a0.scope, a0.scopes = "4xscope", {"1xscope", "4xscope"}
    assert sim.observe_agent(a0)["self"]["zoom"] == 48
    assert [p["id"] for p in sim.observe_agent(a0)["players"]] == ["team-b-0"]


def test_armed_loadout_option_is_an_extension(client):
    msg = client.reset(0, seed=0, options={"loadout": "armed"}, scripted="idle", controlled=["team-a-0"])
    me = msg.obs["team-a-0"]["self"]
    assert me["weapon"] == "ak47" and me["clip"] == 30 and me["reserve"] == 90 and me["cur_weap_idx"] == 0
    assert msg.obs["team-b-1"]["self"]["weapon"] == "ak47"
    msg = client.reset(0, seed=0, scripted="idle", controlled=["team-a-0"])
    assert msg.obs["team-a-0"]["self"]["weapon"] == "fists"  # default = spec behaviour
    with pytest.raises(ProtocolError):
        client.reset(0, options={"loadout": "tank"})


def test_racer_opponent_takes_points_on_the_mock():
    from experiment.survev_rl.mock_bridge import FieldSim

    sim = FieldSim(0, seed="racer-mock", options={"scripted": "racer", "controlled": ["team-a-0", "team-a-1"],
                                                  "objective": {"mode": "race"}, "endOnElimination": False,
                                                  "timeLimit": 25})
    captures = 0
    while not sim.done:
        msg = sim.step(10)
        captures += sum(1 for e in msg["events"] if e["type"] == "capture" and e["team"] == "team-b")
    assert captures >= 2, "racers should keep taking points while team-a idles"
    assert msg["info"]["objective"]["captures"]["team-b"] == captures
    assert msg["info"]["reason"] in ("time_limit", "controlled_dead")
