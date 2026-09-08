from __future__ import annotations

import json

import pytest

from experiment.survev_rl import protocol as P


def test_input_table_matches_spec_numbers():
    assert P.input_to_number("Interact") == 7
    assert P.input_to_number("Reload") == 5
    assert P.input_to_number("Revive") == 8
    assert P.input_to_number("EquipPrimary") == 11
    assert P.input_to_number("EquipSecondary") == 12
    assert P.input_to_number("EquipMelee") == 13
    assert P.input_to_number("EquipNextWeap") == 17
    assert P.input_to_number("UseBandage") == 23
    assert P.input_to_number("UsePainkiller") == 26
    assert P.input_to_name(24) == "UseHealthKit"
    assert P.input_to_name("UseSoda") == "UseSoda"
    with pytest.raises(P.ProtocolError):
        P.input_to_number("Teleport")
    with pytest.raises(P.ProtocolError):
        P.input_to_number(999)


def test_cpc_action_roundtrip_and_release():
    a = P.CpcAction(move=(1.0, 0.0), aim=(0.3, 0.95), fire_hold=True, inputs=["Interact", 5], use_item="bandage")
    j = a.to_json()
    assert j["move"] == {"x": 1.0, "y": 0.0}
    assert j["fire"] == {"start": False, "hold": True}
    assert j["inputs"] == ["Interact", "Reload"]  # numbers are sent as names
    assert j["useItem"] == "bandage"
    back = P.CpcAction.from_json(j)
    assert back.move == (1.0, 0.0) and back.fire_hold and not back.fire_start
    assert back.inputs == ["Interact", "Reload"]
    assert P.CpcAction.release().to_json() == {}
    assert P.CpcAction(move=(0.0, 0.0)).to_json() == {}
    # aim-only is not a release
    assert P.CpcAction(aim=(1.0, 0.0)).to_json()["aim"] == {"x": 1.0, "y": 0.0}


def test_request_builders_and_json_helpers():
    reset = P.make_reset(0, seed="cpc-duo2v2-seed-0", options=P.make_reset_options())
    assert reset["type"] == "reset" and reset["options"]["controlled"] == ["team-a-0", "team-a-1"]
    assert reset["options"]["scripted"] == "chaser" and reset["options"]["timeLimit"] == 60.0
    step = P.make_step(3, {"team-a-0": P.CpcAction(move=(0, 1)), "team-a-1": {}}, ticks=10)
    assert step == {"type": "step", "env_id": 3, "ticks": 10,
                    "actions": {"team-a-0": {"move": {"x": 0.0, "y": 1.0}, "fire": {"start": False, "hold": False},
                                             "useItem": ""}, "team-a-1": {}}}
    batch = P.make_step_batch({0: ({}, 10), 1: ({"team-a-0": P.CpcAction.release()}, 3)})
    assert batch["type"] == "step" and set(batch["envs"]) == {"0", "1"}
    assert batch["envs"]["1"] == {"ticks": 3, "actions": {"team-a-0": {}}}
    with pytest.raises(P.ProtocolError):
        P.make_step(0, {}, ticks=0)
    encoded = P.encode_message(step)
    assert json.loads(encoded) == step
    assert P.decode_message(encoded)["type"] == "step"
    with pytest.raises(P.ProtocolError):
        P.decode_message("not json")
    with pytest.raises(P.ProtocolError):
        P.decode_message('{"no_type": 1}')
    assert P.make_close(2) == {"type": "close", "env_id": 2}


def test_validate_spec_observation_passes(spec_obs):
    P.validate_agent_observation(spec_obs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.__setitem__("enemy_hp", {}),
        lambda o: o["self"].__setitem__("secret", 1),
        lambda o: o["self"]["inventory"].__setitem__("gold", 1),
        lambda o: o["self"]["weapons"][0].__setitem__("durability", 1),
        lambda o: o["players"][0].__setitem__("hp", 55),  # enemy HP must never appear
        lambda o: o["players"][0]["pos"].__setitem__("z", 0),
        lambda o: o["loot"][0].__setitem__("owner", "x"),
        lambda o: o["teammates"][0].__setitem__("inventory", {}),
        lambda o: o["gas"].__setitem__("damage", 1),
        lambda o: o["bullets"][0].__setitem__("damage", 13),
    ],
)
def test_validate_rejects_unknown_keys(spec_obs, mutate):
    mutate(spec_obs)
    with pytest.raises(P.ProtocolError, match="allowlist"):
        P.validate_agent_observation(spec_obs)


def test_obs_message_and_metrics_parsing(spec_obs):
    metrics = {"survival_time": 5.5, "alive_at_end": False, "downed_time": 0.9, "hp_mean": 71.3, "hp_end": 0,
               "damage_dealt": 53, "damage_taken": 120, "kills": 0, "shots": 29, "hits_given": 4,
               "team_win": False, "partner_survival_time": 5.5, "partner_hp_end": 0}
    msg = {
        "type": "obs", "env_id": 0, "t": 3.5, "tick": 350, "done": True,
        "agent_ids": list(P.DUO2V2_AGENT_IDS), "teams": dict(P.DUO2V2_TEAMS),
        "obs": {"team-a-0": spec_obs},
        "events": [{"type": "kill", "t": 6.03, "agent": "team-a-1", "source": "team-b-1"}],
        "info": {"alive_teams": 1, "winner_team": "team-b", "reason": "elimination",
                 "metrics": {"team-a-0": metrics}},
    }
    parsed = P.parse_response(msg)
    assert parsed.done and parsed.info.winner_team == "team-b" and parsed.info.reason == "elimination"
    assert parsed.info.metrics["team-a-0"].damage_taken == 120
    assert parsed.teammates_of("team-a-0") == ["team-a-1"]
    assert parsed.team_of("team-b-1") == "team-b"
    assert parsed.to_json()["info"]["metrics"]["team-a-0"]["shots"] == 29
    with pytest.raises(P.ProtocolError):
        P.Metrics.from_json({**metrics, "secret": 1})
    with pytest.raises(P.ProtocolError, match="bridge error"):
        P.parse_response({"type": "error", "env_id": 0, "message": "boom"})
    with pytest.raises(P.ProtocolError):
        P.parse_response({"type": "closed", "env_id": 0})
    batch = P.parse_batch_response({"type": "obs_batch", "envs": {"0": msg}})
    assert set(batch) == {0} and batch[0].tick == 350
