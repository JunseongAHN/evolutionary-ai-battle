"""summarize(): race captures and the env pickup stats ride along; old records without them still work."""

from experiment.survev_rl.eval import summarize

BASE = {
    "survival_time": 10.0, "hp_mean": 80.0, "hp_end": 0.0, "damage_dealt": 5.0, "damage_taken": 100.0,
    "kills": 0.0, "shots": 3.0, "hits_given": 1.0, "downed_time": 0.0, "partner_survival_time": 9.0,
}
CONTROLLED = ["team-a-0", "team-a-1"]


def record(team_win, captures, pickup):
    return {
        "team_win": team_win,
        "winner_team": "team-a" if team_win else None,
        "duration_s": 60.0,
        "episode_return": {aid: 1.0 for aid in CONTROLLED},
        "metrics": {aid: {**BASE, "captures": c, "team_captures": sum(captures)} for aid, c in zip(CONTROLLED, captures)},
        "pickup": {aid: p for aid, p in zip(CONTROLLED, pickup)},
    }


def test_summarize_captures_and_pickup():
    records = [
        record(True, (2.0, 1.0), ({"gun_pickup_time": 1.5, "armed": 1.0}, {"gun_pickup_time": -1.0, "armed": 0.0})),
        record(False, (0.0, 1.0), ({"gun_pickup_time": 3.5, "armed": 1.0}, {"gun_pickup_time": 2.0, "armed": 1.0})),
    ]
    s = summarize(records, CONTROLLED)
    assert s["episodes"] == 2 and s["win_rate"] == 0.5
    assert s["mean"]["captures"] == 1.0 and s["mean"]["team_captures"] == 2.0  # (3 + 1) / 2
    assert s["per_agent"]["team-a-0"]["captures"] == 1.0 and s["per_agent"]["team-a-1"]["captures"] == 1.0
    assert s["armed_rate"] == 0.75
    assert s["gun_pickup_time_median"] == 2.0  # median of 1.5, 3.5, 2.0 (never-armed agents excluded)


def test_summarize_without_new_fields():
    old = record(False, (0.0, 0.0), ({}, {}))
    for aid in CONTROLLED:
        old["metrics"][aid].pop("captures")
        old["metrics"][aid].pop("team_captures")
    old.pop("pickup")
    s = summarize([old], CONTROLLED)
    assert "captures" not in s["mean"] and "armed_rate" not in s
    assert s["mean"]["survival_time"] == 10.0
