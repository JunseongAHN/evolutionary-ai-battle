"""The cover measure has to count the right steps, or it will flatter a policy that never hides."""

import json
from pathlib import Path

from experiment.survev_rl.cover_usage import is_armed_threat, iter_observations, scan


def _obs(*, weapon="m9", dead=False, downed=False, los_blocked=False, cover_dist=99.0):
    return {
        "team-a-0": {
            "self": {"dead": False, "hp": 100},
            "players": [
                {"weapon": weapon, "dead": dead, "downed": downed, "los_blocked": los_blocked}
            ],
            "obstacles": [{"cover_score": 1.0, "dist": cover_dist}],
        }
    }


def _write(tmp_path: Path, rows, *, as_episode: bool) -> Path:
    p = tmp_path / ("ep.jsonl" if as_episode else "rows.jsonl")
    if as_episode:
        p.write_text(json.dumps({"steps": [{"obs": o} for o in rows]}) + "\n")
    else:
        p.write_text("".join(json.dumps({"obs": o}) + "\n" for o in rows))
    return p


def test_reads_both_log_formats(tmp_path):
    rows = [_obs(), _obs(los_blocked=True)]
    episode = list(iter_observations(_write(tmp_path, rows, as_episode=True)))
    bridge = list(iter_observations(_write(tmp_path, rows, as_episode=False)))
    assert episode == bridge == rows


def test_only_armed_live_enemies_count():
    assert is_armed_threat({"weapon": "m9"})
    for harmless in ({"weapon": "fists"}, {"weapon": ""}, {}, {"weapon": "m9", "dead": True},
                     {"weapon": "m9", "downed": True}):
        assert not is_armed_threat(harmless), harmless


def test_steps_without_an_armed_enemy_are_excluded(tmp_path):
    # standing behind a wall alone in a field is not cover use; counting it would let a policy
    # that simply runs away score as if it were using cover
    log = _write(tmp_path, [_obs(weapon="fists"), _obs(dead=True)], as_episode=False)
    assert scan([log])["steps_with_armed_enemy"] == 0


def test_line_counts_as_broken_only_when_every_enemy_is_blocked(tmp_path):
    both = {
        "team-a-0": {
            "self": {"dead": False},
            "players": [
                {"weapon": "m9", "los_blocked": True},
                {"weapon": "m9", "los_blocked": False},
            ],
            "obstacles": [],
        }
    }
    log = _write(tmp_path, [both], as_episode=False)
    r = scan([log])
    assert r["steps_with_armed_enemy"] == 1 and r["line_broken"] == 0


def test_percentages(tmp_path):
    rows = [_obs(los_blocked=True, cover_dist=3.0), _obs(), _obs(), _obs()]
    r = scan([_write(tmp_path, rows, as_episode=False)])
    assert r["steps_with_armed_enemy"] == 4
    assert r["line_broken"] == 1 and r["line_broken_pct"] == 25.0
    assert r["cover_in_reach"] == 1 and r["cover_in_reach_pct"] == 25.0


def test_cover_out_of_range_and_unusable_cover_do_not_count(tmp_path):
    far = _obs(cover_dist=9.0)
    unusable = _obs(cover_dist=2.0)
    unusable["team-a-0"]["obstacles"] = [{"cover_score": 0.0, "dist": 2.0}]  # a bush: shots pass
    r = scan([_write(tmp_path, [far, unusable], as_episode=False)])
    assert r["cover_in_reach"] == 0


def test_only_the_measured_team_counts(tmp_path):
    row = _obs()
    row["team-b-0"] = {"self": {"dead": False}, "players": [{"weapon": "m9"}], "obstacles": []}
    r = scan([_write(tmp_path, [row], as_episode=False)], team_prefix="team-a")
    assert r["steps_with_armed_enemy"] == 1


def test_max_dist_excludes_steps_that_are_not_a_fight(tmp_path):
    # a policy that never closes in gets walls between itself and everyone else for free; the
    # distance condition is what stops that from reading as cover use
    far = _obs()
    far["team-a-0"]["players"][0].update({"dist": 45.0, "los_blocked": True})
    close = _obs()
    close["team-a-0"]["players"][0].update({"dist": 12.0, "los_blocked": True})
    log = _write(tmp_path, [far, close], as_episode=False)

    assert scan([log])["steps_with_armed_enemy"] == 2
    r = scan([log], max_dist=25.0)
    assert r["steps_with_armed_enemy"] == 1 and r["line_broken"] == 1


def test_max_dist_uses_the_nearest_enemy(tmp_path):
    row = {
        "team-a-0": {
            "self": {"dead": False},
            "players": [
                {"weapon": "m9", "dist": 50.0, "los_blocked": True},
                {"weapon": "m9", "dist": 10.0, "los_blocked": True},
            ],
            "obstacles": [],
        }
    }
    assert scan([_write(tmp_path, [row], as_episode=False)], max_dist=25.0)["steps_with_armed_enemy"] == 1


def _obs_at(x, y, centre=(132.0, 132.0)):
    o = _obs(los_blocked=True)
    o["team-a-0"]["self"]["pos"] = {"x": x, "y": y}
    o["team-a-0"]["players"][0]["dist"] = 20.0
    if centre is not None:
        c = {"x": centre[0], "y": centre[1]}
        o["team-a-0"]["gas"] = {"mode": 0, "rad": 196.0, "pos": c, "rad_new": 196.0, "pos_new": c}
    return o


def test_wandering_is_measured_against_the_gas_centre(tmp_path):
    # one step in the fight, one pinned against the map edge where the v6 retreat ended up
    rows = [_obs_at(132.0, 132.0), _obs_at(1.0, 121.0)]
    r = scan([_write(tmp_path, rows, as_episode=False)], max_dist=25.0, leash_radius=45.0)
    assert r["steps_with_a_position"] == 2
    assert r["beyond_leash"] == 1 and r["beyond_leash_pct"] == 50.0


def test_no_gas_field_means_no_position_report(tmp_path):
    rows = [_obs_at(9999.0, 9999.0, centre=None)]
    r = scan([_write(tmp_path, rows, as_episode=False)], max_dist=25.0)
    assert r["steps_with_a_position"] == 0 and r["beyond_leash_pct"] == 0.0
