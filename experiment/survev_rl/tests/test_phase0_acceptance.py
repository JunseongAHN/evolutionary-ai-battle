"""Phase-0 acceptance sweep (M10): one command that walks the DoD in order.

    python -m pytest experiment/survev_rl/tests/test_phase0_acceptance.py -q

Runs one scripted 2v2 to completion and then checks, in the order the action plan lists them:
the API contract, the observation allowlist and information-set parity, event/HP consistency,
the harness JSONL round trip, and throughput.

It talks to a **real bridge** when ``CPC_BRIDGE_URL`` is set (that is how the Docker setup runs
it, and it is the only way the TS server itself is exercised) and falls back to the in-process
mock otherwise, so the sweep always runs. Which one was used is printed.

Not everything is checkable from Python, and this file does not pretend otherwise:

===== ============================================================ ==========================
M      criterion                                                   verified here?
===== ============================================================ ==========================
M1     reset / step / close, done + winner_team + reason            yes
M2-M4  InputMsg fidelity: displacement, fire delay, interaction     no - vitest `applyCpcAction.test.ts`
       (the *consequences* — looting, firing, dealing damage —      partly, as consequences
       do show up here)
M5     observation freshness == `visibleObjects` after netSync      no - vitest `episode.test.ts`
M6     no information leak: key allowlist, no enemy HP              yes
M7     event correctness: damage sums, down -> kill order           yes
M8     harness JSONL -> validate_episode -> metric vector           yes
M9     seeded layout vs unseeded combat                             no - vitest `determinism.test.ts`
M10    `pnpm cpc:episode` one-liner                                 no - it is a TS entry point
S1-S2  throughput and step latency                                  measured and reported
===== ============================================================ ==========================
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from experiment.core.harness_metrics import compute_metrics
from experiment.core.schema_validation import validate_episode
from experiment.survev_rl.bridge_client import BridgeClient
from experiment.survev_rl.harness_export import add_final_metrics, to_episode_trajectory
from experiment.survev_rl.mock_bridge import MockBridgeThread
from experiment.survev_rl.protocol import validate_agent_observation

SEED = "cpc-phase0-acceptance"
TIME_LIMIT = 60.0
TICKS = 10
CONTROLLED = ["team-a-0", "team-a-1"]


@pytest.fixture(scope="module")
def bridge_url():
    url = os.environ.get("CPC_BRIDGE_URL")
    if url:
        print(f"\n[phase0] real bridge at {url}")
        yield url
        return
    with MockBridgeThread() as server:
        print(f"\n[phase0] mock bridge at {server.url} (set CPC_BRIDGE_URL to use the TS server)")
        yield server.url


@pytest.fixture(scope="module")
def episode(bridge_url) -> dict[str, Any]:
    """One full episode: team-a stands still, the scripted chasers come and kill it.

    Standing still is deliberate — it makes the damage bookkeeping one-directional, which is what
    M7 checks, and it is the scenario the DoD names ("chase and shoot vs stationary").
    """
    messages: list[Any] = []
    with BridgeClient(bridge_url, timeout=60.0) as client:
        first = client.reset(
            env_id=0,
            seed=SEED,
            options={
                "timeLimit": TIME_LIMIT,
                "controlled": CONTROLLED,
                "scripted": "chaser",
                "loadout": "armed",
            },
        )
        messages.append(first)
        started = time.perf_counter()
        step_times: list[float] = []
        message = first
        while not message.done and len(messages) < 1000:
            before = time.perf_counter()
            # {} releases every input: the controlled duo stands still and never fires
            message = client.step(env_id=0, ticks=TICKS, actions={aid: {} for aid in CONTROLLED})
            step_times.append(time.perf_counter() - before)
            messages.append(message)
        wall = time.perf_counter() - started
        client.close(env_id=0)

    return {
        "messages": messages,
        "last": messages[-1],
        "wall": wall,
        "step_times": step_times,
        "game_seconds": messages[-1].t,
    }


# --------------------------------------------------------------------------------------
# M1: API contract


def test_m1_episode_runs_to_a_terminal_state_with_a_reason(episode):
    last = episode["last"]
    assert last.done is True
    assert last.info.reason in ("elimination", "time_limit", "controlled_dead")
    assert last.info.winner_team in (None, "team-a", "team-b")
    assert len(last.agent_ids) == 4
    assert set(last.teams.values()) == {"team-a", "team-b"}
    assert last.info.metrics is not None and set(last.info.metrics) == set(last.agent_ids)


def test_m1_the_first_message_is_a_reset_at_t_zero(episode):
    first = episode["messages"][0]
    assert first.t == 0.0 and first.done is False
    assert set(first.obs) == set(first.agent_ids)


# --------------------------------------------------------------------------------------
# M2-M4 consequences: the adapter's inputs actually did something in the world


def test_m2_m4_scripted_bots_loot_equip_and_fire(episode):
    kinds = {e["type"] for m in episode["messages"] for e in m.events}
    assert "fire" in kinds, "the chasers never fired: the fire input did not reach the engine"
    assert "damage" in kinds, "nobody took damage, so no bullet ever connected"
    # the chasers spawn armed here, so looting is opportunistic; when it happens it must be well formed
    for message in episode["messages"]:
        for event in message.events:
            if event["type"] == "loot":
                assert isinstance(event.get("item"), str) and event["item"]
                assert event.get("count", 0) >= 1


# --------------------------------------------------------------------------------------
# M6: information-set parity


def test_m6_every_observation_is_allowlisted_and_hides_enemy_state(episode):
    for message in episode["messages"]:
        for agent_id, obs in message.obs.items():
            validate_agent_observation(obs, where=f"t={message.t}:{agent_id}")
            for enemy in obs.get("players") or []:
                assert "hp" not in enemy, "an enemy's HP is not sent to a client"
                assert "inventory" not in enemy and "clip" not in enemy
            for mate in obs.get("teammates") or []:
                # group status does carry a teammate's HP, and only these fields
                assert set(mate) <= {"id", "pos", "dist", "hp", "downed", "dead"}


def test_m6_teammates_are_always_known_and_enemies_are_not(episode):
    """Teammates come from the team HUD regardless of sight; enemies must be culled."""
    saw_hidden_enemy = False
    for message in episode["messages"]:
        for agent_id, obs in message.obs.items():
            mates = [m["id"] for m in (obs.get("teammates") or [])]
            assert len(mates) == 1 and mates[0] != agent_id
            if not (obs.get("players") or []):
                saw_hidden_enemy = True
    assert saw_hidden_enemy, "at some point an agent should not have seen any enemy"


# --------------------------------------------------------------------------------------
# M7: events agree with the state they describe


def test_m7_damage_events_account_for_the_hp_that_was_lost(episode):
    """Each damage event's own before/after has to be consistent, and the events an agent
    received have to add up to what the end-of-episode metrics say it took."""
    taken: dict[str, float] = {a: 0.0 for a in episode["last"].agent_ids}
    for message in episode["messages"]:
        for event in message.events:
            if event["type"] != "damage":
                continue
            amount, before, after = event["amount"], event["hp_before"], event["hp_after"]
            # 0 is legal: `player.damage()` ran but nothing was removed, e.g. a downed player's
            # bleed landing inside `downedDamageBuffer` (source and weapon are null for those)
            assert amount >= 0
            if amount == 0:
                assert after == before
                continue
            # A damage event removes either the HP difference or everything that was left. The
            # second case is not redundant: on the hit that downs a player the engine resets HP
            # to 100 (and it then bleeds), so `hp_after` can be *higher* than `hp_before`, and a
            # killing hit zeroes it. Note `downed` is the state *after* the call, so it is also
            # true for later hits on an already-downed player — the `down` event, not this flag,
            # marks the transition.
            assert amount == pytest.approx(before - after, abs=0.5) or amount == pytest.approx(before, abs=0.5), (
                f"damage {amount} does not match hp {before} -> {after}"
            )
            taken[event["agent"]] += amount

    metrics = episode["last"].info.metrics
    for agent_id, total in taken.items():
        assert total == pytest.approx(metrics[agent_id].damage_taken, abs=0.5)


def test_m7_a_down_precedes_the_kill_of_the_same_agent(episode):
    ordered = [e for m in episode["messages"] for e in m.events if e["type"] in ("down", "kill")]
    downed: set[str] = set()
    killed: set[str] = set()
    for event in ordered:
        agent = event["agent"]
        if event["type"] == "down":
            assert agent not in killed, f"{agent} was downed after it was killed"
            downed.add(agent)
        else:
            assert agent not in killed, f"{agent} was killed twice"
            killed.add(agent)
    # in a duo the second death of a pair is a kill without a down (the engine finishes the downed
    # teammate), so a kill without a preceding down is expected; a down without a kill is not
    assert killed, "the stationary duo should have been killed"


def test_m7_fire_events_are_attributed_to_an_armed_agent(episode):
    for message in episode["messages"]:
        for event in message.events:
            if event["type"] != "fire":
                continue
            assert event["agent"] in message.agent_ids
            assert isinstance(event["weapon"], str) and event["weapon"] not in ("", "fists")


# --------------------------------------------------------------------------------------
# M8: data and evaluation are connected


def test_m8_the_episode_becomes_a_valid_harness_episode_with_a_metric_vector(episode, tmp_path):
    record = _as_eval_record(episode)
    trajectory = to_episode_trajectory(record)

    errors = validate_episode(trajectory)
    assert errors == [], f"validate_episode reported {len(errors)} errors: {errors[:5]}"

    metrics = compute_metrics(trajectory)
    add_final_metrics(trajectory, metrics)
    assert set(metrics) == set(episode["last"].agent_ids)

    # the four groups exist for every agent, and the numbers are the engine's
    bridge_metrics = episode["last"].info.metrics
    for agent_id, m in metrics.items():
        assert set(m) >= {"combat", "survival", "cooperation", "movement"}
        assert m["combat"]["damageTaken"] == pytest.approx(bridge_metrics[agent_id].damage_taken, abs=0.6)
        assert m["survival"]["aliveAtEnd"] == bridge_metrics[agent_id].alive_at_end
        assert m["cooperation"]["applicable"] is True

    # combat(damageDealt > 0) for at least one agent, as the DoD asks
    assert any(m["combat"]["damageDealt"] > 0 for m in metrics.values())

    # and it survives a JSONL round trip
    import json

    path = tmp_path / "phase0.jsonl"
    path.write_text(json.dumps(trajectory, default=float) + "\n", encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert validate_episode(reloaded) == []


# --------------------------------------------------------------------------------------
# S1 / S2: throughput and latency, recorded rather than asserted hard


def test_s1_s2_throughput_is_reported(episode):
    wall = episode["wall"]
    game_seconds = episode["game_seconds"]
    speed = game_seconds / wall if wall > 0 else float("inf")
    step_times = episode["step_times"]
    mean_ms = 1000 * sum(step_times) / len(step_times)
    worst_ms = 1000 * max(step_times)
    print(
        f"\n[phase0] {game_seconds:.1f}s game time in {wall:.2f}s wall = {speed:.0f}x real time; "
        f"step {mean_ms:.2f}ms mean / {worst_ms:.2f}ms worst over {len(step_times)} steps"
    )
    # S1 wants >= 20x for one process; recorded, not enforced, because a shared CI box is noisy
    assert speed > 1.0, "the episode ran slower than real time, which no configuration should"


# --------------------------------------------------------------------------------------


def _as_eval_record(episode: dict[str, Any]) -> dict[str, Any]:
    """The message stream in the shape `harness_export` consumes.

    `eval.run_episodes` builds this while it rolls out; here the episode was driven directly, so
    the same record is assembled from the messages: each step holds the observation its action was
    chosen from, and the terminal observation goes in `final_obs`.
    """
    messages = episode["messages"]
    last = episode["last"]
    steps = [
        {
            "t": previous.t,
            "t_next": following.t,
            "obs": dict(previous.obs),
            "actions": {aid: {"cpc": {"move": {"x": 0.0, "y": 0.0}, "aim": {"x": 0.0, "y": 0.0},
                                      "fire": {"start": False, "hold": False}}}
                        for aid in CONTROLLED},
            "rewards": {},
            "events": list(following.events),
        }
        for previous, following in zip(messages, messages[1:])
    ]
    return {
        "episode": 0,
        "seed": SEED,
        "controlled": list(CONTROLLED),
        "agent_ids": list(last.agent_ids),
        "teams": dict(last.teams),
        "ticks": TICKS,
        "length": len(steps),
        "duration_s": last.t,
        "winner_team": last.info.winner_team,
        "reason": last.info.reason,
        "metrics": {aid: m.to_json() for aid, m in (last.info.metrics or {}).items()},
        "steps": steps,
        "final_obs": dict(last.obs),
    }
