from __future__ import annotations

import math
import pathlib
import subprocess
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.move_score import TacticalMoveScorer
from baselines.move_score.move_candidate_utils import get_move_bin_vectors, simulate_candidate_position
from baselines.move_score.move_score_terms import map_boundary_penalty, obstacle_path_collision_penalty


def _snapshot(
    *,
    self_pos: tuple[float, float] = (100.0, 100.0),
    enemy_pos: tuple[float, float] | None = (220.0, 100.0),
    obstacles: list[dict] | None = None,
    map_size: tuple[float, float] = (500.0, 500.0),
    move_speed: float = 20.0,
    radius: float = 10.0,
    fire_range: float = 100.0,
) -> dict:
    state = {
        "self_pos": {"x": self_pos[0], "y": self_pos[1]},
        "enemy_pos": {"x": enemy_pos[0], "y": enemy_pos[1]} if enemy_pos is not None else {"x": 0.0, "y": 0.0},
        "self_hp": 100.0,
        "enemy_hp": 100.0 if enemy_pos is not None else 0.0,
    }
    return {
        "dt": 1.0,
        "state": state,
        "map": {
            "width": map_size[0],
            "height": map_size[1],
            "obstacles": list(obstacles or []),
        },
        "combat": {"fire_range": fire_range},
        "agents": {
            "self": {
                "position": state["self_pos"],
                "radius": radius,
                "move_speed": move_speed,
                "hp": 100.0,
            },
            "enemy": {
                "position": state["enemy_pos"],
                "radius": radius,
                "hp": state["enemy_hp"],
            },
        },
    }


def _obs(snapshot: dict) -> dict:
    state = snapshot["state"]
    return {
        "self_pos": state["self_pos"],
        "enemy_pos": state["enemy_pos"],
        "enemy_hp": state["enemy_hp"],
    }


def _candidate_pos(snapshot: dict, move_bin: int) -> dict[str, float]:
    move_dx, move_dy = get_move_bin_vectors()[move_bin]
    self_pos = snapshot["state"]["self_pos"]
    x, y = simulate_candidate_position(
        self_pos["x"],
        self_pos["y"],
        move_dx,
        move_dy,
        snapshot["agents"]["self"]["move_speed"],
        snapshot["dt"],
    )
    return {"x": x, "y": y}


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def test_move_scorer_does_not_choose_obstacle_collision():
    snapshot = _snapshot(
        self_pos=(100.0, 100.0),
        enemy_pos=(260.0, 100.0),
        obstacles=[{"id": "block", "type": "circle", "x": 125.0, "y": 100.0, "radius": 14.0}],
    )

    selected, _ = TacticalMoveScorer().choose_move(_obs(snapshot), snapshot)

    selected_pos = _candidate_pos(snapshot, selected)
    penalty = obstacle_path_collision_penalty(
        snapshot["state"]["self_pos"],
        selected_pos,
        snapshot["map"]["obstacles"],
        snapshot["agents"]["self"]["radius"],
    )
    assert penalty == 0.0


def test_move_scorer_does_not_choose_out_of_bounds_move():
    snapshot = _snapshot(
        self_pos=(188.0, 100.0),
        enemy_pos=(198.0, 100.0),
        map_size=(200.0, 200.0),
        move_speed=20.0,
        radius=10.0,
    )

    selected, _ = TacticalMoveScorer().choose_move(_obs(snapshot), snapshot)

    selected_pos = _candidate_pos(snapshot, selected)
    assert map_boundary_penalty(selected_pos, 200.0, 200.0, 10.0) == 0.0


def test_move_scorer_approaches_when_enemy_is_far():
    snapshot = _snapshot(self_pos=(100.0, 100.0), enemy_pos=(300.0, 100.0), fire_range=100.0)
    current_dist = _distance(snapshot["state"]["self_pos"], snapshot["state"]["enemy_pos"])

    selected, debug = TacticalMoveScorer().choose_move(_obs(snapshot), snapshot)

    selected_pos = _candidate_pos(snapshot, selected)
    assert _distance(selected_pos, snapshot["state"]["enemy_pos"]) < current_dist
    assert debug["selected_move_bin"] == selected


def test_move_scorer_backs_off_or_strafes_when_enemy_is_too_close():
    snapshot = _snapshot(self_pos=(100.0, 100.0), enemy_pos=(125.0, 100.0), fire_range=100.0)
    current_dist = _distance(snapshot["state"]["self_pos"], snapshot["state"]["enemy_pos"])

    selected, _ = TacticalMoveScorer().choose_move(_obs(snapshot), snapshot)

    selected_pos = _candidate_pos(snapshot, selected)
    assert _distance(selected_pos, snapshot["state"]["enemy_pos"]) >= current_dist


def test_move_scorer_keeps_strafing_near_ideal_range():
    snapshot = _snapshot(self_pos=(420.0, 380.0), enemy_pos=(600.0, 400.0), fire_range=260.0)

    selected, debug = TacticalMoveScorer().choose_move(_obs(snapshot), snapshot)

    assert selected != 0
    assert debug["candidate_scores"][selected]["strafe_score"] > 0.0


def test_move_scorer_returns_score_breakdown_for_all_candidates():
    snapshot = _snapshot()
    scorer = TacticalMoveScorer()

    selected_a, debug_a = scorer.choose_move(_obs(snapshot), snapshot)
    selected_b, debug_b = scorer.choose_move(_obs(snapshot), snapshot)

    assert selected_a == selected_b
    assert debug_a["candidate_scores"] == debug_b["candidate_scores"]
    assert set(debug_a["candidate_scores"]) == set(get_move_bin_vectors())
    for score in debug_a["candidate_scores"].values():
        assert {
            "total",
            "collision_penalty",
            "boundary_penalty",
            "spacing_score",
            "threat_penalty",
            "strafe_score",
            "candidate_pos",
        }.issubset(score)


def test_move_scorer_no_enemy_returns_valid_move():
    snapshot = _snapshot(enemy_pos=None)

    selected, debug = TacticalMoveScorer().choose_move(_obs(snapshot), snapshot)

    assert selected in get_move_bin_vectors()
    assert debug["selected_move_bin"] == selected


def test_move_score_debug_runner_runs():
    result = subprocess.run(
        [
            sys.executable,
            "experiment/baselines/move_score/run_move_score_debug.py",
            "--config",
            "configs/env/manual_enemy_far_right.yaml",
            "--steps",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "selected_move_bin=4" in result.stdout
    assert "action={'move': 4, 'aim': 0, 'fire': 0}" in result.stdout
