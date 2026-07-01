from __future__ import annotations

import math
import pathlib
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
for path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.tactical_baseline import (
    bfs_reachable,
    build_tactical_baseline_bot,
    first_step_to_move_bin,
)
from baselines.tactical_baseline.run_tactical_autoplay import run_tactical_autoplay
from core.cpc_env import CPCEnv
from core.env_config import load_env_config
from core.local_occupancy_grid import build_local_occupancy_grid


def test_bfs_reachable_does_not_cross_blocked_cells():
    blocked = [[False for _ in range(5)] for _ in range(5)]
    for row in range(5):
        blocked[row][2] = True

    reachable, parents = bfs_reachable(blocked, center=(2, 1))

    assert reachable[2][1]
    assert all(not reachable[row][3] for row in range(5))
    assert all(cell[1] < 2 for cell in parents)


def test_bfs_prevents_diagonal_corner_cutting():
    blocked = [
        [True, False],
        [False, True],
    ]

    reachable, _ = bfs_reachable(
        blocked,
        center=(1, 0),
        allow_diagonal=True,
        prevent_corner_cutting=True,
    )

    assert not reachable[0][1]


def test_engage_mode_reduces_distance_when_enemy_far():
    _, debug = _decision("configs/env/autoplay_enemy_far.yaml")
    center = debug["move"]["center_cell"]
    target = debug["move"]["target_cell"]
    enemy = debug["move"]["enemy_cell"]

    assert debug["mode"]["mode"] == "engage"
    assert _cell_distance(target, enemy) < _cell_distance(center, enemy)


def test_kite_mode_increases_distance_when_enemy_close():
    _, debug = _decision("configs/env/autoplay_enemy_close.yaml")
    center = debug["move"]["center_cell"]
    target = debug["move"]["target_cell"]
    enemy = debug["move"]["enemy_cell"]

    assert debug["mode"]["mode"] == "kite"
    assert _cell_distance(target, enemy) > _cell_distance(center, enemy)


def test_hold_range_prefers_ideal_range_and_los():
    _, debug = _decision("configs/env/autoplay_enemy_in_range.yaml")
    move = debug["move"]
    selected_distance = move["selected_score"]["enemy_distance"] * 40.0

    assert debug["mode"]["mode"] == "hold_range"
    assert abs(selected_distance - move["ideal_range"]) < abs(
        move["current_enemy_distance"] - move["ideal_range"]
    )
    assert move["selected_score"]["los"] > 0.0
    assert move["selected_score"]["strafe"] > 0.0


def test_reposition_prefers_los_recovery_when_blocked():
    _, debug = _decision("configs/env/autoplay_los_blocked.yaml")
    move = debug["move"]
    blocked = {tuple(cell) for cell in move["blocked_cells"]}

    assert debug["mode"]["mode"] == "reposition"
    assert move["selected_score"]["los"] > 0.0
    assert all(tuple(cell) not in blocked for cell in move["path"])


def test_tactical_baseline_debug_contains_mode_target_path():
    action, debug = _decision("configs/env/autoplay_obstacle_between.yaml")
    move = debug["move"]
    center = tuple(move["center_cell"])
    next_cell = tuple(move["next_cell"])

    assert {"mode", "move", "aim", "fire", "action"}.issubset(debug)
    assert {"tactical_mode", "target_cell", "path", "next_cell", "move_bin", "selected_score"}.issubset(move)
    assert move["path"][0] == move["center_cell"]
    assert move["path"][-1] == move["target_cell"]
    assert action["move"] == move["move_bin"] == first_step_to_move_bin(center, next_cell)


def test_autoplay_runs_with_mode_conditioned_planner():
    result = run_tactical_autoplay(
        config_path="configs/env/autoplay_los_blocked.yaml",
        steps=3,
        fps=0,
        render=False,
        save_png=False,
        print_debug=False,
    )

    assert result["steps_run"] == 3


def _decision(config_path: str):
    env = CPCEnv.from_config(load_env_config(config_path))
    obs = env.reset()
    snapshot = env.get_debug_state()
    bot = build_tactical_baseline_bot(snapshot)
    grid = build_local_occupancy_grid(snapshot, agent_id="self")
    return bot.act({**obs, "local_occupancy_grid": grid}, snapshot)


def _cell_distance(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
