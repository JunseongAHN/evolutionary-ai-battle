from __future__ import annotations

import pathlib
import subprocess
import sys

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.tactical_baseline import FireRule, build_tactical_baseline_bot
from baselines.tactical_baseline.run_tactical_autoplay import run_tactical_autoplay
from core.cpc_actions import decode_action
from core.cpc_env import CPCEnv
from core.local_occupancy_grid import build_local_occupancy_grid
from core.env_config import load_env_config


def test_tactical_baseline_bot_returns_valid_action():
    env, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_enemy_right.yaml")
    bot = build_tactical_baseline_bot(snapshot)

    action, debug = bot.act({**obs, "local_occupancy_grid": build_local_occupancy_grid(snapshot)}, snapshot)

    assert {"move", "aim", "fire", "move_bin", "aim_bin"}.issubset(action)
    assert action["move"] == action["move_bin"]
    assert action["aim"] == action["aim_bin"]
    decode_action(action)
    assert {"aim", "move", "fire"}.issubset(debug)
    assert debug["action"] == action
    assert env.step_count == 0


def test_tactical_baseline_bot_no_enemy_does_not_crash():
    _, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_enemy_right.yaml")
    obs = {**obs, "enemy_hp": 0.0}
    snapshot["state"]["enemy_hp"] = 0.0
    snapshot["agents"]["enemy"]["hp"] = 0.0
    snapshot["agents"]["enemy"]["alive"] = False
    bot = build_tactical_baseline_bot(snapshot)

    action, debug = bot.act({**obs, "local_occupancy_grid": build_local_occupancy_grid(snapshot)}, snapshot)

    decode_action(action)
    assert action["fire"] == 0
    assert debug["fire"]["reason"] == "no_live_enemy"


def test_tactical_baseline_debug_contains_aim_move_fire_sections():
    _, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_enemy_close.yaml")
    bot = build_tactical_baseline_bot(snapshot)

    _, debug = bot.act({**obs, "local_occupancy_grid": build_local_occupancy_grid(snapshot)}, snapshot)

    assert set(["aim", "move", "fire"]).issubset(debug)
    assert "reason" in debug["aim"]
    assert "reason" in debug["move"]
    assert "reason" in debug["fire"]


def test_fire_rule_blocks_obstacle_line_of_sight():
    _, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_obstacle_between.yaml")
    fire_rule = FireRule()

    fire, debug = fire_rule.decide_fire({**obs, "selected_aim_bin": 0}, snapshot)

    assert fire == 0
    assert debug["line_of_sight"] is False
    assert debug["reason"] == "line_of_sight_blocked"


def test_tactical_autoplay_runs_for_n_steps_without_manual_input():
    result = run_tactical_autoplay(
        config_path="configs/env/autoplay_enemy_far.yaml",
        steps=5,
        fps=0,
        render=False,
        save_png=False,
        print_debug=False,
        stop_on_done=True,
    )

    assert result["steps_run"] == 5


def test_tactical_autoplay_runner_script_runs():
    result = subprocess.run(
        [
            sys.executable,
            "experiment/baselines/tactical_baseline/run_tactical_autoplay.py",
            "--config",
            "configs/env/autoplay_enemy_right.yaml",
            "--steps",
            "2",
            "--fps",
            "0",
            "--print-debug",
            "--print-every",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Reset config=configs/env/autoplay_enemy_right.yaml" in result.stdout
    assert "step=0" in result.stdout
    assert "Finished steps=2" in result.stdout


def _env_obs_snapshot(config_path: str):
    env = CPCEnv.from_config(load_env_config(config_path))
    obs = env.reset()
    return env, obs, env.get_debug_state()
