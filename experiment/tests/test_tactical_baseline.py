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
from baselines.tactical_baseline.run_tactical_autoplay import run_tactical_autoplay, tactical_debug_fields
from core.cpc_actions import decode_action
from core.cpc_env import CPCEnv
from core.local_occupancy_grid import build_local_occupancy_grid
from core.env_config import load_env_config
from gui.pygame_viewer import _panel_lines


def test_tactical_baseline_bot_returns_valid_action():
    env, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_enemy_right.yaml")
    bot = build_tactical_baseline_bot(snapshot)

    action, debug = bot.act({**obs, "local_occupancy_grid": build_local_occupancy_grid(snapshot)}, snapshot)

    assert {"move", "aim_dx", "aim_dy", "fire", "move_bin"}.issubset(action)
    assert action["move"] == action["move_bin"]
    assert action["aim_dx"] == 1.0
    assert action["aim_dy"] == 0.0
    decode_action(action)
    assert {"mode", "aim", "move", "fire"}.issubset(debug)
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


def test_tactical_baseline_aims_at_left_enemy_with_continuous_direction():
    _, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_enemy_left.yaml")
    bot = build_tactical_baseline_bot(snapshot)

    action, debug = bot.act(obs, snapshot)

    assert action["aim_dx"] == -1.0
    assert action["aim_dy"] == 0.0
    assert debug["aim"]["aim_source"] == "enemy_position"
    assert debug["fire"]["sources"]["aim_error"] == "continuous_aim_and_enemy_positions"


def test_tactical_baseline_debug_contains_mode_aim_move_fire_sections():
    _, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_enemy_close.yaml")
    bot = build_tactical_baseline_bot(snapshot)

    _, debug = bot.act({**obs, "local_occupancy_grid": build_local_occupancy_grid(snapshot)}, snapshot)

    assert set(["mode", "aim", "move", "fire"]).issubset(debug)
    assert "reason" in debug["mode"]
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


def test_tactical_autoplay_passes_continuous_aim_to_env():
    result = run_tactical_autoplay(
        config_path="configs/env/autoplay_enemy_left.yaml",
        steps=1,
        fps=0,
        render=False,
        save_png=False,
        print_debug=False,
        stop_on_done=True,
    )

    assert result["info"]["decoded_action"]["aimX"] == -1.0
    assert result["info"]["decoded_action"]["aimY"] == 0.0


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
            "--show-tactical-debug",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Reset config=configs/env/autoplay_enemy_right.yaml" in result.stdout
    assert "step=0" in result.stdout
    assert "step=1" in result.stdout
    assert "tactical_mode=" in result.stdout
    assert "target_cell=" in result.stdout
    assert "next_cell=" in result.stdout
    assert "move_bin=" in result.stdout
    assert "aim_dir=" in result.stdout
    assert "fire=" in result.stdout
    assert "Finished steps=2" in result.stdout


def test_tactical_debug_fields_match_bot_action_and_plan():
    _, obs, snapshot = _env_obs_snapshot("configs/env/autoplay_enemy_in_range.yaml")
    bot = build_tactical_baseline_bot(snapshot)
    action, debug = bot.act({**obs, "local_occupancy_grid": build_local_occupancy_grid(snapshot)}, snapshot)

    overlay = tactical_debug_fields(action, debug)

    assert overlay == {
        "tactical_mode": debug["mode"]["mode"],
        "target_cell": debug["move"]["target_cell"],
        "next_cell": debug["move"]["next_cell"],
        "move_bin": action["move_bin"],
        "aim_dir_x": action["aim_dx"],
        "aim_dir_y": action["aim_dy"],
        "fire": action["fire"],
    }
    panel_lines = _panel_lines({"tactical_debug": overlay}, None)
    assert f"tactical_mode: {overlay['tactical_mode']}" in panel_lines
    assert f"target_cell: {overlay['target_cell']}" in panel_lines
    assert f"next_cell: {overlay['next_cell']}" in panel_lines
    assert f"move_bin: {overlay['move_bin']}" in panel_lines
    assert f"aim_dir: ({overlay['aim_dir_x']:.2f}, {overlay['aim_dir_y']:.2f})" in panel_lines
    assert f"fire: {overlay['fire']}" in panel_lines


def _env_obs_snapshot(config_path: str):
    env = CPCEnv.from_config(load_env_config(config_path))
    obs = env.reset()
    return env, obs, env.get_debug_state()
