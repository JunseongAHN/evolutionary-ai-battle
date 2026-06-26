from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

EXPERIMENT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from baselines.move_score import TacticalMoveScoreBot, TacticalMoveScorer
from core.env_config import load_env_config
from core.local_occupancy_grid import build_local_occupancy_grid, render_grid_to_png
from training.cpc_env import CPCEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPC tactical move-score baseline debug loop.")
    parser.add_argument("--config", default="configs/env/manual_enemy_far_right.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", default="experiment/runs/move_score_debug")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--print-score-breakdown", action="store_true")
    args = parser.parse_args()

    config = load_env_config(args.config)
    env = CPCEnv.from_config(config)
    obs = env.reset(seed=config.seed)
    bot = TacticalMoveScoreBot(TacticalMoveScorer())
    output_dir = pathlib.Path(args.output_dir)

    print(f"Reset config={args.config} seed={config.seed}")
    print_step_header("reset", env.get_debug_state())

    for step_index in range(max(0, args.steps)):
        snapshot = env.get_debug_state()
        action, debug = bot.act(obs, state_snapshot=snapshot)
        print_decision(step_index, snapshot, action, debug, print_breakdown=args.print_score_breakdown)
        obs, reward, done, info = env.step(action)
        print_result(step_index, env, action, reward, done, info)
        if args.save_png:
            saved = save_grid_png(env, output_dir, pathlib.Path(args.config).stem, step_index)
            print(f"saved_png={saved}")
        if done:
            break


def print_step_header(label: str, snapshot: dict[str, Any]) -> None:
    player = snapshot["agents"]["self"]
    enemy = snapshot["agents"]["enemy"]
    print(
        f"{label} | "
        f"player=({player['position']['x']:.1f},{player['position']['y']:.1f}) | "
        f"nearest_enemy=({enemy['position']['x']:.1f},{enemy['position']['y']:.1f})"
    )


def print_decision(
    step_index: int,
    snapshot: dict[str, Any],
    action: dict[str, int],
    debug: dict[str, Any],
    *,
    print_breakdown: bool,
) -> None:
    player = snapshot["agents"]["self"]
    enemy = snapshot["agents"]["enemy"]
    selected = debug["candidate_scores"][debug["selected_move_bin"]]
    print(
        f"step={step_index} | "
        f"player=({player['position']['x']:.1f},{player['position']['y']:.1f}) | "
        f"nearest_enemy=({enemy['position']['x']:.1f},{enemy['position']['y']:.1f}) | "
        f"selected_move_bin={debug['selected_move_bin']} | "
        f"selected_candidate=({selected['candidate_pos'][0]:.1f},{selected['candidate_pos'][1]:.1f}) | "
        f"action={action}"
    )
    if print_breakdown:
        print_score_table(debug["candidate_scores"])


def print_score_table(candidate_scores: dict[int, dict[str, Any]]) -> None:
    print("  bin | total | collision | boundary | spacing | threat | strafe | candidate")
    for move_bin in sorted(candidate_scores):
        row = candidate_scores[move_bin]
        pos = row["candidate_pos"]
        print(
            f"  {move_bin:>3} | "
            f"{row['total']:>7.2f} | "
            f"{row['collision_penalty']:>9.2f} | "
            f"{row['boundary_penalty']:>8.2f} | "
            f"{row['spacing_score']:>7.2f} | "
            f"{row['threat_penalty']:>6.2f} | "
            f"{row['strafe_score']:>6.2f} | "
            f"({pos[0]:.1f},{pos[1]:.1f})"
        )


def print_result(
    step_index: int,
    env: CPCEnv,
    action: dict[str, int],
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> None:
    truncated = bool(done and env.step_count >= env.max_steps)
    terminated = bool(done and not truncated)
    events = important_events(info)
    print(
        f"result step={step_index} | "
        f"env_step={env.step_count} | "
        f"action_sent={action} | "
        f"reward={reward:.4f} | "
        f"done={done} | "
        f"terminated={terminated} | "
        f"truncated={truncated} | "
        f"events={','.join(events) if events else 'none'}"
    )


def important_events(info: dict[str, Any]) -> list[str]:
    events: list[str] = []
    for event in info.get("bullet_events", []):
        event_type = event.get("type")
        if event_type in {"bullet_spawned", "bullet_hit", "bullet_hit_obstacle", "bullet_expired", "bullet_not_spawned"}:
            events.append(str(event_type))
    return events


def save_grid_png(env: CPCEnv, output_dir: pathlib.Path, config_stem: str, step_index: int) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_local_occupancy_grid(env.get_debug_state(), agent_id="self")
    path = output_dir / f"{config_stem}_step_{step_index:03d}_grid.png"
    render_grid_to_png(grid, path)
    return path


if __name__ == "__main__":
    main()
