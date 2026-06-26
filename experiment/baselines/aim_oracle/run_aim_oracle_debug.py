from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiment"
for path in (REPO_ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from experiment.baselines.aim_oracle.tactical_aim_oracle_bot import TacticalAimOracleBot
    from experiment.core.env_config import load_env_config
    from experiment.core.local_occupancy_grid import CHANNEL_ENEMY, build_local_occupancy_grid, render_grid_to_png
    from experiment.training.cpc_actions import AIM_BINS
    from experiment.training.cpc_env import CPCEnv
except ModuleNotFoundError:
    from baselines.aim_oracle.tactical_aim_oracle_bot import TacticalAimOracleBot
    from core.env_config import load_env_config
    from core.local_occupancy_grid import CHANNEL_ENEMY, build_local_occupancy_grid, render_grid_to_png
    from training.cpc_actions import AIM_BINS
    from training.cpc_env import CPCEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local-grid aim-oracle debug baseline.")
    parser.add_argument("--config", required=True, help="Path to an env YAML config.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-dir", default="experiment/runs/aim_oracle_debug")
    parser.add_argument("--save-png", action="store_true", help="Save local occupancy grid PNGs per step.")
    parser.add_argument("--num-aim-bins", type=int, default=AIM_BINS)
    parser.add_argument("--enemy-channel-index", type=int)
    parser.add_argument("--cell-size", type=float)
    parser.add_argument("--stay-move-bin", type=int, default=0)
    args = parser.parse_args()

    run_aim_oracle_debug(
        config_path=args.config,
        steps=args.steps,
        output_dir=pathlib.Path(args.output_dir),
        save_png=args.save_png,
        num_aim_bins=args.num_aim_bins,
        enemy_channel_index=args.enemy_channel_index,
        cell_size=args.cell_size,
        stay_move_bin=args.stay_move_bin,
    )


def run_aim_oracle_debug(
    *,
    config_path: str,
    steps: int,
    output_dir: pathlib.Path,
    save_png: bool,
    num_aim_bins: int,
    enemy_channel_index: int | None,
    cell_size: float | None,
    stay_move_bin: int,
) -> None:
    config = load_env_config(config_path)
    env = CPCEnv.from_config(config)
    obs = env.reset(seed=config.seed)
    initial_grid = build_local_occupancy_grid(env.get_debug_state(), agent_id="self")
    enemy_channel = (
        int(enemy_channel_index)
        if enemy_channel_index is not None
        else initial_grid.channel_index(CHANNEL_ENEMY)
    )
    bot = TacticalAimOracleBot(
        num_aim_bins=num_aim_bins,
        enemy_channel_index=enemy_channel,
        cell_size=float(cell_size if cell_size is not None else initial_grid.cell_size),
        stay_move_bin=stay_move_bin,
        default_aim_bin=0,
    )

    if save_png:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "reset"
        f" | config={config_path}"
        f" | seed={config.seed}"
        f" | grid_shape={initial_grid.shape}"
        f" | channels={list(initial_grid.channel_names)}"
        f" | enemy_channel_index={enemy_channel}"
    )

    for step_index in range(max(0, int(steps))):
        debug_state = env.get_debug_state()
        grid = build_local_occupancy_grid(debug_state, agent_id="self")
        action, oracle_debug = bot.act({**obs, "local_occupancy_grid": grid})
        png_path = _save_grid_png(grid, output_dir, config_path, step_index) if save_png else None
        obs, reward, done, info = env.step(action)
        _print_step(
            step_index=step_index,
            debug_state=debug_state,
            oracle_debug=oracle_debug,
            action=action,
            reward=reward,
            done=done,
            env=env,
            info=info,
            png_path=png_path,
        )
        if done:
            break


def _save_grid_png(grid: Any, output_dir: pathlib.Path, config_path: str, step_index: int) -> pathlib.Path:
    stem = pathlib.Path(config_path).stem
    path = output_dir / f"{stem}_step_{step_index:03d}_grid.png"
    render_grid_to_png(grid, path)
    return path


def _print_step(
    *,
    step_index: int,
    debug_state: dict[str, Any],
    oracle_debug: dict[str, Any],
    action: dict[str, int],
    reward: float,
    done: bool,
    env: CPCEnv,
    info: dict[str, Any],
    png_path: pathlib.Path | None,
) -> None:
    player = _agent(debug_state, "self")
    enemy = _agent(debug_state, "enemy")
    truncated = bool(done and env.step_count >= env.max_steps)
    terminated = bool(done and not truncated)
    events = _important_events(info)
    parts = [
        f"step={step_index}",
        f"player={_format_position(player)}",
        f"enemy={_format_position(enemy)}",
        f"enemy_cell={oracle_debug.get('enemy_cell')}",
        f"local_vector={oracle_debug.get('local_vector')}",
        f"aim_bin={oracle_debug.get('aim_bin')}",
        f"action={action}",
        f"reward={float(reward):.4f}",
        f"done={bool(done)}",
        f"terminated={terminated}",
        f"truncated={truncated}",
        f"reason={oracle_debug.get('reason')}",
    ]
    if events:
        parts.append(f"events={','.join(events)}")
    if png_path is not None:
        parts.append(f"png={png_path}")
    print(" | ".join(parts))


def _agent(debug_state: dict[str, Any], key: str) -> dict[str, Any]:
    agents = debug_state.get("agents", {})
    if key in agents:
        return dict(agents[key])
    state = debug_state.get("state", {})
    if key == "self":
        return {"position": state.get("self_pos", {"x": 0.0, "y": 0.0})}
    return {"position": state.get("enemy_pos", {"x": 0.0, "y": 0.0})}


def _format_position(agent: dict[str, Any]) -> str:
    position = agent.get("position", {})
    return f"({float(position.get('x', 0.0)):.1f},{float(position.get('y', 0.0)):.1f})"


def _important_events(info: dict[str, Any]) -> list[str]:
    events: list[str] = []
    for event in info.get("bullet_events", []):
        event_type = event.get("type")
        if event_type in {"bullet_spawned", "bullet_hit", "bullet_hit_obstacle", "bullet_expired", "bullet_not_spawned"}:
            events.append(str(event_type))
    return events


if __name__ == "__main__":
    main()
