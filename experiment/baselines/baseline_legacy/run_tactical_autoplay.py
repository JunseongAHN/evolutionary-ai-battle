from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiment"
for path in (REPO_ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from experiment.core.cpc_actions import decode_action
    from experiment.core.cpc_env import CPCEnv
    from experiment.core.env_config import load_env_config
    from experiment.core.local_occupancy_grid import build_local_occupancy_grid, render_grid_to_png
except ModuleNotFoundError:
    from core.cpc_actions import decode_action
    from core.cpc_env import CPCEnv
    from core.env_config import load_env_config
    from core.local_occupancy_grid import build_local_occupancy_grid, render_grid_to_png

try:
    from experiment.baselines.baseline_legacy import build_tactical_baseline_bot
except ModuleNotFoundError:
    from baselines.baseline_legacy import build_tactical_baseline_bot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the integrated CPC tactical baseline autoplay loop.")
    parser.add_argument("--config", default="configs/env/autoplay_enemy_right.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action="store_true", help="Open the existing pygame viewer when available.")
    parser.add_argument("--render-substeps", type=int, default=3, help="Interpolated render frames per env step.")
    parser.add_argument("--save-png", action="store_true", help="Save local occupancy grid PNGs.")
    parser.add_argument("--output-dir", default="experiment/runs/tactical_autoplay")
    parser.add_argument("--print-debug", action="store_true", help="Print tactical debug fields to the console.")
    parser.add_argument(
        "--show-tactical-debug",
        action="store_true",
        help="Show tactical debug fields in the render side panel.",
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--stop-on-done", action="store_true", default=True)
    args = parser.parse_args()

    run_tactical_autoplay(
        config_path=args.config,
        steps=args.steps,
        fps=args.fps,
        seed=args.seed,
        render=args.render,
        render_substeps=args.render_substeps,
        save_png=args.save_png,
        output_dir=pathlib.Path(args.output_dir),
        print_debug=args.print_debug,
        show_tactical_debug=args.show_tactical_debug,
        print_every=args.print_every,
        stop_on_done=args.stop_on_done,
    )


def run_tactical_autoplay(
    *,
    config_path: str,
    steps: int,
    fps: float = 10.0,
    seed: int | None = None,
    render: bool = False,
    render_substeps: int = 3,
    save_png: bool = False,
    output_dir: pathlib.Path | str = pathlib.Path("experiment/runs/tactical_autoplay"),
    print_debug: bool = False,
    show_tactical_debug: bool = False,
    print_every: int = 1,
    stop_on_done: bool = True,
) -> dict[str, Any]:
    config = load_env_config(config_path)
    env = CPCEnv.from_config(config)
    reset_seed = config.seed if seed is None else int(seed)
    obs = _reset_observation(env.reset(seed=reset_seed))
    snapshot = get_state_snapshot_if_available(env)
    bot = build_tactical_baseline_bot(snapshot)
    viewer = _create_viewer(render, fps, render_substeps)
    output_path = pathlib.Path(output_dir)
    if save_png:
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reset config={config_path} seed={reset_seed}")
    done = False
    terminated = False
    truncated = False
    steps_run = 0
    last_info: dict[str, Any] = {}
    last_reward = 0.0
    try:
        for step_index in range(max(0, int(steps))):
            frame_start = time.perf_counter()
            snapshot = get_state_snapshot_if_available(env)
            grid = build_local_occupancy_grid(snapshot, agent_id="self")
            bot_obs = {**obs, "local_occupancy_grid": grid}
            action, bot_debug = bot.act(bot_obs, state_snapshot=snapshot)
            env_action = action_for_env(action)
            result = env.step(env_action)
            obs, last_reward, terminated, truncated, done, last_info = unpack_step_result(result)
            steps_run += 1

            if save_png:
                save_debug_frame(env, output_path, pathlib.Path(config_path).stem, step_index)
            if print_debug and step_index % max(1, int(print_every)) == 0:
                print_step_summary(
                    step_index=step_index,
                    env=env,
                    action=action,
                    reward=last_reward,
                    done=done,
                    info=last_info,
                    bot_debug=bot_debug,
                )
            if viewer is not None:
                state = get_state_snapshot_if_available(env)
                state["bullet_events"] = last_info.get("bullet_events", [])
                state["manual_step"] = {
                    "mode": "tactical_autoplay",
                    "current_action": _action_label(action),
                    "controls": ["Q/Esc quit"],
                    "save_button": False,
                }
                if show_tactical_debug:
                    state["tactical_debug"] = tactical_debug_fields(action, bot_debug)
                step_record = build_step_record(action=env_action, reward=last_reward, done=done, info=last_info)
                if not render_autoplay_step(
                    viewer,
                    previous_state=snapshot,
                    current_state=state,
                    step_record=step_record,
                    render_substeps=render_substeps,
                ):
                    break
            if done and stop_on_done:
                break
            if viewer is None:
                sleep_to_match_fps(frame_start, fps)
    finally:
        if viewer is not None:
            viewer.close()

    print(f"Finished steps={steps_run} done={done} terminated={terminated} truncated={truncated}")
    return {
        "steps_run": steps_run,
        "done": bool(done),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward": float(last_reward),
        "info": last_info,
    }


def get_state_snapshot_if_available(env: Any) -> dict[str, Any]:
    if hasattr(env, "get_debug_state"):
        return dict(env.get_debug_state())
    return {}


def action_for_env(action: Mapping[str, Any]) -> dict[str, int | float]:
    env_action: dict[str, int | float] = {
        "move": int(action.get("move", action.get("move_bin", 0))),
        "fire": int(action.get("fire", 0)),
    }
    for key in ("aim_dx", "aim_dy", "aim_angle", "aim_x", "aim_y"):
        if key in action:
            env_action[key] = float(action[key])
    if not any(key in env_action for key in ("aim_dx", "aim_dy", "aim_angle", "aim_x", "aim_y")):
        env_action["aim"] = int(action.get("aim", action.get("aim_bin", 0)))
    return env_action


def unpack_step_result(result: Any) -> tuple[dict[str, Any], float, bool, bool, bool, dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError(f"env.step must return a tuple, got {type(result).__name__}")
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return dict(obs), float(reward), bool(terminated), bool(truncated), done, dict(info)
    if len(result) == 4:
        obs, reward, done, info = result
        truncated = bool(done and bool(info.get("TimeLimit.truncated", False))) if isinstance(info, dict) else False
        terminated = bool(done and not truncated)
        return dict(obs), float(reward), bool(terminated), bool(truncated), bool(done), dict(info)
    raise ValueError(f"env.step returned {len(result)} values; expected 4 or 5")


def _reset_observation(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, _ = result
        return dict(obs)
    return dict(result)


def save_debug_frame(env: CPCEnv, output_dir: pathlib.Path, config_stem: str, step_index: int) -> pathlib.Path:
    grid = build_local_occupancy_grid(env.get_debug_state(), agent_id="self")
    path = output_dir / f"{config_stem}_step_{step_index:03d}_grid.png"
    render_grid_to_png(grid, path)
    return path


def print_step_summary(
    *,
    step_index: int,
    env: CPCEnv,
    action: dict[str, int],
    reward: float,
    done: bool,
    info: dict[str, Any],
    bot_debug: dict[str, Any],
) -> None:
    snapshot = env.get_debug_state()
    self_agent = snapshot["agents"]["self"]
    enemy_agent = snapshot["agents"]["enemy"]
    fire_debug = bot_debug.get("fire", {})
    mode_debug = bot_debug.get("mode", {})
    move_debug = bot_debug.get("move", {})
    aim_debug = bot_debug.get("aim", {})
    events = _important_events(info)
    goal = snapshot.get("goal", {})
    target_dir = aim_debug.get("aim_direction") or aim_debug.get("target_dir") or []
    aim_dir = _decoded_aim_direction(info)
    parts = [
        f"step={step_index}",
        f"env_step={env.step_count}",
        f"player=({_fmt(self_agent['position']['x'])},{_fmt(self_agent['position']['y'])})",
        f"enemy=({_fmt(enemy_agent['position']['x'])},{_fmt(enemy_agent['position']['y'])})",
        f"goal={_debug_position(goal.get('position'))}",
        f"dist_goal={_optional_fmt(goal.get('distance'))}",
        f"goal_count={int(goal.get('reached_count', 0))}",
        f"tactical_mode={mode_debug.get('mode')}",
        f"target_cell={move_debug.get('target_cell')}",
        f"next_cell={move_debug.get('next_cell')}",
        f"move_bin={int(action.get('move_bin', action.get('move', 0)))}",
        f"aim_dir=({_fmt(aim_dir['x'])},{_fmt(aim_dir['y'])})",
        f"target_dir=({_fmt(target_dir[0]) if len(target_dir) > 0 else '-'},"
        f"{_fmt(target_dir[1]) if len(target_dir) > 1 else '-'})",
        f"fire={int(action.get('fire', 0))}",
        f"path={move_debug.get('path')}",
        f"fire_reason={fire_debug.get('reason')}",
        f"reward={float(reward):.4f}",
        f"done={done}",
    ]
    if events:
        parts.append(f"events={','.join(events)}")
    print(" | ".join(parts))


def tactical_debug_fields(action: Mapping[str, Any], bot_debug: Mapping[str, Any]) -> dict[str, Any]:
    mode_debug = bot_debug.get("mode", {})
    move_debug = bot_debug.get("move", {})
    return {
        "tactical_mode": mode_debug.get("mode"),
        "target_cell": move_debug.get("target_cell"),
        "next_cell": move_debug.get("next_cell"),
        "move_bin": int(action.get("move_bin", action.get("move", 0))),
        "aim_dir_x": float(action.get("aim_dx", action.get("aim_x", 1.0))),
        "aim_dir_y": float(action.get("aim_dy", action.get("aim_y", 0.0))),
        "fire": int(action.get("fire", 0)),
    }


def build_step_record(*, action: dict[str, int], reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
    decoded = info.get("decoded_action") or decode_action(action)
    return {
        "agents": {
            "agent": {
                "decoded_action": {
                    "move_x": float(decoded.get("moveX", 0.0)),
                    "move_y": float(decoded.get("moveY", 0.0)),
                    "aim_x": float(decoded.get("aimX", 1.0)),
                    "aim_y": float(decoded.get("aimY", 0.0)),
                    "fire": float(decoded.get("fire", 0.0)),
                }
            }
        },
        "env": {
            "done": bool(done),
            "rewards": {"agent": float(reward)},
            "info": info,
            "action_name": _action_label(action),
        },
    }


def render_autoplay_step(
    viewer: Any,
    *,
    previous_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    step_record: dict[str, Any],
    render_substeps: int,
) -> bool:
    substeps = max(1, int(render_substeps))
    for index in range(1, substeps + 1):
        progress = index / substeps
        state = (
            _interpolated_debug_state(previous_state, current_state, progress)
            if substeps > 1
            else deepcopy(current_state)
        )
        if not viewer.render_step(state, step_record):
            return False
    return True


def _interpolated_debug_state(
    previous_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    progress: float,
) -> dict[str, Any]:
    if progress >= 1.0:
        return deepcopy(current_state)

    state = deepcopy(current_state)
    _interpolate_state_positions(state, previous_state, current_state, progress)
    _interpolate_agent_positions(state, previous_state, current_state, progress)
    _interpolate_projectile_positions(state, progress)
    return state


def _interpolate_state_positions(
    state: dict[str, Any],
    previous_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    progress: float,
) -> None:
    state_positions = state.get("state")
    if not isinstance(state_positions, dict):
        return
    previous_positions = _mapping(previous_state.get("state"))
    current_positions = _mapping(current_state.get("state"))
    for key in ("self_pos", "ally_pos", "enemy_pos"):
        blended = _lerp_position(previous_positions.get(key), current_positions.get(key), progress)
        if blended is not None:
            state_positions[key] = blended


def _interpolate_agent_positions(
    state: dict[str, Any],
    previous_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    progress: float,
) -> None:
    agents = state.get("agents")
    if not isinstance(agents, dict):
        return
    previous_agents = _mapping(previous_state.get("agents"))
    current_agents = _mapping(current_state.get("agents"))
    for agent_id, agent in agents.items():
        if not isinstance(agent, dict):
            continue
        blended = _lerp_position(
            _mapping(previous_agents.get(agent_id)).get("position"),
            _mapping(current_agents.get(agent_id)).get("position"),
            progress,
        )
        if blended is not None:
            agent["position"] = blended


def _interpolate_projectile_positions(state: dict[str, Any], progress: float) -> None:
    for key in ("bullets", "projectiles"):
        projectiles = state.get(key)
        if not isinstance(projectiles, list):
            continue
        for projectile in projectiles:
            if not isinstance(projectile, dict):
                continue
            previous_pos = projectile.get("previous_pos") or projectile.get("spawn_pos")
            current_pos = projectile.get("pos") or projectile.get("position")
            blended = _lerp_position(previous_pos, current_pos, progress)
            if blended is None:
                continue
            if "pos" in projectile:
                projectile["pos"] = dict(blended)
            if "position" in projectile:
                projectile["position"] = dict(blended)


def _lerp_position(start: Any, end: Any, progress: float) -> dict[str, float] | None:
    if not _is_position(start) or not _is_position(end):
        return None
    t = max(0.0, min(1.0, float(progress)))
    return {
        "x": float(start["x"]) + ((float(end["x"]) - float(start["x"])) * t),
        "y": float(start["y"]) + ((float(end["y"]) - float(start["y"])) * t),
    }


def _is_position(value: Any) -> bool:
    return isinstance(value, Mapping) and "x" in value and "y" in value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def sleep_to_match_fps(frame_start: float, fps: float) -> None:
    if float(fps) <= 0.0:
        return
    target_seconds = 1.0 / float(fps)
    elapsed = time.perf_counter() - frame_start
    remaining = target_seconds - elapsed
    if remaining > 0.0:
        time.sleep(remaining)


def _create_viewer(render: bool, fps: float, render_substeps: int) -> Any | None:
    if not render:
        return None
    try:
        from experiment.gui.pygame_viewer import PygameCPCViewer
    except ImportError:
        try:
            from gui.pygame_viewer import PygameCPCViewer
        except ImportError as exc:
            print(f"--render unavailable: {exc}")
            return None
    viewer_fps = 60 if float(fps) <= 0.0 else int(round(float(fps) * max(1, int(render_substeps))))
    return PygameCPCViewer(fps=max(1, viewer_fps), title="CPC Tactical Baseline Autoplay")


def _important_events(info: dict[str, Any]) -> list[str]:
    return [
        str(event.get("type"))
        for event in info.get("events", info.get("bullet_events", []))
        if event.get("type")
    ]


def _action_label(action: Mapping[str, Any]) -> str:
    aim_x = float(action.get("aim_dx", action.get("aim_x", 1.0)))
    aim_y = float(action.get("aim_dy", action.get("aim_y", 0.0)))
    return f"move={int(action.get('move', 0))} aim=({aim_x:.3f},{aim_y:.3f}) fire={int(action.get('fire', 0))}"


def _decoded_aim_direction(info: dict[str, Any]) -> dict[str, float]:
    decoded = info.get("decoded_action", {})
    return {
        "x": float(decoded.get("aimX", 1.0)),
        "y": float(decoded.get("aimY", 0.0)),
    }


def _fmt(value: Any) -> str:
    return f"{float(value):.1f}"


def _debug_position(position: Any) -> str:
    if isinstance(position, Mapping):
        return f"({_fmt(position.get('x', 0.0))},{_fmt(position.get('y', 0.0))})"
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        return f"({_fmt(position[0])},{_fmt(position[1])})"
    return "-"


def _optional_fmt(value: Any) -> str:
    return "-" if value is None else _fmt(value)


if __name__ == "__main__":
    main()
