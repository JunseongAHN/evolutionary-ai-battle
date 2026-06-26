from __future__ import annotations

import argparse
import pathlib
import sys
import time
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
    from experiment.baselines.tactical_baseline import build_tactical_baseline_bot
except ModuleNotFoundError:
    from baselines.tactical_baseline import build_tactical_baseline_bot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the integrated CPC tactical baseline autoplay loop.")
    parser.add_argument("--config", default="configs/env/autoplay_enemy_right.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action="store_true", help="Open the existing pygame viewer when available.")
    parser.add_argument("--save-png", action="store_true", help="Save local occupancy grid PNGs.")
    parser.add_argument("--output-dir", default="experiment/runs/tactical_autoplay")
    parser.add_argument("--print-debug", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--stop-on-done", action="store_true", default=True)
    args = parser.parse_args()

    run_tactical_autoplay(
        config_path=args.config,
        steps=args.steps,
        fps=args.fps,
        seed=args.seed,
        render=args.render,
        save_png=args.save_png,
        output_dir=pathlib.Path(args.output_dir),
        print_debug=args.print_debug,
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
    save_png: bool = False,
    output_dir: pathlib.Path | str = pathlib.Path("experiment/runs/tactical_autoplay"),
    print_debug: bool = False,
    print_every: int = 10,
    stop_on_done: bool = True,
) -> dict[str, Any]:
    config = load_env_config(config_path)
    env = CPCEnv.from_config(config)
    reset_seed = config.seed if seed is None else int(seed)
    obs = _reset_observation(env.reset(seed=reset_seed))
    snapshot = get_state_snapshot_if_available(env)
    bot = build_tactical_baseline_bot(snapshot)
    viewer = _create_viewer(render, fps)
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
                if not viewer.render_step(state, build_step_record(action=env_action, reward=last_reward, done=done, info=last_info)):
                    break
            if done and stop_on_done:
                break
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


def action_for_env(action: dict[str, int]) -> dict[str, int]:
    return {
        "move": int(action.get("move", action.get("move_bin", 0))),
        "aim": int(action.get("aim", action.get("aim_bin", 0))),
        "fire": int(action.get("fire", 0)),
    }


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
    events = _important_events(info)
    parts = [
        f"step={step_index}",
        f"env_step={env.step_count}",
        f"player=({_fmt(self_agent['position']['x'])},{_fmt(self_agent['position']['y'])})",
        f"enemy=({_fmt(enemy_agent['position']['x'])},{_fmt(enemy_agent['position']['y'])})",
        f"action={action_for_env(action)}",
        f"fire_reason={fire_debug.get('reason')}",
        f"reward={float(reward):.4f}",
        f"done={done}",
    ]
    if events:
        parts.append(f"events={','.join(events)}")
    print(" | ".join(parts))


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


def sleep_to_match_fps(frame_start: float, fps: float) -> None:
    if float(fps) <= 0.0:
        return
    target_seconds = 1.0 / float(fps)
    elapsed = time.perf_counter() - frame_start
    remaining = target_seconds - elapsed
    if remaining > 0.0:
        time.sleep(remaining)


def _create_viewer(render: bool, fps: float) -> Any | None:
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
    return PygameCPCViewer(fps=max(1, int(fps)), title="CPC Tactical Baseline Autoplay")


def _important_events(info: dict[str, Any]) -> list[str]:
    events: list[str] = []
    for event in info.get("bullet_events", []):
        event_type = event.get("type")
        if event_type in {"bullet_spawned", "bullet_hit", "bullet_hit_obstacle", "bullet_expired", "bullet_not_spawned"}:
            events.append(str(event_type))
    return events


def _action_label(action: dict[str, int]) -> str:
    return f"move={int(action.get('move', 0))} aim={int(action.get('aim', 0))} fire={int(action.get('fire', 0))}"


def _fmt(value: Any) -> str:
    return f"{float(value):.1f}"


if __name__ == "__main__":
    main()
