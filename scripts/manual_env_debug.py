from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiment"
for path in (REPO_ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.env_config import EnvConfig, load_env_config
from core.cpc_actions import decode_action, vec_to_aim_bin
from core.cpc_env import CPCEnv
from core.local_occupancy_grid import build_local_occupancy_grid, render_grid_to_png
from gui.geometry import screen_to_world
from gui.pygame_viewer import PygameCPCViewer


ACTION_MAP: dict[str, dict[str, int]] = {
    "stay": {"move": 0, "aim": 0, "fire": 0},
    "noop": {"move": 0, "aim": 0, "fire": 0},
    "up": {"move": 1, "aim": 0, "fire": 0},
    "down": {"move": 2, "aim": 0, "fire": 0},
    "left": {"move": 3, "aim": 0, "fire": 0},
    "right": {"move": 4, "aim": 0, "fire": 0},
    "up_left": {"move": 5, "aim": 0, "fire": 0},
    "up_right": {"move": 6, "aim": 0, "fire": 0},
    "down_left": {"move": 7, "aim": 0, "fire": 0},
    "down_right": {"move": 8, "aim": 0, "fire": 0},
    "aim_right": {"move": 0, "aim": 0, "fire": 0},
    "aim_down": {"move": 0, "aim": 4, "fire": 0},
    "aim_left": {"move": 0, "aim": 8, "fire": 0},
    "aim_up": {"move": 0, "aim": 12, "fire": 0},
    "fire": {"move": 0, "aim": 0, "fire": 1},
    "fire_right": {"move": 0, "aim": 0, "fire": 1},
    "fire_down": {"move": 0, "aim": 4, "fire": 1},
    "fire_left": {"move": 0, "aim": 8, "fire": 1},
    "fire_up": {"move": 0, "aim": 12, "fire": 1},
}

DEFAULT_ACTIONS = ["stay", "right", "right", "aim_right", "fire"]
MOVE_NAMES = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "up_left",
    6: "up_right",
    7: "down_left",
    8: "down_right",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPC toy env YAML manual debug viewer.")
    parser.add_argument("--config", default="configs/env/manual_debug.yaml")
    parser.add_argument("--steps", type=int, default=10, help="Console-only scripted step count.")
    parser.add_argument("--actions", help="Console-only comma-separated actions, for example: stay,right,fire")
    parser.add_argument("--no-gui", action="store_true", help="Run scripted console smoke steps instead of opening pygame.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--snapshot-output-dir", default="experiment/runs/manual_grid_debug")
    parser.add_argument("--grid-output-dir", help="Deprecated alias for --snapshot-output-dir.")
    parser.add_argument("--no-grid-png", action="store_true", help="Disable GUI snapshot saving.")
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="In --no-gui mode, save reset and step occupancy-grid PNG/status snapshots.",
    )
    args = parser.parse_args()

    config = load_env_config(args.config)
    snapshot_output_dir = None if args.no_grid_png else pathlib.Path(args.grid_output_dir or args.snapshot_output_dir)
    if args.no_gui:
        if args.save_snapshot and snapshot_output_dir is None:
            raise SystemExit("--save-snapshot requires grid PNG saving; remove --no-grid-png")
        run_scripted_debug(
            config,
            args.config,
            steps=args.steps,
            actions=args.actions,
            snapshot_output_dir=snapshot_output_dir,
            save_snapshot=args.save_snapshot,
        )
        return
    if args.actions:
        print("--actions is used only with --no-gui; GUI mode uses live keyboard controls.")
    run_live_viewer(
        config,
        args.config,
        width=args.width,
        height=args.height,
        fps=args.fps,
        snapshot_output_dir=snapshot_output_dir,
    )


def run_scripted_debug(
    config: EnvConfig,
    config_path: str,
    *,
    steps: int,
    actions: str | None,
    snapshot_output_dir: pathlib.Path | None = None,
    save_snapshot: bool = False,
) -> None:
    env = CPCEnv.from_config(config)
    env.reset(seed=config.seed)
    print(f"Reset config={config_path} seed={config.seed}")
    print_state("reset", env)
    if save_snapshot and snapshot_output_dir is not None:
        saved = export_debug_snapshot(env, config_path, snapshot_output_dir, "step_000_reset")
        print(f"Saved snapshot png={saved['grid_png']} status={saved['status_json']}")

    for step_index, action_name in enumerate(parse_action_names(actions, steps), start=1):
        action = ACTION_MAP[action_name]
        _, reward, done, info = env.step(action)
        print_state(f"step={step_index} action={action_name}", env, reward=reward, done=done, info=info)
        if save_snapshot and snapshot_output_dir is not None:
            saved = export_debug_snapshot(env, config_path, snapshot_output_dir, f"step_{step_index:03d}_{action_name}")
            print(f"Saved snapshot png={saved['grid_png']} status={saved['status_json']}")
        if done:
            break


def run_live_viewer(
    config: EnvConfig,
    config_path: str,
    *,
    width: int,
    height: int,
    fps: int,
    snapshot_output_dir: pathlib.Path | None,
) -> None:
    try:
        viewer = PygameCPCViewer(width=width, height=height, fps=fps, title="CPC Env YAML Manual Debug")
    except ImportError as exc:
        print(str(exc))
        return

    env = CPCEnv.from_config(config)
    env.reset(seed=config.seed)
    pygame = viewer.pygame
    running = True
    force_step = False
    last_aim_bin = 0
    last_step_record = build_step_record(
        action={"move": 0, "aim": last_aim_bin, "fire": 0},
        reward=0.0,
        done=False,
        info={},
        action_name="reset",
    )

    print(f"Reset config={config_path} seed={config.seed}")
    save_control = "G/click save" if snapshot_output_dir is not None else "save disabled"
    print(f"Controls: WASD move, mouse aim, Space/F fire, N/Enter noop step, {save_control}, R reset, Q/Esc quit")
    print_state("reset", env)

    while running:
        force_step = False
        save_snapshot = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if snapshot_output_dir is not None and save_button_rect(viewer).collidepoint(event.pos):
                    save_snapshot = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_g and snapshot_output_dir is not None:
                    save_snapshot = True
                elif event.key == pygame.K_r:
                    env.reset(seed=config.seed)
                    last_aim_bin = 0
                    last_step_record = build_step_record(
                        action={"move": 0, "aim": last_aim_bin, "fire": 0},
                        reward=0.0,
                        done=False,
                        info={},
                        action_name="reset",
                    )
                    print(f"Reset seed={config.seed}")
                    print_state("reset", env)
                elif event.key in (pygame.K_n, pygame.K_RETURN, pygame.K_KP_ENTER):
                    force_step = True

        keys = pygame.key.get_pressed()
        mouse_world = mouse_world_position(viewer, env)
        action, action_name, should_step = action_from_input(pygame, keys, env, mouse_world, last_aim_bin, force_step)
        if not should_step:
            last_step_record = build_step_record(
                action=action,
                reward=float(last_step_record["env"]["rewards"]["agent"]),
                done=bool(last_step_record["env"]["done"]),
                info=last_step_record["env"]["info"],
                action_name="aim_mouse",
            )
        if should_step and not bool(last_step_record["env"]["done"]):
            last_aim_bin = int(action["aim"])
            _, reward, done, info = env.step(action)
            last_step_record = build_step_record(
                action=action,
                reward=reward,
                done=done,
                info=info,
                action_name=action_name,
            )
            print_state(f"step={env.step_count} action={action_name}", env, reward=reward, done=done, info=info)

        if save_snapshot and snapshot_output_dir is not None:
            saved = export_debug_snapshot(
                env,
                config_path,
                snapshot_output_dir,
                f"step_{env.step_count:03d}_{last_step_record['env'].get('action_name', 'manual')}",
            )
            print(f"Saved snapshot png={saved['grid_png']} status={saved['status_json']}")

        running = running and render_viewer(viewer, env, last_step_record, save_enabled=snapshot_output_dir is not None)

    viewer.close()


def mouse_world_position(viewer: PygameCPCViewer, env: CPCEnv) -> dict[str, float]:
    env_map = env.get_debug_state().get("map", {"width": 1000.0, "height": 1000.0})
    return screen_to_world(
        viewer.pygame.mouse.get_pos(),
        env_map,
        (viewer.width - viewer.panel_width, viewer.height),
        viewer.padding,
    )


def action_from_input(
    pygame,
    keys,
    env: CPCEnv,
    mouse_world: dict[str, float],
    last_aim_bin: int,
    force_step: bool,
) -> tuple[dict[str, int], str, bool]:
    move = move_bin_from_keys(pygame, keys)
    aim_bin, aiming = aim_bin_from_mouse(env, mouse_world, last_aim_bin)
    fire = 1 if keys[pygame.K_SPACE] or keys[pygame.K_f] else 0
    should_step = bool(force_step or move != 0 or fire)
    action = {"move": move, "aim": aim_bin, "fire": fire}
    return action, action_name(action, aiming=aiming, force_step=force_step), should_step


def move_bin_from_keys(pygame, keys) -> int:
    left = bool(keys[pygame.K_a])
    right = bool(keys[pygame.K_d])
    up = bool(keys[pygame.K_w])
    down = bool(keys[pygame.K_s])
    x = int(right) - int(left)
    y = int(down) - int(up)
    if x == 0 and y == 0:
        return 0
    if x == 0 and y < 0:
        return 1
    if x == 0 and y > 0:
        return 2
    if x < 0 and y == 0:
        return 3
    if x > 0 and y == 0:
        return 4
    if x < 0 and y < 0:
        return 5
    if x > 0 and y < 0:
        return 6
    if x < 0 and y > 0:
        return 7
    return 8


def aim_bin_from_mouse(env: CPCEnv, mouse_world: dict[str, float], last_aim_bin: int) -> tuple[int, bool]:
    self_pos = env.state.get("self_pos", {"x": 0.0, "y": 0.0})
    aim = {
        "x": float(mouse_world["x"]) - float(self_pos["x"]),
        "y": float(mouse_world["y"]) - float(self_pos["y"]),
    }
    if abs(aim["x"]) <= 1e-6 and abs(aim["y"]) <= 1e-6:
        return int(last_aim_bin), False
    return vec_to_aim_bin(aim), True


def action_name(action: dict[str, int], *, aiming: bool, force_step: bool) -> str:
    parts = [MOVE_NAMES[int(action["move"])]]
    if aiming:
        parts.append(f"aim_{int(action['aim'])}")
    if int(action["fire"]):
        parts.append("fire")
    if force_step and parts == ["stay"]:
        parts.append("step")
    return "+".join(parts)


def render_viewer(viewer: PygameCPCViewer, env: CPCEnv, step_record: dict[str, Any], *, save_enabled: bool) -> bool:
    env_state = env.get_debug_state()
    env_state["bullet_events"] = step_record["env"]["info"].get("bullet_events", [])
    env_state["manual_step"] = {
        "mode": "live",
        "current_action": step_record["env"].get("action_name", "reset"),
        "controls": [
            "WASD move",
            "Mouse aim",
            "Space/F fire",
            "N/Enter step",
            "G/click save" if save_enabled else "Save disabled",
            "R reset",
            "Q/Esc quit",
        ],
        "save_button": save_enabled,
    }
    return viewer.render_step(env_state, step_record, handle_events=False)


def save_button_rect(viewer: PygameCPCViewer):
    return viewer.pygame.Rect(viewer.width - viewer.panel_width + 14, viewer.height - 54, viewer.panel_width - 28, 36)


def export_debug_snapshot(
    env: CPCEnv,
    config_path: str,
    output_dir: pathlib.Path,
    label: str,
) -> dict[str, pathlib.Path]:
    env_state = env.get_debug_state()
    grid = build_local_occupancy_grid(env_state, agent_id="self")
    stem = pathlib.Path(config_path).stem
    safe_label = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label)
    png_path = output_dir / f"{stem}_{safe_label}_grid.png"
    status_path = output_dir / f"{stem}_{safe_label}_status.json"
    render_grid_to_png(grid, png_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "config": config_path,
                "step": env.step_count,
                "env_state": env_state,
                "grid": {
                    "shape": grid.shape,
                    "channels": list(grid.channel_names),
                    "center_cell": grid.center_cell,
                    "origin": grid.origin,
                    "cell_size": grid.cell_size,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"grid_png": png_path, "status_json": status_path}


def build_step_record(
    *,
    action: dict[str, int],
    reward: float,
    done: bool,
    info: dict[str, Any],
    action_name: str,
) -> dict[str, Any]:
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
                },
            },
        },
        "env": {
            "done": bool(done),
            "rewards": {"agent": float(reward)},
            "info": info,
            "action_name": action_name,
        },
    }


def parse_action_names(actions: str | None, steps: int) -> list[str]:
    if actions:
        names = [name.strip() for name in actions.split(",") if name.strip()]
    else:
        names = [DEFAULT_ACTIONS[index % len(DEFAULT_ACTIONS)] for index in range(max(0, steps))]
    unknown = [name for name in names if name not in ACTION_MAP]
    if unknown:
        valid = ", ".join(sorted(ACTION_MAP))
        raise SystemExit(f"unknown action(s): {', '.join(unknown)}; valid actions: {valid}")
    return names[:max(0, steps)] if actions else names


def print_state(
    label: str,
    env: CPCEnv,
    *,
    reward: float | None = None,
    done: bool | None = None,
    info: dict[str, Any] | None = None,
) -> None:
    debug = env.get_debug_state()
    state = debug["state"]
    player = debug["agents"]["self"]
    enemy = debug["agents"]["enemy"]
    fire = (info or {}).get("fire", {})
    events = important_events(info or {})
    truncated = bool(done and env.step_count >= env.max_steps) if done is not None else False
    terminated = bool(done and not truncated) if done is not None else False
    parts = [
        label,
        f"env_step={env.step_count}",
        f"player=({player['position']['x']:.1f},{player['position']['y']:.1f})",
        f"aim_bin={state.get('current_aim_bin', env.current_aim_bin)}",
        f"enemy={enemy['id']}@({enemy['position']['x']:.1f},{enemy['position']['y']:.1f})",
        f"hp self={player['hp']:.1f} enemy={enemy['hp']:.1f}",
    ]
    if reward is not None:
        parts.append(f"reward={reward:.4f}")
    if done is not None:
        parts.append(f"done={done}")
        parts.append(f"terminated={terminated}")
        parts.append(f"truncated={truncated}")
    if fire:
        parts.append(
            "fire="
            f"requested:{fire.get('fire_requested')} "
            f"shot:{fire.get('shot_fired')} "
            f"cooldown:{fire.get('cooldown_remaining_steps_after')}"
        )
    if events:
        parts.append(f"events={','.join(events)}")
    print(" | ".join(parts))


def important_events(info: dict[str, Any]) -> list[str]:
    events: list[str] = []
    for event in info.get("bullet_events", []):
        event_type = event.get("type")
        if event_type in {"bullet_spawned", "bullet_hit", "bullet_hit_obstacle", "bullet_expired", "bullet_not_spawned"}:
            events.append(str(event_type))
    return events


if __name__ == "__main__":
    main()
