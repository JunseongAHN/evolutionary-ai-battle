from __future__ import annotations

import sys

# Direct script execution puts this package directory first, where types.py
# would shadow Python's standard-library types module during argparse import.
if sys.path and sys.path[0].replace("\\", "/").endswith("/hierarchical_baseline"):
    sys.path.pop(0)

import argparse
import pathlib
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiment"
for path in (REPO_ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from experiment.baselines.hierarchical_baseline import HierarchicalBaselineAgent
    from experiment.baselines.hierarchical_baseline.debug import format_debug
    from experiment.core.cpc_env import CPCEnv
    from experiment.core.env_config import load_env_config
    from experiment.core.local_occupancy_grid import build_local_occupancy_grid
except ModuleNotFoundError:
    from baselines.hierarchical_baseline import HierarchicalBaselineAgent
    from baselines.hierarchical_baseline.debug import format_debug
    from core.cpc_env import CPCEnv
    from core.env_config import load_env_config
    from core.local_occupancy_grid import build_local_occupancy_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the function-based hierarchical CPC baseline.")
    parser.add_argument("--config", default="configs/env/autoplay_goal_loop.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--print-debug", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    args = parser.parse_args()
    run_hierarchical_autoplay(
        config_path=args.config,
        steps=args.steps,
        fps=args.fps,
        seed=args.seed,
        render=args.render,
        print_debug=args.print_debug,
        print_every=args.print_every,
    )


def run_hierarchical_autoplay(
    *,
    config_path: str,
    steps: int,
    fps: float = 10.0,
    seed: int | None = None,
    render: bool = False,
    print_debug: bool = False,
    print_every: int = 1,
) -> dict[str, Any]:
    config = load_env_config(config_path)
    env = CPCEnv.from_config(config)
    reset_seed = config.seed if seed is None else int(seed)
    obs = env.reset(seed=reset_seed)
    agent = HierarchicalBaselineAgent()
    viewer = _viewer(render, fps)
    done = False
    last_info: dict[str, Any] = {}
    last_debug: dict[str, Any] = {}
    steps_run = 0
    print(f"Reset config={config_path} seed={reset_seed} baseline=hierarchical")
    try:
        for step_index in range(max(0, int(steps))):
            started = time.perf_counter()
            snapshot = env.get_debug_state()
            grid = build_local_occupancy_grid(snapshot, agent_id="self")
            action, last_debug = agent.act({**obs, "local_occupancy_grid": grid}, snapshot)
            obs, reward, done, last_info = env.step(action)
            steps_run += 1
            if print_debug and step_index % max(1, int(print_every)) == 0:
                player = env.state["self_pos"]
                print(
                    f"step={step_index} | player=({player['x']:.1f},{player['y']:.1f}) | "
                    f"{format_debug(last_debug)} | reward={float(reward):.4f} | done={done}"
                )
            if viewer is not None:
                state = env.get_debug_state()
                state["hierarchical_debug"] = last_debug
                if not viewer.render_step(state, _step_record(action, reward, done, last_info)):
                    break
            elif fps > 0.0:
                remaining = (1.0 / fps) - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
            if done:
                break
    finally:
        if viewer is not None:
            viewer.close()
    print(f"Finished steps={steps_run} done={done} intent={last_debug.get('intent')}")
    return {"steps_run": steps_run, "done": done, "info": last_info, "debug": last_debug, "state": agent.state}


def _viewer(render: bool, fps: float):
    if not render:
        return None
    try:
        from experiment.gui.pygame_viewer import PygameCPCViewer
    except ImportError:
        from gui.pygame_viewer import PygameCPCViewer
    return PygameCPCViewer(fps=max(1, int(fps)), title="CPC Hierarchical Baseline")


def _step_record(action: dict, reward: float, done: bool, info: dict) -> dict:
    decoded = info.get("decoded_action", {})
    return {
        "agents": {"agent": {"decoded_action": {
            "move_x": decoded.get("moveX", 0.0),
            "move_y": decoded.get("moveY", 0.0),
            "aim_x": decoded.get("aimX", 1.0),
            "aim_y": decoded.get("aimY", 0.0),
            "fire": decoded.get("fire", action.get("fire", 0)),
        }}},
        "env": {"done": done, "rewards": {"agent": reward}, "info": info},
    }


if __name__ == "__main__":
    main()
