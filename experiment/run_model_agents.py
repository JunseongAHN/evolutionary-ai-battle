from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from experiment.gameplay_serializer import save_gameplay_result, to_jsonable
    from experiment.policy_agent import PPOPolicyAgent
    from experiment.training.cpc_actions import decode_action
    from experiment.training.cpc_env import CPCEnv
except ModuleNotFoundError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiment.gameplay_serializer import save_gameplay_result, to_jsonable
    from experiment.policy_agent import PPOPolicyAgent
    from experiment.training.cpc_actions import decode_action
    from experiment.training.cpc_env import CPCEnv


def run_model_gameplay(
    checkpoint_a: str,
    checkpoint_b: str | None = None,
    episodes: int = 1,
    max_steps: int = 100,
    device: str = "cpu",
    deterministic: bool = True,
    save_result: str | None = None,
    render_pygame: bool = False,
    render_fps: int = 10,
    pause_on_end: bool = False,
) -> dict[str, Any]:
    if checkpoint_b is not None and not _env_supports_two_agent_control():
        raise NotImplementedError(
            "Two-agent model gameplay requested, but the current env only supports one controlled agent. "
            "Add MultiAgentCPCEnv or runner-level controlled_agent_ids first."
        )

    agent_a = PPOPolicyAgent.from_checkpoint(checkpoint_a, device=device)
    result = _run_single_agent_gameplay(
        agent=agent_a,
        checkpoint_a=checkpoint_a,
        episodes=episodes,
        max_steps=max_steps,
        device=device,
        deterministic=deterministic,
        render_pygame=render_pygame,
        render_fps=render_fps,
        pause_on_end=pause_on_end,
    )
    if save_result:
        save_gameplay_result(result, save_result)
    return result


def run_two_agent_eval(
    checkpoint_a: str,
    checkpoint_b: str | None = None,
    episodes: int = 1,
    device: str = "cpu",
    deterministic: bool = True,
    export_path: str | None = None,
) -> dict[str, Any]:
    return run_model_gameplay(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        episodes=episodes,
        max_steps=100,
        device=device,
        deterministic=deterministic,
        save_result=export_path,
    )


def _run_single_agent_gameplay(
    *,
    agent: Any,
    checkpoint_a: str,
    episodes: int,
    max_steps: int,
    device: str,
    deterministic: bool,
    render_pygame: bool = False,
    render_fps: int = 10,
    pause_on_end: bool = False,
) -> dict[str, Any]:
    episodes = max(1, int(episodes))
    max_steps = max(1, int(max_steps))
    result_episodes = []
    stopped_early = False
    viewer = _create_viewer(render_pygame, render_fps)

    try:
        for episode_index in range(episodes):
            if stopped_early:
                break
            episode_result, episode_stopped_early = _run_single_episode(
                agent=agent,
                checkpoint_a=checkpoint_a,
                episode_index=episode_index,
                max_steps=max_steps,
                deterministic=deterministic,
                viewer=viewer,
            )
            result_episodes.append(episode_result)
            stopped_early = episode_stopped_early
    finally:
        if viewer is not None and pause_on_end and not stopped_early:
            _pause_viewer_on_end(viewer, result_episodes[-1]["steps"][-1] if result_episodes and result_episodes[-1]["steps"] else None)
        if viewer is not None:
            viewer.close()

    return to_jsonable(
        {
            "schema_version": "cpc-common-v0",
            "source": "run_model_gameplay",
            "checkpoints": {
                "agent_a": checkpoint_a,
                "agent_b": None,
            },
            "config": {
                "episodes": episodes,
                "max_steps": max_steps,
                "deterministic": deterministic,
                "device": device,
                "render_pygame": render_pygame,
            },
            "stopped_early": stopped_early,
            "episodes": result_episodes,
        }
    )


def _run_single_episode(
    *,
    agent: Any,
    checkpoint_a: str,
    episode_index: int,
    max_steps: int,
    deterministic: bool,
    viewer: Any | None,
) -> tuple[dict[str, Any], bool]:
    del checkpoint_a
    env = CPCEnv(seed=episode_index, max_steps=max_steps)
    observation = env.reset(seed=episode_index)
    initial_observation = to_jsonable(observation)
    total_reward = 0.0
    steps = []
    done = False
    stopped_early = False

    while not done and env.step_count < max_steps:
        step_index = env.step_count
        state_before_step = env.get_debug_state()
        action_debug = _agent_action_debug(agent, observation, deterministic=deterministic)
        next_observation, reward, done, info = env.step(action_debug["raw_action"])
        total_reward += float(reward)
        truncated = bool(done and env.step_count >= max_steps)
        terminated = bool(done and not truncated)
        env_state = env.get_debug_state()
        env_state["bullet_events"] = info.get("bullet_events", [])
        env_state["aim_debug"] = info.get("aim_debug", {})
        env_state["zone_debug"] = info.get("zone_debug", {})

        step_record = {
            "step": step_index,
            "agents": {
                "agent": {
                    "agent_id": "self",
                    "observation": observation,
                    "raw_action": action_debug["raw_action"],
                    "decoded_action": action_debug["decoded_action"],
                    "policy_debug": action_debug.get("policy_debug", {}),
                }
            },
            "env": {
                "state_before_step": state_before_step,
                "state": env_state,
                "observation_after_step": next_observation,
                "rewards": {"agent": float(reward)},
                "terminated": terminated,
                "truncated": truncated,
                "done": bool(done),
                "info": {
                    "reward_components": info.get("reward_components", {}),
                    "metrics": info.get("metrics", {}),
                    "decoded_actions": {"agent": info.get("decoded_action")},
                    "raw_actions": {"agent": info.get("raw_action")},
                    "damage_delta": info.get("damage_delta", {}),
                    "safe_zone": info.get("safe_zone", {}),
                    "aim_debug": info.get("aim_debug", {}),
                    "zone_debug": info.get("zone_debug", {}),
                    "fire": info.get("fire", {}),
                    "fire_selected": info.get("fire_selected", False),
                    "shot_fired": info.get("shot_fired", False),
                    "bullet_spawned": info.get("bullet_spawned", False),
                    "bullet_count": info.get("bullet_count", 0),
                    "bullet_events": info.get("bullet_events", []),
                    "bullets": info.get("bullets", info.get("projectiles", [])),
                },
            },
        }
        steps.append(step_record)
        if viewer is not None and not viewer.render_step(env_state, step_record):
            stopped_early = True
            break
        observation = next_observation

    return (
        {
            "episode_index": episode_index,
            "initial_observation": initial_observation,
            "steps": steps,
            "episode_return": {"agent": total_reward},
            "episode_length": len(steps),
            "final_metrics": env.metrics.summary(),
            "stopped_early": stopped_early,
        },
        stopped_early,
    )


def _create_viewer(render_pygame: bool, render_fps: int) -> Any | None:
    if not render_pygame:
        return None
    try:
        from experiment.gui.pygame_viewer import PygameCPCViewer
    except ImportError as exc:
        raise ImportError("pygame is required for --render-pygame. Install with: pip install pygame") from exc
    return PygameCPCViewer(fps=render_fps)


def _pause_viewer_on_end(viewer: Any, step_record: dict[str, Any] | None) -> None:
    env_state = (step_record or {}).get("env", {}).get("state", {})
    while viewer.render_step(env_state, step_record):
        pass


def _agent_action_debug(agent: Any, observation: dict[str, Any], *, deterministic: bool) -> dict[str, Any]:
    if hasattr(agent, "act_with_debug"):
        return dict(agent.act_with_debug(observation, deterministic=deterministic))

    raw_action = agent.act(observation, deterministic=deterministic)
    decoded = decode_action(raw_action)
    return {
        "raw_action": dict(raw_action),
        "decoded_action": {
            "move_x": float(decoded["moveX"]),
            "move_y": float(decoded["moveY"]),
            "aim_x": float(decoded["aimX"]),
            "aim_y": float(decoded["aimY"]),
            "fire": float(decoded["fire"]),
        },
        "policy_debug": {},
    }


def _env_supports_two_agent_control() -> bool:
    env = CPCEnv(seed=0, max_steps=1)
    obs = env.reset()
    return isinstance(obs, dict) and "observations" in obs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run loaded PPO model agents in the Python CPC env.")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--export")
    parser.add_argument("--save-result")
    parser.add_argument("--render-pygame", action="store_true")
    parser.add_argument("--render-fps", type=int, default=10)
    parser.add_argument("--pause-on-end", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_model_gameplay(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        deterministic=args.deterministic,
        save_result=args.save_result or args.export,
        render_pygame=args.render_pygame,
        render_fps=args.render_fps,
        pause_on_end=args.pause_on_end,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
