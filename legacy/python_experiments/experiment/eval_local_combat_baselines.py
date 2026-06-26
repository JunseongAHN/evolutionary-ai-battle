from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from experiment.analyze_local_combat_eval import analyze_result, render_markdown
    from experiment.training.cpc_actions import AIM_BINS, decode_action, random_action, vec_to_aim_bin
    from experiment.training.cpc_env import CPCEnv
except ModuleNotFoundError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from analyze_local_combat_eval import analyze_result, render_markdown
    from training.cpc_actions import AIM_BINS, decode_action, random_action, vec_to_aim_bin
    from training.cpc_env import CPCEnv


@dataclass
class BaselineConfig:
    seed: int = 0
    max_episode_steps: int = 100
    stage: str = "local_combat"
    shrink_safe_zone: bool = False
    use_zone_reward: bool = False
    fire_interval_steps: int = 5
    bullet_speed: float = 140.0
    bullet_range: float = 280.0
    bullet_damage: float = 10.0
    bullet_hit_radius: float = 12.0
    randomize_enemy_spawn_direction: bool = True
    enemy_spawn_directions: tuple[str, ...] = (
        "right",
        "left",
        "up",
        "down",
        "upper_right",
        "lower_right",
        "upper_left",
        "lower_left",
    )
    enemy_spawn_direction: str | None = None


class Agent(Protocol):
    def act(self, observation: dict[str, Any], env: CPCEnv) -> dict[str, int]:
        ...


class RandomBaseline:
    def __init__(self, seed: int = 0):
        import random

        self.rng = random.Random(seed)

    def act(self, observation: dict[str, Any], env: CPCEnv) -> dict[str, int]:
        del observation, env
        return random_action(self.rng)


class Aim0FireBaseline:
    def act(self, observation: dict[str, Any], env: CPCEnv) -> dict[str, int]:
        del observation, env
        return {"move": 0, "aim": 0, "fire": 1}


class ScriptedAimBaseline:
    def act(self, observation: dict[str, Any], env: CPCEnv) -> dict[str, int]:
        del observation
        dx = float(env.state["enemy_pos"]["x"]) - float(env.state["self_pos"]["x"])
        dy = float(env.state["enemy_pos"]["y"]) - float(env.state["self_pos"]["y"])
        aim = vec_to_aim_bin({"x": dx, "y": dy}, AIM_BINS)
        distance = math.hypot(dx, dy)
        fire = int(distance <= env.fire_range)
        return {"move": 0, "aim": aim, "fire": fire}


class CheckpointBaseline:
    def __init__(self, checkpoint_path: str, device: str):
        from experiment.policy_agent import PPOPolicyAgent

        self.agent = PPOPolicyAgent.from_checkpoint(checkpoint_path, device=device)

    def act(self, observation: dict[str, Any], env: CPCEnv) -> dict[str, int]:
        del env
        return self.agent.act(observation, deterministic=True)


def run_policy(policy_name: str, agent: Agent, cfg: Any, episodes: int) -> dict[str, Any]:
    result_episodes = []
    for episode_index in range(max(1, int(episodes))):
        env = CPCEnv(
            seed=int(cfg.seed) + episode_index,
            max_steps=int(cfg.max_episode_steps),
            randomize_enemy_spawn_direction=bool(cfg.randomize_enemy_spawn_direction),
            enemy_spawn_directions=cfg.enemy_spawn_directions,
            enemy_spawn_direction=cfg.enemy_spawn_direction,
            stage=cfg.stage,
            shrink_safe_zone=cfg.shrink_safe_zone,
            use_zone_reward=cfg.use_zone_reward,
            fire_interval_steps=cfg.fire_interval_steps,
            bullet_speed=cfg.bullet_speed,
            bullet_range=cfg.bullet_range,
            bullet_damage=cfg.bullet_damage,
            bullet_hit_radius=cfg.bullet_hit_radius,
        )
        observation = env.reset(seed=int(cfg.seed) + episode_index)
        total_reward = 0.0
        steps = []
        done = False
        while not done and env.step_count < int(cfg.max_episode_steps):
            step_index = env.step_count
            action = agent.act(observation, env)
            decoded = decode_action(action)
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps.append(_compact_step(step_index, action, decoded, reward, info, env))
        result_episodes.append(
            {
                "episode_index": episode_index,
                "steps": steps,
                "episode_return": {"agent": total_reward},
                "episode_length": len(steps),
                "final_metrics": env.metrics.summary(),
                "stopped_early": False,
            }
        )
    return {
        "schema_version": "cpc-common-v0",
        "source": f"eval_local_combat_baselines:{policy_name}",
        "config": {"policy": policy_name, "episodes": episodes, "max_steps": int(cfg.max_episode_steps)},
        "episodes": result_episodes,
    }


def evaluate_baselines(config: str | Path, checkpoint_a: str | None, episodes: int, device: str = "cpu") -> dict[str, Any]:
    cfg = load_baseline_config(config)
    policies: list[tuple[str, Agent]] = [
        ("random", RandomBaseline(seed=int(cfg.seed))),
        ("aim0_fire", Aim0FireBaseline()),
        ("scripted_aim", ScriptedAimBaseline()),
    ]
    if checkpoint_a:
        policies.append(("checkpoint", CheckpointBaseline(checkpoint_a, device)))

    rows = []
    analyses = {}
    for policy_name, agent in policies:
        result = run_policy(policy_name, agent, cfg, episodes)
        analysis = analyze_result(result)
        analyses[policy_name] = analysis
        aggregate = analysis.get("aggregate", {})
        warning_names = aggregate.get("warnings", {})
        rows.append(
            {
                "policy": policy_name,
                "return": aggregate.get("total_reward", 0.0),
                "damage_dealt_ratio": aggregate.get("damage_dealt_ratio", 0.0),
                "damage_taken_ratio": aggregate.get("damage_taken_ratio", 0.0),
                "damage_trade_ratio": aggregate.get("damage_trade_ratio", 0.0),
                "hit_ratio": aggregate.get("hit_ratio", 0.0),
                "missed_shot_rate": aggregate.get("missed_shot_rate", 0.0),
                "aim_bin_0_rate": aggregate.get("aim_bin_0_rate", 0.0),
                "exact_aim_match_rate": aggregate.get("exact_aim_match_rate", 0.0),
                "self_dead": _mean_metric(analysis, "self_dead"),
                "enemy_dead": _mean_metric(analysis, "enemy_dead"),
                "warnings": warning_names,
            }
        )
    return {"rows": rows, "analyses": analyses}


def load_baseline_config(path: str | Path) -> BaselineConfig:
    cfg = BaselineConfig()
    values = _read_simple_yaml(Path(path))
    for key, value in values.items():
        if hasattr(cfg, key):
            current = getattr(cfg, key)
            setattr(cfg, key, _coerce_value(value, current))
    return cfg


def render_baseline_markdown(result: dict[str, Any]) -> str:
    lines = ["# Local Combat Baseline Comparison", ""]
    lines.append(
        "| policy | return | damage_dealt_ratio | damage_taken_ratio | damage_trade_ratio | hit_ratio | "
        "missed_shot_rate | aim_bin_0_rate | exact_aim_match_rate | self_dead | enemy_dead | warnings |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in result["rows"]:
        lines.append(
            f"| {row['policy']} | {row['return']:.4f} | {row['damage_dealt_ratio']:.4f} | "
            f"{row['damage_taken_ratio']:.4f} | {row['damage_trade_ratio']:.4f} | {row['hit_ratio']:.4f} | "
            f"{row['missed_shot_rate']:.4f} | {row['aim_bin_0_rate']:.4f} | "
            f"{row['exact_aim_match_rate']:.4f} | {row['self_dead']:.4f} | {row['enemy_dead']:.4f} | "
            f"{row['warnings']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _compact_step(
    step_index: int,
    action: dict[str, int],
    decoded: dict[str, Any],
    reward: float,
    info: dict[str, Any],
    env: CPCEnv,
) -> dict[str, Any]:
    return {
        "t": step_index,
        "action": {
            "raw": dict(action),
            "decoded": {
                "move": {"x": decoded["moveX"], "y": decoded["moveY"]},
                "aim": {"x": decoded["aimX"], "y": decoded["aimY"]},
                "fire": decoded["fire"],
            },
        },
        "aim": {
            "aim_bin": info.get("aim_debug", {}).get("aim_bin"),
            "ideal_aim_bin": info.get("aim_debug", {}).get("ideal_aim_bin"),
            "aim_bin_error": info.get("aim_debug", {}).get("aim_bin_error"),
            "alignment": info.get("aim_debug", {}).get("aim_alignment", 0.0),
            "angle_error_deg": info.get("aim_debug", {}).get("angle_error_deg", 0.0),
        },
        "fire": {
            "requested": info.get("fire", {}).get("fire_requested", False),
            "shot_fired": info.get("fire", {}).get("shot_fired", False),
            "blocked_reason": info.get("fire", {}).get("fire_blocked_reason"),
            "cooldown_before": info.get("fire", {}).get("cooldown_remaining_steps_before"),
            "cooldown_after": info.get("fire", {}).get("cooldown_remaining_steps_after"),
        },
        "range": info.get("range_debug", {}),
        "events": info.get("bullet_events", []),
        "bullets": info.get("bullets", []),
        "reward": float(reward),
        "reward_components": info.get("reward_components", {}),
        "state_after": {
            "self": {"hp": env.state.get("self_hp"), "pos": env.state.get("self_pos")},
            "enemy": {"hp": env.state.get("enemy_hp"), "pos": env.state.get("enemy_pos")},
            "dist": {"enemy": info.get("range_debug", {}).get("distance_to_enemy")},
        },
        "metrics_delta": {
            "damage_dealt_delta": info.get("damage_delta", {}).get("enemy_hp", 0.0),
            "damage_taken_delta": info.get("damage_delta", {}).get("self_hp", 0.0),
        },
    }


def _mean_metric(analysis: dict[str, Any], key: str) -> float:
    episodes = analysis.get("episodes", [])
    if not episodes:
        return 0.0
    return sum(float(ep.get("metrics", {}).get(key, 0.0)) for ep in episodes) / len(episodes)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not path.exists():
        return values
    current_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            current_list_key = None
            continue
        if current_list_key is not None and stripped.startswith("- "):
            values[current_list_key].append(stripped[2:].strip().strip('"').strip("'"))
            continue
        current_list_key = None
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            values[key] = []
            current_list_key = key
            continue
        values[key] = value.strip('"').strip("'")
    return values


def _coerce_value(value: Any, current: Any) -> Any:
    if isinstance(current, tuple):
        if isinstance(value, list):
            return tuple(str(item) for item in value)
        return tuple(item.strip() for item in str(value).split(",") if item.strip())
    if value in ("null", "None", ""):
        return None
    if isinstance(current, bool):
        return str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 local combat baselines.")
    parser.add_argument("--config", default="experiment/configs/local_combat_micro.yaml")
    parser.add_argument("--checkpoint-a")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-md")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    result = evaluate_baselines(args.config, args.checkpoint_a, args.episodes, device=args.device)
    markdown = render_baseline_markdown(result)
    print(markdown)
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
