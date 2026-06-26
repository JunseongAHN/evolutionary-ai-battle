from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from experiment.analyze_local_combat_eval import analyze_result
    from experiment.training.cpc_actions import decode_action, vec_to_aim_bin
    from experiment.training.cpc_env import CPCEnv
    from experiment.training.train_ppo import load_config
except ModuleNotFoundError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from analyze_local_combat_eval import analyze_result
    from training.cpc_actions import decode_action, vec_to_aim_bin
    from training.cpc_env import CPCEnv
    from training.train_ppo import load_config


DEFAULT_CONFIG = "experiment/configs/local_combat_in_range.yaml"


def run_scripted_ideal_aim_same_reset(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    episodes: int = 1,
    max_steps: int | None = None,
    seed: int = 0,
    enemy_spawn_direction: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path, smoke=False)
    if max_steps is not None:
        cfg.max_episode_steps = int(max_steps)
    if enemy_spawn_direction is not None:
        cfg.enemy_spawn_direction = enemy_spawn_direction
        cfg.randomize_enemy_spawn_direction = False
    result_episodes = []

    for episode_index in range(max(1, int(episodes))):
        env = CPCEnv(
            seed=int(seed) + episode_index,
            max_steps=int(cfg.max_episode_steps),
            randomize_enemy_spawn_direction=cfg.randomize_enemy_spawn_direction,
            enemy_spawn_directions=cfg.enemy_spawn_directions,
            enemy_spawn_direction=cfg.enemy_spawn_direction,
            stage=cfg.stage,
            shrink_safe_zone=cfg.shrink_safe_zone,
            use_zone_reward=cfg.use_zone_reward,
            enemy_move=cfg.enemy_move,
            enemy_fire=cfg.enemy_fire,
            stationary_target_mode=cfg.stationary_target_mode,
            enemy_spawn_distance_min=cfg.enemy_spawn_distance_min,
            enemy_spawn_distance_max=cfg.enemy_spawn_distance_max,
            fire_interval_steps=cfg.fire_interval_steps,
            bullet_speed=cfg.bullet_speed,
            bullet_range=cfg.bullet_range,
            bullet_damage=cfg.bullet_damage,
            bullet_hit_radius=cfg.bullet_hit_radius,
        )
        observation = env.reset(seed=int(seed) + episode_index)
        target_dir = {
            "x": float(observation["target_dir_x"]),
            "y": float(observation["target_dir_y"]),
        }
        ideal_aim_bin = vec_to_aim_bin(target_dir)
        total_reward = 0.0
        steps: list[dict[str, Any]] = []
        done = False
        while not done and env.step_count < int(cfg.max_episode_steps):
            step_index = env.step_count
            action = {"move": 0, "aim": ideal_aim_bin, "fire": 1}
            decoded = decode_action(action)
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps.append(
                {
                    "t": step_index,
                    "distance_to_enemy": float(info.get("range_debug", {}).get("distance_to_enemy", 0.0)),
                    "ideal_aim_bin": int(info.get("aim_debug", {}).get("ideal_aim_bin", ideal_aim_bin)),
                    "action": {
                        "raw": dict(action),
                        "decoded": {
                            "move": {"x": decoded["moveX"], "y": decoded["moveY"]},
                            "aim": {"x": decoded["aimX"], "y": decoded["aimY"]},
                            "fire": decoded["fire"],
                        },
                    },
                    "fire": {
                        "requested": info.get("fire", {}).get("fire_requested", False),
                        "shot_fired": info.get("fire", {}).get("shot_fired", False),
                        "blocked_reason": info.get("fire", {}).get("fire_blocked_reason"),
                    },
                    "aim": {
                        "aim_bin": info.get("aim_debug", {}).get("aim_bin"),
                        "ideal_aim_bin": info.get("aim_debug", {}).get("ideal_aim_bin"),
                        "aim_bin_error": info.get("aim_debug", {}).get("aim_bin_error"),
                        "alignment": info.get("aim_debug", {}).get("aim_alignment", 0.0),
                        "angle_error_deg": info.get("aim_debug", {}).get("angle_error_deg", 0.0),
                    },
                    "range": info.get("range_debug", {}),
                    "events": info.get("bullet_events", []),
                    "bullets": info.get("bullets", []),
                    "reward": float(reward),
                    "reward_components": info.get("reward_components", {}),
                    "metrics_delta": {
                        "damage_dealt_delta": info.get("damage_delta", {}).get("enemy_hp", 0.0),
                        "damage_taken_delta": info.get("damage_delta", {}).get("self_hp", 0.0),
                    },
                }
            )
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

    result = {
        "schema_version": "cpc-common-v0",
        "source": "eval_scripted_ideal_aim_same_reset",
        "config": {
            "config_path": str(config_path),
            "episodes": max(1, int(episodes)),
            "max_steps": int(cfg.max_episode_steps),
            "seed": int(seed),
            "stage": cfg.stage,
            "enemy_move": bool(cfg.enemy_move),
            "enemy_fire": bool(cfg.enemy_fire),
            "stationary_target_mode": bool(cfg.stationary_target_mode),
            "bullet_range": float(cfg.bullet_range),
            "enemy_spawn_distance_min": cfg.enemy_spawn_distance_min,
            "enemy_spawn_distance_max": cfg.enemy_spawn_distance_max,
            "enemy_spawn_direction": cfg.enemy_spawn_direction,
            "randomize_enemy_spawn_direction": bool(cfg.randomize_enemy_spawn_direction),
        },
        "episodes": result_episodes,
    }
    analysis = analyze_result(result)
    return {
        **result,
        "analysis": analysis,
    }


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    analysis = result["analysis"]
    aggregate = analysis.get("aggregate", {})
    episodes = analysis.get("episodes", [])
    return {
        "episode_count": analysis.get("episode_count", 0),
        "bullet_range": result.get("config", {}).get("bullet_range", 0.0),
        "mean_distance_to_enemy": aggregate.get("avg_distance_to_enemy", 0.0),
        "mean_shot_fired_count": aggregate.get("shot_fired_count", 0.0),
        "mean_self_bullet_hit_count": aggregate.get("self_bullet_hit_count", 0.0),
        "mean_bullet_hit_per_shot": aggregate.get("bullet_hit_per_shot", 0.0),
        "warnings": aggregate.get("warnings", {}),
        "episodes": [
            {
                "episode_index": episode.get("episode_index", idx),
                "distance_to_enemy": episode.get("metrics", {}).get("avg_distance_to_enemy", 0.0),
                "ideal_aim_bins": sorted(
                    {
                        step.get("ideal_aim_bin")
                        for step in result.get("episodes", [])[idx].get("steps", [])
                        if step.get("ideal_aim_bin") is not None
                    }
                ),
                "shot_fired_count": episode.get("metrics", {}).get("shot_fired_count", 0.0),
                "self_bullet_hit_count": episode.get("metrics", {}).get("self_bullet_hit_count", 0.0),
                "bullet_hit_per_shot": episode.get("metrics", {}).get("bullet_hit_per_shot", 0.0),
                "bullet_lifecycle": episode.get("bullet_lifecycle", []),
                "warnings": episode.get("warnings", []),
            }
            for idx, episode in enumerate(episodes)
        ],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("Scripted ideal-aim same-reset probe")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe ideal-aim shooting on the same reset sample.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enemy-spawn-direction")
    parser.add_argument("--save-result")
    args = parser.parse_args()

    result = run_scripted_ideal_aim_same_reset(
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        enemy_spawn_direction=args.enemy_spawn_direction,
    )
    summary = summarize(result)
    print_summary(summary)
    if args.save_result:
        path = Path(args.save_result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"saved_result: {path}")


if __name__ == "__main__":
    main()
