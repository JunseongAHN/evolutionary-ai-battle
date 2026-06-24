from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from experiment.checkpointing import load_checkpoint
except ModuleNotFoundError:
    import sys

    EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiment.checkpointing import load_checkpoint

if __package__:
    from .ppo_policy import MultiDiscreteActorCritic
    from .torchrl_env import TorchRLCPCEnv
else:
    from ppo_policy import MultiDiscreteActorCritic
    from torchrl_env import TorchRLCPCEnv


@torch.no_grad()
def eval_checkpoint(
    checkpoint: str | Path,
    *,
    episodes: int = 3,
    sampled: bool = False,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
) -> dict:
    device = torch.device(device)
    checkpoint_data = load_checkpoint(checkpoint, map_location=device)
    cfg = checkpoint_data.get("config", {})
    policy = MultiDiscreteActorCritic(hidden_dim=int(checkpoint_data.get("hidden_dim", cfg.get("hidden_dim", 64))))
    policy.load_state_dict(checkpoint_data["policy_state_dict"])
    policy.to(device)
    policy.eval()

    returns = []
    lengths = []
    metrics = []
    for episode in range(episodes):
        env = TorchRLCPCEnv(
            seed=int(cfg.get("seed", 0)) + episode,
            max_steps=int(cfg.get("max_episode_steps", 50)),
            device=device,
            randomize_enemy_spawn_direction=bool(cfg.get("randomize_enemy_spawn_direction", False)),
            enemy_spawn_directions=cfg.get("enemy_spawn_directions"),
            enemy_spawn_direction=cfg.get("enemy_spawn_direction"),
            stage=str(cfg.get("stage", "local_combat")),
            shrink_safe_zone=bool(cfg.get("shrink_safe_zone", False)),
            use_zone_reward=bool(cfg.get("use_zone_reward", False)),
            enemy_move=bool(cfg.get("enemy_move", True)),
            enemy_fire=bool(cfg.get("enemy_fire", True)),
            stationary_target_mode=bool(cfg.get("stationary_target_mode", False)),
            fire_interval_steps=cfg.get("fire_interval_steps"),
            bullet_speed=cfg.get("bullet_speed"),
            bullet_range=cfg.get("bullet_range"),
            bullet_damage=cfg.get("bullet_damage"),
            bullet_hit_radius=cfg.get("bullet_hit_radius"),
        )
        obs = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        last_metrics = {}
        while not done:
            if sampled or not deterministic:
                output = policy.sample_action(obs)
                action = output.action
            else:
                move_logits, aim_logits, fire_logits, _ = policy(obs)
                action = {
                    "move": move_logits.argmax(dim=-1).squeeze(),
                    "aim": aim_logits.argmax(dim=-1).squeeze(),
                    "fire": fire_logits.argmax(dim=-1).squeeze(),
                }
            step_td = obs.clone()
            step_td["move"] = action["move"]
            step_td["aim"] = action["aim"]
            step_td["fire"] = action["fire"]
            obs = env.step(step_td)["next"]
            done = bool(obs["done"].reshape(-1)[0].item())
            episode_return += float(obs["reward"].reshape(-1)[0].item())
            episode_length += 1
            last_metrics = _metrics_from_td(obs)
        returns.append(episode_return)
        lengths.append(episode_length)
        metrics.append(last_metrics)

    report = {
        "mean_episode_return": _mean(returns),
        "mean_episode_length": _mean(lengths),
        "mean_metrics": _mean_metrics(metrics),
        "episodes": episodes,
        "checkpoint": str(checkpoint),
        "selection_metric": checkpoint_data.get("selection_metric"),
        "selection_mode": checkpoint_data.get("selection_mode"),
        "selection_value": checkpoint_data.get("selection_value"),
        "update": checkpoint_data.get("update"),
        "global_step": checkpoint_data.get("global_step"),
    }
    return report


def _metrics_from_td(td) -> dict[str, float]:
    metrics = {}
    for key in ("avg_ally_distance", "isolation_rate", "damage_dealt", "damage_taken"):
        try:
            metrics[key] = float(td["metrics", key].reshape(-1)[0].item())
        except Exception:
            metrics[key] = 0.0
    return metrics


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = ("avg_ally_distance", "isolation_rate", "damage_dealt", "damage_taken")
    return {key: _mean([item.get(key, 0.0) for item in metrics]) for key in keys}


def debug_print_reset_samples_from_checkpoint(
    checkpoint: str | Path,
    *,
    samples: int,
    device: str | torch.device = "cpu",
) -> None:
    device = torch.device(device)
    checkpoint_data = load_checkpoint(checkpoint, map_location=device)
    cfg = checkpoint_data.get("config", {})
    env = TorchRLCPCEnv(
        seed=int(cfg.get("seed", 0)),
        max_steps=int(cfg.get("max_episode_steps", 50)),
        device=device,
        randomize_enemy_spawn_direction=bool(cfg.get("randomize_enemy_spawn_direction", False)),
        enemy_spawn_directions=cfg.get("enemy_spawn_directions"),
        enemy_spawn_direction=cfg.get("enemy_spawn_direction"),
        enemy_spawn_distance_min=cfg.get("enemy_spawn_distance_min"),
        enemy_spawn_distance_max=cfg.get("enemy_spawn_distance_max"),
        stage=str(cfg.get("stage", "local_combat")),
        shrink_safe_zone=bool(cfg.get("shrink_safe_zone", False)),
        use_zone_reward=bool(cfg.get("use_zone_reward", False)),
        enemy_move=bool(cfg.get("enemy_move", True)),
        enemy_fire=bool(cfg.get("enemy_fire", True)),
        stationary_target_mode=bool(cfg.get("stationary_target_mode", False)),
        fire_interval_steps=cfg.get("fire_interval_steps"),
        bullet_speed=cfg.get("bullet_speed"),
        bullet_range=cfg.get("bullet_range"),
        bullet_damage=cfg.get("bullet_damage"),
        bullet_hit_radius=cfg.get("bullet_hit_radius"),
    )
    for index in range(max(0, int(samples))):
        td = env.reset()
        metrics = _metrics_from_td(td)
        debug_state = env.cpc_env.get_debug_state() if hasattr(env.cpc_env, "get_debug_state") else {}
        if not isinstance(debug_state, dict):
            debug_state = {}
        distance = float(
            debug_state.get("range_debug", {}).get("distance_to_enemy", td["distance_to_enemy"].reshape(-1)[0].item())
        )
        bullet_range = float(debug_state.get("combat", {}).get("bullet_range", cfg.get("bullet_range", 0.0)))
        payload = {
            "sample": index,
            "distance_to_enemy": distance,
            "bullet_range": bullet_range,
            "within_bullet_range": distance < bullet_range,
            "enemy_move": bool(debug_state.get("map", {}).get("enemy_move", getattr(env.cpc_env, "enemy_move", True))),
            "enemy_fire": bool(debug_state.get("map", {}).get("enemy_fire", getattr(env.cpc_env, "enemy_fire", True))),
            "stationary_target_mode": bool(
                debug_state.get("map", {}).get(
                    "stationary_target_mode",
                    getattr(env.cpc_env, "stationary_target_mode", False),
                )
            ),
            "self_pos": debug_state.get("state", {}).get("self_pos"),
            "enemy_pos": debug_state.get("state", {}).get("enemy_pos"),
            "metrics": metrics,
        }
        if payload["self_pos"] is None:
            payload["self_pos"] = {
                "x": float(td["self_pos"].reshape(-1)[0].item()),
                "y": float(td["self_pos"].reshape(-1)[1].item()),
            }
        if payload["enemy_pos"] is None:
            payload["enemy_pos"] = {
                "x": float(td["enemy_pos"].reshape(-1)[0].item()),
                "y": float(td["enemy_pos"].reshape(-1)[1].item()),
            }
        print(json.dumps(payload, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PR3 PPO smoke checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--debug-reset-samples", type=int, default=0)
    args = parser.parse_args()
    if args.debug_reset_samples > 0:
        debug_print_reset_samples_from_checkpoint(
            args.checkpoint,
            samples=args.debug_reset_samples,
            device=args.device,
        )
        return
    print(json.dumps(
        eval_checkpoint(
            args.checkpoint,
            episodes=args.episodes,
            sampled=args.sampled,
            device=args.device,
            deterministic=args.deterministic or not args.sampled,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
