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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PR3 PPO smoke checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sampled", action="store_true")
    args = parser.parse_args()
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
