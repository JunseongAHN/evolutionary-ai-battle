from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from ppo_policy import MultiDiscreteActorCritic, flatten_observation
    from torchrl_env import TorchRLCPCEnv
except ImportError:
    from .ppo_policy import MultiDiscreteActorCritic, flatten_observation
    from .torchrl_env import TorchRLCPCEnv


@dataclass
class PPOConfig:
    seed: int = 0
    device: str = "cpu"
    total_steps: int = 256
    rollout_steps: int = 64
    num_epochs: int = 2
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 0.0003
    max_grad_norm: float = 0.5
    max_episode_steps: int = 50
    hidden_dim: int = 64
    run_dir: str = "experiment/runs"


def load_config(path: str | Path | None, *, smoke: bool = False) -> PPOConfig:
    cfg = PPOConfig()
    if path is not None:
        values = _read_simple_yaml(Path(path))
        for key, value in values.items():
            if hasattr(cfg, key):
                current = getattr(cfg, key)
                setattr(cfg, key, _coerce_value(value, current))
    if smoke:
        cfg.total_steps = min(cfg.total_steps, 256)
        cfg.rollout_steps = min(cfg.rollout_steps, 64)
        cfg.num_epochs = min(cfg.num_epochs, 2)
        cfg.minibatch_size = min(cfg.minibatch_size, 32)
    return cfg


def collect_rollout(env: TorchRLCPCEnv, policy: MultiDiscreteActorCritic, cfg: PPOConfig) -> dict[str, Any]:
    obs_td = env.reset()
    observations = []
    moves = []
    aims = []
    fires = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    episode_returns = []
    episode_lengths = []
    current_return = 0.0
    current_length = 0
    last_metrics: dict[str, float] = {}

    for _ in range(cfg.rollout_steps):
        features = flatten_observation(obs_td).squeeze(0)
        with torch.no_grad():
            output = policy.sample_action(obs_td)

        step_td = obs_td.clone()
        step_td["move"] = output.action["move"].detach()
        step_td["aim"] = output.action["aim"].detach()
        step_td["fire"] = output.action["fire"].detach()
        next_td = env.step(step_td)["next"]

        reward = float(next_td["reward"].reshape(-1)[0].item())
        done = bool(next_td["done"].reshape(-1)[0].item())

        observations.append(features.detach())
        moves.append(output.action["move"].detach().reshape(()))
        aims.append(output.action["aim"].detach().reshape(()))
        fires.append(output.action["fire"].detach().reshape(()))
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=features.device))
        dones.append(torch.tensor(done, dtype=torch.float32, device=features.device))
        log_probs.append(output.log_prob.detach().reshape(()))
        values.append(output.value.detach().reshape(-1)[0])

        current_return += reward
        current_length += 1
        last_metrics = _metrics_from_td(next_td)

        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            current_return = 0.0
            current_length = 0
            obs_td = env.reset()
        else:
            obs_td = next_td

    with torch.no_grad():
        next_value = policy.value(obs_td).detach().reshape(-1)[0]

    return {
        "observations": torch.stack(observations),
        "actions": {
            "move": torch.stack(moves).long(),
            "aim": torch.stack(aims).long(),
            "fire": torch.stack(fires).long(),
        },
        "rewards": torch.stack(rewards),
        "dones": torch.stack(dones),
        "log_probs": torch.stack(log_probs),
        "values": torch.stack(values),
        "next_value": next_value,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "last_metrics": last_metrics,
    }


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step]
        bootstrap_value = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * bootstrap_value * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: MultiDiscreteActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, float]:
    obs = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["log_probs"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = obs.shape[0]
    indices = torch.arange(batch_size, device=obs.device)
    last_stats: dict[str, float] = {}

    for _ in range(cfg.num_epochs):
        shuffled = indices[torch.randperm(batch_size, device=obs.device)]
        for start in range(0, batch_size, cfg.minibatch_size):
            mb = shuffled[start:start + cfg.minibatch_size]
            mb_actions = {key: value[mb] for key, value in actions.items()}
            new_log_probs, entropy, values = policy.evaluate_actions(obs[mb], mb_actions)
            log_ratio = new_log_probs - old_log_probs[mb]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[mb] - values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs[mb] - new_log_probs).mean()
                clip_fraction = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean()
            last_stats = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_loss.item()),
                "approx_kl": float(approx_kl.item()),
                "clip_fraction": float(clip_fraction.item()),
            }
    return last_stats


def train_ppo(cfg: PPOConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_dir = _make_run_dir(cfg)
    env = TorchRLCPCEnv(seed=cfg.seed, max_steps=cfg.max_episode_steps, device=device)
    policy = MultiDiscreteActorCritic(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    metrics_path = run_dir / "metrics.csv"
    rows = []
    steps = 0

    while steps < cfg.total_steps:
        rollout = collect_rollout(env, policy, cfg)
        steps += int(rollout["rewards"].shape[0])
        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["dones"],
            rollout["values"],
            rollout["next_value"],
            cfg.gamma,
            cfg.gae_lambda,
        )
        losses = ppo_update(policy, optimizer, rollout, advantages, returns, cfg)
        row = {
            "update": len(rows) + 1,
            "step": steps,
            "episodic_return_mean": _mean(rollout["episode_returns"]),
            "episode_length_mean": _mean(rollout["episode_lengths"]),
            **losses,
            **rollout["last_metrics"],
        }
        rows.append(row)
        _write_metrics(metrics_path, rows)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": asdict(cfg),
            "obs_dim": policy.encoder[0].in_features,
            "hidden_dim": cfg.hidden_dim,
        },
        checkpoint_path,
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "metrics_csv": str(metrics_path),
        "last_metrics": rows[-1] if rows else {},
    }


def _metrics_from_td(td) -> dict[str, float]:
    metrics = {}
    for key in ("avg_ally_distance", "isolation_rate", "damage_dealt", "damage_taken"):
        try:
            metrics[key] = float(td["metrics", key].reshape(-1)[0].item())
        except Exception:
            metrics[key] = 0.0
    return metrics


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "update",
        "step",
        "episodic_return_mean",
        "episode_length_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "avg_ally_distance",
        "isolation_rate",
        "damage_dealt",
        "damage_taken",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, 0.0) for key in fieldnames})


def _make_run_dir(cfg: PPOConfig) -> Path:
    root = Path(cfg.run_dir)
    run_dir = root / f"ppo_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _read_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _coerce_value(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal PPO smoke train on TorchRLCPCEnv.")
    parser.add_argument("--config", default="experiment/configs/ppo_smoke.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = train_ppo(load_config(args.config, smoke=args.smoke))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
