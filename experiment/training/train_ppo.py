from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from experiment.checkpointing import (
        save_checkpoint,
        save_selected_checkpoint_if_needed,
        selected_paths,
    )
    from experiment.analyze_local_combat_eval import analyze_result
except ModuleNotFoundError:
    import sys

    EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiment.checkpointing import (
        save_checkpoint,
        save_selected_checkpoint_if_needed,
        selected_paths,
    )
    from experiment.analyze_local_combat_eval import analyze_result

if __package__:
    from .cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS, decode_action
    from .ppo_policy import OBS_DIM, OBS_KEYS, MultiDiscreteActorCritic, flatten_observation
    from .torchrl_env import METRIC_KEYS, REWARD_COMPONENT_KEYS, TorchRLCPCEnv
else:
    from cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS, decode_action
    from ppo_policy import OBS_DIM, OBS_KEYS, MultiDiscreteActorCritic, flatten_observation
    from torchrl_env import METRIC_KEYS, REWARD_COMPONENT_KEYS, TorchRLCPCEnv


@dataclass
class PPOConfig:
    seed: int = 0
    device: str = "auto"
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
    stage: str = "local_combat"
    shrink_safe_zone: bool = False
    use_zone_reward: bool = False
    enemy_move: bool = True
    enemy_fire: bool = True
    stationary_target_mode: bool = False
    fire_interval_steps: int = 5
    bullet_speed: float = 140.0
    bullet_range: float = 280.0
    bullet_damage: float = 10.0
    bullet_hit_radius: float = 12.0
    selection_metric: str = "stage1_combat_quality"
    selection_mode: str = "max"
    selection_eval_episodes: int = 2
    eval_analysis_interval_steps: int = 10000
    eval_analysis_episodes: int = 2
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
    enemy_spawn_distance_min: float | None = None
    enemy_spawn_distance_max: float | None = None
    config_path: str | None = None


def load_config(path: str | Path | None, *, smoke: bool = False) -> PPOConfig:
    cfg = PPOConfig()
    if path is not None:
        config_path = resolve_config_path(path)
        cfg.config_path = str(config_path)
        values = _read_simple_yaml(config_path)
        for key, value in values.items():
            if hasattr(cfg, key):
                current = getattr(cfg, key)
                setattr(cfg, key, _coerce_value(value, current))
        for key in ("enemy_spawn_distance_min", "enemy_spawn_distance_max"):
            if key in values and values[key] not in ("", None):
                setattr(cfg, key, float(values[key]))
        _validate_stage1_config(config_path, values)
    if smoke:
        cfg.total_steps = min(cfg.total_steps, 256)
        cfg.rollout_steps = min(cfg.rollout_steps, 64)
        cfg.num_epochs = min(cfg.num_epochs, 2)
        cfg.minibatch_size = min(cfg.minibatch_size, 32)
        cfg.selection_eval_episodes = min(cfg.selection_eval_episodes, 2)
    return cfg


def resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("config path is required")
    raw_path = Path(path)
    candidates = [raw_path]
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = experiment_root.parent
    if not raw_path.is_absolute():
        candidates.extend(
            [
                experiment_root / raw_path,
                repo_root / raw_path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return raw_path.resolve(strict=False)


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
    reward_component_sums = {f"reward_{key}": 0.0 for key in REWARD_COMPONENT_KEYS}

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
        for key, value in _reward_components_from_td(next_td).items():
            reward_component_sums[key] += value

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
        "reward_components_mean": {
            key: value / max(1, cfg.rollout_steps)
            for key, value in reward_component_sums.items()
        },
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


def train_ppo(cfg: PPOConfig, *, progress: bool = False) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = _make_run_dir(cfg)
    env = TorchRLCPCEnv(
        seed=cfg.seed,
        max_steps=cfg.max_episode_steps,
        device=device,
        randomize_enemy_spawn_direction=cfg.randomize_enemy_spawn_direction,
        enemy_spawn_directions=cfg.enemy_spawn_directions,
        enemy_spawn_direction=cfg.enemy_spawn_direction,
        enemy_spawn_distance_min=cfg.enemy_spawn_distance_min,
        enemy_spawn_distance_max=cfg.enemy_spawn_distance_max,
        stage=cfg.stage,
        shrink_safe_zone=cfg.shrink_safe_zone,
        use_zone_reward=cfg.use_zone_reward,
        enemy_move=cfg.enemy_move,
        enemy_fire=cfg.enemy_fire,
        stationary_target_mode=cfg.stationary_target_mode,
        fire_interval_steps=cfg.fire_interval_steps,
        bullet_speed=cfg.bullet_speed,
        bullet_range=cfg.bullet_range,
        bullet_damage=cfg.bullet_damage,
        bullet_hit_radius=cfg.bullet_hit_radius,
    )
    policy = MultiDiscreteActorCritic(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    metrics_path = run_dir / "metrics.csv"
    paths = selected_paths(run_dir)
    selected_checkpoint_path = (
        paths["checkpoint_min_reward"]
        if cfg.selection_mode == "min"
        else paths["checkpoint_max_reward"]
    )
    rows = []
    steps = 0
    next_eval_analysis_step = (
        int(cfg.eval_analysis_interval_steps)
        if int(cfg.eval_analysis_interval_steps) > 0
        else None
    )

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
        eval_mean_episode_reward = None
        eval_analysis = None
        eval_analysis_path = None
        if cfg.selection_metric == "eval_mean_episode_reward":
            eval_mean_episode_reward = evaluate_policy_mean_return(
                policy,
                cfg,
                episodes=cfg.selection_eval_episodes,
                device=device,
                seed_offset=len(rows) * max(1, cfg.selection_eval_episodes),
            )
            selection_value = eval_mean_episode_reward
        elif cfg.selection_metric == "stage1_combat_quality":
            eval_analysis = evaluate_policy_local_combat_analysis(
                policy,
                cfg,
                episodes=cfg.selection_eval_episodes,
                device=device,
                seed_offset=len(rows) * max(1, cfg.selection_eval_episodes),
            )
            eval_analysis_path = run_dir / f"selection_eval_analysis_step_{steps}.json"
            eval_analysis_path.write_text(json.dumps(eval_analysis, indent=2, sort_keys=True), encoding="utf-8")
            eval_mean_episode_reward = float(eval_analysis.get("aggregate", {}).get("total_reward", 0.0))
            selection_value = stage1_combat_quality_score(eval_analysis)
        elif cfg.selection_metric == "episodic_return_mean":
            selection_value = _mean(rollout["episode_returns"])
        else:
            raise ValueError(
                "selection_metric must be 'eval_mean_episode_reward', 'episodic_return_mean', "
                "'stage1_combat_quality', "
                f"got {cfg.selection_metric!r}"
            )
        if next_eval_analysis_step is not None and steps >= next_eval_analysis_step:
            if eval_analysis is None:
                eval_analysis = evaluate_policy_local_combat_analysis(
                    policy,
                    cfg,
                    episodes=cfg.eval_analysis_episodes,
                    device=device,
                    seed_offset=10_000 + len(rows) * max(1, cfg.eval_analysis_episodes),
                )
            eval_analysis_path = run_dir / f"eval_analysis_step_{steps}.json"
            eval_analysis_path.write_text(json.dumps(eval_analysis, indent=2, sort_keys=True), encoding="utf-8")
            while next_eval_analysis_step is not None and steps >= next_eval_analysis_step:
                next_eval_analysis_step += int(cfg.eval_analysis_interval_steps)
        row = {
            "update": len(rows) + 1,
            "step": steps,
            "episodic_return_mean": _mean(rollout["episode_returns"]),
            "episode_length_mean": _mean(rollout["episode_lengths"]),
            "eval_mean_episode_reward": eval_mean_episode_reward,
            "selection_metric": cfg.selection_metric,
            "selection_value": selection_value,
            "enemy_move": float(bool(cfg.enemy_move)),
            "enemy_fire": float(bool(cfg.enemy_fire)),
            "stationary_target_mode": float(bool(cfg.stationary_target_mode)),
            "is_selected_checkpoint": False,
            "checkpoint_latest": str(paths["checkpoint_latest"]),
            "checkpoint_selected": str(paths["checkpoint_selected"]),
            "checkpoint_min_reward": str(paths["checkpoint_min_reward"]),
            "checkpoint_max_reward": str(paths["checkpoint_max_reward"]),
            **losses,
            **rollout["last_metrics"],
            **rollout["reward_components_mean"],
        }
        if eval_analysis is not None:
            row.update(_eval_analysis_row(eval_analysis))
            row["eval_analysis_path"] = "" if eval_analysis_path is None else str(eval_analysis_path)
        save_checkpoint(
            paths["checkpoint_latest"],
            policy=policy,
            optimizer=optimizer,
            config={**asdict(cfg), "resolved_device": str(device)},
            update=row["update"],
            global_step=steps,
            metrics=row,
            selection_metric=cfg.selection_metric,
            selection_mode=cfg.selection_mode,
            selection_value=selection_value,
            action_metadata=_action_metadata(),
            observation_metadata=_observation_metadata(),
        )
        is_selected, current_selected_value = save_selected_checkpoint_if_needed(
            run_dir=run_dir,
            latest_checkpoint=paths["checkpoint_latest"],
            selected_checkpoint=selected_checkpoint_path,
            metadata_path=paths["selected_reward_checkpoint"],
            update=row["update"],
            global_step=steps,
            metrics=row,
            selection_metric=cfg.selection_metric,
            selection_mode=cfg.selection_mode,
            selection_value=selection_value,
        )
        row["is_selected_checkpoint"] = is_selected
        if is_selected:
            shutil.copy2(selected_checkpoint_path, paths["checkpoint_selected"])
        rows.append(row)
        _write_metrics(metrics_path, rows)
        if progress:
            progress_payload = {
                "update": row["update"],
                "step": row["step"],
                "total_steps": cfg.total_steps,
                "episodic_return_mean": row["episodic_return_mean"],
                "eval_mean_episode_reward": row["eval_mean_episode_reward"],
                "selection_metric": row["selection_metric"],
                "selection_value": row["selection_value"],
                "is_selected_checkpoint": row["is_selected_checkpoint"],
            }
            if eval_analysis is not None:
                progress_payload["eval_analysis"] = eval_analysis["aggregate"]
                progress_payload["eval_analysis_path"] = "" if eval_analysis_path is None else str(eval_analysis_path)
            print(
                json.dumps(progress_payload),
                file=sys.stderr,
                flush=True,
            )

    checkpoint_path = run_dir / "checkpoint.pt"
    shutil.copy2(paths["checkpoint_latest"], checkpoint_path)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    selected_metadata = _read_json_if_exists(paths["selected_reward_checkpoint"])
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_latest": str(paths["checkpoint_latest"]),
        "checkpoint_selected": str(paths["checkpoint_selected"]),
        "checkpoint_min_reward": str(paths["checkpoint_min_reward"]),
        "checkpoint_max_reward": str(paths["checkpoint_max_reward"]),
        "selected_reward_checkpoint": str(paths["selected_reward_checkpoint"]),
        "selection_metric": cfg.selection_metric,
        "selection_mode": cfg.selection_mode,
        "selection_value": selected_metadata.get("selection_value"),
        "updates": len(rows),
        "global_step": steps,
        "metrics_csv": str(metrics_path),
        "device": str(device),
        "last_metrics": rows[-1] if rows else {},
    }


@torch.no_grad()
def evaluate_policy_mean_return(
    policy: MultiDiscreteActorCritic,
    cfg: PPOConfig,
    *,
    episodes: int,
    device: torch.device,
    seed_offset: int = 0,
) -> float:
    returns = []
    was_training = policy.training
    policy.eval()
    for episode in range(max(1, int(episodes))):
        env = TorchRLCPCEnv(
            seed=int(cfg.seed) + seed_offset + episode,
            max_steps=int(cfg.max_episode_steps),
            device=device,
            randomize_enemy_spawn_direction=cfg.randomize_enemy_spawn_direction,
            enemy_spawn_directions=cfg.enemy_spawn_directions,
            enemy_spawn_direction=cfg.enemy_spawn_direction,
            enemy_spawn_distance_min=cfg.enemy_spawn_distance_min,
            enemy_spawn_distance_max=cfg.enemy_spawn_distance_max,
            stage=cfg.stage,
            shrink_safe_zone=cfg.shrink_safe_zone,
            use_zone_reward=cfg.use_zone_reward,
            enemy_move=cfg.enemy_move,
            enemy_fire=cfg.enemy_fire,
            stationary_target_mode=cfg.stationary_target_mode,
            fire_interval_steps=cfg.fire_interval_steps,
            bullet_speed=cfg.bullet_speed,
            bullet_range=cfg.bullet_range,
            bullet_damage=cfg.bullet_damage,
            bullet_hit_radius=cfg.bullet_hit_radius,
        )
        obs = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            move_logits, aim_logits, fire_logits, _ = policy(obs)
            step_td = obs.clone()
            step_td["move"] = move_logits.argmax(dim=-1).squeeze()
            step_td["aim"] = aim_logits.argmax(dim=-1).squeeze()
            step_td["fire"] = fire_logits.argmax(dim=-1).squeeze()
            obs = env.step(step_td)["next"]
            episode_return += float(obs["reward"].reshape(-1)[0].item())
            done = bool(obs["done"].reshape(-1)[0].item())
        returns.append(episode_return)
    if was_training:
        policy.train()
    return _mean(returns)


@torch.no_grad()
def evaluate_policy_local_combat_analysis(
    policy: MultiDiscreteActorCritic,
    cfg: PPOConfig,
    *,
    episodes: int,
    device: torch.device,
    seed_offset: int = 0,
) -> dict[str, Any]:
    was_training = policy.training
    policy.eval()
    result_episodes = []
    fire_logits_sum = torch.zeros(2, dtype=torch.float32)
    fire_probs_sum = torch.zeros(2, dtype=torch.float32)
    deterministic_fire_count = 0
    total_steps = 0
    for episode in range(max(1, int(episodes))):
        env = TorchRLCPCEnv(
            seed=int(cfg.seed) + seed_offset + episode,
            max_steps=int(cfg.max_episode_steps),
            device=device,
            randomize_enemy_spawn_direction=cfg.randomize_enemy_spawn_direction,
            enemy_spawn_directions=cfg.enemy_spawn_directions,
            enemy_spawn_direction=cfg.enemy_spawn_direction,
            enemy_spawn_distance_min=cfg.enemy_spawn_distance_min,
            enemy_spawn_distance_max=cfg.enemy_spawn_distance_max,
            stage=cfg.stage,
            shrink_safe_zone=cfg.shrink_safe_zone,
            use_zone_reward=cfg.use_zone_reward,
            enemy_move=cfg.enemy_move,
            enemy_fire=cfg.enemy_fire,
            stationary_target_mode=cfg.stationary_target_mode,
            fire_interval_steps=cfg.fire_interval_steps,
            bullet_speed=cfg.bullet_speed,
            bullet_range=cfg.bullet_range,
            bullet_damage=cfg.bullet_damage,
            bullet_hit_radius=cfg.bullet_hit_radius,
        )
        cpc_env = env.cpc_env
        obs = cpc_env.reset(seed=int(cfg.seed) + seed_offset + episode)
        done = False
        total_reward = 0.0
        steps = []
        while not done and cpc_env.step_count < int(cfg.max_episode_steps):
            step_index = cpc_env.step_count
            obs_td = env._td_from_obs(
                obs,
                reward=None,
                done=False,
                terminated=False,
                truncated=False,
                info={},
            )
            move_logits, aim_logits, fire_logits, _ = policy(obs_td)
            fire_logits_cpu = fire_logits.detach().to("cpu").reshape(-1, fire_logits.shape[-1]).mean(dim=0)
            fire_probs_cpu = torch.softmax(fire_logits.detach(), dim=-1).to("cpu").reshape(-1, fire_logits.shape[-1]).mean(dim=0)
            fire_logits_sum += fire_logits_cpu
            fire_probs_sum += fire_probs_cpu
            deterministic_fire_count += int(fire_logits_cpu.argmax(dim=-1).item())
            action = {
                "move": int(move_logits.argmax(dim=-1).reshape(-1)[0].item()),
                "aim": int(aim_logits.argmax(dim=-1).reshape(-1)[0].item()),
                "fire": int(fire_logits.argmax(dim=-1).reshape(-1)[0].item()),
            }
            decoded = decode_action(action)
            obs, reward, done, info = cpc_env.step(action)
            total_reward += float(reward)
            steps.append(_compact_eval_step(step_index, action, decoded, reward, info, cpc_env))
            total_steps += 1
        result_episodes.append(
            {
                "episode_index": episode,
                "steps": steps,
                "episode_return": {"agent": total_reward},
                "episode_length": len(steps),
                "final_metrics": cpc_env.metrics.summary(),
                "stopped_early": False,
            }
        )
    if was_training:
        policy.train()
    analysis = analyze_result(
        {
            "schema_version": "cpc-common-v0",
            "source": "train_ppo_eval_analysis",
            "config": {
                "episodes": max(1, int(episodes)),
                "max_steps": int(cfg.max_episode_steps),
                "stage": cfg.stage,
                "enemy_move": bool(cfg.enemy_move),
                "enemy_fire": bool(cfg.enemy_fire),
                "stationary_target_mode": bool(cfg.stationary_target_mode),
                "bullet_range": float(cfg.bullet_range),
            },
            "episodes": result_episodes,
        }
    )
    mean_fire_logits = fire_logits_sum / float(max(total_steps, 1))
    mean_fire_probs = fire_probs_sum / float(max(total_steps, 1))
    analysis["fire_diagnostics"] = {
        "mean_logits": [float(value) for value in mean_fire_logits.tolist()],
        "mean_probs": [float(value) for value in mean_fire_probs.tolist()],
        "sampled_fire_rate": None,
        "deterministic_fire_rate": deterministic_fire_count / max(total_steps, 1),
        "deterministic_fire_action": int(mean_fire_logits.argmax(dim=-1).item()) if mean_fire_logits.numel() else 0,
        "stochastic_eval": False,
    }
    return analysis


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _metrics_from_td(td) -> dict[str, float]:
    metrics = {}
    for key in METRIC_KEYS:
        try:
            metrics[key] = float(td["metrics", key].reshape(-1)[0].item())
        except Exception:
            metrics[key] = 0.0
    return metrics


def _reward_components_from_td(td) -> dict[str, float]:
    components = {}
    for key in REWARD_COMPONENT_KEYS:
        try:
            components[f"reward_{key}"] = float(td["reward_components", key].reshape(-1)[0].item())
        except Exception:
            components[f"reward_{key}"] = 0.0
    return components


def stage1_combat_quality_score(analysis: dict[str, Any]) -> float:
    aggregate = analysis.get("aggregate", {})
    warning_count = sum(int(value) for value in aggregate.get("warnings", {}).values())
    damage_trade_ratio = float(aggregate.get("damage_trade_ratio", 0.0))
    bullet_hit_per_shot = float(aggregate.get("bullet_hit_per_shot", 0.0))
    damage_dealt_ratio = float(aggregate.get("damage_dealt_ratio", 0.0))
    return (
        damage_trade_ratio
        + (0.20 * bullet_hit_per_shot)
        + (0.10 * min(damage_dealt_ratio, 1.0))
        - (0.05 * warning_count)
    )


def _eval_analysis_row(analysis: dict[str, Any]) -> dict[str, float]:
    aggregate = analysis.get("aggregate", {})
    config = analysis.get("config", {})
    fire = analysis.get("fire_diagnostics", {})
    warning_count = sum(int(value) for value in aggregate.get("warnings", {}).values())
    mean_logits = fire.get("mean_logits", [0.0, 0.0])
    mean_probs = fire.get("mean_probs", [0.0, 0.0])
    return {
        "eval_analysis_total_reward": float(aggregate.get("total_reward", 0.0)),
        "eval_analysis_damage_taken": float(aggregate.get("damage_taken", 0.0)),
        "eval_analysis_damage_dealt_ratio": float(aggregate.get("damage_dealt_ratio", 0.0)),
        "eval_analysis_damage_taken_ratio": float(aggregate.get("damage_taken_ratio", 0.0)),
        "eval_analysis_damage_trade_ratio": float(aggregate.get("damage_trade_ratio", 0.0)),
        "eval_analysis_bullet_range": float(aggregate.get("bullet_range", 0.0)),
        "eval_analysis_avg_distance_to_enemy": float(aggregate.get("avg_distance_to_enemy", 0.0)),
        "eval_analysis_max_distance_to_enemy": float(aggregate.get("max_distance_to_enemy", 0.0)),
        "eval_analysis_distance_over_bullet_range_rate": float(aggregate.get("distance_over_bullet_range_rate", 0.0)),
        "eval_analysis_within_bullet_range_rate": float(aggregate.get("within_bullet_range_rate", 0.0)),
        "eval_analysis_fire_head_logit_0": float(mean_logits[0] if len(mean_logits) > 0 else 0.0),
        "eval_analysis_fire_head_logit_1": float(mean_logits[1] if len(mean_logits) > 1 else 0.0),
        "eval_analysis_fire_head_prob_0": float(mean_probs[0] if len(mean_probs) > 0 else 0.0),
        "eval_analysis_fire_head_prob_1": float(mean_probs[1] if len(mean_probs) > 1 else 0.0),
        "eval_analysis_fire_deterministic_action": float(fire.get("deterministic_fire_action", 0.0)),
        "eval_analysis_fire_deterministic_rate": float(fire.get("deterministic_fire_rate", 0.0)),
        "eval_analysis_enemy_move": float(bool(config.get("enemy_move", True))),
        "eval_analysis_enemy_fire": float(bool(config.get("enemy_fire", True))),
        "eval_analysis_stationary_target_mode": float(bool(config.get("stationary_target_mode", False))),
        "eval_analysis_fire_requested_count": float(aggregate.get("fire_requested_count", 0.0)),
        "eval_analysis_shot_fired_count": float(aggregate.get("shot_fired_count", 0.0)),
        "eval_analysis_fire_blocked_cooldown_count": float(aggregate.get("fire_blocked_cooldown_count", 0.0)),
        "eval_analysis_self_bullet_spawn_count": float(aggregate.get("self_bullet_spawn_count", 0.0)),
        "eval_analysis_self_bullet_hit_count": float(aggregate.get("self_bullet_hit_count", 0.0)),
        "eval_analysis_self_bullet_missed_count": float(aggregate.get("self_bullet_missed_count", 0.0)),
        "eval_analysis_self_bullet_alive_at_episode_end": float(aggregate.get("self_bullet_alive_at_episode_end", 0.0)),
        "eval_analysis_enemy_bullet_spawn_count": float(aggregate.get("enemy_bullet_spawn_count", 0.0)),
        "eval_analysis_enemy_bullet_hit_self_count": float(aggregate.get("enemy_bullet_hit_self_count", 0.0)),
        "eval_analysis_hit_ratio": float(aggregate.get("hit_ratio", 0.0)),
        "eval_analysis_missed_shot_rate": float(aggregate.get("missed_shot_rate", 0.0)),
        "eval_analysis_bullet_hit_per_shot": float(aggregate.get("bullet_hit_per_shot", 0.0)),
        "eval_analysis_aim_bin_0_rate": float(aggregate.get("aim_bin_0_rate", 0.0)),
        "eval_analysis_dominant_aim_bin": float(aggregate.get("dominant_aim_bin", 0.0) or 0.0),
        "eval_analysis_dominant_aim_bin_rate": float(aggregate.get("dominant_aim_bin_rate", 0.0)),
        "eval_analysis_exact_aim_match_rate": float(aggregate.get("exact_aim_match_rate", 0.0)),
        "eval_analysis_shot_exact_aim_rate": float(aggregate.get("shot_exact_aim_rate", 0.0)),
        "eval_analysis_good_range_rate": float(aggregate.get("good_range_rate", 0.0)),
        "eval_analysis_shot_good_range_rate": float(aggregate.get("shot_good_range_rate", 0.0)),
        "eval_analysis_reward_hacking_warning_count": float(warning_count),
    }


def _compact_eval_step(
    step_index: int,
    action: dict[str, int],
    decoded: dict[str, Any],
    reward: float,
    info: dict[str, Any],
    cpc_env,
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
            "self": {"hp": cpc_env.state.get("self_hp"), "pos": cpc_env.state.get("self_pos")},
            "enemy": {"hp": cpc_env.state.get("enemy_hp"), "pos": cpc_env.state.get("enemy_pos")},
            "dist": {"enemy": info.get("range_debug", {}).get("distance_to_enemy")},
        },
        "metrics_delta": {
            "damage_dealt_delta": info.get("damage_delta", {}).get("enemy_hp", 0.0),
            "damage_taken_delta": info.get("damage_delta", {}).get("self_hp", 0.0),
        },
    }


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "update",
        "step",
        "episodic_return_mean",
        "episode_length_mean",
        "eval_mean_episode_reward",
        "selection_metric",
        "selection_value",
        "enemy_move",
        "enemy_fire",
        "stationary_target_mode",
        "is_selected_checkpoint",
        "checkpoint_latest",
        "checkpoint_selected",
        "checkpoint_min_reward",
        "checkpoint_max_reward",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        *METRIC_KEYS,
        "eval_analysis_total_reward",
        "eval_analysis_damage_taken",
        "eval_analysis_damage_dealt_ratio",
        "eval_analysis_damage_taken_ratio",
        "eval_analysis_damage_trade_ratio",
        "eval_analysis_bullet_range",
        "eval_analysis_avg_distance_to_enemy",
        "eval_analysis_max_distance_to_enemy",
        "eval_analysis_distance_over_bullet_range_rate",
        "eval_analysis_within_bullet_range_rate",
        "eval_analysis_fire_head_logit_0",
        "eval_analysis_fire_head_logit_1",
        "eval_analysis_fire_head_prob_0",
        "eval_analysis_fire_head_prob_1",
        "eval_analysis_fire_deterministic_action",
        "eval_analysis_fire_deterministic_rate",
        "eval_analysis_enemy_move",
        "eval_analysis_enemy_fire",
        "eval_analysis_stationary_target_mode",
        "eval_analysis_fire_requested_count",
        "eval_analysis_shot_fired_count",
        "eval_analysis_fire_blocked_cooldown_count",
        "eval_analysis_self_bullet_spawn_count",
        "eval_analysis_self_bullet_hit_count",
        "eval_analysis_self_bullet_missed_count",
        "eval_analysis_self_bullet_alive_at_episode_end",
        "eval_analysis_enemy_bullet_spawn_count",
        "eval_analysis_enemy_bullet_hit_self_count",
        "eval_analysis_hit_ratio",
        "eval_analysis_missed_shot_rate",
        "eval_analysis_bullet_hit_per_shot",
        "eval_analysis_aim_bin_0_rate",
        "eval_analysis_dominant_aim_bin",
        "eval_analysis_dominant_aim_bin_rate",
        "eval_analysis_exact_aim_match_rate",
        "eval_analysis_shot_exact_aim_rate",
        "eval_analysis_good_range_rate",
        "eval_analysis_shot_good_range_rate",
        "eval_analysis_reward_hacking_warning_count",
        "eval_analysis_path",
        *[f"reward_{key}" for key in REWARD_COMPONENT_KEYS],
        "total_reward",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key, 0.0) for key in fieldnames}
            csv_row["total_reward"] = row.get("episodic_return_mean", 0.0)
            writer.writerow(csv_row)


def _make_run_dir(cfg: PPOConfig) -> Path:
    root = Path(cfg.run_dir)
    run_dir = root / f"ppo_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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
    if value == "null" or value == "None":
        return None
    if isinstance(current, bool):
        return str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        return value
    return value


def _action_metadata() -> dict[str, int]:
    return {
        "move_bins": MOVE_BINS,
        "aim_bins": AIM_BINS,
        "fire_bins": FIRE_BINS,
    }


def _observation_metadata() -> dict[str, Any]:
    return {
        "observation_keys": list(OBS_KEYS),
        "obs_dim": OBS_DIM,
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_stage1_config(path: Path, values: dict[str, Any]) -> None:
    if not path.name.startswith("local_combat_"):
        return
    required_keys = (
        "enemy_move",
        "enemy_fire",
        "stationary_target_mode",
        "enemy_spawn_distance_min",
        "enemy_spawn_distance_max",
        "bullet_range",
    )
    missing = [key for key in required_keys if key not in values or values[key] in ("", None)]
    if missing:
        raise ValueError(
            f"Stage 1 config {path} is missing required field(s): {', '.join(missing)}"
        )


def debug_print_reset_samples(cfg: PPOConfig, *, samples: int, config_path: str | None = None) -> None:
    source_path_text = config_path or cfg.config_path
    source_path = resolve_config_path(source_path_text) if source_path_text else None
    raw_yaml = _read_simple_yaml(source_path) if source_path is not None else {}
    device = resolve_device(cfg.device)
    env = TorchRLCPCEnv(
        seed=cfg.seed,
        max_steps=cfg.max_episode_steps,
        device=device,
        randomize_enemy_spawn_direction=cfg.randomize_enemy_spawn_direction,
        enemy_spawn_directions=cfg.enemy_spawn_directions,
        enemy_spawn_direction=cfg.enemy_spawn_direction,
        enemy_spawn_distance_min=cfg.enemy_spawn_distance_min,
        enemy_spawn_distance_max=cfg.enemy_spawn_distance_max,
        stage=cfg.stage,
        shrink_safe_zone=cfg.shrink_safe_zone,
        use_zone_reward=cfg.use_zone_reward,
        enemy_move=cfg.enemy_move,
        enemy_fire=cfg.enemy_fire,
        stationary_target_mode=cfg.stationary_target_mode,
        fire_interval_steps=cfg.fire_interval_steps,
        bullet_speed=cfg.bullet_speed,
        bullet_range=cfg.bullet_range,
        bullet_damage=cfg.bullet_damage,
        bullet_hit_radius=cfg.bullet_hit_radius,
    )
    print(json.dumps(
        {
            "config_path": config_path or cfg.config_path,
            "raw_yaml": {
                "bullet_range": raw_yaml.get("bullet_range"),
                "enemy_spawn_distance_min": raw_yaml.get("enemy_spawn_distance_min"),
                "enemy_spawn_distance_max": raw_yaml.get("enemy_spawn_distance_max"),
                "enemy_move": raw_yaml.get("enemy_move"),
                "enemy_fire": raw_yaml.get("enemy_fire"),
                "stationary_target_mode": raw_yaml.get("stationary_target_mode"),
            },
            "ppo_config": {
                "bullet_range": cfg.bullet_range,
                "enemy_spawn_distance_min": cfg.enemy_spawn_distance_min,
                "enemy_spawn_distance_max": cfg.enemy_spawn_distance_max,
                "enemy_move": cfg.enemy_move,
                "enemy_fire": cfg.enemy_fire,
                "stationary_target_mode": cfg.stationary_target_mode,
            },
            "env_internal": {
                "bullet_range": getattr(env.cpc_env, "fire_range", None),
                "enemy_spawn_distance_min": getattr(env.cpc_env, "enemy_spawn_distance_min", None),
                "enemy_spawn_distance_max": getattr(env.cpc_env, "enemy_spawn_distance_max", None),
                "enemy_move": getattr(env.cpc_env, "enemy_move", None),
                "enemy_fire": getattr(env.cpc_env, "enemy_fire", None),
                "stationary_target_mode": getattr(env.cpc_env, "stationary_target_mode", None),
            },
        },
        sort_keys=True,
    ))
    for index in range(max(0, int(samples))):
        td = env.reset()
        metrics = _metrics_from_td(td)
        debug_state = env.cpc_env.get_debug_state() if hasattr(env.cpc_env, "get_debug_state") else {}
        if not isinstance(debug_state, dict):
            debug_state = {}
        distance = float(
            debug_state.get("range_debug", {}).get("distance_to_enemy", td["distance_to_enemy"].reshape(-1)[0].item())
        )
        bullet_range = float(debug_state.get("combat", {}).get("bullet_range", cfg.bullet_range))
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
    parser = argparse.ArgumentParser(description="Run a minimal PPO smoke train on TorchRLCPCEnv.")
    parser.add_argument("--config", default="experiment/configs/ppo_smoke.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--debug-reset-samples", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.debug_reset_samples > 0:
        debug_print_reset_samples(cfg, samples=args.debug_reset_samples, config_path=args.config)
        return
    result = train_ppo(cfg, progress=args.progress)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
