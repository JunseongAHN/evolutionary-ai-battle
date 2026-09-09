"""CleanRL-style PPO for MultiDiscrete actions on ``SurvevVecEnv`` (torch only).

* ``ActorCritic``: MLP obs -> 256 -> 256 with one categorical head per action dimension
  (independent factorized policy) and a separate value MLP. ``forward`` returns concatenated
  logits + value so the actor can be exported to ONNX as-is.
* ``train``: rollouts of ``rollout_steps`` per row, GAE(lambda), clipped surrogate, entropy
  bonus, clipped value loss, advantage normalization, grad clipping, linear LR annealing,
  optional early stop on ``target_kl``. Rows of dead agents (``infos[row]["active"]`` false)
  are masked out of the losses (``mask_inactive``).
* Logging: ``progress.csv`` + ``log.jsonl`` per update (losses + episode metrics window),
  ``episodes.jsonl`` per finished episode, optional TensorBoard (guarded import).
* Checkpoints: ``torch.save`` of state dicts + every config needed to rebuild the model.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

EPISODE_METRIC_KEYS: tuple[str, ...] = (
    "survival_time",
    "hp_mean",
    "hp_end",
    "damage_dealt",
    "damage_taken",
    "kills",
    "shots",
    "hits_given",
    "downed_time",
    "partner_survival_time",
    "partner_hp_end",
    # race objective (bridge metrics) and the env's pickup stat; blank in progress.csv when absent
    "captures",
    "team_captures",
    "armed",
)
LOG_COLUMNS: tuple[str, ...] = (
    "update",
    "global_step",
    "time_s",
    "sps",
    "learning_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "n_episodes",
    "win_rate",
    "episode_return",
    "episode_length",
) + EPISODE_METRIC_KEYS


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000  # agent transitions (rows x vec steps)
    rollout_steps: int = 128  # per row
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float | None = None
    hidden_dim: int = 256
    seed: int = 0
    device: str = "cpu"
    checkpoint_every: int = 10  # updates
    mask_inactive: bool = True
    tensorboard: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PPOConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(name)


def _layer_init(layer: nn.Linear, std: float = float(np.sqrt(2)), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared-nothing actor / critic MLPs; one categorical head per action dimension."""

    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.nvec = tuple(int(n) for n in nvec)
        self.hidden_dim = int(hidden_dim)
        self.actor_body = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([_layer_init(nn.Linear(hidden_dim, n), std=0.01) for n in self.nvec])
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def logits(self, obs: torch.Tensor) -> list[torch.Tensor]:
        h = self.actor_body(obs)
        return [head(h) for head in self.heads]

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ONNX-friendly: (concatenated logits [B, sum(nvec)], value [B, 1])."""
        return torch.cat(self.logits(obs), dim=-1), self.critic(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dists = [Categorical(logits=l) for l in self.logits(obs)]
        if action is None:
            if deterministic:
                action = torch.stack([d.probs.argmax(dim=-1) for d in dists], dim=-1)
            else:
                action = torch.stack([d.sample() for d in dists], dim=-1)
        logprob = torch.stack([d.log_prob(action[..., i]) for i, d in enumerate(dists)], dim=-1).sum(-1)
        entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        return action, logprob, entropy, self.critic(obs).squeeze(-1)

    def split_logits(self, flat: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(flat, list(self.nvec), dim=-1))


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer | None,
    cfg: PPOConfig,
    meta: Mapping[str, Any] | None = None,
    global_step: int = 0,
    update: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": agent.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "ppo_config": cfg.to_dict(),
        "obs_dim": agent.obs_dim,
        "nvec": list(agent.nvec),
        "hidden_dim": agent.hidden_dim,
        "global_step": int(global_step),
        "update": int(update),
        "meta": dict(meta or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[ActorCritic, dict[str, Any]]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    agent = ActorCritic(ckpt["obs_dim"], ckpt["nvec"], ckpt.get("hidden_dim", 256)).to(device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    return agent, ckpt


# --------------------------------------------------------------------------------------
# logging helpers
# --------------------------------------------------------------------------------------
class EpisodeStats:
    """Window of finished-episode metrics gathered from ``infos`` (harness headline numbers)."""

    def __init__(self, maxlen: int = 200) -> None:
        self.window: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.total_episodes = 0

    def add_from_infos(self, infos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        new: list[dict[str, Any]] = []
        for info in infos:
            if "episode_metrics" not in info:
                continue
            rec = {
                "agent_id": info.get("agent_id"),
                "env_id": info.get("env_id"),
                "seed": info.get("seed"),
                "episode_return": float(info.get("episode_return", 0.0)),
                "episode_length": int(info.get("episode_length", 0)),
                "episode_time": float(info.get("episode_time", 0.0)),
                "team_win": bool(info.get("team_win", False)),
                "winner_team": info.get("winner_team"),
                "reason": info.get("reason"),
                **{k: float(v) for k, v in (info.get("episode_metrics") or {}).items()},
            }
            self.window.append(rec)
            new.append(rec)
            self.total_episodes += 1
        return new

    def summary(self) -> dict[str, float]:
        if not self.window:
            return {"n_episodes": 0}
        out: dict[str, float] = {"n_episodes": float(len(self.window))}
        out["win_rate"] = float(np.mean([r["team_win"] for r in self.window]))
        out["episode_return"] = float(np.mean([r["episode_return"] for r in self.window]))
        out["episode_length"] = float(np.mean([r["episode_length"] for r in self.window]))
        for k in EPISODE_METRIC_KEYS:
            vals = [r[k] for r in self.window if k in r]
            if vals:
                out[k] = float(np.mean(vals))
        return out


class RunLogger:
    def __init__(self, out_dir: str | Path, tensorboard: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "progress.csv"
        self.jsonl_path = self.out_dir / "log.jsonl"
        self.episodes_path = self.out_dir / "episodes.jsonl"
        self._csv_started = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self.writer = SummaryWriter(str(self.out_dir / "tb"))
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[ppo] tensorboard unavailable ({exc}); continuing without it")

    def log_update(self, row: Mapping[str, Any]) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(LOG_COLUMNS), extrasaction="ignore")
            if not self._csv_started:
                writer.writeheader()
                self._csv_started = True
            writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if self.writer is not None:
            step = int(row.get("global_step", 0))
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "global_step":
                    self.writer.add_scalar(k, float(v), step)

    def log_episodes(self, records: Sequence[Mapping[str, Any]], global_step: int) -> None:
        if not records:
            return
        with self.episodes_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps({"global_step": global_step, **rec}, default=float) + "\n")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def train(
    cfg: PPOConfig,
    env: Any,
    out_dir: str | Path,
    meta: Mapping[str, Any] | None = None,
    resume: str | Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run PPO on a ``SurvevVecEnv``-like object; returns the final log row."""
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"ppo": cfg.to_dict(), "meta": dict(meta or {})}, indent=2, default=str), encoding="utf-8"
    )

    n_rows, obs_dim = env.n_rows, env.obs_dim
    nvec = tuple(env.action_space.nvec)
    agent = ActorCritic(obs_dim, nvec, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    global_step, start_update = 0, 1
    if resume is not None:
        ckpt = torch.load(Path(resume), map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step, start_update = int(ckpt.get("global_step", 0)), int(ckpt.get("update", 0)) + 1

    batch_size = n_rows * cfg.rollout_steps
    minibatch_size = max(1, batch_size // cfg.num_minibatches)
    num_updates = max(1, cfg.total_steps // batch_size)
    logger = RunLogger(out_dir, tensorboard=cfg.tensorboard)
    stats = EpisodeStats()

    T, N, D = cfg.rollout_steps, n_rows, len(nvec)
    obs_buf = torch.zeros((T, N, obs_dim), dtype=torch.float32, device=device)
    act_buf = torch.zeros((T, N, D), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    done_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    val_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    mask_buf = torch.ones((T, N), dtype=torch.float32, device=device)

    next_obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    next_done = torch.zeros(N, dtype=torch.float32, device=device)
    next_active = torch.ones(N, dtype=torch.float32, device=device)
    start_time = time.time()
    last_row: dict[str, Any] = {}

    for update in range(start_update, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for group in optimizer.param_groups:
                group["lr"] = frac * cfg.learning_rate

        # -- rollout ------------------------------------------------------------------------
        for step in range(T):
            global_step += N
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            mask_buf[step] = next_active if cfg.mask_inactive else 1.0
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            act_buf[step] = action
            logp_buf[step] = logprob
            val_buf[step] = value
            obs_np, reward_np, done_np, infos = env.step(action.cpu().numpy())
            rew_buf[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done_np.astype(np.float32), device=device)
            # rows whose env just reset are active again; otherwise follow the alive flag
            next_active = torch.as_tensor(
                np.array([1.0 if d else (1.0 if inf.get("active", True) else 0.0) for d, inf in zip(done_np, infos)],
                         dtype=np.float32),
                device=device,
            )
            new_eps = stats.add_from_infos(infos)
            logger.log_episodes(new_eps, global_step)

        # -- GAE ----------------------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_nonterminal, next_values = 1.0 - next_done, next_value
                else:
                    next_nonterminal, next_values = 1.0 - done_buf[t + 1], val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_values * next_nonterminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, D)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)
        b_mask = mask_buf.reshape(-1)

        # -- optimisation ------------------------------------------------------------------
        inds = np.arange(batch_size)
        clip_fracs: list[float] = []
        approx_kl = torch.zeros(())
        pg_loss = v_loss = entropy_loss = torch.zeros(())
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start : start + minibatch_size]
                mask = b_mask[mb]
                denom = mask.sum().clamp(min=1.0)
                _, new_logp, entropy, new_val = agent.get_action_and_value(b_obs[mb], b_act[mb])
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / denom
                    clip_fracs.append(float((((ratio - 1.0).abs() > cfg.clip_coef).float() * mask).sum() / denom))
                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    m = (mb_adv * mask).sum() / denom
                    s = torch.sqrt((((mb_adv - m) ** 2) * mask).sum() / denom)
                    mb_adv = (mb_adv - m) / (s + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = (torch.max(pg1, pg2) * mask).sum() / denom
                if cfg.clip_vloss:
                    v_unclipped = (new_val - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(new_val - b_val[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * (torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2) * mask).sum() / denom
                else:
                    v_loss = 0.5 * (((new_val - b_ret[mb]) ** 2) * mask).sum() / denom
                entropy_loss = (entropy * mask).sum() / denom
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if cfg.target_kl is not None and float(approx_kl) > cfg.target_kl:
                break

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y
        elapsed = time.time() - start_time
        row: dict[str, Any] = {
            "update": update,
            "global_step": global_step,
            "time_s": round(elapsed, 2),
            "sps": round(global_step / max(elapsed, 1e-9), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "policy_loss": float(pg_loss.detach()),
            "value_loss": float(v_loss.detach()),
            "entropy": float(entropy_loss.detach()),
            "approx_kl": float(approx_kl.detach()),
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "explained_variance": explained_var,
            **stats.summary(),
        }
        logger.log_update(row)
        last_row = row
        if progress:
            ep = f"eps={int(row.get('n_episodes', 0))} win={row.get('win_rate', float('nan')):.2f} " \
                 f"surv={row.get('survival_time', float('nan')):.1f}s ret={row.get('episode_return', float('nan')):.2f}" \
                 if row.get("n_episodes") else "eps=0"
            if "team_captures" in row:
                ep += f" tcap={row['team_captures']:.2f}"
            if "armed" in row:
                ep += f" armed={row['armed']:.2f}"
            print(
                f"[ppo] upd {update}/{num_updates} step {global_step} sps {row['sps']:.0f} "
                f"pl {row['policy_loss']:.3f} vl {row['value_loss']:.3f} ent {row['entropy']:.2f} "
                f"kl {row['approx_kl']:.4f} {ep}",
                flush=True,
            )
        if update % cfg.checkpoint_every == 0 or update == num_updates:
            save_checkpoint(out_dir / "checkpoint_latest.pt", agent, optimizer, cfg, meta, global_step, update)

    save_checkpoint(out_dir / "checkpoint_final.pt", agent, optimizer, cfg, meta, global_step, num_updates)
    logger.close()
    return last_row
