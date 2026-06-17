from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import torch
from torch import nn
from tqdm import tqdm

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from adapters.torchrl_env import CpcTorchRLEnv
from core.env_core import PythonBattleCoreEnv
from training.configs import TrainingConfig


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_actor_critic(
    env: CpcTorchRLEnv,
    cfg: TrainingConfig,
    device: torch.device,
):
    actor_net = nn.Sequential(
        nn.Linear(cfg.obs_dim, cfg.hidden_dim, device=device),
        nn.Tanh(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim, device=device),
        nn.Tanh(),
        nn.Linear(cfg.hidden_dim, 2 * cfg.action_dim, device=device),
        NormalParamExtractor(),
    )

    policy_base = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    policy = ProbabilisticActor(
        module=policy_base,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.Linear(cfg.obs_dim, cfg.hidden_dim, device=device),
        nn.Tanh(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim, device=device),
        nn.Tanh(),
        nn.Linear(cfg.hidden_dim, 1, device=device),
    )

    value = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return policy, value


def train_ppo(
    core_env_factory: Callable[[], PythonBattleCoreEnv],
    cfg: TrainingConfig = TrainingConfig(),
):
    torch.manual_seed(cfg.seed)
    device = make_device()

    print(f"Using device: {device}")

    env = CpcTorchRLEnv(
        core_env_factory=core_env_factory,
        train_cfg=cfg,
        device=device,
    )

    check_env_specs(env)

    policy, value = build_actor_critic(env, cfg, device)

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        split_trajs=False,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.gae_lambda,
        value_network=value,
        average_gae=True,
        device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=bool(cfg.entropy_coef),
        entropy_coeff=cfg.entropy_coef,
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    logs = defaultdict(list)
    pbar = tqdm(total=cfg.total_frames)

    for batch_idx, batch in enumerate(collector):
        # batch is a TensorDict collected from env-policy interaction.
        batch = batch.to(device)

        for _ in range(cfg.num_epochs):
            advantage(batch)

            flat = batch.reshape(-1)
            replay_buffer.extend(flat.detach().cpu())

            num_updates = max(1, cfg.frames_per_batch // cfg.sub_batch_size)

            for _ in range(num_updates):
                subdata = replay_buffer.sample(cfg.sub_batch_size).to(device)

                loss_vals = loss_module(subdata)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optimizer.step()

        reward_mean = batch["next", "reward"].mean().item()
        done_count = batch["next", "done"].sum().item()

        logs["reward_mean"].append(reward_mean)
        logs["done_count"].append(done_count)

        pbar.update(batch.numel())
        pbar.set_description(
            f"batch={batch_idx} reward_mean={reward_mean:.4f} done={int(done_count)}"
        )

        collector.update_policy_weights_()

    pbar.close()

    return {
        "policy": policy,
        "value": value,
        "logs": dict(logs),
        "config": cfg,
    }


@torch.no_grad()
def evaluate_policy(
    core_env_factory: Callable[[], PythonBattleCoreEnv],
    policy,
    cfg: TrainingConfig = TrainingConfig(),
    episodes: int = 5,
):
    device = make_device()
    env = CpcTorchRLEnv(core_env_factory, cfg, device=device)

    returns: list[float] = []

    with set_exploration_type(ExplorationType.MODE):
        for _ in range(episodes):
            td = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                td = policy(td)
                td_next = env.step(td)
                reward = float(td_next["next", "reward"].item())
                done = bool(td_next["next", "done"].item())
                ep_return += reward
                td = td_next["next"]

            returns.append(ep_return)

    return {
        "returns": returns,
        "mean_return": sum(returns) / max(1, len(returns)),
    }
