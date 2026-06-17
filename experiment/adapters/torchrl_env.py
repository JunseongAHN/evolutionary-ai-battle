from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

try:
    from torchrl.data import Bounded, Composite, Unbounded
except ImportError:
    from torchrl.data import (
        BoundedTensorSpec as Bounded,
        CompositeSpec as Composite,
        UnboundedContinuousTensorSpec as Unbounded,
    )

from core.env_core import PythonBattleCoreEnv
from core.reward import compute_agent_reward
from core.schema import AgentId, BattleAction, MultiAgentAction, SCHEMA_VERSION, TacticalObservation
from core.vectorizer import vectorize_observation
from training.configs import TrainingConfig
from training.scripted_policies import build_scripted_action


CoreEnvFactory = Callable[[], PythonBattleCoreEnv]


@dataclass(frozen=True)
class AdapterConfig:
    learning_agent_id: str = "team-a-0"
    obs_dim: int = 20
    action_dim: int = 5
    seed: int = 0
    action_low: tuple[float, float, float, float, float] = (-1.0, -1.0, -1.0, -1.0, 0.0)
    action_high: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)


class CpcTorchRLEnv(EnvBase):
    batch_locked = True

    def __init__(
        self,
        core_env_factory: CoreEnvFactory | None = None,
        train_cfg: TrainingConfig | AdapterConfig | None = None,
        device: torch.device | str = "cpu",
    ):
        self.train_cfg = train_cfg or AdapterConfig()
        self.learning_agent_id = self.train_cfg.learning_agent_id
        self.core_env_factory = core_env_factory or PythonBattleCoreEnv
        self.core_env: PythonBattleCoreEnv = self.core_env_factory()
        self.latest_observations: dict[AgentId, TacticalObservation] | None = None
        self.rng = random.Random(self.train_cfg.seed)

        device = torch.device(device)
        super().__init__(device=device, batch_size=torch.Size([]))

        obs_low = torch.full((self.train_cfg.obs_dim,), -float("inf"), dtype=torch.float32, device=device)
        obs_high = torch.full((self.train_cfg.obs_dim,), float("inf"), dtype=torch.float32, device=device)
        action_low = torch.tensor(self.train_cfg.action_low, dtype=torch.float32, device=device)
        action_high = torch.tensor(self.train_cfg.action_high, dtype=torch.float32, device=device)

        self.observation_spec = Composite(
            observation=Bounded(
                low=obs_low,
                high=obs_high,
                shape=torch.Size([self.train_cfg.obs_dim]),
                dtype=torch.float32,
                device=device,
            ),
            shape=torch.Size([]),
            device=device,
        )
        self.action_spec = Bounded(
            low=action_low,
            high=action_high,
            shape=torch.Size([self.train_cfg.action_dim]),
            dtype=torch.float32,
            device=device,
        )
        self.reward_spec = Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=device)

    def _set_seed(self, seed: int | None) -> int | None:
        if seed is None:
            seed = self.train_cfg.seed
        self.rng.seed(seed)
        return seed

    def _make_obs_tensor(self, obs: TacticalObservation) -> torch.Tensor:
        vector = vectorize_observation(obs, obs_dim=self.train_cfg.obs_dim)
        return torch.tensor(vector, dtype=torch.float32, device=self.device)

    def _reset(self, tensordict=None) -> TensorDict:
        seed = self.rng.randint(0, 2**31 - 1)
        self.core_env = self.core_env_factory()
        observations = self.core_env.reset(seed=seed)
        self.latest_observations = observations
        obs = observations[self.learning_agent_id]
        return TensorDict(
            {
                "observation": self._make_obs_tensor(obs),
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _tensor_to_learning_action(self, action_tensor: torch.Tensor) -> BattleAction:
        move_x, move_y, aim_x, aim_y, fire = [
            float(x) for x in action_tensor.detach().to("cpu").float().tolist()
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.core_env.episode_id,
            "step": self.core_env.step_index,
            "agent_id": self.learning_agent_id,
            "action": {
                "move_x": max(-1.0, min(1.0, move_x)),
                "move_y": max(-1.0, min(1.0, move_y)),
                "aim_x": max(-1.0, min(1.0, aim_x)),
                "aim_y": max(-1.0, min(1.0, aim_y)),
                "fire": max(0.0, min(1.0, fire)),
            },
            "source": {"policy_type": "future_policy", "policy_id": "torchrl-ppo-v0"},
        }

    def _build_multi_agent_action(self, action_tensor: torch.Tensor) -> MultiAgentAction:
        if self.latest_observations is None:
            raise RuntimeError("latest_observations is None. Call reset() before step().")

        actions: dict[AgentId, BattleAction] = {}
        for agent_id in self.core_env.agent_ids:
            if agent_id == self.learning_agent_id:
                actions[agent_id] = self._tensor_to_learning_action(action_tensor)
            else:
                actions[agent_id] = build_scripted_action(self.latest_observations[agent_id])
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.core_env.episode_id,
            "step": self.core_env.step_index,
            "actions": actions,
        }

    def _step(self, tensordict: TensorDict) -> TensorDict:
        step = self.core_env.step(self._build_multi_agent_action(tensordict["action"]))
        self.latest_observations = step["observations"]
        obs = step["observations"][self.learning_agent_id]
        reward = compute_agent_reward(step, self.learning_agent_id)
        terminated = bool(step["terminated"])
        truncated = bool(step["truncated"])
        done = terminated or truncated
        return TensorDict(
            {
                "observation": self._make_obs_tensor(obs),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )

