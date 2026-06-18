from __future__ import annotations

import random
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

if __package__:
    from .cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS
    from .cpc_env import CPCEnv
    from .torchrl_specs import categorical_spec, composite_spec, unbounded_spec
else:
    from cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS
    from cpc_env import CPCEnv
    from torchrl_specs import categorical_spec, composite_spec, unbounded_spec


REWARD_COMPONENT_KEYS = (
    "survival",
    "ally_support",
    "damage",
    "pressure_response",
    "isolation",
    "self_preservation",
    "damage_taken",
)

METRIC_KEYS = (
    "avg_ally_distance",
    "isolation_rate",
    "teammate_under_pressure_response",
    "damage_dealt",
    "damage_taken",
)


class TorchRLCPCEnv(EnvBase):
    """Thin TorchRL adapter for the PR1 toy CPC environment.

    Action keys are flat (`move`, `aim`, `fire`) instead of nested under
    `action` to keep sampling and test setup simple across TorchRL versions.
    """

    batch_locked = True

    def __init__(
        self,
        *,
        seed: int = 0,
        max_steps: int = 50,
        device: torch.device | str = "cpu",
        env: CPCEnv | None = None,
    ):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.cpc_env = env or CPCEnv(seed=self.seed, max_steps=max_steps)
        device = torch.device(device)
        super().__init__(device=device, batch_size=torch.Size([]))

        self.observation_spec = self._make_observation_spec()
        self.action_spec = self._make_action_spec()
        self.reward_spec = unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device)
        self.done_spec = composite_spec(
            device=self.device,
            done=unbounded_spec(shape=(1,), dtype=torch.bool, device=self.device),
            terminated=unbounded_spec(shape=(1,), dtype=torch.bool, device=self.device),
            truncated=unbounded_spec(shape=(1,), dtype=torch.bool, device=self.device),
        )

    def _set_seed(self, seed: int | None) -> int:
        if seed is None:
            seed = self.seed
        self.seed = int(seed)
        self.rng.seed(self.seed)
        self.cpc_env.reset(seed=self.seed)
        return self.seed

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        seed = self.seed if tensordict is None else self._seed_from_tensordict(tensordict)
        obs = self.cpc_env.reset(seed=seed)
        return self._td_from_obs(
            obs,
            reward=None,
            done=False,
            terminated=False,
            truncated=False,
            info={},
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = self._action_from_tensordict(tensordict)
        obs, reward, done, info = self.cpc_env.step(action)
        truncated = bool(done and self.cpc_env.step_count >= self.cpc_env.max_steps)
        terminated = bool(done and not truncated)
        return self._td_from_obs(
            obs,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _make_observation_spec(self):
        return composite_spec(
            device=self.device,
            self_hp=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            ally_hp=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            enemy_hp=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            self_pos=unbounded_spec(shape=(2,), dtype=torch.float32, device=self.device),
            ally_pos=unbounded_spec(shape=(2,), dtype=torch.float32, device=self.device),
            enemy_pos=unbounded_spec(shape=(2,), dtype=torch.float32, device=self.device),
            distance_to_ally=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            ally_under_pressure=unbounded_spec(shape=(1,), dtype=torch.bool, device=self.device),
            self_low_hp=unbounded_spec(shape=(1,), dtype=torch.bool, device=self.device),
            step_count=unbounded_spec(shape=(1,), dtype=torch.int64, device=self.device),
            decoded_action=unbounded_spec(shape=(5,), dtype=torch.float32, device=self.device),
            reward_components=composite_spec(
                device=self.device,
                **{
                    key: unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device)
                    for key in REWARD_COMPONENT_KEYS
                },
            ),
            metrics=composite_spec(
                device=self.device,
                **{
                    key: unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device)
                    for key in METRIC_KEYS
                },
            ),
        )

    def _make_action_spec(self):
        return composite_spec(
            device=self.device,
            move=categorical_spec(MOVE_BINS, device=self.device),
            aim=categorical_spec(AIM_BINS, device=self.device),
            fire=categorical_spec(FIRE_BINS, device=self.device),
        )

    def _td_from_obs(
        self,
        obs: dict[str, Any],
        *,
        reward: float | None,
        done: bool,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> TensorDict:
        data: dict[str, Any] = {
            "self_hp": self._float_1(obs["self_hp"]),
            "ally_hp": self._float_1(obs["ally_hp"]),
            "enemy_hp": self._float_1(obs["enemy_hp"]),
            "self_pos": self._pos(obs["self_pos"]),
            "ally_pos": self._pos(obs["ally_pos"]),
            "enemy_pos": self._pos(obs["enemy_pos"]),
            "distance_to_ally": self._float_1(obs["distance_to_ally"]),
            "ally_under_pressure": self._bool_1(obs["ally_under_pressure"]),
            "self_low_hp": self._bool_1(obs["self_low_hp"]),
            "step_count": torch.tensor([int(obs["step_count"])], dtype=torch.int64, device=self.device),
            "done": self._bool_1(done),
            "terminated": self._bool_1(terminated),
            "truncated": self._bool_1(truncated),
            "decoded_action": self._float_action(info.get("decoded_action", self._zero_decoded_action())),
            "reward_components": self._reward_components(info.get("reward_components", {})),
            "metrics": self._metrics(info.get("metrics", {})),
        }
        if reward is not None:
            data["reward"] = self._float_1(reward)
        return TensorDict(data, batch_size=torch.Size([]), device=self.device)

    def _action_from_tensordict(self, tensordict: TensorDict) -> dict[str, int]:
        return {
            "move": self._int_from_td(tensordict, "move"),
            "aim": self._int_from_td(tensordict, "aim"),
            "fire": self._int_from_td(tensordict, "fire"),
        }

    def _int_from_td(self, tensordict: TensorDict, key: str) -> int:
        if key in tensordict.keys():
            value = tensordict[key]
        elif ("action", key) in tensordict.keys(True):
            value = tensordict["action", key]
        else:
            raise KeyError(f"Missing TorchRL CPC action key: {key}")
        return int(value.detach().to("cpu").reshape(-1)[0].item())

    def _seed_from_tensordict(self, tensordict: TensorDict) -> int:
        if "seed" not in tensordict.keys():
            return self.seed
        return int(tensordict["seed"].detach().to("cpu").reshape(-1)[0].item())

    def _float_1(self, value: float) -> torch.Tensor:
        return torch.tensor([float(value)], dtype=torch.float32, device=self.device)

    def _bool_1(self, value: bool) -> torch.Tensor:
        return torch.tensor([bool(value)], dtype=torch.bool, device=self.device)

    def _pos(self, value: dict[str, float]) -> torch.Tensor:
        return torch.tensor([float(value["x"]), float(value["y"])], dtype=torch.float32, device=self.device)

    def _float_action(self, action: dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [
                float(action["moveX"]),
                float(action["moveY"]),
                float(action["aimX"]),
                float(action["aimY"]),
                float(action["fire"]),
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _reward_components(self, components: dict[str, float]) -> TensorDict:
        return TensorDict(
            {key: self._float_1(float(components.get(key, 0.0))) for key in REWARD_COMPONENT_KEYS},
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _metrics(self, metrics: dict[str, float]) -> TensorDict:
        return TensorDict(
            {key: self._float_1(float(metrics.get(key, 0.0))) for key in METRIC_KEYS},
            batch_size=torch.Size([]),
            device=self.device,
        )

    def _zero_decoded_action(self) -> dict[str, float]:
        return {
            "moveX": 0.0,
            "moveY": 0.0,
            "aimX": 0.0,
            "aimY": 0.0,
            "fire": 0.0,
        }


__all__ = ["TorchRLCPCEnv"]
