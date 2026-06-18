from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.distributions import Categorical

try:
    from cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS
except ImportError:
    from .cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS


OBS_KEYS = (
    "self_hp",
    "ally_hp",
    "enemy_hp",
    "self_pos",
    "ally_pos",
    "enemy_pos",
    "distance_to_ally",
    "ally_under_pressure",
    "self_low_hp",
    "step_count",
)

OBS_DIM = 16


@dataclass(frozen=True)
class PolicyOutput:
    action: dict[str, torch.Tensor]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


def flatten_observation(obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    parts = [
        _as_float_feature(obs["self_hp"]) / 100.0,
        _as_float_feature(obs["ally_hp"]) / 100.0,
        _as_float_feature(obs["enemy_hp"]) / 100.0,
        _as_float_feature(obs["self_pos"]) / 1000.0,
        _as_float_feature(obs["ally_pos"]) / 1000.0,
        _as_float_feature(obs["enemy_pos"]) / 1000.0,
        _as_float_feature(obs["distance_to_ally"]) / 1000.0,
        _as_float_feature(obs["ally_under_pressure"]),
        _as_float_feature(obs["self_low_hp"]),
        _as_float_feature(obs["step_count"]) / 100.0,
    ]
    return torch.cat(parts, dim=-1)


def _as_float_feature(value: torch.Tensor) -> torch.Tensor:
    tensor = value.float()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.move_head = nn.Linear(hidden_dim, MOVE_BINS)
        self.aim_head = nn.Linear(hidden_dim, AIM_BINS)
        self.fire_head = nn.Linear(hidden_dim, FIRE_BINS)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Mapping[str, torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = obs if isinstance(obs, torch.Tensor) else flatten_observation(obs)
        hidden = self.encoder(features)
        return (
            self.move_head(hidden),
            self.aim_head(hidden),
            self.fire_head(hidden),
            self.value_head(hidden),
        )

    def sample_action(self, obs: Mapping[str, torch.Tensor] | torch.Tensor) -> PolicyOutput:
        move_logits, aim_logits, fire_logits, value = self.forward(obs)
        move_dist = Categorical(logits=move_logits)
        aim_dist = Categorical(logits=aim_logits)
        fire_dist = Categorical(logits=fire_logits)
        move = move_dist.sample()
        aim = aim_dist.sample()
        fire = fire_dist.sample()
        log_prob = move_dist.log_prob(move) + aim_dist.log_prob(aim) + fire_dist.log_prob(fire)
        entropy = move_dist.entropy() + aim_dist.entropy() + fire_dist.entropy()
        return PolicyOutput(
            action={"move": move.squeeze(), "aim": aim.squeeze(), "fire": fire.squeeze()},
            log_prob=log_prob.squeeze(-1) if log_prob.ndim > 0 else log_prob,
            entropy=entropy.squeeze(-1) if entropy.ndim > 0 else entropy,
            value=value.squeeze(-1),
        )

    def evaluate_actions(
        self,
        obs: Mapping[str, torch.Tensor] | torch.Tensor,
        actions: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        move_logits, aim_logits, fire_logits, value = self.forward(obs)
        move_dist = Categorical(logits=move_logits)
        aim_dist = Categorical(logits=aim_logits)
        fire_dist = Categorical(logits=fire_logits)
        move = actions["move"].long()
        aim = actions["aim"].long()
        fire = actions["fire"].long()
        log_prob = move_dist.log_prob(move) + aim_dist.log_prob(aim) + fire_dist.log_prob(fire)
        entropy = move_dist.entropy() + aim_dist.entropy() + fire_dist.entropy()
        return log_prob, entropy, value.squeeze(-1)

    def value(self, obs: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return self.forward(obs)[3].squeeze(-1)
