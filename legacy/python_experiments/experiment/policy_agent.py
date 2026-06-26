from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from experiment.checkpointing import load_checkpoint
except ModuleNotFoundError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from experiment.checkpointing import load_checkpoint

try:
    from experiment.training.cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS, decode_action, random_action
    from experiment.training.ppo_policy import OBS_DIM, MultiDiscreteActorCritic
except ModuleNotFoundError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    if str(EXPERIMENT_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
    from training.cpc_actions import AIM_BINS, FIRE_BINS, MOVE_BINS, decode_action, random_action
    from training.ppo_policy import OBS_DIM, MultiDiscreteActorCritic


class BaseAgent:
    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> dict[str, int]:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> dict[str, int]:
        del observation, deterministic
        return random_action(self.rng)


class PPOPolicyAgent(BaseAgent):
    def __init__(
        self,
        policy: MultiDiscreteActorCritic,
        *,
        device: str | torch.device = "cpu",
        checkpoint: dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ):
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.policy.eval()
        self.checkpoint = checkpoint or {}
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str | torch.device = "cpu"):
        device = torch.device(device)
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        cfg = checkpoint.get("config", {})
        hidden_dim = int(checkpoint.get("hidden_dim", cfg.get("hidden_dim", 64)))
        policy = MultiDiscreteActorCritic(hidden_dim=hidden_dim)
        try:
            policy.load_state_dict(checkpoint["policy_state_dict"])
        except RuntimeError as exc:
            metadata = checkpoint.get("observation_metadata", {})
            checkpoint_obs_dim = metadata.get("obs_dim", checkpoint.get("obs_dim", "unknown"))
            raise RuntimeError(
                f"Checkpoint observation dimension mismatch: checkpoint obs_dim={checkpoint_obs_dim}, "
                f"current obs_dim={OBS_DIM}. Retrain the PPO checkpoint with the current observation schema."
            ) from exc
        return cls(policy, device=device, checkpoint=checkpoint, checkpoint_path=checkpoint_path)

    @torch.no_grad()
    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> dict[str, int]:
        model_input = observation_to_model_input(observation, self.device)
        if deterministic:
            move_logits, aim_logits, fire_logits, _ = self.policy(model_input)
            action = {
                "move": int(move_logits.argmax(dim=-1).reshape(-1)[0].item()),
                "aim": int(aim_logits.argmax(dim=-1).reshape(-1)[0].item()),
                "fire": int(fire_logits.argmax(dim=-1).reshape(-1)[0].item()),
            }
        else:
            output = self.policy.sample_action(model_input)
            action = {
                "move": int(output.action["move"].reshape(-1)[0].item()),
                "aim": int(output.action["aim"].reshape(-1)[0].item()),
                "fire": int(output.action["fire"].reshape(-1)[0].item()),
            }
        _validate_action(action)
        return action

    @torch.no_grad()
    def act_with_debug(self, observation: Mapping[str, Any], deterministic: bool = True) -> dict[str, Any]:
        model_input = observation_to_model_input(observation, self.device)
        move_logits, aim_logits, fire_logits, value = self.policy(model_input)
        log_prob = None
        if deterministic:
            action = {
                "move": int(move_logits.argmax(dim=-1).reshape(-1)[0].item()),
                "aim": int(aim_logits.argmax(dim=-1).reshape(-1)[0].item()),
                "fire": int(fire_logits.argmax(dim=-1).reshape(-1)[0].item()),
            }
        else:
            output = self.policy.sample_action(model_input)
            action = {
                "move": int(output.action["move"].reshape(-1)[0].item()),
                "aim": int(output.action["aim"].reshape(-1)[0].item()),
                "fire": int(output.action["fire"].reshape(-1)[0].item()),
            }
            log_prob = float(output.log_prob.detach().to("cpu").reshape(-1)[0].item())
        _validate_action(action)
        return {
            "raw_action": action,
            "decoded_action": _snake_decoded_action(decode_action(action)),
            "policy_debug": {
                "log_prob": log_prob,
                "value": float(value.detach().to("cpu").reshape(-1)[0].item()),
                "move_logits": move_logits.detach().to("cpu").reshape(-1).tolist(),
                "aim_logits": aim_logits.detach().to("cpu").reshape(-1).tolist(),
                "fire_logits": fire_logits.detach().to("cpu").reshape(-1).tolist(),
            },
        }


def observation_to_model_input(observation: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "self_hp": _float_1(observation["self_hp"], device),
        "ally_hp": _float_1(observation["ally_hp"], device),
        "enemy_hp": _float_1(observation["enemy_hp"], device),
        "self_pos": _pos(observation["self_pos"], device),
        "ally_pos": _pos(observation["ally_pos"], device),
        "enemy_pos": _pos(observation["enemy_pos"], device),
        "distance_to_ally": _float_1(observation["distance_to_ally"], device),
        "ally_under_pressure": _bool_1(observation["ally_under_pressure"], device),
        "self_low_hp": _bool_1(observation["self_low_hp"], device),
        "step_count": torch.tensor([int(observation["step_count"])], dtype=torch.int64, device=device),
        "target_dir": torch.tensor(
            [float(observation.get("target_dir_x", 0.0)), float(observation.get("target_dir_y", 0.0))],
            dtype=torch.float32,
            device=device,
        ),
        "aim_alignment": _float_1(observation.get("aim_alignment", 0.0), device),
        "can_fire": _bool_1(observation.get("can_fire", False), device),
        "weapon_cooldown_fraction": _float_1(observation.get("weapon_cooldown_fraction", 0.0), device),
        "distance_to_center": _float_1(observation.get("distance_to_center", 0.0), device),
        "safe_radius": _float_1(observation.get("safe_radius", 1.0), device),
        "safe_margin_fraction": _float_1(observation.get("safe_margin_fraction", 0.0), device),
        "outside_safe_zone": _bool_1(observation.get("outside_safe_zone", False), device),
    }


def _float_1(value: Any, device: torch.device) -> torch.Tensor:
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


def _bool_1(value: Any, device: torch.device) -> torch.Tensor:
    return torch.tensor([bool(value)], dtype=torch.bool, device=device)


def _pos(value: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(value["x"]), float(value["y"])], dtype=torch.float32, device=device)


def _validate_action(action: Mapping[str, int]) -> None:
    if not 0 <= int(action["move"]) < MOVE_BINS:
        raise ValueError(f"move out of bounds: {action['move']}")
    if not 0 <= int(action["aim"]) < AIM_BINS:
        raise ValueError(f"aim out of bounds: {action['aim']}")
    if not 0 <= int(action["fire"]) < FIRE_BINS:
        raise ValueError(f"fire out of bounds: {action['fire']}")


def _snake_decoded_action(decoded: Mapping[str, Any]) -> dict[str, float]:
    return {
        "move_x": float(decoded["moveX"]),
        "move_y": float(decoded["moveY"]),
        "aim_x": float(decoded["aimX"]),
        "aim_y": float(decoded["aimY"]),
        "fire": float(decoded["fire"]),
    }
