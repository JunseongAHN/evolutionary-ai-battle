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
    "damage_dealt_ratio",
    "damage_taken_ratio",
    "bullet_hit",
    "bullet_hit_reward",
    "missed_shot",
    "missed_shot_penalty",
    "aim_bin_exact",
    "aim_bin_wrong",
    "bad_aim_shot_penalty",
    "aim_alignment",
    "shot_fired_reward",
    "no_fire_ready_penalty",
    "good_range",
    "too_close",
    "too_far",
    "kill",
    "death",
    "timeout_hp_lead",
    "accuracy_bonus",
    "no_shot_episode",
    "death_without_shooting",
    "death_without_damage",
    "zone_pressure",
    "return_to_zone",
    "move_deeper_outside_zone",
    "near_edge_outward",
)

METRIC_KEYS = (
    "avg_ally_distance",
    "isolation_rate",
    "teammate_under_pressure_response",
    "damage_dealt",
    "damage_taken",
    "enemy_max_hp",
    "self_max_hp",
    "damage_dealt_ratio",
    "damage_taken_ratio",
    "damage_trade_ratio",
    "enemy_hp_remaining_ratio",
    "self_hp_remaining_ratio",
    "kill_rate",
    "enemy_dead",
    "self_dead",
    "survival_steps",
    "mean_aim_alignment",
    "aim_bin_0_rate",
    "aim_bin_entropy",
    "exact_aim_match_rate",
    "within_1_bin_aim_rate",
    "bad_aim_rate",
    "shot_exact_aim_rate",
    "shot_near_aim_rate",
    "shot_off_target_rate",
    "shot_bad_aim_rate",
    "bullet_hit_per_shot",
    "fire_requested_count",
    "shot_fired_count",
    "off_target_shot_count",
    "bullet_hit_count",
    "missed_shot_count",
    "hit_ratio",
    "missed_shot_rate",
    "avg_distance_to_enemy",
    "good_range_rate",
    "too_close_rate",
    "too_far_rate",
    "total_return",
    "mean_step_reward",
    "reward_hacking_warning_count",
    "outside_safe_zone_rate",
    "near_edge_outward_count",
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
        randomize_enemy_spawn_direction: bool = False,
        enemy_spawn_directions: list[str] | tuple[str, ...] | None = None,
        enemy_spawn_direction: str | None = None,
        stage: str = "local_combat",
        shrink_safe_zone: bool = False,
        use_zone_reward: bool = False,
        enemy_move: bool = True,
        enemy_fire: bool = True,
        stationary_target_mode: bool = False,
        enemy_spawn_distance_min: float | None = None,
        enemy_spawn_distance_max: float | None = None,
        fire_interval_steps: int | None = None,
        bullet_speed: float | None = None,
        bullet_range: float | None = None,
        bullet_damage: float | None = None,
        bullet_hit_radius: float | None = None,
    ):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.cpc_env = env or CPCEnv(
            seed=self.seed,
            max_steps=max_steps,
            randomize_enemy_spawn_direction=randomize_enemy_spawn_direction,
            enemy_spawn_directions=enemy_spawn_directions,
            enemy_spawn_direction=enemy_spawn_direction,
            stage=stage,
            shrink_safe_zone=shrink_safe_zone,
            use_zone_reward=use_zone_reward,
            enemy_move=enemy_move,
            enemy_fire=enemy_fire,
            stationary_target_mode=stationary_target_mode,
            enemy_spawn_distance_min=enemy_spawn_distance_min,
            enemy_spawn_distance_max=enemy_spawn_distance_max,
            fire_interval_steps=fire_interval_steps,
            bullet_speed=bullet_speed,
            bullet_range=bullet_range,
            bullet_damage=bullet_damage,
            bullet_hit_radius=bullet_hit_radius,
        )
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
        seed = None if tensordict is None else self._seed_from_tensordict(tensordict)
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
            safe_radius=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            distance_to_enemy=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            can_fire=unbounded_spec(shape=(1,), dtype=torch.bool, device=self.device),
            weapon_cooldown_fraction=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            target_dir=unbounded_spec(shape=(2,), dtype=torch.float32, device=self.device),
            aim_alignment=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            distance_to_center=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            safe_margin_fraction=unbounded_spec(shape=(1,), dtype=torch.float32, device=self.device),
            outside_safe_zone=unbounded_spec(shape=(1,), dtype=torch.bool, device=self.device),
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
            "safe_radius": self._float_1(obs["safe_radius"]),
            "distance_to_enemy": self._float_1(obs["distance_to_enemy"]),
            "can_fire": self._bool_1(obs["can_fire"]),
            "weapon_cooldown_fraction": self._float_1(obs["weapon_cooldown_fraction"]),
            "target_dir": torch.tensor(
                [float(obs["target_dir_x"]), float(obs["target_dir_y"])],
                dtype=torch.float32,
                device=self.device,
            ),
            "aim_alignment": self._float_1(obs["aim_alignment"]),
            "distance_to_center": self._float_1(obs["distance_to_center"]),
            "safe_margin_fraction": self._float_1(obs["safe_margin_fraction"]),
            "outside_safe_zone": self._bool_1(obs["outside_safe_zone"]),
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
