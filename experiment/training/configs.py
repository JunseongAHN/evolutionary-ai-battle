from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    learning_agent_id: str = "team-a-0"

    obs_dim: int = 20
    action_dim: int = 5

    frames_per_batch: int = 1024
    total_frames: int = 50_000
    sub_batch_size: int = 256
    num_epochs: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 1e-3
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    seed: int = 0
    hidden_dim: int = 256

    # Action bounds: move_x, move_y, aim_x, aim_y, fire
    action_low: tuple[float, float, float, float, float] = (-1.0, -1.0, -1.0, -1.0, 0.0)
    action_high: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)