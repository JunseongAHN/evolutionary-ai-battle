from __future__ import annotations

from ..types import BaselineConfig, Control


def build_action(control: Control, config: BaselineConfig) -> tuple[dict[str, int | float], dict]:
    del config
    move_bin = int(control.move_bin) if 0 <= int(control.move_bin) <= 8 else 0
    fire = 1 if int(control.fire) == 1 else 0
    action: dict[str, int | float] = {
        "move": move_bin,
        "move_bin": move_bin,
        "aim_dx": float(control.aim_dx),
        "aim_dy": float(control.aim_dy),
        "fire": fire,
    }
    return action, {
        "reason": "continuous_env_action_built",
        "move_bin": move_bin,
        "aim_dir": [float(control.aim_dx), float(control.aim_dy)],
        "fire": fire,
    }


__all__ = ["build_action"]
