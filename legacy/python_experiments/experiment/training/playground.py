from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .train_ppo import debug_print_reset_samples, load_config, resolve_config_path
except ImportError:
    EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
    if str(EXPERIMENT_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
    from training.train_ppo import debug_print_reset_samples, load_config, resolve_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Playground for local combat reset debugging.")
    parser.add_argument(
        "--config",
        default="experiment/configs/local_combat_in_range.yaml",
        help="Path to the PPO YAML config to inspect.",
    )
    parser.add_argument("--samples", type=int, default=10, help="Number of reset samples to print.")
    parser.add_argument("--smoke", action="store_true", help="Apply PPO smoke limits while loading the config.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path, smoke=args.smoke)
    debug_print_reset_samples(cfg, samples=args.samples, config_path=str(config_path))


if __name__ == "__main__":
    main()
