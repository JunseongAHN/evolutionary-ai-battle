from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiment"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from debug.manual_control import run_manual_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPC Python core env manual debug harness.")
    parser.add_argument("--agent-id", default="team-a-0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=700)
    args = parser.parse_args()

    run_manual_debug(
        seed=args.seed,
        controlled_agent_id=args.agent_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
