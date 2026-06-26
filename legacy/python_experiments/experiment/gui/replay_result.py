from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
    REPO_ROOT = EXPERIMENT_ROOT.parent
    for path in (EXPERIMENT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

from experiment.gui.pygame_viewer import PygameCPCViewer


def replay_result(result_path: str | Path, *, fps: int = 10) -> None:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    viewer = PygameCPCViewer(fps=fps, title="CPC Gameplay Replay")
    try:
        for episode in result.get("episodes", []):
            for step_record in episode.get("steps", []):
                env_state = step_record.get("env", {}).get("state", {})
                if not viewer.render_step(env_state, step_record):
                    return
    finally:
        viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved model gameplay JSON result in pygame.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    replay_result(args.result, fps=args.fps)


if __name__ == "__main__":
    main()
