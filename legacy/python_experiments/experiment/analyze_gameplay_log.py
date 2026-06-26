from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze_gameplay_log(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    if result_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = sum(1 for record in records if record.get("type") == "step")
        end_records = [record for record in records if record.get("type") == "episode_end"]
        final_metrics = end_records[-1].get("final_metrics", {}) if end_records else {}
        mode = "jsonl"
    else:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        steps = sum(len(episode.get("steps", [])) for episode in episodes)
        final_metrics = episodes[-1].get("final_metrics", {}) if episodes else {}
        mode = _mode_from_data(data)
    return {
        "path": str(result_path),
        "mode": mode,
        "size_kb": round(result_path.stat().st_size / 1024.0, 2),
        "steps": steps,
        "final_metrics": final_metrics,
    }


def _mode_from_data(data: dict[str, Any]) -> str:
    version = str(data.get("format_version", ""))
    if "summary" in version:
        return "summary"
    if "compact" in version:
        return "compact"
    return "full"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact summary of a saved gameplay log.")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_gameplay_log(args.result), indent=2))


if __name__ == "__main__":
    main()
