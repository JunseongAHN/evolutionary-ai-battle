"""Does the controller actually obey the intent it was given?

One evaluation per intent, same checkpoint, same seeds, same opponents — only the order changes. If
the four runs come back with the same numbers, the intent is decoration: the policy learned one way
to fight and ignores what it was told. If they separate (a pushing run trades more damage at closer
range than a retreating one), the planner's choice reaches the game, which is the whole point of
conditioning the controller on it.

    python -m experiment.survev_rl.eval_intents --checkpoint runs/cover_intent_v1/checkpoint_final.pt \
        --bridge ws://127.0.0.1:8799 --episodes 20

Numbers come from ``eval.main`` itself (it returns its summary), so this adds no measurement of its
own — it only runs the same evaluation once per intent and puts the rows side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .eval import main as eval_main

#: what a fight looks like, in the order a reader wants it. `eval` puts the episode-level numbers at
#: the top of its summary and the per-agent means under "mean", so a lookup checks both.
COLUMNS: tuple[str, ...] = (
    "win_rate",
    "mean_duration_s",
    "survival_time",
    "damage_dealt",
    "damage_taken",
    "kills",
    "shots",
    "hp_end",
)


def cell(row: dict[str, Any], column: str) -> float | None:
    value = row.get(column, (row.get("mean") or {}).get(column))
    return float(value) if isinstance(value, (int, float)) else None


def endings(path: Path) -> dict[str, int]:
    """How the episodes ended. The question `retreat` has to answer is whether contact was broken:
    a controller told to withdraw and still ending every episode in elimination did not obey, however
    different its damage numbers look."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        reason = record.get("reason") or (record.get("info") or {}).get("reason") or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def run_one(intent: str, args: argparse.Namespace) -> dict[str, Any]:
    argv = [
        "--checkpoint", args.checkpoint,
        "--episodes", str(args.episodes),
        "--n-envs", str(args.n_envs),
        "--seed", str(args.seed),
        "--intent", intent,
        "--device", args.device,
    ]
    if args.mock:
        argv.append("--mock")
    else:
        argv += ["--bridge", args.bridge]
    if args.deterministic:
        argv.append("--deterministic")
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        argv += ["--out", str(Path(args.out_dir) / f"intent_{intent}.jsonl")]
    return eval_main(argv)


def spread(rows: dict[str, dict[str, Any]], column: str) -> float:
    """max - min across intents; 0 means every intent played the same."""
    values = [v for v in (cell(r, column) for r in rows.values()) if v is not None]
    return max(values) - min(values) if values else 0.0


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--intents", default="push,hold_angle,trade,retreat")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=1000, help="same seeds for every intent, so only the order differs")
    p.add_argument("--bridge", default="ws://127.0.0.1:8765")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=None, help="write each intent's episodes and summary here")
    args = p.parse_args(argv)

    intents = [i for i in args.intents.split(",") if i]
    rows: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, int]] = {}
    for intent in intents:
        print(f"\n=== intent: {intent} ===", flush=True)
        rows[intent] = run_one(intent, args)
        if args.out_dir:
            ends[intent] = endings(Path(args.out_dir) / f"intent_{intent}.jsonl")

    width = max(len(i) for i in intents) + 2
    print("\n" + "intent".ljust(width) + "".join(c.rjust(17) for c in COLUMNS))
    for intent, row in rows.items():
        cells = "".join(
            (f"{cell(row, c):.2f}" if cell(row, c) is not None else "-").rjust(17) for c in COLUMNS
        )
        print(intent.ljust(width) + cells)
    if ends:
        print("\nhow the episodes ended:")
        for intent, counts in ends.items():
            summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
            print(f"  {intent:<12} {summary}")
    print("\nspread (max - min across intents):")
    for column in COLUMNS:
        print(f"  {column:<16} {spread(rows, column):.2f}")
    print(
        "\nA spread of ~0 on every column means the intent changed nothing: the controller fights one\n"
        "way whatever it is told, and conditioning it was wasted. Differences in damage taken and\n"
        "duration are what obeying looks like."
    )
    if args.out_dir:
        path = Path(args.out_dir) / "intents_summary.json"
        path.write_text(json.dumps({"summaries": rows, "endings": ends}, indent=2, default=float), encoding="utf-8")
        print(f"wrote {path}")
    return rows


if __name__ == "__main__":
    main()
