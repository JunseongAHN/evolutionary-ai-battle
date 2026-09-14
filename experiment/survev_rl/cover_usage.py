"""How often a policy actually puts something between itself and a gun.

Win rate cannot tell you this: a policy that wins by out-aiming everyone in the open scores the
same as one that wins from behind a wall. So this counts, over the steps where an armed enemy is
in view, how often

* every one of those enemies' lines of sight is broken (``los_blocked``), and
* usable cover sits within reach (``cover_score > 0`` within ``--cover-range``).

Both flags come from the server, which computes them with the engine's own bullet rule, so nothing
here credits hiding behind something a shot passes through.

Steps with no armed enemy in view are excluded: standing behind a wall alone in a field is not
cover use, and including it would let a policy that simply runs away score well.

That exclusion is not enough on its own. Random actions score 28% line-broken on the axis layout,
because a policy that never closes in leaves walls sitting between itself and everyone else. So
``--max-dist`` restricts the count to steps where the nearest armed enemy is actually within
fighting range: wandering at 40 m earns nothing there, and only breaking a line you are genuinely
exposed to does.

Usage::

    python -m experiment.survev_rl.cover_usage \
        --label "controller v6" runs/cover_v6.jsonl \
        --label "scripted"      runs/scripted.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

#: a player who can shoot back; fists do not make someone worth taking cover from
_UNARMED = ("", "fists")


def is_armed_threat(p: Mapping[str, Any]) -> bool:
    return (
        not p.get("dead")
        and not p.get("downed")
        and str(p.get("weapon") or "fists") not in _UNARMED
    )


def iter_observations(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield ``{agent: observation}`` rows from either log format.

    ``eval.py --out`` writes one record per episode with a ``steps`` list; the bridge row logs
    write one record per step. Both carry the same observations, so both are read here rather
    than forcing the caller to know which one they have.
    """
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        steps = rec.get("steps")
        if isinstance(steps, list):  # eval.py episode record
            for step in steps:
                yield step.get("obs") or {}
        else:  # bridge row
            yield rec.get("obs") or {}


def centre_of(obs: Mapping[str, Any]) -> tuple[float, float] | None:
    """The play area's centre, as the server reports it in ``gas.pos``.

    The circle has not started closing in these scenarios, so its radius says nothing — but its
    centre is the play area's, which is what "how far did it wander" has to be measured against.
    """
    pos = (obs.get("gas") or {}).get("pos")
    if not pos:
        return None
    return float(pos.get("x", 0.0)), float(pos.get("y", 0.0))


def scan(
    paths: list[Path],
    *,
    team_prefix: str = "team-a",
    cover_range: float = 6.0,
    max_dist: float | None = None,
    leash_radius: float = 45.0,
) -> dict[str, float]:
    seen = blocked = near_cover = 0
    #: how far from the play area's centre the agent stood, over the same steps. A policy that
    #: withdraws by leaving the map scores well on "line broken" for the wrong reason, so the
    #: verdict needs this next to it.
    from_centre: list[float] = []
    for path in paths:
        for obs_by_agent in iter_observations(path):
            for agent, o in obs_by_agent.items():
                if not agent.startswith(team_prefix):
                    continue
                me = o.get("self") if isinstance(o.get("self"), dict) else o
                if not me or me.get("dead"):
                    continue
                enemies = [p for p in (o.get("players") or []) if is_armed_threat(p)]
                if not enemies:
                    continue
                if max_dist is not None:
                    nearest = min(float(p.get("dist", 1e9)) for p in enemies)
                    if nearest > max_dist:
                        continue
                seen += 1
                if all(p.get("los_blocked") for p in enemies):
                    blocked += 1
                if any(
                    float(ob.get("cover_score") or 0.0) > 0.0
                    and float(ob.get("dist", 99.0)) < cover_range
                    for ob in (o.get("obstacles") or [])
                ):
                    near_cover += 1
                centre = centre_of(o)
                here = me.get("pos")
                if centre and here:
                    from_centre.append(
                        math.hypot(
                            float(here.get("x", 0.0)) - centre[0],
                            float(here.get("y", 0.0)) - centre[1],
                        )
                    )
    ordered = sorted(from_centre)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    beyond = sum(1 for d in ordered if d > leash_radius)
    return {
        "max_dist": max_dist if max_dist is not None else 0.0,
        "leash_radius": leash_radius,
        "steps_with_a_position": len(ordered),
        "median_dist_from_centre": round(median, 1),
        "beyond_leash": beyond,
        "beyond_leash_pct": round(100.0 * beyond / len(ordered), 1) if ordered else 0.0,
        "steps_with_armed_enemy": seen,
        "line_broken": blocked,
        "line_broken_pct": 100.0 * blocked / seen if seen else 0.0,
        "cover_in_reach": near_cover,
        "cover_in_reach_pct": 100.0 * near_cover / seen if seen else 0.0,
    }


def format_row(label: str, r: Mapping[str, Any]) -> str:
    if not r["steps_with_armed_enemy"]:
        return f"{label:<28} no steps with an armed enemy in view"
    row = (
        f"{label:<28} armed enemy in view: {r['steps_with_armed_enemy']:5d} steps"
        f" | line broken: {r['line_broken']:5d} ({r['line_broken_pct']:.0f}%)"
        f" | usable cover within {{range}}m: {r['cover_in_reach']:5d} ({r['cover_in_reach_pct']:.0f}%)"
    )
    if r.get("steps_with_a_position"):
        row += (
            f"\n{'':<28} from the play-area centre: median {r['median_dist_from_centre']:.0f} m"
            f" | beyond the {r['leash_radius']:.0f} m leash: {r['beyond_leash_pct']:.0f}%"
        )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME",
        help="name for the next log path; repeat to compare several",
    )
    ap.add_argument(
        "logs",
        nargs="+",
        help="jsonl logs, one entry per --label; an entry may list several paths comma-separated",
    )
    ap.add_argument("--team-prefix", default="team-a", help="which side to measure")
    ap.add_argument(
        "--cover-range", type=float, default=6.0, help="how close cover counts as in reach"
    )
    ap.add_argument(
        "--leash-radius",
        type=float,
        default=45.0,
        help="how far from the play-area centre counts as having wandered off",
    )
    ap.add_argument(
        "--max-dist",
        type=float,
        default=None,
        help="only count steps with an armed enemy this close, i.e. steps in an actual fight",
    )
    ap.add_argument("--json-out", type=Path, help="also write the numbers here")
    args = ap.parse_args(argv)

    labels = args.label or [Path(p).stem for p in args.logs]
    if len(labels) != len(args.logs):
        ap.error(f"{len(labels)} labels for {len(args.logs)} logs")

    out: dict[str, Any] = {}
    for label, log in zip(labels, args.logs, strict=True):
        paths = [Path(p) for p in str(log).split(",") if p]
        r = scan(
            paths,
            team_prefix=args.team_prefix,
            cover_range=args.cover_range,
            max_dist=args.max_dist,
            leash_radius=args.leash_radius,
        )
        out[label] = r
        print(format_row(label, r).replace("{range}", f"{args.cover_range:.0f}"))
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
