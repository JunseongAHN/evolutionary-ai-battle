"""Draw an episode from above, so a person can see what the agents actually did.

    python -m experiment.slm.render_replay --episodes runs/live_like_controller.jsonl --out out/replay

Numbers say a controller wins more; they do not say whether it looks sane next to you. This draws the
ground truth of one episode — cover, both duos' paths, where shots were fired, where damage landed,
where someone went down — and leaves the judging to the reader.

Paths are the agents' own positions from their observations, so nothing is drawn that the server did
not record. Standard library only; the page is HTML+SVG and a headless browser turns it into a PNG.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

TEAM_COLOUR = {"team-a": "#2e86c1", "team-b": "#c0392b"}
MATE_COLOUR = {"team-a-0": "#2e86c1", "team-a-1": "#5dade2", "team-b-0": "#c0392b", "team-b-1": "#e6807a"}


def episodes(path: Path) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Read either log this project writes, as (rows, meta) per episode.

    * the bridge/episode runner writes one row per step, episodes run together in one file;
    * ``eval.py`` writes one object per episode with its steps inside and the metrics alongside.

    Both carry `t`, an `obs` per agent and the step's `events`, which is all the drawing needs. With
    ``eval.py --full-obs`` the observation is the whole thing (cover, enemies); without it, it is a
    per-agent summary that still has `pos`, so a path can be drawn either way.
    """
    out: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "steps" in record:  # eval.py: one episode per line
            meta = {
                "reason": record.get("reason"),
                "winner_team": record.get("team_win") and "team-a" or record.get("winner_team"),
                "metrics": record.get("metrics") or {},
                "t": float(record.get("duration_s", 0.0)),
            }
            out.append(([dict(step) for step in record["steps"]], meta))
            continue
        if current and float(record.get("t", 0.0)) < float(current[-1].get("t", 0.0)):
            out.append((current, _meta_from_rows(current)))
            current = []
        current.append(record)
    if current:
        out.append((current, _meta_from_rows(current)))
    return out


def _meta_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    info = (rows[-1].get("info") or {}) if rows else {}
    return {
        "reason": info.get("reason"),
        "winner_team": info.get("winner_team"),
        "metrics": info.get("metrics") or {},
        "t": float(rows[-1].get("t", 0.0)) if rows else 0.0,
    }


def _pos(obs: dict[str, Any]) -> dict[str, float] | None:
    """`self.pos` in a full observation, `pos` in eval's summary."""
    me = obs.get("self") if isinstance(obs.get("self"), dict) else obs
    pos = (me or {}).get("pos")
    return pos if isinstance(pos, dict) else None


def _weapon(obs: dict[str, Any]) -> str:
    me = obs.get("self") if isinstance(obs.get("self"), dict) else obs
    return str((me or {}).get("weapon") or "")


def bounds(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for row in rows:
        for obs in (row.get("obs") or {}).values():
            pos = _pos(obs)
            if pos:
                xs.append(float(pos["x"]))
                ys.append(float(pos["y"]))
            for o in (obs.get("obstacles") or []):
                xs.append(float(o["pos"]["x"]))
                ys.append(float(o["pos"]["y"]))
    if not xs:
        return 0.0, 0.0, 1.0, 1.0
    pad = 6.0
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def svg(rows: list[dict[str, Any]], size: int = 720) -> str:
    x0, y0, x1, y1 = bounds(rows)
    span = max(x1 - x0, y1 - y0)
    scale = (size - 40) / span
    sx = lambda x: 20 + (x - x0) * scale
    sy = lambda y: size - 20 - (y - y0) * scale  # world y grows upward

    parts = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect width="{size}" height="{size}" fill="#eef3e8"/>']

    # cover, from whatever any agent saw (the union is the map as far as anyone knows)
    seen: dict[int, dict[str, Any]] = {}
    for row in rows:
        for obs in (row.get("obs") or {}).values():
            for o in (obs.get("obstacles") or []):
                if o.get("collidable"):
                    seen[int(o["id"])] = o
    for o in seen.values():
        parts.append(f'<circle cx="{sx(o["pos"]["x"]):.1f}" cy="{sy(o["pos"]["y"]):.1f}" r="{max(3.0, 2.2 * scale):.1f}" '
                     'fill="#9aa89a" stroke="#7a887a"/>')

    # paths
    tracks: dict[str, list[tuple[float, float]]] = {}
    # where an agent was still empty-handed: the live-like check found the controller spends half a
    # match without a gun, and a path alone does not show it
    unarmed: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        for agent, obs in (row.get("obs") or {}).items():
            pos = _pos(obs)
            if not pos:
                continue
            tracks.setdefault(agent, []).append((float(pos["x"]), float(pos["y"])))
            if not _weapon(obs) or _weapon(obs) == "fists":
                unarmed.setdefault(agent, []).append((float(pos["x"]), float(pos["y"])))
    for agent, points in tracks.items():
        if len(points) < 2:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
        colour = MATE_COLOUR.get(agent, "#555")
        parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.5" opacity="0.85"/>')
        bx, by = points[0]
        parts.append(f'<circle cx="{sx(bx):.1f}" cy="{sy(by):.1f}" r="5" fill="white" stroke="{colour}" stroke-width="2.5"/>')
        ex, ey = points[-1]
        parts.append(f'<circle cx="{sx(ex):.1f}" cy="{sy(ey):.1f}" r="6" fill="{colour}"/>')
        parts.append(f'<text x="{sx(ex) + 9:.1f}" y="{sy(ey) + 4:.1f}" font-size="13" font-weight="700" fill="{colour}">{html.escape(agent)}</text>')

    for agent, points in unarmed.items():
        for x, y in points[::3]:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2" fill="#f39c12" opacity="0.8"/>')

    # events: a tick for every shot, a cross where damage landed, a ring where someone went down
    for row in rows:
        for e in (row.get("events") or []):
            kind, pos = e.get("type"), e.get("pos")
            if not pos:
                continue
            x, y = sx(float(pos["x"])), sy(float(pos["y"]))
            colour = MATE_COLOUR.get(str(e.get("agent") or e.get("source") or ""), "#555")
            if kind == "fire":
                d = e.get("dir") or {"x": 0, "y": 0}
                parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + float(d["x"]) * 9:.1f}" '
                             f'y2="{y - float(d["y"]) * 9:.1f}" stroke="{colour}" stroke-width="1" opacity="0.45"/>')
            elif kind == "damage":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#e74c3c" opacity="0.8"/>')
            elif kind in ("down", "kill"):
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="none" stroke="#111" stroke-width="2"/>')
                parts.append(f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" font-size="12" fill="#111">{kind} {html.escape(str(e.get("agent","")))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def summary(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [f"t={float(meta.get('t', 0.0)):.1f}s  ended: {meta.get('reason')}  winner: {meta.get('winner_team')}"]
    for agent, m in sorted((meta.get("metrics") or {}).items()):
        lines.append(f"{agent}: survived {m.get('survival_time', 0):.1f}s  dealt {m.get('damage_dealt', 0):.0f}  "
                     f"taken {m.get('damage_taken', 0):.0f}  kills {m.get('kills', 0)}  shots {m.get('shots', 0)}")
    # what each agent was holding at the end, and when it first had a gun
    first_gun = {}
    for row in rows:
        for agent, obs in (row.get("obs") or {}).items():
            if agent not in first_gun and _weapon(obs) not in ("", "fists"):
                first_gun[agent] = float(row.get("t", 0.0))
    lines.append("first held a gun: " + (", ".join(f"{a} {t:.1f}s" for a, t in sorted(first_gun.items())) or "nobody"))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", required=True, help="episode JSONL (bridge log or eval --out)")
    ap.add_argument("--out", required=True, help="directory for the HTML pages")
    ap.add_argument("--limit", type=int, default=4, help="how many episodes to draw")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    out = Path(args.out) / "html"
    out.mkdir(parents=True, exist_ok=True)
    eps = episodes(Path(args.episodes))[: args.limit]
    for i, (rows, meta) in enumerate(eps, 1):
        page = f"""<title>replay {i}</title>
<style>body{{margin:0;padding:16px 20px;background:#fff;font-family:'Noto Sans CJK KR',sans-serif;width:1180px}}
h1{{font-size:20px;margin:0 0 2px}} .sub{{color:#666;font-size:13px;margin-bottom:10px}}
.wrap{{display:flex;gap:20px}} pre{{background:#1f2430;color:#e6e1cf;padding:12px 14px;border-radius:6px;font-size:13px;line-height:1.5}}
.legend{{font-size:12px;color:#555;margin-top:8px}}</style>
<h1>{html.escape(args.label or Path(args.episodes).stem)} — episode {i}</h1>
<div class="sub">위에서 본 한 판. 흰 점이 시작, 굵은 점이 끝. 가는 선은 사격 방향, 붉은 점은 피해가 닿은 자리, 검은 원은 다운/사망.</div>
<div class="wrap"><div>{svg(rows)}
<div class="legend">● 회색 = 엄폐물 · 파랑 계열 = team-a(우리) · 빨강 계열 = team-b(상대) · 주황 점 = 아직 맨손</div></div>
<div><pre>{html.escape(summary(rows, meta))}</pre></div></div>"""
        (out / f"{i:02d}.html").write_text(page, encoding="utf-8")
    print(f"wrote {len(eps)} pages to {out}")


if __name__ == "__main__":
    main()
