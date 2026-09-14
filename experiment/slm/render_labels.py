"""Render planner decisions next to the state they were made in, one page per case, for a person to audit.

    python -m experiment.slm.render_labels --cases experiment/slm/cases_v1.json \
        --run v0-4B=/work/out/slm/cmp_v0_4b.jsonl --run teacher-27B=/work/out/slm/cmp_27b.jsonl \
        --out /work/out/slm/review

Writes `html/<case>.html` and `html/summary.html`; screenshot them to PNG with headless chromium.

The map is rebuilt from the state block text alone — distance plus an 8-point bearing — so it shows
exactly what the planner was told and no more: positions are quantised to 45 degrees, and anything the
block leaves out (walls, exact angles) is not drawn. The "expected" column is the hand-written answer key
in `bench_planner.EXPECTED`; it is on the page so the key can be audited too, not just the models.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import re
import statistics
from pathlib import Path

from experiment.slm.bench_planner import EXPECTED

BEARING = {"E": 0, "NE": 45, "N": 90, "NW": 135, "W": 180, "SW": 225, "S": 270, "SE": 315}

#: why the answer key says what it says, in the prompt's own rules; shown so a reviewer can disagree
WHY = {
    "spawn_unarmed": "맨손, 적 없음 → 총부터 줍는다",
    "enemy_in_view_unarmed": "맨손인데 적이 보임 → 총을 줍거나 물러난다 (3m 안이면 주먹)",
    "enemy_in_view_armed": "무장했고 적이 보임 → 교전 (30hp 미만일 때만 후퇴)",
    "taking_fire": "맞고 있음 → 교전 또는 후퇴",
    "race_point_quiet": '조용하고 거점이 보임 → move_to "point"',
    "teammate_downed_under_fire": "팀원 다운 + 무장한 적 25m 안 → 리바이브 전에 교전/후퇴",
    "idle_armed_no_point": '할 일 없음 → 사람 곁에 (follow 또는 move_to "team-a-1")',
    "revive_under_fire": "플레이테스트 원문: 팀원 다운, 적 21m → 교전/후퇴",
    "revive_under_fire_unmasked": "위와 같은 상황, 스킬 마스크 없이 (대조군)",
}


def parse_block(block: str) -> dict:
    """The block's bracketed lines as drawable things: (kind, label, dist, bearing, extra)."""
    things: list[dict] = []
    info: dict = {"things": things, "header": "", "weapon": "", "can_do": "", "heard": [], "other": []}

    def entity(kind: str, text: str, **extra) -> None:
        m = re.match(r"^(.*?)\s*(\d+)m (N|NE|E|SE|S|SW|W|NW|here)\b\s*(.*)$", text.strip())
        if m:
            things.append({"kind": kind, "label": m.group(1).strip(), "dist": int(m.group(2)),
                           "bearing": m.group(3), "rest": m.group(4).strip(), **extra})

    for line in block.splitlines():
        body = line.strip()[1:-1]
        key, _, value = body.partition(": ")
        if body.startswith("t="):
            info["header"] = body
        elif key == "weapon":
            info["weapon"] = value
        elif key.startswith("teammate "):
            state, _, where = value.partition(", ")
            entity("mate", f"{key.split()[1]} {where}", state=state)
        elif key == "enemies seen":
            for e in value.split(" | "):
                entity("enemy", e)
        elif key == "loot":
            for item in value.split(", "):
                entity("loot", item)
        elif key == "cover":
            for item in value.split(", "):
                entity("cover", item)
        elif key == "point":
            entity("point", "point " + value)
        elif key == "shots heard":
            info["heard"] = value.split(", ")
        elif key == "can do":
            info["can_do"] = value
        else:
            info["other"].append(line.strip())
    return info


def svg_map(info: dict, size: int = 640) -> str:
    c = size / 2
    far = max([t["dist"] for t in info["things"]] + [30])
    scale = (c - 50) / far
    marks: list[str] = []
    labels: list[str] = []
    placed: list[tuple[float, float, float, float]] = [(c - 14, c - 14, c + 14, c + 14)]

    def label(x: float, y: float, text: str, px: int, color: str, bold: bool = False) -> None:
        # nudge up or down until clear of the labels already placed; flip to the left at the right edge
        w = sum(px if ord(ch) > 0x1100 else px * 0.6 for ch in text)
        lx = x - 12 - w if x + 12 + w > size - 4 else x + 12
        for dy in (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90):
            box = (lx, y + dy - px * 0.6, lx + w, y + dy + px * 0.5)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                break
        placed.append(box)
        weight = ' font-weight="700"' if bold else ""
        labels.append(f'<text x="{lx:.1f}" y="{y + dy + px * 0.4:.1f}" font-size="{px}"{weight} fill="{color}">'
                      f"{html.escape(text)}</text>")

    for ring in (10, 25, 50, 100):
        if ring <= far * 1.05:
            dash = ' stroke-dasharray="6 5"' if ring == 25 else ""
            color = "#c0392b" if ring == 25 else "#b9b2a3"
            marks.append(f'<circle cx="{c}" cy="{c}" r="{ring * scale:.1f}" fill="none" stroke="{color}"{dash}/>')
            note = " (리바이브 안전거리)" if ring == 25 else ""
            marks.append(f'<text x="{c + 4}" y="{c - ring * scale - 4:.1f}" font-size="12" fill="{color}">{ring}m{note}</text>')
    marks.append(f'<text x="{c - 5}" y="18" font-size="16" font-weight="700" fill="#555">N</text>')

    order = {"enemy": 0, "mate": 1, "point": 2, "loot": 3, "cover": 4}
    for t in sorted(info["things"], key=lambda t: order[t["kind"]]):
        a = math.radians(BEARING.get(t["bearing"], 0))
        x, y = c + t["dist"] * scale * math.cos(a), c - t["dist"] * scale * math.sin(a)
        text = f'{t["label"]} {t["dist"]}m {t["rest"]}'.strip()
        if t["kind"] == "enemy":
            armed = "no gun" not in t["rest"]
            marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{"#c0392b" if armed else "#fff"}" '
                         f'stroke="#c0392b" stroke-width="3" opacity="0.85"/>')
            if "DOWNED" in t["rest"]:
                marks.append(f'<path d="M{x-8:.1f},{y-8:.1f} L{x+8:.1f},{y+8:.1f} M{x+8:.1f},{y-8:.1f} L{x-8:.1f},{y+8:.1f}" '
                             'stroke="#000" stroke-width="2"/>')
            label(x, y, text, 14, "#c0392b", bold=True)
        elif t["kind"] == "mate":
            down = t.get("state") in ("DOWNED", "DEAD")
            marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{"#fff" if down else "#27ae60"}" '
                         'stroke="#27ae60" stroke-width="3"/>')
            label(x, y, f'사람 {text} [{t["state"]}]', 14, "#1e8449", bold=True)
        elif t["kind"] == "point":
            marks.append(f'<rect x="{x - 10:.1f}" y="{y - 10:.1f}" width="20" height="20" fill="#8e44ad" '
                         f'transform="rotate(45 {x:.1f} {y:.1f})"/>')
            label(x, y, text, 14, "#8e44ad", bold=True)
        elif t["kind"] == "loot":
            marks.append(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="#d4a017"/>')
            label(x, y, text, 12, "#7d6608")
        else:
            marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#95a5a6" opacity="0.7"/>')
            label(x, y, text, 11, "#566573")
    for shot in info["heard"]:
        m = re.search(r"\b(N|NE|E|SE|S|SW|W|NW)\b", shot)
        if m:
            a = math.radians(BEARING[m.group(1)])
            label(c + (c - 30) * math.cos(a), c - (c - 30) * math.sin(a), f"총성 {shot}", 13, "#e67e22", bold=True)
    marks.append(f'<circle cx="{c}" cy="{c}" r="12" fill="#2e86c1" stroke="#1b4f72" stroke-width="3"/>')
    label(c, c + 20, "CPC (나)", 14, "#1b4f72", bold=True)
    return "\n".join([f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">',
                      f'<rect width="{size}" height="{size}" fill="#f4f1ea"/>', *marks, *labels, "</svg>"])


def decision_key(text: str) -> str:
    d = json.loads(text)
    params = ", ".join(str(v) for v in d.get("params", {}).values())
    return f'{d["skill"]}({params})'


def model_panel(name: str, rows: list[dict]) -> str:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[decision_key(r["text"]) if r["valid"] else "INVALID"].append(r)
    fit = sum(r["appropriate"] for r in rows)
    parts = [f'<div class="model"><h3>{html.escape(name)} <span class="score">{fit}/{len(rows)} 적절</span></h3><table>']
    for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ok = rs[0]["appropriate"]
        says = collections.Counter((json.loads(r["text"]).get("say") or "—") for r in rs if r["valid"])
        commit = statistics.median(json.loads(r["text"])["commit_ms"] for r in rs if r["valid"]) if rs[0]["valid"] else 0
        say_text = " · ".join(f"“{html.escape(s)}”" + (f"×{n}" if n > 1 else "") for s, n in says.most_common(4))
        parts.append(f'<tr class="{"ok" if ok else "bad"}"><td class="mark">{"✓" if ok else "X"}</td>'
                     f'<td class="n">×{len(rs)}</td><td class="key">{html.escape(key)}</td>'
                     f'<td class="commit">{commit:.0f}ms</td><td class="say">{say_text}</td></tr>')
    parts.append("</table></div>")
    return "\n".join(parts)


STYLE = """<style>
body{margin:0;padding:18px 22px;background:#fff;color:#222;font-family:'Noto Sans CJK KR','Noto Sans KR',sans-serif;width:1560px}
h1{font-size:24px;margin:0 0 4px} .sub{color:#666;font-size:14px;margin-bottom:12px}
.wrap{display:flex;gap:22px} .right{flex:1;min-width:0}
pre{background:#1f2430;color:#e6e1cf;padding:12px 14px;border-radius:6px;font-size:14px;line-height:1.45;white-space:pre-wrap;margin:0 0 12px;font-family:'Noto Sans Mono CJK KR',monospace}
.key-box{border:2px solid #2e86c1;background:#eef6fc;border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:15px}
.key-box b{color:#1b4f72}
.model{border:1px solid #ddd;border-radius:6px;padding:8px 12px;margin-bottom:10px}
.model h3{margin:0 0 6px;font-size:17px} .score{font-weight:400;color:#555;font-size:15px;margin-left:8px}
table{border-collapse:collapse;width:100%;font-size:14px} td{padding:3px 6px;vertical-align:top}
tr.ok td.mark{color:#1e8449;font-weight:700} tr.bad td.mark{color:#c0392b;font-weight:700} tr.bad td.key{color:#c0392b}
td.n{width:36px;color:#555} td.key{font-family:monospace;white-space:nowrap} td.commit{color:#888;white-space:nowrap} td.say{color:#444}
.legend{font-size:12px;color:#666;margin-top:6px}
.sum td,.sum th{border-bottom:1px solid #e5e5e5;padding:6px 10px;font-size:15px;text-align:left} .sum th{background:#f4f4f4}
</style>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--run", action="append", required=True, help="LABEL=path.jsonl from bench_planner --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    runs = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs[label] = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    out = Path(args.out) / "html"
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, case in enumerate(cases, 1):
        name = case["name"]
        info = parse_block(case["block"])
        expected = " / ".join(sorted(EXPECTED.get(name, set()))) or "(채점 없음)"
        panels = "\n".join(model_panel(label, [r for r in rows if r["case"] == name]) for label, rows in runs.items())
        page = f"""<title>{html.escape(name)}</title>{STYLE}
<h1>{i}. {html.escape(name)}</h1>
<div class="sub">상태 블록(모델이 받는 전부)과 그로부터 복원한 지도 · 방위는 45° 단위 · 모델별 10회 샘플(temperature 0.7) · 스킬 마스크: {html.escape(info["can_do"] or "없음")}</div>
<div class="wrap"><div>{svg_map(info)}
<div class="legend">● 파랑 = CPC · 초록 = 사람 팀원(흰색=다운) · 빨강 = 적(흰색=총 없음, X=다운) · ◆ 보라 = 거점 · ■ 노랑 = 아이템 · 회색 = 엄폐물</div></div>
<div class="right"><pre>{html.escape(case["block"])}</pre>
<div class="key-box"><b>정답 기준 (Claude가 작성, 검토 대상):</b> {html.escape(expected)}<br>{html.escape(WHY.get(name, ""))}</div>
{panels}</div></div>"""
        (out / f"{i:02d}_{name}.html").write_text(page, encoding="utf-8")
        summary_rows.append((i, name, expected, {label: sum(r["appropriate"] for r in rows if r["case"] == name) for label, rows in runs.items()}))

    head = "".join(f"<th>{html.escape(label)}</th>" for label in runs)
    body = []
    for i, name, expected, scores in summary_rows:
        body.append(f"<tr><td>{i}</td><td>{html.escape(name)}</td><td>{html.escape(expected)}</td>"
                    + "".join(f"<td>{scores[label]}/10</td>" for label in runs) + "</tr>")
    graded = [n for _, n, _, _ in summary_rows if n != "revive_under_fire_unmasked"]
    totals = "".join(f"<td><b>{sum(r['appropriate'] for r in rows if r['case'] in graded)}/{10 * len(graded)}</b></td>" for rows in runs.values())
    lat = "".join(f"<td>{statistics.median(r['wall_ms'] for r in rows):.0f} ms</td>" for rows in runs.values())
    (out / "00_summary.html").write_text(f"""<title>summary</title>{STYLE}
<h1>cases_v1 — 모델별 적절한 결정 수</h1>
<div class="sub">같은 9개 상황 · 같은 게임 프롬프트(prompt_ko) · 같은 문법/마스크 · 상황당 10회 · 합계는 대조군(마스크 없음) 제외</div>
<table class="sum"><tr><th>#</th><th>상황</th><th>정답 기준 (Claude)</th>{head}</tr>{''.join(body)}
<tr><td></td><td><b>합계</b></td><td></td>{totals}</tr><tr><td></td><td>결정 지연 중앙값</td><td></td>{lat}</tr></table>""", encoding="utf-8")
    print(f"wrote {len(summary_rows) + 1} pages to {out}")


if __name__ == "__main__":
    main()
