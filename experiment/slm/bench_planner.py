"""Week-3 zero-shot check of the System 2 planner behind llama-server.

For each state block: send the fixed planner prefix + the block, sample under the skill GBNF
grammar, and record latency, token counts, whether the reply parses into a skill the server can run,
and the decision itself so a person can judge whether it fits the situation.

    python -m experiment.slm.bench_planner --url http://127.0.0.1:8090 \
        --blocks /work/out/slm/blocks.json --grammar /work/out/slm/skill.gbnf --runs 3

No dependencies beyond the standard library. The prefix is identical on every call on purpose: it is
what llama-server's prompt cache reuses, so only the block at the end costs prefill.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

#: The seven skills the server implements, with the params the grammar allows.
SKILLS: dict[str, str] = {
    "move_to": 'params {"pos": {"x": X, "y": Y}} - walk to a point (world coordinates, y grows north)',
    "follow": 'params {"target": AGENT_ID, "distance": N} - stay near a teammate',
    "loot": "params {} - pick up what you need next: a gun, then ammo for it",
    "heal": "params {} - use a bandage or healthkit until hp is full",
    "engage": 'params {"target": AGENT_ID, "style": "push"|"hold_angle"|"trade"} - fight an enemy',
    "retreat": 'params {"away_from": AGENT_ID, "distance": N} - back off from a threat',
    "revive": 'params {"target": AGENT_ID} - pick up a downed teammate',
}

SYSTEM_PROMPT = """You are the tactical brain of a teammate in a 2v2 top-down battle royale (surviv.io).
A separate motor system aims, moves and shoots every tick; you only choose WHAT to do next.

Read the state block and reply with exactly one JSON decision:
{"skill": NAME, "params": {...}, "commit_ms": MS, "say": SHORT_KOREAN_OR_NULL}

Skills:
""" + "\n".join(f"- {name}: {desc}" for name, desc in SKILLS.items()) + """

Rules:
- The state block is the only ground truth. Never name an enemy that is not listed.
- Without a gun you deal almost no damage at range: get one before a fight unless the enemy is on top of you.
- A downed teammate dies unless revived; revive when no enemy is close.
- commit_ms is how long you keep this decision (300-3000). Shorter when the situation is changing fast.
- "say" is an optional short Korean chat line to your human teammate (under 15 characters), or null.
- Directions are compass points: N is up the screen."""


def call(url: str, block: str, grammar: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": block},
        ],
        "grammar": grammar,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "cache_prompt": True,
        # both families think by default; the grammar forbids it anyway, this keeps the template honest
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        # llama-server explains a rejected grammar or template in the body; do not lose it
        raise SystemExit(f"llama-server {exc.code}: {exc.read().decode()[:400]}") from exc
    wall = time.perf_counter() - started
    text = data["choices"][0]["message"]["content"]
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    return {
        "wall_ms": wall * 1000,
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "cached_tokens": timings.get("cache_n"),
    }


def check(text: str) -> tuple[bool, str]:
    """Is this something `episode.resolveSkill` would accept?"""
    try:
        decision = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"not JSON: {exc}"
    if decision.get("skill") not in SKILLS:
        return False, f"unknown skill {decision.get('skill')!r}"
    if not isinstance(decision.get("params"), dict):
        return False, "params is not an object"
    if not isinstance(decision.get("commit_ms"), int):
        return False, "commit_ms missing"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="write every call as JSONL")
    args = ap.parse_args()

    blocks = json.loads(Path(args.blocks).read_text())
    grammar = Path(args.grammar).read_text()

    call(args.url, blocks[0]["block"], grammar, args.temperature, args.max_tokens)  # warm the prefix cache

    rows: list[dict[str, Any]] = []
    for case in blocks:
        print(f"\n## {case['name']}  (t={case['t']:.1f}s)")
        for line in case["block"].splitlines():
            print(f"   {line}")
        for run in range(args.runs):
            r = call(args.url, case["block"], grammar, args.temperature, args.max_tokens)
            ok, why = check(r["text"])
            r.update({"case": case["name"], "run": run, "valid": ok, "why": why})
            rows.append(r)
            print(f"   -> {r['wall_ms']:5.0f} ms  {'OK ' if ok else 'BAD'}  {r['text']}")

    walls = [r["wall_ms"] for r in rows]
    valid = sum(r["valid"] for r in rows)
    tps = [r["predicted_per_second"] for r in rows if r["predicted_per_second"]]
    completion = [r["completion_tokens"] for r in rows if r["completion_tokens"]]
    prompt = [r["prompt_tokens"] for r in rows if r["prompt_tokens"]]
    print(f"\n=== {args.label or args.url}: {len(rows)} decisions ===")
    print(f"valid JSON skill: {valid}/{len(rows)}")
    print(f"latency ms: median {statistics.median(walls):.0f}  p90 {sorted(walls)[int(0.9 * (len(walls) - 1))]:.0f}  max {max(walls):.0f}")
    if tps:
        print(f"decode tok/s: median {statistics.median(tps):.0f}")
    if completion and prompt:
        print(f"tokens: prompt median {statistics.median(prompt):.0f}, completion median {statistics.median(completion):.0f}")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
