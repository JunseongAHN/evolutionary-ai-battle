"""Measure a llama-server: decode speed alone and aggregate throughput under concurrency.

    python experiment/slm/measure_server.py --url http://127.0.0.1:8091 --concurrency 1 4 8

Uses the native `/completion` endpoint with `ignore_eos` so every request decodes exactly
`--tokens` tokens — the numbers are then comparable across models and settings instead of depending
on when the model decides to stop. The prompt is sized like a teacher-labeling request (a planner
prefix plus a state block, ~1k tokens), and each concurrent request gets a different tail so the
cached prefix is shared while the slots still do real work.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PREFIX = (
    "You label decisions for a teammate in a 2v2 top-down battle royale. For the state block below, "
    "write a one-sentence rationale, then the decision as JSON with skill, params, commit_ms and a short "
    "Korean chat line, in the voice of the given persona.\n\n"
) + "Rules and skill reference:\n" + "\n".join(
    f"- rule {i}: keep the teammate alive, fight what you can see, never name what is not listed." for i in range(60)
)


def block(i: int) -> str:
    return (
        f"\n\nPersona: {['cautious', 'aggressive', 'looter', 'chatty'][i % 4]}\n"
        f"[t={i}s you=team-a-0 {40 + i % 50}hp]\n[weapon: ak47 {i % 30}/60 | scope 1xscope]\n"
        f"[teammate team-a-1: DOWNED, {2 + i % 9}m SE]\n[enemies seen: team-b-0 {15 + i % 20}m W ak47]\n"
        "[can do: move_to, engage, retreat]\nRationale:"
    )


def one(url: str, i: int, tokens: int) -> dict:
    body = {"prompt": PREFIX + block(i), "n_predict": tokens, "ignore_eos": True, "cache_prompt": True,
            "temperature": 0.7}
    req = urllib.request.Request(f"{url}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    t = data.get("timings", {})
    return {"wall": time.perf_counter() - started, "predicted": t.get("predicted_n", tokens),
            "decode_tps": t.get("predicted_per_second"), "prompt_n": t.get("prompt_n"),
            "prompt_tps": t.get("prompt_per_second")}


def run(url: str, concurrency: int, tokens: int, rounds: int) -> dict:
    results: list[dict] = []
    lock = threading.Lock()

    errors: list[str] = []

    def worker(w: int) -> None:
        for r in range(rounds):
            try:
                out = one(url, w * 100 + r, tokens)
            except Exception as exc:  # a thread's exception is otherwise swallowed silently
                body = exc.read().decode()[:300] if hasattr(exc, "read") else ""
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc} {body}")
                continue
            with lock:
                results.append(out)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - started
    if errors:
        raise SystemExit(f"{len(errors)} request(s) failed at concurrency {concurrency}: {errors[0]}")
    total = sum(r["predicted"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "aggregate_tps": total / wall,
        "per_request_decode_tps": statistics.median(r["decode_tps"] for r in results if r["decode_tps"]),
        "median_request_s": statistics.median(r["wall"] for r in results),
        "prompt_tokens": statistics.median(r["prompt_n"] for r in results if r["prompt_n"] is not None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--tokens", type=int, default=128, help="tokens decoded per request")
    ap.add_argument("--rounds", type=int, default=3, help="requests per worker")
    args = ap.parse_args()

    one(args.url, 999, 16)  # warm up and cache the prefix
    print(f"{'conc':>4} {'reqs':>4} {'agg tok/s':>10} {'per-req tok/s':>14} {'req s':>7} {'prompt tok':>10}")
    for c in args.concurrency:
        r = run(args.url, c, args.tokens, args.rounds)
        print(f"{r['concurrency']:>4} {r['requests']:>4} {r['aggregate_tps']:>10.1f} {r['per_request_decode_tps']:>14.1f} "
              f"{r['median_request_s']:>7.2f} {r['prompt_tokens']:>10.0f}")


if __name__ == "__main__":
    main()
