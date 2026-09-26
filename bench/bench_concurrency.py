#!/usr/bin/env python3
"""
Concurrency test: does the API actually overlap requests, or serialise them?

The previous revision held a single global asyncio.Lock around inference, so N
concurrent clients were strictly serial: total wall time equalled the sum of
the per-request latencies. This measures wall time and per-request latency at
several concurrency levels and reports the implied effective parallelism.

Usage:
  python bench_concurrency.py --base-url http://127.0.0.1:8101 --token tok \
      --levels 1,2,4,8 --out concurrency.json
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import post


def one(base, token, fx, fmt, timeout):
    return post(base, "/v1/audio/transcriptions", token, fx["path"], fmt, timeout=timeout)


def run_level(base, token, fixtures, fmt, concurrency, timeout):
    # Enough work that every level has a full queue to chew on.
    jobs = []
    while len(jobs) < concurrency * len(fixtures):
        jobs.extend(fixtures[:concurrency])
    jobs = jobs[: concurrency * len(fixtures)]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(lambda fx: one(base, token, fx, fmt, timeout), jobs)
        )
    wall = time.perf_counter() - start

    lat = [r[0] for r in results]
    codes = [r[1] for r in results]
    audio = sum(fx["duration"] for fx in jobs)
    mean_lat = statistics.mean(lat)

    return {
        "concurrency": concurrency,
        "n_requests": len(jobs),
        "wall_s": round(wall, 3),
        "total_audio_s": round(audio, 1),
        "errors": sum(1 for c in codes if c != 200),
        "latency_mean_s": round(mean_lat, 3),
        "latency_p95_s": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 3),
        # sum of individual latencies / wall clock: 1.0 means perfectly
        # serialised, N means N requests genuinely in flight.
        "effective_parallelism": round(sum(lat) / wall, 3),
        "rtf_wallclock": round(wall / audio, 4),
        "audio_x_realtime": round(audio / wall, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--fixtures", default="/home/ai-agent/.hermes/cache/scratch/asr_fixtures/ground_truth.json")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--format", default="json")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    fixtures = json.load(open(args.fixtures))
    out = {"label": args.label, "base_url": args.base_url, "levels": []}

    # Warmup so level-1 numbers are not paying first-request autotune.
    one(args.base_url, args.token, fixtures[0], args.format, args.timeout)

    print(
        f"{'conc':>5}{'reqs':>6}{'wall_s':>9}{'lat_mean':>10}{'lat_p95':>9}"
        f"{'errors':>8}{'par':>8}{'audio_x':>9}"
    )
    for c in [int(x) for x in args.levels.split(",")]:
        r = run_level(args.base_url, args.token, fixtures, args.format, c, args.timeout)
        out["levels"].append(r)
        print(
            f"{r['concurrency']:>5}{r['n_requests']:>6}{r['wall_s']:>9}"
            f"{r['latency_mean_s']:>10}{r['latency_p95_s']:>9}{r['errors']:>8}"
            f"{r['effective_parallelism']:>8}{r['audio_x_realtime']:>9}"
        )

    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
