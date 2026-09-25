#!/usr/bin/env python3
"""
A/B benchmark: interleave requests between two running servers.

The GPU on this host is shared with other resident services, so wall-clock
latency drifts over a run. Measuring server A fully, then server B fully,
attributes that drift to whichever ran second. Alternating the requests makes
the contention hit both variants equally.

Usage:
  python bench_ab.py --a http://127.0.0.1:8100 --b http://127.0.0.1:8101 \
      --token tok --label-a baseline --label-b opt --out bench_ab.json
"""
import argparse
import json
import os
import statistics
import sys

import jiwer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import normalise, post


def percentile(values, pct):
    """Linear-interpolation percentile. Index arithmetic on small n is unstable
    (a naive int(0.95*n) on 36 samples selects the 2nd-worst value, which swings
    wildly between runs)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarise(rows):
    """
    rows must be in the same order as their reference transcripts, so the
    corpus WER pairs reference[i] with hypothesis[i] positionally. Errors are
    excluded from the WER pairing (they would otherwise misalign the rest).
    """
    lat = [r["latency_s"] for r in rows]
    rtf = [r["rtf"] for r in rows]
    total_audio = sum(r["duration"] for r in rows)
    ok = [r for r in rows if r["http"] == 200 and r.get("ref_text") is not None]

    return {
        "n_requests": len(rows),
        "errors": len(rows) - len(ok),
        "total_audio_s": round(total_audio, 1),
        "latency_mean_s": round(statistics.mean(lat), 4),
        "latency_p50_s": round(statistics.median(lat), 4),
        "latency_p90_s": round(percentile(lat, 90), 4),
        "latency_p95_s": round(percentile(lat, 95), 4),
        "rtf_mean": round(statistics.mean(rtf), 4),
        "rtf_wallclock": round(sum(lat) / total_audio, 4),
        "wer_corpus": round(
            jiwer.wer(
                " ".join(r["ref_text"] for r in ok),
                " ".join(r["hyp_text"] for r in ok),
            ),
            4,
        )
        if ok
        else None,
        "wer_mean_per_utt": round(statistics.mean([r["wer"] for r in rows]), 4),
        "chars_ref": sum(r["ref_chars"] for r in rows),
        "chars_hyp": sum(r["hyp_chars"] for r in rows),
        "compression": round(
            sum(r["hyp_chars"] for r in rows) / max(1, sum(r["ref_chars"] for r in rows)), 4
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--fixtures", default="/home/ai-agent/.hermes/cache/scratch/asr_fixtures/ground_truth.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="json,verbose_json")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    fixtures = json.load(open(args.fixtures))
    out = {"fixtures": args.fixtures, "repeats": args.repeats, "runs": {}}

    for fmt in args.formats.split(","):
        if args.warmup:
            for base in (args.a, args.b):
                post(base, "/v1/audio/transcriptions", args.token, fixtures[0]["path"], fmt)

        # Interleave: for each fixture, hit A then B (and B then A on odd reps,
        # so neither variant systematically benefits from ordering).
        rows_a, rows_b = [], []
        for r in range(args.repeats):
            for i, fx in enumerate(fixtures):
                order = [(args.a, args.label_a, rows_a), (args.b, args.label_b, rows_b)]
                if (r + i) % 2:
                    order.reverse()
                for base, label, sink in order:
                    lat, code, text = post(
                        base, "/v1/audio/transcriptions", args.token, fx["path"], fmt
                    )
                    ref, hyp = normalise(fx["text"]), normalise(text)
                    wer = 1.0 if not hyp else jiwer.wer(ref, hyp)
                    sink.append(
                        {
                            "id": fx["id"],
                            "rep": r,
                            "duration": fx["duration"],
                            "http": code,
                            "latency_s": round(lat, 4),
                            "rtf": round(lat / fx["duration"], 4),
                            "wer": round(wer, 4),
                            "ref_chars": len(ref),
                            "hyp_chars": len(hyp),
                            "ref_text": ref,
                            "hyp_text": hyp,
                        }
                    )

        sa, sb = summarise(rows_a), summarise(rows_b)
        out["runs"][fmt] = {
            args.label_a: {"summary": sa, "rows": rows_a},
            args.label_b: {"summary": sb, "rows": rows_b},
        }

        print(f"\n########## {fmt} ##########")
        print(f"{'metric':<24}{args.label_a:>14}{args.label_b:>14}{'delta':>14}")
        for k in sa:
            va, vb = sa[k], sb[k]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = vb - va
                pct = (d / va * 100) if va else 0.0
                print(f"{k:<24}{va:>14}{vb:>14}{pct:>13.1f}%")
            else:
                print(f"{k:<24}{str(va):>14}{str(vb):>14}{'':>14}")

    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
