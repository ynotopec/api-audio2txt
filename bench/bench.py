#!/usr/bin/env python3
"""
Benchmark api-audio2txt against a labelled fixture set.

Measures, per response_format:
  - latency (mean / p50 / p95) and real-time factor (RTF = latency / audio_duration)
  - WER against the LibriSpeech reference transcript
  - text truncation rate (hypothesis far shorter than reference)

Usage:
  python bench.py --base-url http://127.0.0.1:8000 --label baseline \
      --token "$(...)" --out bench_baseline.json
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid

import jiwer

# LibriSpeech references are UPPERCASE with no punctuation; hypotheses are
# normal prose. Normalising both to lowercase, unpunctuated words is what makes
# the WER comparison meaningful.
NORMALISER = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
    ]
)


def post(base, path, token, filename, fmt, extra=None, timeout=300):
    boundary = uuid.uuid4().hex
    parts = []

    def field(name, value):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    with open(filename, "rb") as f:
        payload = f.read()

    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{filename.split("/")[-1]}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + payload
        + b"\r\n"
    )
    field("model", "whisper-1")
    field("response_format", fmt)
    for k, v in (extra or {}).items():
        field(k, v)

    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{base}{path}",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
            code = resp.status
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        code = e.code
    latency = time.perf_counter() - start

    if fmt == "json":
        return latency, code, json.loads(text).get("text", "")
    if fmt == "verbose_json":
        return latency, code, json.loads(text).get("text", "")
    return latency, code, text


def normalise(s):
    return NORMALISER(s)


def percentile(values, pct):
    """Linear-interpolation percentile; index arithmetic on small n is unstable."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--path", default="/v1/audio/transcriptions")
    ap.add_argument("--token", required=True)
    ap.add_argument("--fixtures", default="/home/ai-agent/.hermes/cache/scratch/asr_fixtures/ground_truth.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="json,verbose_json")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    fixtures = json.load(open(args.fixtures))
    results = {"label": args.label, "base_url": args.base_url, "runs": {}}

    # Warmup: first request pays cudnn.benchmark autotune + lazy alloc.
    if args.warmup:
        post(args.base_url, args.path, args.token, fixtures[0]["path"], "json")

    for fmt in args.formats.split(","):
        rows = []
        for fx in fixtures:
            lat, code, text = post(args.base_url, args.path, args.token, fx["path"], fmt)
            ref, hyp = normalise(fx["text"]), normalise(text)
            if not hyp:
                wer = 1.0
            else:
                wer = jiwer.wer(ref, hyp)
            rows.append(
                {
                    "id": fx["id"],
                    "duration": fx["duration"],
                    "http": code,
                    "latency_s": round(lat, 3),
                    "rtf": round(lat / fx["duration"], 4),
                    "wer": round(wer, 4),
                    "ref_chars": len(ref),
                    "hyp_chars": len(hyp),
                    "truncated": len(hyp) < 0.5 * len(ref),
                    "hyp_text": hyp,
                }
            )
            print(
                f"[{args.label}/{fmt}] {fx['id']} {fx['duration']:>6.1f}s "
                f"lat={lat:6.2f}s rtf={lat/fx['duration']:.3f} wer={wer:.3f} http={code}"
            )

        lat = [r["latency_s"] for r in rows]
        rtf = [r["rtf"] for r in rows]
        wers = [r["wer"] for r in rows]
        total_audio = sum(r["duration"] for r in rows)
        ok = [r for r in rows if r["http"] == 200]
        summary = {
            "n": len(rows),
            "total_audio_s": round(total_audio, 1),
            "errors": sum(1 for r in rows if r["http"] != 200),
            "latency_mean_s": round(statistics.mean(lat), 3),
            "latency_p50_s": round(statistics.median(lat), 3),
            "latency_p95_s": round(percentile(lat, 95), 3),
            "rtf_mean": round(statistics.mean(rtf), 4),
            "rtf_total_wallclock": round(sum(lat) / total_audio, 4),
            # Corpus-level WER: one concatenated reference vs one concatenated hypothesis.
            "wer_corpus": round(
                jiwer.wer(
                    " ".join(normalise(f["text"]) for f in fixtures),
                    " ".join(r["hyp_text"] for r in ok),
                ),
                4,
            )
            if ok
            else None,
            "wer_mean_per_utt": round(statistics.mean(wers), 4),
            "truncated_count": sum(1 for r in rows if r["truncated"]),
            "chars_ref": sum(r["ref_chars"] for r in rows),
            "chars_hyp": sum(r["hyp_chars"] for r in rows),
        }
        results["runs"][fmt] = {"summary": summary, "rows": rows}
        print(f"\n=== {args.label} / {fmt} ===")
        print(json.dumps(summary, indent=2), "\n")

    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
