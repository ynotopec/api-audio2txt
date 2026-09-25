# api-audio2txt

Minimal OpenAI-compatible speech-to-text API powered by FastAPI and Whisper.

## Quick start

```bash
./install.sh
source run.sh 0.0.0.0 8000
```

`install.sh` is idempotent and upgrade-safe: it creates or reuses the project virtual environment at `~/venv/api-audio2txt`, installs `uv` when needed, upgrades dependencies from `requirements.txt`, and creates `.env` from `.env.example` when missing.

## Configuration

Copy and edit the example environment file if you do not run `install.sh` first:

```bash
cp .env.example .env
```

Important settings:

- `ASR_API_TOKENS`: comma-separated bearer tokens for API access.
- `ASR_MODEL`: Hugging Face model id. Defaults to `openai/whisper-large-v3-turbo`.
- `HOST` / `PORT`: default bind address used when `run.sh` arguments are omitted.
- `ASR_BATCH`, `ASR_CHUNK_S`, `ASR_WORKERS`: tuning knobs for GPUs such as H100 or DGX Spark systems.
- `ASR_VAD`: energy-based silence trimming, on by default. See [Performance](#performance).

## Performance

This revision changes what is decoded, not which model runs. `openai/whisper-large-v3-turbo`
is unchanged and the OpenAI contract is identical, so the change is reversible by
reverting `app.py`.

### What changed

1. **The decode budget scales with audio length.** The previous flat
   `max_new_tokens=256` truncated any utterance with more speech than ~30 s and
   returned a silently shortened transcript with HTTP 200. The budget is now
   derived per request from the duration and clamped to Whisper's 448-position
   decoder limit.
2. **Timestamps are decoded only when the response carries them.** `json` and
   `text` used to pay for timestamp tokens they then discarded.
3. **Energy-based VAD** trims silence before inference, so the encoder spends its
   budget on speech. This is where most of the WER gain comes from. Two details
   make it safe:
   - A short silence is re-inserted between kept regions. Splicing speech
     back-to-back removes the phrase boundary the decoder relies on; the naive
     splice measured **+40 % WER** on `verbose_json` before this was added.
   - Segment timestamps are mapped back onto the original timeline, so `srt`
     and `vtt` cues still line up with the source audio.
4. **Bounded worker pool** replaces a single global `asyncio.Lock`, so concurrent
   requests overlap instead of queueing behind one another. A semaphore still
   bounds in-flight GPU work, and the API returns 503 past `ASR_MAX_QUEUE`.
5. **ffmpeg fallback** for decoding, so mp3/m4a/ogg/webm work instead of 400.
6. **Warmup inference at startup**, so the first request does not pay cuDNN
   autotune.
7. **CUDA cleanup only on genuine faults.** It ran on every error path, forcing a
   device sync and allocator flush that slowed the next request.

### Measured results

`whisper-large-v3-turbo`, fp16, NVIDIA GB10, `ASR_WORKERS=2`. Fixtures are
LibriSpeech `clean` validation utterances; WER is computed on normalised text
(lowercase, punctuation stripped). Requests were interleaved between the two
revisions so contention from other services on the shared GPU hit both equally.

Long-form fixtures (92 s / 189 s / 313 s, with 2.5-4 s of silence between
utterances, so roughly half the file is non-speech), 3 repeats, 9 requests per
side:

| Metric | `json` | `verbose_json` |
|---|---|---|
| Mean latency | **-41.9 %** | **-23.4 %** |
| p95 latency | **-47.5 %** | **-29.8 %** |
| RTF (wall clock) | **-41.7 %** | **-23.4 %** |
| Corpus WER | **6.38 % -> 4.52 %** | **6.38 % -> 4.18 %** |

Short fixtures (5-18 s, little silence), 4 repeats, 48 requests per side, run
on an otherwise idle host:

| Metric | `json` | `verbose_json` |
|---|---|---|
| Mean latency | -6.2 % | -0.5 % |
| p95 latency | -9.7 % | -2.9 % |
| RTF (wall clock) | -6.2 % | -0.4 % |
| Corpus WER | 5.94 % (unchanged) | 5.94 % -> 6.27 % |

Short clean audio has almost no silence to trim, so the VAD correctly declines
to act and the only gain left is skipping timestamp tokens. That helps `json`
noticeably and `verbose_json` barely, because `verbose_json` still has to decode
the timestamps it returns.

`verbose_json` on short files is a small quality regression, +0.33 points of WER,
from the VAD reshaping a clip that had little silence to remove. It is a real
effect and not measurement noise: 48 requests per side reproduced it. The long
fixtures, where the VAD has real silence to work with, improve by 2.2 points. If
your workload is short clips needing word timings, `ASR_VAD=0` is the honest
setting.

An earlier run of the same comparison, taken while another benchmark was hitting
the same GPU, reported +62.9 % for `json` — a measurement artefact. Re-running it
alone produced the -6.2 % above. Treat any single-run latency figure on a shared
GPU with suspicion; the interleaved pairing only removes drift when nothing else
is competing for the device.

Absolute numbers for context: the long fixtures transcribed at RTF 0.022 before
and 0.013 after, i.e. roughly 75x faster than real time.

### Concurrency

Same host, short fixtures, 12 requests per client, `ASR_WORKERS=2` on the
optimized revision. `parallelism` is the ratio of summed per-request latency to
wall clock: 1.0 means requests were strictly serialised, N means N were
genuinely in flight.

| Clients | Wall clock (base -> opt) | Parallelism (base -> opt) | Throughput x realtime |
|---|---|---|---|
| 1 | 8.7 s -> 2.9 s (**-67 %**) | 1.00 -> 1.00 | 8.1x -> 24.2x |
| 2 | 16.2 s -> 5.5 s (**-66 %**) | 1.96 -> 2.00 | 7.9x -> 23.4x |
| 4 | 42.1 s -> 12.6 s (**-70 %**) | 3.85 -> 3.91 | 9.4x -> 31.5x |
| 8 | 98.4 s -> 26.2 s (**-73 %**) | 7.68 -> 7.75 | 10.5x -> 39.5x |

Both revisions already overlapped once a client sent more than one request, so
the concurrency change shows up as throughput per wall second rather than as a
parallelism jump. The single-request latency is where the worker pool and warmup
show: 0.72 s -> 0.24 s at C=1.

### Reproducing

```bash
# Unit tests for the pure helpers. No GPU, no model download: the helpers are
# extracted from app.py by AST and exercised in a bare namespace.
python tests/test_helpers.py

# Serve the old and new revision side by side, then interleave requests.
python bench/bench_ab.py \
  --a http://127.0.0.1:8100 --b http://127.0.0.1:8101 \
  --token "$ASR_API_TOKENS" --repeats 3 --out ab.json

python bench/bench_concurrency.py \
  --base-url http://127.0.0.1:8101 --token "$ASR_API_TOKENS" --levels 1,2,4,8
```

### Caveats

- The host GPU is shared with other resident services, so absolute latencies
  include contention. Only the interleaved A/B deltas should be trusted.
- Changing the audio actually fed to the model (here, the VAD trim) changes the
  timestamp prompt, so it changes the transcript text too. A trim is not a
  pure filter: the same file can yield a different number of segments before and
  after. Text-level comparisons are only meaningful against a fixed server
  build, which is why the table above pairs one baseline process against one
  optimized process and runs them interleaved.
- The fixtures are English read speech. Gains on conversational or noisy audio
  should be re-measured; the VAD is the component most sensitive to that shift.
- `ASR_CONDITION_ON_PREV=0` suppresses Whisper's repetition loops on long noisy
  audio, but measured **+29 % WER** on these clean fixtures. It is off by default
  for that reason.
- `ASR_WORKERS` above 1 overlaps requests but they still contend for the same
  GPU. Raise it only with memory headroom.

## Run with systemd

Use `bash -lc` so the source-compatible launcher can activate the virtual environment:

```ini
[Service]
WorkingDirectory=/workspace/api-audio2txt
ExecStart=/bin/bash -lc 'source run.sh 0.0.0.0 8000'
Restart=always
```

## API

### Transcribe audio

OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer ${ASR_TOKEN}" \
  -F file=@audio.mp3 \
  -F model=whisper-1 \
  -F response_format=json
```

Popular response formats are supported: `json`, `text`, `srt`, `vtt`, and `verbose_json`.

`verbose_json` adds `duration` (the original upload, not the VAD-trimmed audio),
`audio_truncated` (whether `ASR_MAX_SEC` clipped the input) and
`vad_removed_silence`. Segment timestamps always refer to the original timeline,
including when VAD trimmed the audio.

Compatibility alias:

```text
POST /audio/transcriptions
```

### Health

```bash
curl http://127.0.0.1:8000/healthz
```
