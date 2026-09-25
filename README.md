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
- `ASR_BATCH`: the number of audio chunks evaluated together by the Transformers
  pipeline. The default is `4`; reduce it if VRAM is limited.
- `ASR_REQUEST_BATCH`: maximum number of compatible concurrent HTTP requests
  combined into one GPU pipeline call. The default is `4`.
- `ASR_BATCH_WAIT_MS`: maximum micro-batching window (default `15` ms). This tiny
  latency cost lets requests arriving together share a GPU batch.
- `ASR_QUEUE_SIZE`: maximum pending work. Once full, the API returns `429` with a
  `Retry-After` header instead of accumulating requests that will time out.
- `ASR_CHUNK_S` / `ASR_STRIDE_S`: chunk size and overlap. The 30 s / 5 s defaults
  preserve words around chunk boundaries; reducing the stride can save work but
  can also reduce transcription quality.
- `ASR_DECODE_WORKERS`: concurrent CPU audio decoding/resampling jobs. Decoding
  runs separately from the serialized GPU inference queue, so uploads can be
  prepared while another request is transcribed.
- `ASR_CPU_THREADS`: PyTorch CPU threads, primarily useful for CPU-only serving.
- `ASR_MAX_UPLOAD_MB`: rejects oversized input before it consumes unbounded RAM.

The common `json` and `text` paths do not request timestamps from Whisper. This
avoids timestamp decoding overhead without changing transcript text. Timestamp
decoding remains enabled for `verbose_json`, `srt`, and `vtt`, or when the client
explicitly sends `timestamp_granularities`.

## Performance and quality tuning

Measure changes on representative audio rather than optimizing only one clip.
Track median and tail latency, real-time factor (`processing seconds / audio
seconds`), peak VRAM, and word error rate (WER) against a fixed reference set.

Recommended order:

1. Keep `ASR_NUM_BEAMS=1`; beam search usually adds substantial latency.
2. Tune `ASR_REQUEST_BATCH` and `ASR_BATCH` together (`2`, `4`, then `8`) while
   watching VRAM. Re-test both throughput and tail latency after each change.
3. Keep the default 5 s stride for quality. Only lower it after a WER comparison.
4. Supply `language` when it is known to avoid language-detection work and reduce
   the chance of selecting the wrong language.
5. Leave `ASR_EMPTY_CACHE_ON_ERROR=0`. Emptying the CUDA allocator cache on every
   error causes allocator churn and slows the next request.

One scheduler owns inference in each process, but it dynamically batches compatible
concurrent requests (same language and timestamp mode). This avoids CUDA contention
and model duplication while using GPU parallelism instead of processing every HTTP
request serially. A request that times out while queued is discarded before
inference. Use one application process per GPU; do not use multiple Uvicorn workers
on the same device. Scale beyond the capacity of one tuned batch worker with one
replica per GPU and put a bounded load-balancing queue in front of the replicas.

For overload handling, clients should retry only `429` and transient `5xx`
responses, honor `Retry-After`, use exponential backoff with jitter, and impose a
retry limit. They should not blindly retry validation errors. Set the client timeout
above the measured P95 plus queue allowance, but keep `ASR_QUEUE_SIZE` low enough
that accepted requests can normally finish inside that timeout.

### OSS alternatives reviewed

The right engine depends on the deployment target; these are useful baselines to
benchmark against this Transformers implementation:

| Project | Best fit | Relevant trade-off |
| --- | --- | --- |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | NVIDIA GPU services and batched workloads | CTranslate2, quantization, batched transcription, and integrated VAD can substantially improve throughput; output can differ, so compare WER and timestamp behavior. |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | CPU, edge, Apple Silicon, and small standalone deployments | Native runtime and quantized models reduce the serving footprint; GPU-server throughput should be benchmarked on the actual target. |
| [WhisperX](https://github.com/m-bain/whisperX) | Accurate word alignment and speaker-aware workflows | Adds alignment and optional diarization stages, improving timestamp utility at the cost of extra models, time, and memory. |
| [Distil-Whisper](https://github.com/huggingface/distil-whisper) | English workloads where a smaller model is acceptable | Lower inference cost, but language coverage and accuracy characteristics differ from multilingual large-v3-turbo. |

For a drop-in service optimization, benchmark `faster-whisper` first. Keep this
implementation when broad Transformers compatibility and minimal backend-specific
code matter more. Do not compare only advertised speed: use the same model or an
explicit quality target, decoding settings, audio set, concurrency, and hardware.

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

Compatibility alias:

```text
POST /audio/transcriptions
```

### Health

```bash
curl http://127.0.0.1:8000/healthz
```
