# app.py — FastAPI ASR OpenAI-compatible — Whisper v3 Turbo
# Fixes:
# - évite de bloquer l'event-loop Uvicorn
# - force num_beams=1 pour éviter beam_search coûteux/bloqué
# - limite max_new_tokens
# - regroupe les requêtes concurrentes en micro-lots GPU
# - ajoute timeout applicatif
# - logs start/end/error
# - support json/text/srt/vtt/verbose_json
# - évite return_timestamps="sentence" invalide/douteux

import os
import io
import json
import time
import uuid
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# =============================================================================
# Config
# =============================================================================

MID = os.getenv("ASR_MODEL", "openai/whisper-large-v3-turbo")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

SR = int(os.getenv("ASR_SR", "16000"))
CHUNK = int(os.getenv("ASR_CHUNK_S", "30"))
BATCH = max(1, int(os.getenv("ASR_BATCH", "4")))

MAX_SEC = int(os.getenv("ASR_MAX_SEC", "5400"))  # 90 min max
MAX_UPLOAD_BYTES = int(os.getenv("ASR_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
ASR_TIMEOUT = int(os.getenv("ASR_TIMEOUT", "180"))
ASR_MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "256"))
ASR_STRIDE = int(os.getenv("ASR_STRIDE_S", "5"))

# Très important pour éviter des comportements de génération lents/bloqués.
ASR_NUM_BEAMS = int(os.getenv("ASR_NUM_BEAMS", "1"))

# Une seule inférence GPU à la fois par process.
ASR_DECODE_WORKERS = max(1, int(os.getenv("ASR_DECODE_WORKERS", "2")))
ASR_REQUEST_BATCH = max(1, int(os.getenv("ASR_REQUEST_BATCH", "4")))
ASR_BATCH_WAIT_MS = max(0, int(os.getenv("ASR_BATCH_WAIT_MS", "15")))
ASR_QUEUE_SIZE = max(1, int(os.getenv("ASR_QUEUE_SIZE", "64")))
ASR_EMPTY_CACHE_ON_ERROR = os.getenv("ASR_EMPTY_CACHE_ON_ERROR", "0") == "1"
ASR_CPU_THREADS = max(
    1, int(os.getenv("ASR_CPU_THREADS", str(min(8, os.cpu_count() or 1))))
)

AUTH_FILE = os.getenv("AUTH_FILE", "auth_tokens.txt")
ASR_API_TOKENS = os.getenv("ASR_API_TOKENS", os.getenv("API_TOKEN", ""))


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("api-audio2txt")


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="api-audio2txt", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Perf CPU/GPU
# =============================================================================

try:
    torch.set_num_threads(ASR_CPU_THREADS)
except Exception:
    pass

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# =============================================================================
# Auth
# =============================================================================

AUTH_TOKENS = set()


def _load_tokens(path: str = AUTH_FILE) -> set[str]:
    tokens = {
        token.strip()
        for token in ASR_API_TOKENS.replace("\n", ",").split(",")
        if token.strip()
    }

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            tokens.update(line.strip() for line in f if line.strip())

    return tokens


@app.on_event("startup")
async def _startup() -> None:
    global AUTH_TOKENS, asr_queue, asr_batch_task
    AUTH_TOKENS = _load_tokens()
    asr_queue = asyncio.Queue(maxsize=ASR_QUEUE_SIZE)
    asr_batch_task = asyncio.create_task(_asr_batch_loop())
    log.info("Loaded %d auth token(s)", len(AUTH_TOKENS))


@app.on_event("shutdown")
async def _shutdown() -> None:
    if asr_batch_task is not None:
        asr_batch_task.cancel()
        try:
            await asr_batch_task
        except asyncio.CancelledError:
            pass
    asr_executor.shutdown(wait=False, cancel_futures=True)
    decode_executor.shutdown(wait=False, cancel_futures=True)


def _check_auth(authorization: Optional[str]) -> None:
    if not AUTH_TOKENS:
        log.warning("Authentication is disabled because no API tokens are configured")
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ", 1)[-1].strip()

    if token not in AUTH_TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")


# =============================================================================
# Model
# =============================================================================

CUDA = torch.cuda.is_available()
DEV = 0 if CUDA else -1
DT = torch.float16 if CUDA else torch.float32

log.info("Loading ASR model: %s", MID)
log.info("CUDA=%s device=%s dtype=%s", CUDA, DEV, DT)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MID,
    torch_dtype=DT,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)

model.eval()

if CUDA:
    model.to("cuda")

processor = AutoProcessor.from_pretrained(MID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=CHUNK,
    stride_length_s=ASR_STRIDE,
    batch_size=BATCH,
    torch_dtype=DT,
    device=DEV,
)

# A single executor still owns the GPU, but the scheduler below combines several
# HTTP requests into one pipeline call. This raises throughput without launching
# competing CUDA kernels or duplicating the model in multiple Uvicorn workers.
asr_executor = ThreadPoolExecutor(max_workers=1)
decode_executor = ThreadPoolExecutor(max_workers=ASR_DECODE_WORKERS)
asr_queue: Optional[asyncio.Queue] = None
asr_batch_task: Optional[asyncio.Task] = None

log.info(
    "ASR ready: chunk=%ss stride=%ss batch=%s timeout=%ss max_new_tokens=%s "
    "num_beams=%s request_batch=%s batch_wait_ms=%s queue=%s "
    "decode_workers=%s cpu_threads=%s",
    CHUNK,
    ASR_STRIDE,
    BATCH,
    ASR_TIMEOUT,
    ASR_MAX_NEW_TOKENS,
    ASR_NUM_BEAMS,
    ASR_REQUEST_BATCH,
    ASR_BATCH_WAIT_MS,
    ASR_QUEUE_SIZE,
    ASR_DECODE_WORKERS,
    ASR_CPU_THREADS,
)


# =============================================================================
# Utils
# =============================================================================

def _timestamp_mode(raw: Optional[str], response_format: str = "json") -> Any:
    """
    OpenAI envoie parfois:
      timestamp_granularities[]=word
      timestamp_granularities=["word"]
      timestamp_granularities=word

    Transformers attend:
      return_timestamps=True
      ou return_timestamps="word"

    On évite "sentence", qui est douteux ici.
    """
    if not raw:
        # Timestamp decoding adds work and is unnecessary for the common JSON/text
        # path. Subtitle and verbose responses still get segment timestamps.
        return response_format in {"srt", "vtt", "verbose_json"}

    text = str(raw).strip()

    try:
        value = json.loads(text)

        if isinstance(value, str):
            return "word" if value == "word" else True

        if isinstance(value, list):
            return "word" if "word" in value else True

    except Exception:
        pass

    return "word" if text == "word" else True


def _segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result.get("chunks") or result.get("segments") or []


def _timestamp(segment: Dict[str, Any]) -> Tuple[float, float]:
    ts = segment.get("timestamp", [0, 0])

    if isinstance(ts, dict):
        return float(ts.get("start", 0) or 0), float(ts.get("end", 0) or 0)

    start = float(ts[0] or 0) if ts and ts[0] is not None else 0.0
    end = float(ts[1] or 0) if ts and len(ts) > 1 and ts[1] is not None else 0.0

    return start, end


def _format_time(seconds: float, srt: bool = True) -> str:
    seconds = max(float(seconds), 0.0)

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    sep = "," if srt else "."
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def _to_srt(segments: List[Dict[str, Any]]) -> str:
    out: List[str] = []

    for i, segment in enumerate(segments, 1):
        start, end = _timestamp(segment)
        text = segment.get("text", "").strip()

        out.extend(
            [
                str(i),
                f"{_format_time(start, True)} --> {_format_time(end, True)}",
                text,
                "",
            ]
        )

    return "\n".join(out).strip()


def _to_vtt(segments: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]

    for segment in segments:
        start, end = _timestamp(segment)
        text = segment.get("text", "").strip()

        out.extend(
            [
                f"{_format_time(start, False)} --> {_format_time(end, False)}",
                text,
                "",
            ]
        )

    return "\n".join(out).strip()


def _safe_cuda_cleanup() -> None:
    # empty_cache() forces allocator churn and normally makes the next request
    # slower. It remains available as an escape hatch for deployments that see
    # CUDA OOM errors.
    if CUDA and ASR_EMPTY_CACHE_ON_ERROR:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _load_audio_to_array(data: bytes) -> Tuple[Any, float]:
    try:
        buffer = io.BytesIO(data)
        buffer.seek(0)

        wav, sr = torchaudio.load(buffer)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {exc}",
        )

    if wav.dim() != 2 or wav.size(0) < 1:
        raise HTTPException(status_code=400, detail="Invalid audio tensor")

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    if MAX_SEC and wav.size(1) > SR * MAX_SEC:
        wav = wav[:, : SR * MAX_SEC]

    duration = wav.size(1) / float(SR)
    array = wav.squeeze(0).contiguous().numpy()

    return array, duration


async def _read_upload(file: UploadFile) -> bytes:
    """Read incrementally so an oversized upload is rejected before it is copied."""
    chunks: List[bytes] = []
    size = 0

    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if MAX_UPLOAD_BYTES and size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio file is too large")
        chunks.append(chunk)

    return b"".join(chunks)


async def _decode_audio(data: bytes) -> Tuple[Any, float]:
    """Keep container decoding and resampling off Uvicorn's event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(decode_executor, _load_audio_to_array, data)


def _run_asr_batch_sync(
    arrays: List[Any],
    language: Optional[str],
    timestamp_mode: Any,
) -> List[Dict[str, Any]]:
    generate_kwargs: Dict[str, Any] = {
        "num_beams": ASR_NUM_BEAMS,
        "do_sample": False,
        "max_new_tokens": ASR_MAX_NEW_TOKENS,
    }

    if language:
        generate_kwargs["language"] = language

    input_payloads = [
        {"array": array, "sampling_rate": SR}
        for array in arrays
    ]

    with torch.inference_mode():
        results = pipe(
            input_payloads,
            return_timestamps=timestamp_mode,
            generate_kwargs=generate_kwargs,
        )

    if not isinstance(results, list) or len(results) != len(arrays):
        raise RuntimeError(
            f"Unexpected ASR batch result: {type(results)} "
            f"({len(results) if isinstance(results, list) else 'n/a'} results)"
        )
    if not all(isinstance(result, dict) for result in results):
        raise RuntimeError("ASR batch contains a non-dictionary result")

    return results


async def _asr_batch_loop() -> None:
    """Collect a short burst of requests and infer compatible jobs together."""
    assert asr_queue is not None
    loop = asyncio.get_running_loop()

    while True:
        first = await asr_queue.get()
        jobs = [first]

        if ASR_BATCH_WAIT_MS > 0:
            await asyncio.sleep(ASR_BATCH_WAIT_MS / 1000.0)

        while len(jobs) < ASR_REQUEST_BATCH:
            try:
                jobs.append(asr_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # A single pipeline invocation must share generation options. Split a
        # micro-batch only when clients request different languages/timestamps.
        groups: Dict[Tuple[Optional[str], Any], List[Any]] = {}
        for job in jobs:
            _array, language, timestamp_mode, future = job
            if not future.cancelled():
                groups.setdefault((language, timestamp_mode), []).append(job)

        try:
            for (language, timestamp_mode), group in groups.items():
                arrays = [job[0] for job in group]
                try:
                    results = await loop.run_in_executor(
                        asr_executor,
                        _run_asr_batch_sync,
                        arrays,
                        language,
                        timestamp_mode,
                    )
                except Exception as exc:
                    for *_unused, future in group:
                        if not future.done():
                            future.set_exception(exc)
                else:
                    for job, result in zip(group, results):
                        future = job[3]
                        if not future.done():
                            future.set_result(result)
        finally:
            for _job in jobs:
                asr_queue.task_done()


async def _run_asr_async(
    array: Any,
    language: Optional[str],
    timestamp_mode: Any,
) -> Dict[str, Any]:
    global asr_queue, asr_batch_task
    loop = asyncio.get_running_loop()
    if asr_queue is None:
        # Supports direct invocation in tests and non-ASGI embedding.
        asr_queue = asyncio.Queue(maxsize=ASR_QUEUE_SIZE)
        asr_batch_task = asyncio.create_task(_asr_batch_loop())

    future = loop.create_future()
    try:
        asr_queue.put_nowait((array, language, timestamp_mode, future))
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=429,
            detail="ASR queue is full; retry later",
            headers={"Retry-After": "1"},
        )

    try:
        return await asyncio.wait_for(future, timeout=ASR_TIMEOUT)
    except asyncio.TimeoutError:
        future.cancel()
        raise


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "api-audio2txt is running",
        "docs": "/docs",
        "health": "/healthz",
    }


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": MID,
        "cuda": CUDA,
        "device": DEV,
        "queue_depth": asr_queue.qsize() if asr_queue is not None else 0,
        "queue_capacity": ASR_QUEUE_SIZE,
        "request_batch_size": ASR_REQUEST_BATCH,
    }


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    model_name: str = Form(None, alias="model"),
    language: str = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: str = Form(None),
):
    _check_auth(authorization)

    if response_format not in {"json", "verbose_json", "text", "srt", "vtt"}:
        raise HTTPException(status_code=400, detail="Invalid response format specified")

    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()

    log.info(
        "[%s] STT start filename=%s model=%s language=%s response_format=%s",
        request_id,
        file.filename,
        model_name,
        language,
        response_format,
    )

    try:
        data = await _read_upload(file)

        if not data:
            raise HTTPException(status_code=400, detail="Empty audio file")

        array, duration = await _decode_audio(data)

        log.info(
            "[%s] audio loaded duration=%.2fs bytes=%d",
            request_id,
            duration,
            len(data),
        )

        timestamp_mode = _timestamp_mode(timestamp_granularities, response_format)

        result = await _run_asr_async(
            array=array,
            language=language,
            timestamp_mode=timestamp_mode,
        )

        text = result.get("text", "") or ""
        segments = _segments(result)

        elapsed = time.perf_counter() - started

        log.info(
            "[%s] STT done elapsed=%.2fs duration=%.2fs text_chars=%d segments=%d",
            request_id,
            elapsed,
            duration,
            len(text),
            len(segments),
        )

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        log.error("[%s] STT timeout after %.2fs", request_id, elapsed)
        _safe_cuda_cleanup()
        raise HTTPException(
            status_code=504,
            detail=f"ASR timeout after {ASR_TIMEOUT}s",
        )

    except HTTPException:
        raise

    except Exception as exc:
        elapsed = time.perf_counter() - started
        log.exception("[%s] STT error after %.2fs: %s", request_id, elapsed, exc)
        _safe_cuda_cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"ASR pipeline error: {exc}",
        )

    finally:
        try:
            await file.close()
        except Exception:
            pass

    if response_format == "json":
        return JSONResponse({"text": text})

    if response_format == "verbose_json":
        return JSONResponse(
            {
                "text": text,
                "segments": segments,
                "duration": duration,
            }
        )

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format == "srt":
        return PlainTextResponse(_to_srt(segments))

    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(segments))

    # response_format is validated before decoding/inference.
    raise AssertionError("unreachable response format")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        workers=1,
        reload=False,
    )
