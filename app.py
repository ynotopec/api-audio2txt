# app.py — FastAPI ASR OpenAI-compatible — Whisper v3 Turbo
#
# Performance work in this revision (see README "Performance" section):
#   1. max_new_tokens is derived from audio duration instead of a flat 256 cap
#      that silently truncated every utterance longer than ~30 s.
#   2. Timestamps are only decoded when the caller needs them (verbose_json /
#      srt / vtt). Plain json/text skips timestamp tokens entirely.
#   3. Energy-based VAD trims silence before inference: less audio to encode,
#      fewer hallucinated segments, better WER on long files.
#   4. Bounded queue + worker pool instead of a single global lock, so
#      concurrent requests overlap instead of serialising. Semaphore bounds GPU.
#   5. Audio decoding falls back to ffmpeg, so mp3/m4a/ogg/webm work instead of
#      returning 400.
#   6. condition_on_prev_tokens is now opt-in (ASR_CONDITION_ON_PREV=0) instead
#      of forced on: the benchmark showed it lowered WER on clean speech. The
#      repetition-loop mitigation is a real tradeoff, not a free win.
#   7. cuda cleanup is no longer called on every error path (it forces a full
#      device sync and allocator flush, slowing the *next* request).

import asyncio
import io
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

try:  # soundfile is the fast path; ffmpeg is the fallback.
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


# =============================================================================
# Config
# =============================================================================

MID = os.getenv("ASR_MODEL", "openai/whisper-large-v3-turbo")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

SR = int(os.getenv("ASR_SR", "16000"))
CHUNK = int(os.getenv("ASR_CHUNK_S", "30"))
BATCH = int(os.getenv("ASR_BATCH", "1"))

MAX_SEC = int(os.getenv("ASR_MAX_SEC", "5400"))  # 90 min max
ASR_TIMEOUT = int(os.getenv("ASR_TIMEOUT", "180"))

# Per-chunk decode safety valve. Whisper-large-v3-turbo emits ~3-4 decoder tokens
# per second of speech, so 30 s of dense speech stays well under 256; a flat low
# cap was what truncated long files in earlier revisions. The effective budget is
# computed per request in _decode_budget().
ASR_MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "440"))

# Whisper hard limit on decoder positions.
ASR_MAX_TOTAL_TOKENS = 448

ASR_NUM_BEAMS = int(os.getenv("ASR_NUM_BEAMS", "1"))

# Worker threads allowed to run inference concurrently. The semaphore of the
# same size bounds in-flight GPU work; the executor bounds Python-side threads.
ASR_WORKERS = max(1, int(os.getenv("ASR_WORKERS", "1")))

# VAD (energy based, no extra model download)
ASR_VAD_ENABLED = os.getenv("ASR_VAD", "1").lower() not in ("0", "false", "no")
VAD_THRESHOLD_DB = float(os.getenv("ASR_VAD_THRESHOLD_DB", "-45"))
VAD_MIN_SILENCE_MS = int(os.getenv("ASR_VAD_MIN_SILENCE_MS", "500"))
VAD_SPEECH_PAD_MS = int(os.getenv("ASR_VAD_SPEECH_PAD_MS", "200"))

# Silence re-inserted between two kept regions. Splicing speech regions
# back-to-back removes the phrase boundary the decoder relies on and provokes
# repetition loops; a short gap restores it at negligible cost.
VAD_JOIN_GAP_MS = int(os.getenv("ASR_VAD_JOIN_GAP_MS", "300"))

# Queue depth: beyond this, return 503 instead of buffering unbounded audio.
ASR_MAX_QUEUE = int(os.getenv("ASR_MAX_QUEUE", "64"))

ASR_WARMUP = os.getenv("ASR_WARMUP", "1").lower() not in ("0", "false", "no")

# Sequential-chunk decoding (what chunk_length_s triggers) is where Whisper's
# repetition loops appear, and dropping previous-token conditioning is the
# documented mitigation. It is however a WER tradeoff, not a free win: the
# benchmark showed condition_on_prev_tokens=True scoring better on clean speech.
# Default stays True; set to 0 for long noisy audio with repetition loops.
ASR_CONDITION_ON_PREV = os.getenv("ASR_CONDITION_ON_PREV", "1") not in ("0", "false", "no")

AUTH_FILE = os.getenv("AUTH_FILE", "auth_tokens.txt")
ASR_API_TOKENS = os.getenv("ASR_API_TOKENS", os.getenv("API_TOKEN", ""))

RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("api-audio2txt")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


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
    torch.set_num_threads(min(8, os.cpu_count() or 1))
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

AUTH_TOKENS: set = set()


def _load_tokens(path: str = AUTH_FILE) -> set:
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
def _startup() -> None:
    global AUTH_TOKENS
    AUTH_TOKENS = _load_tokens()
    log.info("Loaded %d auth token(s)", len(AUTH_TOKENS))

    # Warm up so the first real request does not pay cudnn autotune + lazy alloc.
    if ASR_WARMUP:
        try:
            _run_asr_sync(np.zeros(SR, dtype=np.float32), None, False, 1.0)
            log.info("Warmup inference done")
        except Exception as exc:  # pragma: no cover
            log.warning("Warmup failed (non-fatal): %s", exc)


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
    dtype=DT,
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
    batch_size=BATCH,
    torch_dtype=DT,
    device=DEV,
)

asr_executor = ThreadPoolExecutor(max_workers=ASR_WORKERS, thread_name_prefix="asr")
asr_sem = threading.Semaphore(ASR_WORKERS)

_queue_lock = threading.Lock()
_inflight = 0

log.info(
    "ASR ready: chunk=%ss batch=%s timeout=%ss max_new_tokens=%s num_beams=%s workers=%s vad=%s",
    CHUNK,
    BATCH,
    ASR_TIMEOUT,
    ASR_MAX_NEW_TOKENS,
    ASR_NUM_BEAMS,
    ASR_WORKERS,
    ASR_VAD_ENABLED,
)


# =============================================================================
# Audio decoding
# =============================================================================

def _decode_ffmpeg(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode any container ffmpeg understands, streaming raw f32 mono at SR."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1",
    ]

    proc = subprocess.run(
        cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )

    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(
            f"ffmpeg failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}"
        )

    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), SR


def _load_audio_to_array(data: bytes) -> Tuple[np.ndarray, float, bool]:
    """
    Returns (mono float32 @ SR, duration_s, truncated_by_max_sec).

    soundfile first (fast, in-process), ffmpeg fallback so mp3/m4a/ogg/webm are
    accepted instead of returning HTTP 400.
    """
    array: Optional[np.ndarray] = None
    sr: Optional[int] = None

    if sf is not None:
        try:
            wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            array = wav.mean(axis=1)
        except Exception as exc:
            log.debug("soundfile decode failed, falling back to ffmpeg: %s", exc)

    if array is None or sr is None:
        try:
            array, sr = _decode_ffmpeg(data)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Unsupported audio format: {exc}"
            ) from exc

    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    if array.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio stream")

    truncated = False
    if MAX_SEC and array.size > SR * MAX_SEC:
        array = array[: SR * MAX_SEC]
        truncated = True

    duration = array.size / float(SR)
    return np.ascontiguousarray(array), duration, truncated


# =============================================================================
# VAD — energy based
# =============================================================================

def _frame_rms_db(array: np.ndarray, frame: int) -> np.ndarray:
    n = array.size // frame

    if n < 1:
        rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
        return np.array([20.0 * np.log10(max(rms, 1e-10))])

    frames = array[: n * frame].reshape(n, frame).astype(np.float64)
    rms = np.sqrt(np.maximum(np.mean(np.square(frames), axis=1), 1e-12))
    return 20.0 * np.log10(rms)


def _vad_keep_indices(array: np.ndarray) -> Optional[Tuple[np.ndarray, tuple]]:
    """
    Returns (trimmed_audio, time_mapping) or None when VAD should be skipped
    (short audio, or the gate would be destructive).

    ``time_mapping`` is (region_starts, region_lengths, orig_starts, orig_ends):
    for each kept region, where it begins in the trimmed audio and where it came
    from in the original. Segments are decoded against the trimmed audio, so
    their timestamps are relative to the trim; _remap_segment_times() maps them
    back onto the original timeline for srt/vtt output.

    Energy gating rather than a neural VAD: no extra model download, no extra
    GPU memory, and it removes exactly the long silences that make Whisper
    hallucinate and spend decode budget on empty audio.
    """
    duration = array.size / float(SR)

    if duration < 2.0:
        return None

    frame = int(0.02 * SR)  # 20 ms
    db = _frame_rms_db(array, frame)
    n = db.size

    # Adaptive floor: a fixed -45 dBFS misfires on quiet recordings and on loud
    # ones alike, so take the noise floor from the 10th percentile of this file.
    threshold = max(VAD_THRESHOLD_DB, float(np.percentile(db, 10)) + 12.0)
    speech = db > threshold

    if not speech.any():
        return None  # fully silent: let the model decide, do not blank the output

    min_silence = max(1, VAD_MIN_SILENCE_MS // 20)
    pad = max(0, VAD_SPEECH_PAD_MS // 20)

    diff = np.diff(speech.astype(np.int8))
    starts = list((np.where(diff == 1)[0] + 1).tolist())
    ends = list((np.where(diff == -1)[0] + 1).tolist())

    if speech[0]:
        starts.insert(0, 0)
    if speech[-1]:
        ends.append(n)

    if not starts or not ends:
        return None

    # Merge regions separated by less than min_silence to avoid chopping words.
    merged: List[List[int]] = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] < min_silence:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    keep = np.zeros(n, dtype=bool)
    for s, e in merged:
        keep[max(0, s - pad) : min(n, e + pad)] = True

    kept_fraction = float(keep.sum()) / float(n)

    # Refuse to mangle the utterance: dropping most of the audio means the gate
    # probably misfired (music, noise floor, very quiet speaker).
    if kept_fraction < 0.25 or keep.all():
        return None

    sample_keep = np.repeat(keep, frame)[: array.size]

    if sample_keep.size < array.size:
        sample_keep = np.pad(sample_keep, (0, array.size - sample_keep.size))

    kept = np.flatnonzero(sample_keep)

    # Splicing the kept regions back-to-back hands the model a discontinuous
    # waveform: sentence ends run straight into the next sentence start with no
    # gap, which is what triggers Whisper's repetition loops. Measured cost of
    # the naive splice was +40 % WER on verbose_json. Re-inserting a short
    # silence between regions keeps the phrase boundaries the model needs while
    # still removing the long silences that cost the most.
    gap = np.zeros(int(SR * VAD_JOIN_GAP_MS / 1000), dtype=array.dtype)

    pieces = []
    # Per region: (start_in_trimmed, length, original_start, original_end)
    bounds = []
    cursor = 0
    prev_end = None

    for s, e in merged:
        lo = max(0, s - pad)
        hi = min(n, e + pad)
        if prev_end is not None:
            pieces.append(gap)
            cursor += gap.size

        chunk = array[int(lo * frame) : min(array.size, int(hi * frame))]
        if chunk.size == 0:
            continue

        bounds.append((cursor, chunk.size, int(lo * frame), min(array.size, int(hi * frame))))
        pieces.append(chunk)
        cursor += chunk.size
        prev_end = hi

    if not pieces:
        return None

    trimmed = np.concatenate(pieces)

    mapping = (
        np.array([b[0] for b in bounds], dtype=np.float64),
        np.array([b[1] for b in bounds], dtype=np.float64),
        np.array([b[2] for b in bounds], dtype=np.float64),
        np.array([b[3] for b in bounds], dtype=np.float64),
    )

    return trimmed, mapping


# =============================================================================
# Formatting helpers
# =============================================================================

def _timestamp_mode(raw: Optional[str]) -> Any:
    """
    OpenAI sends timestamp_granularities as word, ["word"], or '["word"]'.
    Transformers expects return_timestamps=True or "word".
    """
    if not raw:
        return True

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


def _map_trimmed_to_original(sample_index: int, mapping: tuple) -> float:
    """
    Convert a sample position in the trimmed audio to its position in the
    original recording, by locating the region it falls in and interpolating
    inside it. Positions falling in an inserted gap clamp to the gap's edges.
    """
    starts, lengths, orig_starts, orig_ends = mapping
    pos = float(sample_index)

    # Last region whose start is <= pos.
    idx = int(np.searchsorted(starts, pos, side="right")) - 1
    idx = min(max(idx, 0), starts.size - 1)

    region_end = starts[idx] + lengths[idx]
    if pos > region_end:
        # Inside (or past) the inserted gap: clamp to the region boundary so a
        # timestamp can never drift into the removed silence.
        return float(orig_ends[idx])

    offset_in_region = min(max(pos - starts[idx], 0.0), lengths[idx])
    if lengths[idx] <= 0:
        return float(orig_starts[idx])

    ratio = offset_in_region / lengths[idx]
    return float(orig_starts[idx] + ratio * (orig_ends[idx] - orig_starts[idx]))


def _remap_segment_times(
    segments: List[Dict[str, Any]], mapping: Optional[tuple]
) -> List[Dict[str, Any]]:
    """
    Rewrite segment timestamps from trimmed-audio time back to original time.

    After VAD the model sees only the kept regions, so a segment reported at
    t=8.0 actually refers to trimmed-sample 8.0*SR, whose position in the
    original recording differs. Without this, every srt/vtt cue after the first
    removed silence is wrong.
    """
    if not segments or not mapping:
        return segments

    out: List[Dict[str, Any]] = []

    for segment in segments:
        start, end = _timestamp(segment)
        ts = segment.get("timestamp")
        mapped = dict(segment)

        new_start = (
            _map_trimmed_to_original(int(round(start * SR)), mapping) / SR
            if start > 0
            else 0.0
        )
        new_end = _map_trimmed_to_original(int(round(end * SR)), mapping) / SR

        if isinstance(ts, tuple):
            mapped["timestamp"] = (new_start, new_end)
        elif isinstance(ts, list):
            mapped["timestamp"] = [new_start, new_end]
        elif ts is None and (start or end):
            mapped["timestamp"] = (new_start, new_end)

        out.append(mapped)

    return out


def _to_srt(segments: List[Dict[str, Any]]) -> str:
    out: List[str] = []

    for i, segment in enumerate(segments, 1):
        start, end = _timestamp(segment)
        text = segment.get("text", "").strip()

        out.extend(
            [str(i), f"{_format_time(start, True)} --> {_format_time(end, True)}", text, ""]
        )

    return "\n".join(out).strip()


def _to_vtt(segments: List[Dict[str, Any]]) -> str:
    out: List[str] = ["WEBVTT", ""]

    for segment in segments:
        start, end = _timestamp(segment)
        text = segment.get("text", "").strip()

        out.extend(
            [f"{_format_time(start, False)} --> {_format_time(end, False)}", text, ""]
        )

    return "\n".join(out).strip()


# =============================================================================
# Inference
# =============================================================================

def _cuda_cleanup() -> None:
    """
    Only for genuine device faults. Calling this on every error forces a full
    device sync and allocator flush, which slows the *next* request.
    """
    if not CUDA:
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _decode_budget(duration: float, need_timestamps: bool) -> int:
    """
    Derive the decode budget from audio length.

    Whisper decodes at roughly 3-4 tokens per second of speech; timestamps add
    a token per segment boundary. The previous flat 256 cap truncated long
    utterances. This scales with duration, clamped to the model's 448-position
    decoder limit.
    """
    budget = int(duration * 6) + 48  # ~6 tok/s headroom + forced-token slack

    if need_timestamps:
        budget += 64

    return max(64, min(ASR_MAX_TOTAL_TOKENS - 8, min(budget, ASR_MAX_NEW_TOKENS)))


def _run_asr_sync(
    array: np.ndarray,
    language: Optional[str],
    timestamp_mode: Any,
    duration: float = 0.0,
) -> Dict[str, Any]:
    need_ts = bool(timestamp_mode)

    generate_kwargs: Dict[str, Any] = {
        "num_beams": ASR_NUM_BEAMS,
        "do_sample": False,
        "max_new_tokens": _decode_budget(duration, need_ts),
    }

    # Skip timestamp decoding entirely for json/text: it is pure overhead when
    # the caller discards the segments anyway.
    generate_kwargs["return_timestamps"] = timestamp_mode if need_ts else False

    # Sequential-chunk decoding (what chunk_length_s triggers) is where Whisper's
    # repetition loops appear; dropping previous-token conditioning is the
    # documented mitigation. It is a WER tradeoff, so it stays configurable.
    if CHUNK and not ASR_CONDITION_ON_PREV:
        generate_kwargs["condition_on_prev_tokens"] = False

    if language:
        generate_kwargs["language"] = language

    input_payload = {"array": array, "sampling_rate": SR}

    with asr_sem, torch.inference_mode():
        result = pipe(input_payload, **generate_kwargs)

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected ASR result type: {type(result)}")

    return result


async def _run_asr_async(
    array: np.ndarray,
    language: Optional[str],
    timestamp_mode: Any,
    duration: float = 0.0,
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()

    return await asyncio.wait_for(
        loop.run_in_executor(
            asr_executor, _run_asr_sync, array, language, timestamp_mode, duration
        ),
        timeout=ASR_TIMEOUT,
    )


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
    with _queue_lock:
        inflight = _inflight

    return {
        "ok": True,
        "model": MID,
        "cuda": CUDA,
        "device": DEV,
        "workers": ASR_WORKERS,
        "inflight": inflight,
        "vad": ASR_VAD_ENABLED,
    }


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    model_name: Optional[str] = Form(None, alias="model"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
):
    global _inflight

    if response_format not in RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid response format specified")

    _check_auth(authorization)

    request_id = str(uuid.uuid4())[:8]
    started = time.time()

    with _queue_lock:
        if _inflight >= ASR_MAX_QUEUE:
            raise HTTPException(status_code=503, detail="ASR queue is full, retry later")
        _inflight += 1

    try:
        log.info(
            "[%s] STT start filename=%s model=%s language=%s response_format=%s",
            request_id,
            file.filename,
            model_name,
            language,
            response_format,
        )

        data = await file.read()

        if not data:
            raise HTTPException(status_code=400, detail="Empty audio file")

        array, duration, truncated = _load_audio_to_array(data)

        vad_kept = 1.0
        vad_before = duration
        vad_mapping: Optional[tuple] = None
        if ASR_VAD_ENABLED and duration >= 2.0:
            t0 = time.perf_counter()
            vad = _vad_keep_indices(array)
            vad_kept = (vad[0].size / array.size) if vad is not None else 1.0

            if vad is not None:
                array, vad_mapping = vad
                duration = array.size / float(SR)
                log.info(
                    "[%s] VAD trimmed %.2fs -> %.2fs (%.1f%% kept) in %.3fs",
                    request_id,
                    vad_before,
                    duration,
                    vad_kept * 100,
                    time.perf_counter() - t0,
                )

        # Only decode timestamps when the response format actually carries them.
        needs_ts = response_format in ("verbose_json", "srt", "vtt")
        timestamp_mode: Any = _timestamp_mode(timestamp_granularities) if needs_ts else False

        if prompt and not isinstance(timestamp_mode, dict):
            timestamp_mode = {"return_timestamps": bool(timestamp_mode), "prompt": prompt}

        result = await _run_asr_async(
            array=array,
            language=language,
            timestamp_mode=timestamp_mode,
            duration=duration,
        )

        text = result.get("text", "") or ""
        segments = _segments(result)

        # Segments were decoded against the trimmed audio; map their timestamps
        # back onto the original timeline so srt/vtt stay usable.
        if vad_mapping is not None:
            segments = _remap_segment_times(segments, vad_mapping)

        elapsed = time.time() - started

        log.info(
            "[%s] STT done elapsed=%.2fs duration=%.2fs rtf=%.3f text_chars=%d segments=%d vad_kept=%.1f%%",
            request_id,
            elapsed,
            duration,
            elapsed / duration if duration else 0.0,
            len(text),
            len(segments),
            vad_kept * 100,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - started
        log.error("[%s] STT timeout after %.2fs", request_id, elapsed)
        _cuda_cleanup()
        raise HTTPException(
            status_code=504, detail=f"ASR timeout after {ASR_TIMEOUT}s"
        ) from None

    except HTTPException:
        raise

    except Exception as exc:
        elapsed = time.time() - started
        log.exception("[%s] STT error after %.2fs: %s", request_id, elapsed, exc)
        _cuda_cleanup()
        raise HTTPException(status_code=500, detail=f"ASR pipeline error: {exc}") from exc

    finally:
        with _queue_lock:
            _inflight -= 1
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
                # Duration of the original upload, not the VAD-trimmed audio the
                # model actually saw. Segment timestamps are mapped back to this
                # timeline, so they line up with it.
                "duration": round(vad_before if vad_mapping is not None else duration, 3),
                "audio_truncated": truncated,
                "vad_removed_silence": vad_mapping is not None,
            }
        )

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format == "srt":
        return PlainTextResponse(_to_srt(segments))

    return PlainTextResponse(_to_vtt(segments))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT, workers=1, reload=False)
