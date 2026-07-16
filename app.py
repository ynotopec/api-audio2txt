# app.py — FastAPI ASR OpenAI-compatible — Whisper v3 Turbo
# Fixes:
# - évite de bloquer l'event-loop Uvicorn
# - force num_beams=1 pour éviter beam_search coûteux/bloqué
# - limite max_new_tokens
# - sérialise l'accès GPU avec asr_lock
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
BATCH = int(os.getenv("ASR_BATCH", "1"))

MAX_SEC = int(os.getenv("ASR_MAX_SEC", "5400"))  # 90 min max
ASR_TIMEOUT = int(os.getenv("ASR_TIMEOUT", "180"))
ASR_MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "256"))

# Très important pour éviter des comportements de génération lents/bloqués.
ASR_NUM_BEAMS = int(os.getenv("ASR_NUM_BEAMS", "1"))

# Une seule inférence GPU à la fois par process.
ASR_WORKERS = int(os.getenv("ASR_WORKERS", "1"))

AUTH_FILE = os.getenv("AUTH_FILE", "auth_tokens.txt")


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

app = FastAPI(title="api-audio2txt", version="1.1.0")

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
    torch.set_num_threads(min(4, os.cpu_count() or 1))
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
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


@app.on_event("startup")
def _startup() -> None:
    global AUTH_TOKENS
    AUTH_TOKENS = _load_tokens()
    log.info("Loaded %d auth token(s)", len(AUTH_TOKENS))


def _check_auth(authorization: Optional[str]) -> None:
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
    batch_size=BATCH,
    torch_dtype=DT,
    device=DEV,
)

asr_executor = ThreadPoolExecutor(max_workers=ASR_WORKERS)
asr_lock = asyncio.Lock()

log.info(
    "ASR ready: chunk=%ss batch=%s timeout=%ss max_new_tokens=%s num_beams=%s workers=%s",
    CHUNK,
    BATCH,
    ASR_TIMEOUT,
    ASR_MAX_NEW_TOKENS,
    ASR_NUM_BEAMS,
    ASR_WORKERS,
)


# =============================================================================
# Utils
# =============================================================================

def _timestamp_mode(raw: Optional[str]) -> Any:
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
    if CUDA:
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


def _run_asr_sync(
    array: Any,
    language: Optional[str],
    timestamp_mode: Any,
) -> Dict[str, Any]:
    generate_kwargs: Dict[str, Any] = {
        "num_beams": ASR_NUM_BEAMS,
        "do_sample": False,
        "max_new_tokens": ASR_MAX_NEW_TOKENS,
    }

    if language:
        generate_kwargs["language"] = language

    input_payload = {
        "array": array,
        "sampling_rate": SR,
    }

    with torch.inference_mode():
        result = pipe(
            input_payload,
            return_timestamps=timestamp_mode,
            generate_kwargs=generate_kwargs,
        )

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected ASR result type: {type(result)}")

    return result


async def _run_asr_async(
    array: Any,
    language: Optional[str],
    timestamp_mode: Any,
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()

    async with asr_lock:
        return await asyncio.wait_for(
            loop.run_in_executor(
                asr_executor,
                _run_asr_sync,
                array,
                language,
                timestamp_mode,
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
    return {
        "ok": True,
        "model": MID,
        "cuda": CUDA,
        "device": DEV,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    model_name: str = Form(None, alias="model"),
    language: str = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: str = Form(None),
):
    _check_auth(authorization)

    request_id = str(uuid.uuid4())[:8]
    started = time.time()

    log.info(
        "[%s] STT start filename=%s model=%s language=%s response_format=%s",
        request_id,
        file.filename,
        model_name,
        language,
        response_format,
    )

    try:
        data = await file.read()

        if not data:
            raise HTTPException(status_code=400, detail="Empty audio file")

        array, duration = _load_audio_to_array(data)

        log.info(
            "[%s] audio loaded duration=%.2fs bytes=%d",
            request_id,
            duration,
            len(data),
        )

        timestamp_mode = _timestamp_mode(timestamp_granularities)

        result = await _run_asr_async(
            array=array,
            language=language,
            timestamp_mode=timestamp_mode,
        )

        text = result.get("text", "") or ""
        segments = _segments(result)

        elapsed = time.time() - started

        log.info(
            "[%s] STT done elapsed=%.2fs duration=%.2fs text_chars=%d segments=%d",
            request_id,
            elapsed,
            duration,
            len(text),
            len(segments),
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - started
        log.error("[%s] STT timeout after %.2fs", request_id, elapsed)
        _safe_cuda_cleanup()
        raise HTTPException(
            status_code=504,
            detail=f"ASR timeout after {ASR_TIMEOUT}s",
        )

    except HTTPException:
        raise

    except Exception as exc:
        elapsed = time.time() - started
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

    raise HTTPException(
        status_code=400,
        detail="Invalid response format specified",
    )


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
