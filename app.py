# app.py — FastAPI ASR (Whisper v3 Turbo) — ultra-ultra optimisé
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import os, io, json, torch, torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Perf CPU/GPU globales ----
try:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
except Exception:
    pass
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---- Auth (chargée au démarrage) ----
AUTH_TOKENS = set()
def _load_tokens(p="auth_tokens.txt"):
    if os.path.exists(p):
        with open(p) as f: return {t.strip() for t in f if t.strip()}
    return set()

@app.on_event("startup")
def _startup():  # nosec
    global AUTH_TOKENS
    AUTH_TOKENS = _load_tokens()

# ---- Modèle (chargé une seule fois) ----
MID = os.getenv("ASR_MODEL", "openai/whisper-large-v3-turbo")
CUDA = torch.cuda.is_available()
DEV  = 0 if CUDA else -1
DT   = torch.float16 if CUDA else torch.float32
SR   = int(os.getenv("ASR_SR", "16000"))
CHUNK = int(os.getenv("ASR_CHUNK_S", "30"))         # longueur de chunk (s)
BATCH = int(os.getenv("ASR_BATCH", "2" if CUDA else "1"))
MAX_SEC = int(os.getenv("ASR_MAX_SEC", "5400"))      # coupe les fichiers > 90 min

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MID, torch_dtype=DT, low_cpu_mem_usage=True, use_safetensors=True
)
model.eval()
if CUDA: model.to("cuda")
proc = AutoProcessor.from_pretrained(MID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=proc.tokenizer,
    feature_extractor=proc.feature_extractor,
    return_timestamps=True,
    chunk_length_s=CHUNK,
    batch_size=BATCH,
    torch_dtype=DT,
    device=DEV,
    # SDPA = attention efficace (par défaut sur torch récent, on force quand même)
    model_kwargs={"attn_implementation": "sdpa"},
)

# ---- Utils compacts ----
def _gran(raw):
    if not raw: return "sentence"
    try:
        v = json.loads(raw)
        if isinstance(v, str): return "word" if v == "word" else "sentence"
        return "word" if any(x == "word" for x in v) else "sentence"
    except Exception:
        return "word" if str(raw) == "word" else "sentence"

def _segs(res): return res.get("chunks") or res.get("segments") or []
def _ts(sg):
    ts = sg.get("timestamp", [0, 0])
    if isinstance(ts, dict): return float(ts.get("start", 0) or 0), float(ts.get("end", 0) or 0)
    a = float(ts[0] or 0) if ts and ts[0] is not None else 0.0
    b = float(ts[1] or 0) if ts and len(ts) > 1 and ts[1] is not None else 0.0
    return a, b
def _fmt(t, srt=True):
    t=max(float(t),0.0); h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int((t%1)*1000)
    return f"{h:02}:{m:02}:{s:02}{',' if srt else '.'}{ms:03}"
def _srt(segs):
    out=[]
    for i,sg in enumerate(segs,1):
        a,b=_ts(sg); out += [f"{i}", f"{_fmt(a)} --> {_fmt(b)}", sg.get("text",""), ""]
    return "\n".join(out).strip()
def _vtt(segs):
    lines=["WEBVTT",""]
    for sg in segs:
        a,b=_ts(sg); lines += [f"{_fmt(a,0)} --> {_fmt(b,0)}", sg.get("text",""), ""]
    return "\n".join(lines).strip()

# ---- Route principale ----
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    model: str = Form(...),                    # compat, ignoré
    language: str = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: str = Form(None),
):
    # Auth (simple & strict)
    if not (authorization and authorization.startswith("Bearer ")): 
        raise HTTPException(401, "Unauthorized")
    if authorization.split(" ",1)[-1].strip() not in AUTH_TOKENS:
        raise HTTPException(401, "Unauthorized")

    # Lecture + prétraitement
    try:
        buf = io.BytesIO(await file.read()); buf.seek(0)
        wav, sr = torchaudio.load(buf)  # (C, T)
    except Exception as e:
        raise HTTPException(400, f"Unsupported audio format: {e}")

    if wav.dim()!=2 or wav.size(0)<1: raise HTTPException(400, "Invalid audio tensor")
    if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)        # mono
    if sr != SR:        wav = torchaudio.functional.resample(wav, sr, SR)
    # Coupe fichiers très longs (sécurité/perf)
    if MAX_SEC and wav.size(1) > SR * MAX_SEC:
        wav = wav[:, : SR * MAX_SEC]
    arr = wav.squeeze(0).numpy()

    # Inférence
    try:
        with torch.inference_mode():
            gran = _gran(timestamp_granularities)
            inp = {"array": arr, "sampling_rate": SR}
            if language: inp["language"] = language
            res = pipe(inp, return_timestamps=gran)
    except Exception as e:
        raise HTTPException(500, f"ASR pipeline error: {e}")

    segs, text = _segs(res), res.get("text","")

    # Formats de sortie
    if response_format == "json":         return JSONResponse({"text": text})
    if response_format == "verbose_json": return JSONResponse({"text": text, "segments": segs})
    if response_format == "text":         return PlainTextResponse(text)
    if response_format == "srt":          return PlainTextResponse(_srt(segs))
    if response_format == "vtt":          return PlainTextResponse(_vtt(segs))
    raise HTTPException(400, "Invalid response format specified")

@app.get("/healthz")
def healthz(): return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8000")))
