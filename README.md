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
