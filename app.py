from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import io
import soundfile as sf
import resampy
import numpy as np
from datasets import load_dataset

import whisper_timestamped

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
#openai/whisper-large-v3"
#openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
#    max_new_tokens=128,
#    chunk_length_s=30,
#    batch_size=4,
    return_timestamps="word",
#sentence",
#word",
    device=device,
)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    model: str = Form(...),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: list[str] = Form(None),
):
    # Check authorization
    if authorization != "Bearer EMPTY":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Read the audio file
    audio_bytes = await file.read()
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Ensure the audio is mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample the audio to 16kHz
    target_sample_rate = 16000
    audio = resampy.resample(audio, sample_rate, target_sample_rate)

    # Process the audio file through the pipeline
    result = pipe({"array": audio, "sampling_rate": target_sample_rate})

    # Prepare the response based on the specified format
    if response_format == "json":
        response_data = {"text": result["text"]}
    elif response_format == "text":
        response_data = result["text"]
    elif response_format == "srt":
        # Generate SRT format (this is a placeholder, actual implementation needed)
        response_data = generate_srt(result)
    elif response_format == "verbose_json":
        # Generate verbose JSON (this is a placeholder, actual implementation needed)
        response_data = {"words": result["chunks"]}
        #response_data = result
    elif response_format == "vtt":
        # Generate VTT format (this is a placeholder, actual implementation needed)
        response_data = generate_vtt(result)
    else:
        raise HTTPException(status_code=400, detail="Invalid response format specified")

    return JSONResponse(response_data)

def generate_srt(result):
    # Placeholder for SRT generation logic
    return "SRT content"

def generate_vtt(result):
    # Placeholder for VTT generation logic
    return "VTT content"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
