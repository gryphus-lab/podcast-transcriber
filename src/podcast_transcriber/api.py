"""FastAPI application exposing transcription and conversion services."""

import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

# Suppress noisy warnings before importing heavy libs
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import HF_TOKEN, LANGUAGE, OUTPUT_DIR, SUPPORTED_FORMATS, WHISPER_MODEL
from .utils.formatter import format_transcript, write_transcript
from .utils.transcriber import transcribe_audio

app = FastAPI(
    title="Podcast Transcriber API",
    description="Transcribe audio with speaker diarization and convert audio to MP4.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=WHISPER_MODEL),
    language: str = Form(default=LANGUAGE),
    diarize: bool = Form(default=True),
    hf_token: str = Form(default=""),
    output_format: str = Form(default="txt"),
):
    """Transcribe an uploaded audio file.

    Returns the transcript in the requested format (txt, srt, json).
    """
    # Validate format
    if output_format not in ("txt", "srt", "json"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{output_format}'. Use txt, srt, or json.",
        )

    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            ),
        )

    # Use env token if none provided
    token = hf_token or HF_TOKEN

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Transcribe
        result = transcribe_audio(
            audio_path=tmp_path,
            model_name=model,
            language=language,
            diarize=diarize,
            hf_token=token,
        )

        # Format output
        transcript = format_transcript(result, output_format=output_format)

        # Write to output dir
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = write_transcript(
            transcript,
            audio_path=Path(file.filename or "upload"),
            output_dir=OUTPUT_DIR,
            output_format=output_format,
        )

        if output_format == "json":
            import json

            return JSONResponse(content=json.loads(transcript))
        else:
            return FileResponse(
                path=str(output_path),
                media_type="text/plain",
                filename=output_path.name,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/convert")
async def api_convert(
    file: UploadFile = File(...),
    output_format: str = Form(default="mp4"),
):
    """Convert an audio file to MP4 (or other format) using FFmpeg.

    Supported output formats: mp4, mp3, wav, flac, ogg.
    """
    allowed_outputs = {"mp4", "mp3", "wav", "flac", "ogg", "mkv", "webm"}
    if output_format not in allowed_outputs:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output format '{output_format}'. "
                f"Supported: {', '.join(sorted(allowed_outputs))}"
            ),
        )

    # Check ffmpeg is available
    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed on this server.",
        )

    # Save uploaded file
    input_suffix = Path(file.filename or "input").suffix or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = Path(tmp_in.name)

    # Output path
    stem = Path(file.filename or "output").stem
    output_path = OUTPUT_DIR / f"{stem}.{output_format}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Run ffmpeg conversion
        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-i", str(input_path),
        ]

        # Add codec settings based on output format
        if output_format == "mp4":
            # For audio-only MP4, use AAC codec
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        elif output_format == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
        elif output_format == "wav":
            cmd.extend(["-c:a", "pcm_s16le"])
        elif output_format == "flac":
            cmd.extend(["-c:a", "flac"])
        elif output_format in ("ogg", "webm"):
            cmd.extend(["-c:a", "libopus", "-b:a", "128k"])
        elif output_format == "mkv":
            cmd.extend(["-c:a", "copy"])

        cmd.append(str(output_path))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg conversion failed: {result.stderr[:500]}",
            )

        return FileResponse(
            path=str(output_path),
            media_type=f"audio/{output_format}" if output_format != "mp4" else "video/mp4",
            filename=output_path.name,
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Conversion timed out.")

    finally:
        input_path.unlink(missing_ok=True)


def start():
    """Entry point for `transcribe-api` script command."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "podcast_transcriber.api:app",
        host=host,
        port=port,
        reload=False,
    )
