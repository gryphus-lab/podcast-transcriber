"""FastAPI application exposing transcription and conversion services."""

import asyncio
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Annotated

# Suppress noisy warnings before importing heavy libs
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

from .config import (  # noqa: E402
    HF_TOKEN,
    LANGUAGE,
    OUTPUT_DIR,
    SUPPORTED_FORMATS,
    WHISPER_MODEL,
)
from .utils.converter import convert_audio, get_media_type  # noqa: E402
from .utils.formatter import format_transcript, write_transcript  # noqa: E402
from .utils.transcriber import transcribe_audio  # noqa: E402

app = FastAPI(
    title="Podcast Transcriber API",
    description="Transcribe audio with speaker diarization and convert audio to MP4.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/transcribe",
    responses={
        400: {"description": "Invalid input format or parameters"},
        500: {"description": "Transcription or server error"},
    },
)
async def api_transcribe(
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model: Annotated[str, Form()] = WHISPER_MODEL,
    language: Annotated[str, Form()] = LANGUAGE,
    diarize: Annotated[bool, Form()] = True,
    hf_token: Annotated[str, Form()] = "",
    output_format: Annotated[str, Form()] = "txt",
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
    tmp_path = Path(tempfile.mkstemp(suffix=suffix)[1])
    content = await file.read()
    await asyncio.to_thread(tmp_path.write_bytes, content)

    try:
        # Transcribe (run in thread to avoid blocking event loop)
        result = await asyncio.to_thread(
            transcribe_audio,
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
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        tmp_path.unlink(missing_ok=True)


@app.post(
    "/convert",
    responses={
        400: {"description": "Invalid output format"},
        500: {"description": "FFmpeg conversion error or timeout"},
    },
)
async def api_convert(
    file: Annotated[UploadFile, File(description="Audio file to convert")],
    output_format: Annotated[str, Form()] = "mp4",
):
    """Convert an audio file to MP4 (or other format) using FFmpeg.

    Supported output formats: mp4, mp3, wav, flac, ogg, mkv, webm.
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
    tmp_path = Path(tempfile.mkstemp(suffix=input_suffix)[1])
    content = await file.read()
    await asyncio.to_thread(tmp_path.write_bytes, content)

    # Output path
    stem = Path(file.filename or "output").stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{stem}.{output_format}"

    try:
        # Convert using shared utility
        await convert_audio(
            input_path=tmp_path,
            output_path=output_path,
            output_format=output_format,
            bitrate="192k",
            timeout=300,
        )

        return FileResponse(
            path=str(output_path),
            media_type=get_media_type(output_format),
            filename=output_path.name,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    except TimeoutError as e:
        raise HTTPException(status_code=500, detail="Conversion timed out.") from e

    finally:
        tmp_path.unlink(missing_ok=True)


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
