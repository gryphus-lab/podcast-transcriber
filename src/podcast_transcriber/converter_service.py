"""Standalone FastAPI service for audio/video format conversion via FFmpeg."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .utils.converter import convert_audio, get_media_type

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))

app = FastAPI(
    title="Audio Converter API",
    description="Convert audio files between formats using FFmpeg.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    ffmpeg_available = shutil.which("ffmpeg") is not None
    return {"status": "ok", "ffmpeg": ffmpeg_available}


@app.post(
    "/convert",
    responses={
        400: {"description": "Invalid output format"},
        500: {"description": "FFmpeg not available, conversion failed, or timeout"},
    },
)
async def convert(
    file: Annotated[UploadFile, File(description="Audio file to convert")],
    output_format: Annotated[str, Form()] = "mp4",
    audio_bitrate: Annotated[str, Form()] = "192k",
):
    """Convert an uploaded audio file to the specified format.

    Args:
        file: The audio file to convert.
        output_format: Target format (mp4, mp3, wav, flac, ogg, mkv, webm, aac).
        audio_bitrate: Audio bitrate for lossy formats (default: 192k).
    """
    allowed_outputs = {"mp4", "mp3", "wav", "flac", "ogg", "mkv", "webm", "aac"}
    if output_format not in allowed_outputs:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output format '{output_format}'. "
                f"Supported: {', '.join(sorted(allowed_outputs))}"
            ),
        )

    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=500, detail="FFmpeg is not installed on this server."
        )

    # Save uploaded file
    input_suffix = Path(file.filename or "input").suffix or ".m4a"
    tmp_path = Path(tempfile.mkstemp(suffix=input_suffix)[1])
    content = await file.read()
    await asyncio.to_thread(tmp_path.write_bytes, content)

    # Build output path
    stem = Path(file.filename or "output").stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{stem}.{output_format}"

    try:
        # Convert using shared utility
        await convert_audio(
            input_path=tmp_path,
            output_path=output_path,
            output_format=output_format,
            bitrate=audio_bitrate,
            timeout_seconds=300,
        )

        return FileResponse(
            path=str(output_path),
            media_type=get_media_type(output_format),
            filename=output_path.name,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    except TimeoutError as e:
        raise HTTPException(
            status_code=500, detail="Conversion timed out (5 min)."
        ) from e

    finally:
        tmp_path.unlink(missing_ok=True)
