"""Standalone FastAPI service for audio/video format conversion via FFmpeg."""

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .utils.converter import (
    SUPPORTED_OUTPUT_FORMATS,
    convert_audio,
    format_supported_outputs,
    get_media_type,
)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 500 * 1024 * 1024))  # 500MB default

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
    background_tasks: BackgroundTasks,
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
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output format '{output_format}'. "
                f"Supported: {format_supported_outputs(SUPPORTED_OUTPUT_FORMATS)}"
            ),
        )

    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=500, detail="FFmpeg is not installed on this server."
        )

    # Save uploaded file with bounded streaming
    input_suffix = Path(file.filename or "input").suffix or ".m4a"
    fd, tmp_file = tempfile.mkstemp(suffix=input_suffix)
    os.close(fd)  # Close fd before writing
    tmp_path = Path(tmp_file)
    
    # Stream upload in bounded chunks
    cumulative_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                cumulative_size += len(chunk)
                if cumulative_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum size of {MAX_UPLOAD_SIZE} bytes.",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}") from e

    # Build output path with unique filename to prevent concurrent overwrites
    stem = Path(file.filename or "output").stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unique_name = f"{stem}_{uuid.uuid4().hex}.{output_format}"
    output_path = OUTPUT_DIR / unique_name
    download_name = f"{stem}.{output_format}"

    try:
        # Convert using shared utility
        await convert_audio(
            input_path=tmp_path,
            output_path=output_path,
            output_format=output_format,
            bitrate=audio_bitrate,
            timeout_seconds=300,
        )

        background_tasks.add_task(output_path.unlink, missing_ok=True)

        return FileResponse(
            path=str(output_path),
            media_type=get_media_type(output_format),
            filename=download_name,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    except TimeoutError as e:
        raise HTTPException(
            status_code=500, detail="Conversion timed out (5 min)."
        ) from e

    finally:
        tmp_path.unlink(missing_ok=True)
