"""Standalone FastAPI service for audio/video format conversion via FFmpeg."""

import shutil
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile

from .config import MAX_UPLOAD_SIZE, OUTPUT_DIR
from .utils.converter import SUPPORTED_OUTPUT_FORMATS, handle_conversion_request

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
    return await handle_conversion_request(
        file=file,
        output_format=output_format,
        allowed_formats=SUPPORTED_OUTPUT_FORMATS,
        max_upload_size=MAX_UPLOAD_SIZE,
        output_dir=OUTPUT_DIR,
        background_tasks=background_tasks,
        bitrate=audio_bitrate,
    )
