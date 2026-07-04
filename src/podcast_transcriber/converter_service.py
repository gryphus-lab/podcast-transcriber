"""Standalone FastAPI service for audio/video format conversion via FFmpeg."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

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
        cmd = ["ffmpeg", "-y", "-i", str(tmp_path)]

        # Codec settings per format
        codec_map = {
            "mp4": ["-c:a", "aac", "-b:a", audio_bitrate],
            "mp3": ["-c:a", "libmp3lame", "-b:a", audio_bitrate],
            "wav": ["-c:a", "pcm_s16le"],
            "flac": ["-c:a", "flac"],
            "ogg": ["-c:a", "libopus", "-b:a", "128k"],
            "webm": ["-c:a", "libopus", "-b:a", "128k"],
            "mkv": ["-c:a", "copy"],
            "aac": ["-c:a", "aac", "-b:a", audio_bitrate],
        }
        cmd.extend(codec_map.get(output_format, ["-c:a", "copy"]))
        cmd.append(str(output_path))

        # Run ffmpeg asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg failed: {stderr.decode()[:500]}",
            )

        # Determine media type
        media_types = {
            "mp4": "video/mp4",
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "mkv": "video/x-matroska",
            "aac": "audio/aac",
        }

        return FileResponse(
            path=str(output_path),
            media_type=media_types.get(output_format, "application/octet-stream"),
            filename=output_path.name,
        )

    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=500, detail="Conversion timed out (5 min)."
        ) from e

    finally:
        tmp_path.unlink(missing_ok=True)
