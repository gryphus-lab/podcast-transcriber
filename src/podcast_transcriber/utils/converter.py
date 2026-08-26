"""Shared FFmpeg conversion helpers."""

import asyncio
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .uploads import save_upload_to_temp

CONVERSION_TIMEOUT_SECONDS = 300

CODEC_MAP = {
    "mp4": ["-c:a", "aac"],
    "mp3": ["-c:a", "libmp3lame"],
    "wav": ["-c:a", "pcm_s16le"],
    "flac": ["-c:a", "flac"],
    "ogg": ["-c:a", "libopus", "-b:a", "128k"],
    "webm": ["-c:a", "libopus", "-b:a", "128k"],
    "mkv": ["-c:a", "copy"],
    "aac": ["-c:a", "aac"],
}

BITRATE_FORMATS = {"mp4", "mp3", "aac"}

MEDIA_TYPES = {
    "mp4": "video/mp4",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
    "mkv": "video/x-matroska",
    "aac": "audio/aac",
}

SUPPORTED_OUTPUT_FORMATS = frozenset(CODEC_MAP)
PODCAST_API_OUTPUT_FORMATS = SUPPORTED_OUTPUT_FORMATS - {"aac"}


def format_supported_outputs(output_formats: set[str] | frozenset[str]) -> str:
    """Return output formats as a stable comma-separated list."""
    return ", ".join(sorted(output_formats))


def get_media_type(output_format: str) -> str:
    """Return the HTTP media type for a converted file format."""
    return MEDIA_TYPES.get(output_format, "application/octet-stream")


async def convert_audio(
    input_path: Path,
    output_path: Path,
    output_format: str,
    bitrate: str = "192k",
    timeout_seconds: int = 300,
) -> None:
    """Convert an audio file with FFmpeg.

    Raises:
        RuntimeError: If FFmpeg exits unsuccessfully.
        TimeoutError: If conversion exceeds the configured timeout.
    """
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    codec_args = list(CODEC_MAP.get(output_format, ["-c:a", "copy"]))

    if output_format in BITRATE_FORMATS:
        codec_args.extend(["-b:a", bitrate])

    cmd.extend(codec_args)
    cmd.append(str(output_path))

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        async with asyncio.timeout(timeout_seconds):
            _, stderr = await process.communicate()
    except TimeoutError as e:
        process.kill()
        await process.wait()
        raise TimeoutError("Conversion timed out.") from e

    if process.returncode != 0:
        detail = stderr.decode(errors="replace")[:500]
        raise RuntimeError(f"FFmpeg conversion failed: {detail}")


async def handle_conversion_request(
    file: UploadFile,
    output_format: str,
    allowed_formats: frozenset[str],
    max_upload_size: int,
    output_dir: Path,
    background_tasks: BackgroundTasks,
    bitrate: str = "192k",
) -> FileResponse:
    """Validate, stream, convert, and return an uploaded file as a FileResponse.

    Shared by the transcriber API and the standalone converter service.

    Raises:
        HTTPException: 400 (bad format), 413 (too large, via streaming),
            or 500 (ffmpeg missing / conversion failure / timeout).
    """
    if output_format not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output format '{output_format}'. "
                f"Supported: {format_supported_outputs(allowed_formats)}"
            ),
        )

    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=500, detail="FFmpeg is not installed on this server."
        )

    # Stream the upload to a temp file (bounded by max_upload_size).
    input_suffix = Path(file.filename or "input").suffix or ".m4a"
    tmp_path = await save_upload_to_temp(file, input_suffix, max_upload_size)

    # Unique output name prevents concurrent requests overwriting each other.
    stem = Path(file.filename or "output").stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_{uuid.uuid4().hex}.{output_format}"
    download_name = f"{stem}.{output_format}"

    try:
        await convert_audio(
            input_path=tmp_path,
            output_path=output_path,
            output_format=output_format,
            bitrate=bitrate,
            timeout_seconds=CONVERSION_TIMEOUT_SECONDS,
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
        raise HTTPException(status_code=500, detail="Conversion timed out.") from e
    finally:
        tmp_path.unlink(missing_ok=True)
