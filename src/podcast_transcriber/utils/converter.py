"""Shared FFmpeg conversion helpers."""

import asyncio
from pathlib import Path

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
        raise TimeoutError("Conversion timed out.") from e

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {stderr.decode()[:500]}")
