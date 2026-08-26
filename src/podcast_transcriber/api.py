"""FastAPI application exposing transcription and conversion services."""

# Suppress noisy warnings BEFORE importing heavy libs (whisperx, pyannote, ...)
from .utils.warnings_filter import suppress_noisy_output

suppress_noisy_output()

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Annotated  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import (  # noqa: E402
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

from .config import (  # noqa: E402
    HF_TOKEN,
    LANGUAGE,
    MAX_UPLOAD_SIZE,
    OUTPUT_DIR,
    SUPPORTED_FORMATS,
    WHISPER_MODEL,
)
from .utils.converter import (  # noqa: E402
    PODCAST_API_OUTPUT_FORMATS,
    convert_audio,
    format_supported_outputs,
    get_media_type,
)
from .utils.formatter import format_transcript, write_transcript  # noqa: E402
from .utils.transcriber import transcribe_audio  # noqa: E402
from .utils.uploads import save_upload_to_temp  # noqa: E402

CONVERSION_TIMEOUT_SECONDS = 300

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
    background_tasks: BackgroundTasks,
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

    # Save uploaded file to temp location with bounded streaming
    tmp_path = await save_upload_to_temp(file, suffix, MAX_UPLOAD_SIZE)

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

        # write_transcript creates the output dir as needed
        output_path = write_transcript(
            transcript,
            audio_path=Path(file.filename or "upload"),
            output_dir=OUTPUT_DIR,
            output_format=output_format,
        )

        background_tasks.add_task(output_path.unlink, missing_ok=True)

        if output_format == "json":
            return JSONResponse(content=json.loads(transcript))
        else:
            return FileResponse(
                path=str(output_path),
                media_type="text/plain",
                filename=output_path.name,
            )

    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).exception("Transcription failed")
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
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio file to convert")],
    output_format: Annotated[str, Form()] = "mp4",
):
    """Convert an audio file to MP4 (or other format) using FFmpeg.

    Supported output formats: mp4, mp3, wav, flac, ogg, mkv, webm.
    """
    if output_format not in PODCAST_API_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid output format '{output_format}'. "
                f"Supported: {format_supported_outputs(PODCAST_API_OUTPUT_FORMATS)}"
            ),
        )

    # Check ffmpeg is available
    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed on this server.",
        )

    # Save uploaded file with bounded streaming
    input_suffix = Path(file.filename or "input").suffix or ".m4a"
    tmp_path = await save_upload_to_temp(file, input_suffix, MAX_UPLOAD_SIZE)

    # Output path with unique filename to prevent concurrent overwrites
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
            bitrate="192k",
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


def start() -> None:
    """Entry point for `transcribe-api` script command."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "podcast_transcriber.api:app",
        host=host,
        port=port,
        reload=False,
    )
