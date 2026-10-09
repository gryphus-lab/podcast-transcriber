"""FastAPI application exposing transcription and conversion services."""

# Suppress noisy warnings BEFORE importing heavy libs (whisperx, pyannote, ...)
from .utils.warnings_filter import suppress_noisy_output

suppress_noisy_output()

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Annotated, Optional  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import (  # noqa: E402
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .config import (  # noqa: E402
    HF_TOKEN,
    LANGUAGE,
    MAX_UPLOAD_SIZE,
    OUTPUT_DIR,
    SUPPORTED_FORMATS,
    WHISPER_MODEL,
)
from .job_backend import (  # noqa: E402
    build_store,
    enqueue_transcription,
    get_backend_name,
    result_filename,
    results_dir,
    save_job_input,
)
from .jobs import Job, JobStatus  # noqa: E402
from .utils.converter import (  # noqa: E402
    PODCAST_API_OUTPUT_FORMATS,
    handle_conversion_request,
)
from .utils.formatter import format_transcript, write_transcript  # noqa: E402
from .utils.transcriber import transcribe_audio  # noqa: E402
from .utils.uploads import save_upload_to_temp  # noqa: E402

# Static assets (the single-page UI) live alongside this module.
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Podcast Transcriber API",
    description=(
        "Transcribe audio with speaker diarization and convert audio to MP4. "
        "Transcriptions can run synchronously (/transcribe) or as async jobs "
        "(POST /jobs, then poll GET /jobs/{job_id} or use a callback webhook)."
    ),
    version="0.2.0",
)

# Job store, selected by JOB_BACKEND (memory | rq). The in-process default needs
# no broker; set JOB_BACKEND=rq for the durable Redis-backed queue.
_store = build_store()


# ---------------------------------------------------------------------------
# Async job schemas
# ---------------------------------------------------------------------------


class JobAccepted(BaseModel):
    """Returned by POST /jobs when a transcription is enqueued."""

    job_id: str
    status: JobStatus
    status_url: str = Field(description="Poll this URL for the result.")


class JobView(BaseModel):
    """Full job state returned by GET /jobs/{job_id} and sent to webhooks."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    source_name: Optional[str] = None
    output_format: str = "txt"
    result_url: Optional[str] = Field(
        None, description="Download URL for the saved transcript (when finished)."
    )
    transcript: Optional[str] = None


class JobSummary(BaseModel):
    """Compact job row for the jobs table (no transcript payload)."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    source_name: Optional[str] = None
    output_format: str = "txt"
    result_url: Optional[str] = None


def _result_url(job: Job) -> Optional[str]:
    """Download URL for a finished job's transcript, if available."""
    if job.status is JobStatus.DONE and job.result is not None:
        return f"/jobs/{job.id}/result"
    return None


def _to_view(job: Job) -> JobView:
    return JobView(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        source_name=job.source_name,
        output_format=job.output_format,
        result_url=_result_url(job),
        transcript=None if job.result is None else job.result.transcript,
    )


def _to_summary(job: Job) -> JobSummary:
    return JobSummary(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        source_name=job.source_name,
        output_format=job.output_format,
        result_url=_result_url(job),
    )


def _validate_transcription_inputs(output_format: str, filename: str) -> str:
    """Validate format + extension; return the lower-cased suffix. Raises 400."""
    if output_format not in ("txt", "srt", "json"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{output_format}'. Use txt, srt, or json.",
        )
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            ),
        )
    return suffix


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
    # Validate format + file extension (shared with the async /jobs route).
    suffix = _validate_transcription_inputs(output_format, file.filename or "")

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
    return await handle_conversion_request(
        file=file,
        output_format=output_format,
        allowed_formats=PODCAST_API_OUTPUT_FORMATS,
        max_upload_size=MAX_UPLOAD_SIZE,
        output_dir=OUTPUT_DIR,
        background_tasks=background_tasks,
    )


# ---------------------------------------------------------------------------
# Async transcription jobs (submit + poll / webhook)
# ---------------------------------------------------------------------------


@app.post(
    "/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["jobs"],
    responses={
        400: {"description": "Invalid input format or parameters"},
    },
)
async def submit_job(
    response: Response,
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model: Annotated[str, Form()] = WHISPER_MODEL,
    language: Annotated[str, Form()] = LANGUAGE,
    diarize: Annotated[bool, Form()] = True,
    hf_token: Annotated[str, Form()] = "",
    output_format: Annotated[str, Form()] = "txt",
    callback_url: Annotated[str, Form()] = "",
) -> JobAccepted:
    """Submit an async transcription job and return immediately with a job id.

    Transcription runs in the background; poll GET /jobs/{job_id} for the
    result, or supply ``callback_url`` to receive it via webhook.
    """
    suffix = _validate_transcription_inputs(output_format, file.filename or "")

    # Read the upload (bounded) and stage it so any backend/worker can read it.
    tmp_path = await save_upload_to_temp(file, suffix, MAX_UPLOAD_SIZE)
    try:
        audio_bytes = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    job_id = uuid.uuid4().hex
    audio_path = await asyncio.to_thread(save_job_input, job_id, audio_bytes, suffix)

    params = {
        "job_id": job_id,
        "audio_path": str(audio_path),
        "source_name": file.filename or f"upload{suffix}",
        "model": model,
        "language": language,
        "diarize": diarize,
        "hf_token": hf_token or HF_TOKEN,
        "output_format": output_format,
    }
    job = enqueue_transcription(_store, params, callback_url or None)

    status_url = f"/jobs/{job.id}"
    response.headers["Location"] = status_url
    return JobAccepted(job_id=job.id, status=job.status, status_url=status_url)


@app.get("/jobs", tags=["jobs"])
def list_jobs() -> list[JobSummary]:
    """List all jobs (running and completed), newest first - powers the UI."""
    return [_to_summary(job) for job in _store.list()]


@app.get(
    "/jobs/{job_id}",
    tags=["jobs"],
    responses={404: {"description": "No job with that id."}},
)
def get_job(job_id: str) -> JobView:
    """Return the status (and, when finished, the transcript) of a job."""
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return _to_view(job)


@app.get(
    "/jobs/{job_id}/result",
    tags=["jobs"],
    responses={404: {"description": "No such job, or its result is not available."}},
)
def get_job_result(job_id: str) -> FileResponse:
    """Download a finished job's saved transcript file.

    404s if the job isn't done or the file is missing.
    """
    job = _store.get(job_id)
    if job is None or job.status is not JobStatus.DONE:
        raise HTTPException(
            status_code=404, detail=f"No completed result for job: {job_id}"
        )
    fmt = job.output_format or "txt"
    path = results_dir() / (job.result_file or result_filename(job_id, fmt))
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Result file not found for job: {job_id}"
        )
    media_type = "application/json" if fmt == "json" else "text/plain"
    return FileResponse(path, media_type=media_type, filename=path.name)


# ---------------------------------------------------------------------------
# UI + banner
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the single-page transcription console."""
    index_html = _STATIC_DIR / "index.html"
    if not index_html.exists():
        return HTMLResponse(
            "<h1>Podcast Transcriber</h1><p>UI asset missing. "
            'API docs at <a href="/docs">/docs</a>.</p>'
        )
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.get("/api", tags=["meta"])
def banner() -> dict:
    """Service banner, including the active job backend."""
    return {
        "service": "podcast-transcriber",
        "job_backend": get_backend_name(),
        "docs": "/docs",
    }


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
