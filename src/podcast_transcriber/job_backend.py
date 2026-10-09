"""Job backend selection and shared transcription work.

Two interchangeable backends expose the same ``submit`` / ``get`` / ``list``
surface:

- ``memory`` (default): in-process :class:`~podcast_transcriber.jobs.JobStore`.
  Zero dependencies, no broker - great for local/dev and single instances.
- ``rq``: durable, out-of-process :class:`~podcast_transcriber.jobs_rq.RQJobStore`
  (Redis + RQ), a lightweight JobRunr-style queue that survives restarts and
  scales across workers. Requires the ``rq`` extra and ``JOB_BACKEND=rq``.

The API calls :func:`enqueue_transcription`, which adapts the parameters to
whichever backend is active, so the route code stays backend-agnostic.

Because the RQ worker runs in a separate process, the uploaded audio is saved to
a shared directory (:func:`save_job_input`) and passed to the worker by path.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional

import httpx

from .config import HF_TOKEN
from .jobs import Job, TranscriptionResult
from .utils.formatter import format_transcript, write_transcript
from .utils.transcriber import transcribe_audio

logger = logging.getLogger(__name__)


def get_backend_name() -> str:
    """Return the configured backend name ('memory' or 'rq')."""
    return os.environ.get("JOB_BACKEND", "memory").strip().lower()


def jobs_dir() -> Path:
    """Directory where per-job inputs (uploaded audio) are staged."""
    return Path(os.environ.get("JOBS_DIR", "jobs"))


def results_dir() -> Path:
    """Directory where per-job transcript files are written."""
    return Path(os.environ.get("OUTPUT_DIR", "output"))


def result_filename(job_id: str, output_format: str = "txt") -> str:
    """Per-job transcript filename, suffixed with the job id so runs are kept."""
    return f"transcript-{job_id}.{output_format}"


# ---------------------------------------------------------------------------
# Input staging + shared work
# ---------------------------------------------------------------------------


def save_job_input(job_id: str, audio_bytes: bytes, suffix: str) -> Path:
    """Persist uploaded audio for a job so any backend/worker can read it."""
    directory = jobs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}{suffix}"
    path.write_bytes(audio_bytes)
    return path


def run_transcription(params: dict) -> TranscriptionResult:
    """Transcribe the staged audio and return the formatted transcript.

    ``params`` keys: job_id, audio_path, source_name, model, language, diarize,
    hf_token, output_format. Cleans up the staged input afterwards.
    """
    audio_path = Path(params["audio_path"])
    output_format = params.get("output_format", "txt")
    source_name = params.get("source_name") or audio_path.name
    try:
        result = transcribe_audio(
            audio_path=audio_path,
            model_name=params.get("model", "large-v3"),
            language=params.get("language", "en"),
            diarize=params.get("diarize", True),
            hf_token=params.get("hf_token") or HF_TOKEN,
        )
        transcript = format_transcript(result, output_format=output_format)
    finally:
        # The staged upload is no longer needed once transcription is done.
        Path(audio_path).unlink(missing_ok=True)

    return TranscriptionResult(
        transcript=transcript,
        output_format=output_format,
        source_name=source_name,
    )


def persist_transcript(job_id: str, result: TranscriptionResult) -> str:
    """Write a finished transcript to the results dir; return its filename."""
    directory = results_dir()
    filename = result_filename(job_id, result.output_format)
    # Name the output after the job (traversal-safe) via write_transcript.
    write_transcript(
        result.transcript,
        audio_path=Path(filename).with_suffix(""),
        output_dir=directory,
        output_format=result.output_format,
    )
    logger.info("Job %s transcript saved to %s", job_id, filename)
    return filename


# ---------------------------------------------------------------------------
# Backend selection + enqueue
# ---------------------------------------------------------------------------


def build_store():
    """Instantiate the configured job store."""
    if get_backend_name() == "rq":
        from .jobs_rq import RQJobStore

        return RQJobStore()

    from .jobs import JobStore

    return JobStore()


def enqueue_transcription(store, params: dict, callback_url: Optional[str]) -> Job:
    """Enqueue a transcription on ``store``, adapting to its backend type.

    - RQ backend: pass ``params`` straight through (RQ serializes the work).
    - In-process backend: bind ``params`` into a zero-arg callable and provide a
      webhook ``on_complete`` hook.
    """
    source_name = params.get("source_name")
    output_format = params.get("output_format", "txt")

    # RQ store is detected structurally to avoid importing the optional module.
    if store.__class__.__name__ == "RQJobStore":
        return store.submit(
            params=params,
            source_name=source_name,
            output_format=output_format,
            callback_url=callback_url,
        )

    work = partial(run_transcription, params)
    return store.submit(
        work,
        source_name=source_name,
        output_format=output_format,
        callback_url=callback_url,
        on_complete=partial(_on_memory_job_complete, job_id=params["job_id"]),
    )


def _on_memory_job_complete(job: Job, job_id: str) -> None:
    """Completion hook for the in-process backend.

    Persists the transcript to a per-job file (recording the filename on the
    job) and, if a callback_url was supplied, delivers the webhook. Both steps
    are best-effort and never raise into the worker thread.
    """
    if job.status.value == "done" and job.result is not None:
        try:
            job.result_file = persist_transcript(job_id, job.result)
        except Exception:  # noqa: BLE001 - best-effort persistence
            logger.exception("Job %s transcript persistence failed", job_id)

    if job.callback_url:
        _deliver_callback(job)


def _deliver_callback(job: Job) -> None:
    """Webhook delivery for the in-process backend (best-effort)."""
    payload = {
        "job_id": job.id,
        "status": job.status.value,
        "transcript": None if job.result is None else job.result.transcript,
        "output_format": None if job.result is None else job.result.output_format,
        "error": job.error,
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            client.post(job.callback_url, json=payload)
    except Exception:  # noqa: BLE001 - best-effort webhook
        logger.exception("Callback POST to %s failed", job.callback_url)
