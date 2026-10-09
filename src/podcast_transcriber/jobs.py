"""In-process async job store for long-running transcriptions.

Transcribing a podcast can take minutes (model load + inference + diarization),
which is far too long to hold an HTTP request open. The API saves the uploaded
audio, submits a job (returning immediately with an id), runs it on a background
thread pool, and the caller either polls for the result or supplies a callback
URL that receives the result via webhook.

This is a single-process store (dict + lock + ThreadPoolExecutor) with no
external dependencies - adequate for a single service instance. For multiple
replicas or durability across restarts, set JOB_BACKEND=rq to use the
Redis/RQ-backed store in :mod:`podcast_transcriber.jobs_rq`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TranscriptionResult:
    """The outcome of a successful transcription job."""

    transcript: str
    output_format: str
    source_name: str


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[TranscriptionResult] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    # Original uploaded file name, for display in the UI.
    source_name: Optional[str] = None
    # Output format requested (txt | srt | json), for display + download naming.
    output_format: str = "txt"
    # Relative path (under the results dir) of the persisted transcript file,
    # set once a job finishes successfully. Powers the UI's "result" link.
    result_file: Optional[str] = None


class JobStore:
    """Thread-safe registry of jobs backed by a bounded thread pool."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="transcribe-job"
        )

    def submit(
        self,
        work: Callable[[], TranscriptionResult],
        *,
        source_name: Optional[str] = None,
        output_format: str = "txt",
        callback_url: Optional[str] = None,
        on_complete: Optional[Callable[[Job], None]] = None,
    ) -> Job:
        """Register a job and schedule ``work`` to run on the pool.

        ``work`` is a zero-arg callable (bind params with functools.partial).
        ``on_complete`` is invoked with the finished Job (used for webhooks).
        """
        job = Job(
            id=uuid.uuid4().hex,
            source_name=source_name,
            output_format=output_format,
            callback_url=callback_url,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, work, on_complete)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Return all known jobs, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _set(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)

    def _run(
        self,
        job: Job,
        work: Callable[[], TranscriptionResult],
        on_complete: Optional[Callable[[Job], None]],
    ) -> None:
        self._set(job.id, status=JobStatus.RUNNING, started_at=_now())
        try:
            result = work()
            self._set(
                job.id,
                status=JobStatus.DONE,
                result=result,
                finished_at=_now(),
            )
            logger.info("Job %s done (%s)", job.id, result.output_format)
        except Exception as exc:  # noqa: BLE001 - record any failure on the job
            self._set(
                job.id,
                status=JobStatus.ERROR,
                error=str(exc),
                finished_at=_now(),
            )
            logger.error("Job %s failed: %s", job.id, exc)

        if on_complete is not None:
            # Never let a callback failure crash the worker thread.
            try:
                on_complete(self.get(job.id))
            except Exception as exc:  # noqa: BLE001
                logger.error("Job %s on_complete hook failed: %s", job.id, exc)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
