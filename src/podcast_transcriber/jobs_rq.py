"""RQ-backed job store (durable, out-of-process).

A lightweight, JobRunr-style alternative to the in-process
:class:`~podcast_transcriber.jobs.JobStore`. Like JobRunr, jobs are persisted
(here in Redis) and executed by separate worker processes, so they survive API
restarts and scale horizontally. Enabled with ``JOB_BACKEND=rq``; otherwise the
zero-dependency in-process store is used.

Requires the ``rq`` extra:  uv sync --extra rq
Run a worker:  rq worker --url "$REDIS_URL" transcriptions

The uploaded audio is saved to a shared directory before enqueue (see
``job_backend.save_job_input``), so the worker - which does not share memory
with the API - can read it by path.
"""

from __future__ import annotations

import logging
import os
from datetime import timezone
from typing import Optional

import httpx

from .jobs import Job, JobStatus, TranscriptionResult

logger = logging.getLogger(__name__)

# RQ queue name and webhook timeout.
QUEUE_NAME = "transcriptions"
_CALLBACK_TIMEOUT_S = 15.0
# Keep finished jobs (and their results) in Redis for this long.
_RESULT_TTL_S = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Worker-side functions (must be importable by the RQ worker; no closures)
# ---------------------------------------------------------------------------


def run_transcription_job(params: dict) -> dict:
    """Execute a transcription on an RQ worker and return a plain dict.

    ``params`` carries the saved audio path and transcription options. The
    returned dict is JSON-serializable so the result stored in Redis stays
    portable. The transcript is also persisted to the results dir here, so a
    polling client can download it even after the worker exits.
    """
    # Imported here (not at module import) to keep the base install light.
    from .job_backend import persist_transcript, run_transcription

    result = run_transcription(params)
    result_file = persist_transcript(params["job_id"], result)
    return {
        "transcript": result.transcript,
        "output_format": result.output_format,
        "source_name": result.source_name,
        "result_file": result_file,
    }


def deliver_callback(job, connection, result, *args, **kwargs) -> None:
    """RQ ``on_success`` hook: POST the finished result to the callback URL."""
    callback_url = (job.meta or {}).get("callback_url")
    if not callback_url:
        return
    payload = {
        "job_id": job.id,
        "status": JobStatus.DONE.value,
        "transcript": (result or {}).get("transcript"),
        "output_format": (result or {}).get("output_format"),
    }
    _post_callback(callback_url, payload)


def deliver_callback_failure(job, connection, exc_type, exc_value, tb) -> None:
    """RQ ``on_failure`` hook: POST the failure to the callback URL."""
    callback_url = (job.meta or {}).get("callback_url")
    if not callback_url:
        return
    payload = {
        "job_id": job.id,
        "status": JobStatus.ERROR.value,
        "error": str(exc_value),
        "transcript": None,
    }
    _post_callback(callback_url, payload)


def _post_callback(url: str, payload: dict) -> None:
    try:
        with httpx.Client(timeout=_CALLBACK_TIMEOUT_S) as client:
            client.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001 - best-effort webhook
        logger.error("Callback POST to %s failed: %s", url, exc)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

# RQ status string -> our JobStatus.
_STATUS_MAP = {
    "queued": JobStatus.PENDING,
    "deferred": JobStatus.PENDING,
    "scheduled": JobStatus.PENDING,
    "started": JobStatus.RUNNING,
    "finished": JobStatus.DONE,
    "failed": JobStatus.ERROR,
    "stopped": JobStatus.ERROR,
    "canceled": JobStatus.ERROR,
}


class RQJobStore:
    """Durable job store backed by Redis + RQ.

    Exposes the same ``submit`` / ``get`` / ``list`` surface as the in-process
    JobStore so the API code is backend-agnostic. The actual work runs in a
    separate RQ worker process (``rq worker transcriptions``).
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        # Imported lazily so the base install (without the rq extra) still works.
        from redis import Redis
        from rq import Queue

        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis = Redis.from_url(url)
        self._queue = Queue(QUEUE_NAME, connection=self._redis)

    def submit(
        self,
        *,
        params: Optional[dict] = None,
        source_name: Optional[str] = None,
        output_format: str = "txt",
        callback_url: Optional[str] = None,
    ) -> Job:
        """Enqueue a transcription job on the RQ queue.

        The work is identified by ``params`` (a plain dict passed to
        :func:`run_transcription_job` on the worker), because RQ serializes the
        function reference + args rather than a closure. Webhook delivery uses
        RQ's own on_success/on_failure hooks.
        """
        from rq import Callback

        if params is None:
            raise ValueError("RQJobStore.submit requires params=<transcription kwargs>")

        job_id = params["job_id"]
        rq_job = self._queue.enqueue(
            run_transcription_job,
            params,
            job_id=job_id,
            result_ttl=_RESULT_TTL_S,
            failure_ttl=_RESULT_TTL_S,
            on_success=Callback(deliver_callback) if callback_url else None,
            on_failure=Callback(deliver_callback_failure) if callback_url else None,
            meta={
                "callback_url": callback_url,
                "source_name": source_name,
                "output_format": output_format,
            },
        )
        return self._to_job(rq_job)

    def get(self, job_id: str) -> Optional[Job]:
        from rq.exceptions import NoSuchJobError
        from rq.job import Job as RQJob

        try:
            rq_job = RQJob.fetch(job_id, connection=self._redis)
        except NoSuchJobError:
            return None
        return self._to_job(rq_job)

    def list(self) -> list[Job]:
        """Return jobs across the queue registries, newest first."""
        from rq.job import Job as RQJob
        from rq.registry import (
            FailedJobRegistry,
            FinishedJobRegistry,
            StartedJobRegistry,
        )

        ids: list[str] = list(self._queue.job_ids)
        for registry in (
            StartedJobRegistry(queue=self._queue),
            FinishedJobRegistry(queue=self._queue),
            FailedJobRegistry(queue=self._queue),
        ):
            ids.extend(registry.get_job_ids())

        unique_ids = list(dict.fromkeys(ids))
        jobs: list[Job] = [
            self._to_job(rq_job)
            for rq_job in RQJob.fetch_many(unique_ids, connection=self._redis)
            if rq_job is not None
        ]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def _to_job(self, rq_job) -> Job:
        status = _STATUS_MAP.get(rq_job.get_status(refresh=True), JobStatus.PENDING)
        meta = rq_job.meta or {}

        result: Optional[TranscriptionResult] = None
        result_file: Optional[str] = None
        if status is JobStatus.DONE and isinstance(rq_job.result, dict):
            payload = rq_job.result
            result = TranscriptionResult(
                transcript=payload.get("transcript", ""),
                output_format=payload.get("output_format", "txt"),
                source_name=payload.get("source_name", ""),
            )
            result_file = payload.get("result_file")

        error: Optional[str] = None
        if status is JobStatus.ERROR:
            latest = rq_job.latest_result()
            error = (
                latest.exc_string
                if latest is not None
                else (rq_job.exc_info or "Job failed")
            )

        return Job(
            id=rq_job.id,
            status=status,
            created_at=_aware(rq_job.created_at),
            started_at=_aware(rq_job.started_at),
            finished_at=_aware(rq_job.ended_at),
            result=result,
            error=error,
            callback_url=meta.get("callback_url"),
            source_name=meta.get("source_name"),
            output_format=meta.get("output_format", "txt"),
            result_file=result_file,
        )


def _aware(dt):
    """RQ stores naive UTC datetimes; tag them as UTC for the API views."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
