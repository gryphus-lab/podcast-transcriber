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
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from functools import partial
from pathlib import Path
from typing import Optional

import httpx

from .config import HF_TOKEN
from .jobs import Job, TranscriptionResult
from .utils.formatter import format_transcript, write_transcript
from .utils.transcriber import transcribe_audio

logger = logging.getLogger(__name__)


# Run the heavy transcription (model load + inference + diarization) in a
# short-lived child process by default, so that if the OS OOM-kills it on a
# large file the parent API survives: the job is marked ``error`` with a clear
# message instead of the whole container being killed and relaunched by
# ``restart: unless-stopped`` (which, with the in-memory backend, also loses the
# job). A spawned child additionally releases ALL of the ML stack's native
# memory back to the OS on exit - something an in-thread run can't guarantee.
# Set ISOLATE_TRANSCRIPTION=0 to run in-thread (e.g. for debugging/profiling).
def _isolation_enabled() -> bool:
    return os.environ.get("ISOLATE_TRANSCRIPTION", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


# Hard wall-clock ceiling for a single isolated transcription. A VM-wide OOM
# (as opposed to a cgroup-scoped one) does not always surface as a clean
# ``BrokenProcessPool``: the OS may kill the child such that the parent's
# ``Future.result()`` neither returns nor raises, so the pool silently respawns
# the worker and re-thrashes while the job sits in ``running`` forever. This
# timeout guarantees a terminal outcome: if a run exceeds it, we treat it as a
# resource failure, tear the pool down, and let the job be recorded as ``error``
# instead of looping. 0/empty disables the ceiling.
def _transcription_timeout() -> Optional[float]:
    raw = os.environ.get("TRANSCRIPTION_TIMEOUT", "3600").strip()
    try:
        seconds = float(raw)
    except ValueError:
        return 3600.0
    return seconds if seconds > 0 else None


# Shared actionable guidance for an out-of-memory / killed worker.
_OOM_GUIDANCE = (
    "Transcription worker was killed or stalled - almost certainly out of "
    "memory. Long audio with 'large-v3' + diarization can need ~10-12 GB, which "
    "exceeds a default Docker Desktop VM. Options: run the native 'transcribe' "
    "CLI on the host (full machine RAM, no VM ceiling - recommended for long "
    "files), use a smaller model (e.g. 'medium'), disable diarization, shorten "
    "the audio, or raise BOTH the container limit (TRANSCRIBER_MEMORY_LIMIT) and "
    "the Docker Desktop VM memory (Settings -> Resources -> Memory)."
)


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


def _transcribe_to_text(params: dict) -> str:
    """Pure transcription step: audio path + options -> formatted transcript.

    Importable at module top level (no closures) so it can run in a spawned
    child process. Does NOT touch the staged file or build the result dataclass;
    the parent owns cleanup and job bookkeeping.
    """
    # Test seam: a spawned child re-imports this module fresh, so a monkeypatch
    # in the parent can't reach it. This lets a test deterministically simulate
    # the OS OOM-killer SIGKILL'ing the worker, exercising the real subprocess
    # boundary. Never set in production params.
    if params.get("_test_kill_worker"):  # pragma: no cover - runs in child
        os.kill(os.getpid(), getattr(__import__("signal"), "SIGKILL"))
    # Test seam: block the child so the parent's result(timeout=...) fires,
    # exercising the timeout -> force-kill -> MemoryError path. Never in prod.
    if params.get("_test_sleep_seconds"):  # pragma: no cover - runs in child
        import time as _time

        _time.sleep(float(params["_test_sleep_seconds"]))
    return format_transcript(
        transcribe_audio(
            audio_path=Path(params["audio_path"]),
            model_name=params.get("model", "large-v3"),
            language=params.get("language", "en"),
            diarize=params.get("diarize", True),
            hf_token=params.get("hf_token") or HF_TOKEN,
        ),
        output_format=params.get("output_format", "txt"),
    )


def _transcribe_isolated(params: dict) -> str:
    """Run :func:`_transcribe_to_text` in a short-lived child process.

    Isolation gives two guarantees against an out-of-memory event on a large
    file:

    * The parent API process survives. If the child is SIGKILL'd by the OOM
      killer the pool raises ``BrokenProcessPool``, which we translate into a
      ``MemoryError`` so the job is recorded as a clean failure.
    * The job always reaches a terminal state. A VM-wide OOM can kill the child
      without the parent's ``Future.result()`` ever returning or raising, so the
      pool would silently respawn and re-thrash while the job hangs on
      ``running``. A hard :func:`_transcription_timeout` bounds the wait: on
      expiry we forcibly kill any surviving child and raise ``MemoryError`` so
      the run fails instead of looping forever.

    ``spawn`` gives the child a fresh interpreter (no inherited torch/CUDA
    state) and guarantees its native memory is reclaimed on exit.
    """
    from concurrent.futures.process import BrokenProcessPool

    ctx = mp.get_context("spawn")
    # One worker, one task: the pool exists only to sandbox a single run so a
    # kill is observable instead of fatal to the parent.
    executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    try:
        future = executor.submit(_transcribe_to_text, params)
        try:
            return future.result(timeout=_transcription_timeout())
        except FutureTimeout as exc:
            logger.error(
                "Transcription exceeded the %s s ceiling; treating as OOM/stall "
                "and failing the job.",
                _transcription_timeout(),
            )
            _kill_pool_processes(executor)
            raise MemoryError(_OOM_GUIDANCE) from exc
    except BrokenProcessPool as exc:
        raise MemoryError(_OOM_GUIDANCE) from exc
    finally:
        # Don't wait on a possibly-dead pool; just tear it down. Any child still
        # alive after a timeout was already killed above.
        executor.shutdown(wait=False, cancel_futures=True)


def _kill_pool_processes(executor: ProcessPoolExecutor) -> None:
    """Force-terminate any child processes still owned by ``executor``.

    After a timeout the worker may still be alive and holding memory (the whole
    point of the timeout is that it did not die cleanly). ``shutdown`` alone
    won't reclaim it promptly, so SIGKILL it directly to free the RAM before the
    next job starts.
    """
    procs = getattr(executor, "_processes", None) or {}
    # Snapshot via tuple(): killing a child can trigger the pool to mutate its
    # own _processes dict, and iterating a live view while it changes raises
    # RuntimeError. A tuple copy is independent of that mutation.
    for proc in tuple(procs.values()):
        try:
            if proc.is_alive():
                proc.kill()
        except (ValueError, OSError):  # pragma: no cover - process already gone
            pass


def run_transcription(params: dict) -> TranscriptionResult:
    """Transcribe the staged audio and return the formatted transcript.

    ``params`` keys: job_id, audio_path, source_name, model, language, diarize,
    hf_token, output_format. Cleans up the staged input afterwards.

    By default the transcription runs in an isolated child process (see
    :func:`_isolation_enabled`) so an out-of-memory kill fails just this job
    rather than taking down the API container.
    """
    audio_path = Path(params["audio_path"])
    output_format = params.get("output_format", "txt")
    source_name = params.get("source_name") or audio_path.name
    try:
        if _isolation_enabled():
            transcript = _transcribe_isolated(params)
        else:
            transcript = _transcribe_to_text(params)
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
        # Persist the transcript before the job flips to `done`, so a client
        # that polls until `done` can immediately download the result file.
        finalize=persist_transcript,
        on_complete=_on_memory_job_complete,
    )


def _on_memory_job_complete(job: Job) -> None:
    """Completion hook for the in-process backend: deliver the webhook.

    Transcript persistence happens in the ``finalize`` step (before the job is
    marked done); this hook only delivers the optional webhook callback. Runs on
    the worker thread and must never raise.
    """
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
