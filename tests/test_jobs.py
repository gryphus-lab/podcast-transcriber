"""Tests for the in-process job store and job backend."""

import time
from unittest.mock import patch

import pytest

from podcast_transcriber import job_backend
from podcast_transcriber.jobs import Job, JobStatus, JobStore, TranscriptionResult


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    """Point JOBS_DIR / OUTPUT_DIR at a temp dir so tests don't touch the repo."""
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))


def _wait(store, job_id, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job.status in (JobStatus.DONE, JobStatus.ERROR):
            return job
        time.sleep(0.01)
    return store.get(job_id)


def test_store_runs_job_to_done():
    store = JobStore()
    result = TranscriptionResult(
        transcript="hello\n", output_format="txt", source_name="a.mp3"
    )

    job = store.submit(lambda: result, source_name="a.mp3", output_format="txt")
    done = _wait(store, job.id)

    assert done.status is JobStatus.DONE
    assert done.result.transcript == "hello\n"
    store.shutdown()


def test_finalize_runs_before_done_and_sets_result_file():
    # finalize must complete (and its return value be recorded on the job)
    # atomically with the DONE transition, so a client that observes `done`
    # can rely on result_file being set.
    store = JobStore()
    result = TranscriptionResult(
        transcript="hi\n", output_format="txt", source_name="a.mp3"
    )
    seen_status_at_finalize = {}

    def finalize(job_id, res):
        # When finalize runs the job must not yet be marked done.
        seen_status_at_finalize["status"] = store.get(job_id).status
        assert res is result
        return "latest-abc.txt"

    job = store.submit(lambda: result, finalize=finalize)
    done = _wait(store, job.id)

    assert seen_status_at_finalize["status"] is not JobStatus.DONE
    assert done.status is JobStatus.DONE
    assert done.result_file == "latest-abc.txt"
    store.shutdown()


def test_finalize_failure_does_not_fail_the_job():
    # A persistence failure in finalize is best-effort: the job still completes
    # (status done), just without a result_file.
    store = JobStore()
    result = TranscriptionResult(
        transcript="hi\n", output_format="txt", source_name="a.mp3"
    )

    def finalize(_job_id, _res):
        raise OSError("disk full")

    job = store.submit(lambda: result, finalize=finalize)
    done = _wait(store, job.id)

    assert done.status is JobStatus.DONE
    assert done.result_file is None
    store.shutdown()


def _done_job(callback_url=None):
    return Job(
        id="job-1",
        status=JobStatus.DONE,
        result=TranscriptionResult(
            transcript="hi\n", output_format="txt", source_name="a.mp3"
        ),
        callback_url=callback_url,
        output_format="txt",
    )


def test_memory_complete_posts_webhook_when_callback_url_set():
    job = _done_job(callback_url="https://hook.test/done")
    with patch("podcast_transcriber.job_backend.httpx.Client") as mock_client:
        post = mock_client.return_value.__enter__.return_value.post
        job_backend._on_memory_job_complete(job)

    post.assert_called_once()
    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "https://hook.test/done"
    assert kwargs["json"]["status"] == "done"
    assert kwargs["json"]["transcript"] == "hi\n"


def test_memory_complete_noop_without_callback_url():
    job = _done_job(callback_url=None)
    with patch("podcast_transcriber.job_backend.httpx.Client") as mock_client:
        job_backend._on_memory_job_complete(job)
    mock_client.assert_not_called()


def test_memory_complete_swallows_webhook_failure():
    job = _done_job(callback_url="https://hook.test/down")
    with patch("podcast_transcriber.job_backend.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = RuntimeError(
            "connection refused"
        )
        # Must not raise out of the worker thread.
        job_backend._on_memory_job_complete(job)


def test_store_records_error():
    store = JobStore()

    def boom():
        raise RuntimeError("transcription blew up")

    job = store.submit(boom)
    done = _wait(store, job.id)

    assert done.status is JobStatus.ERROR
    assert "blew up" in done.error
    assert done.result is None
    store.shutdown()


def test_store_list_newest_first():
    store = JobStore()
    r = TranscriptionResult(transcript="x", output_format="txt", source_name="s")
    j1 = store.submit(lambda: r)
    j2 = store.submit(lambda: r)
    _wait(store, j1.id)
    _wait(store, j2.id)

    listed = store.list()
    ids = [j.id for j in listed]
    assert j1.id in ids
    assert j2.id in ids
    # Newest first: j2 was created after j1.
    assert ids.index(j2.id) <= ids.index(j1.id)
    store.shutdown()


def test_persist_transcript_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    result = TranscriptionResult(
        transcript="line one\n", output_format="txt", source_name="ep.mp3"
    )

    filename = job_backend.persist_transcript("abc123", result)

    written = job_backend.results_dir() / filename
    assert written.exists()
    assert written.read_text() == "line one\n"


def test_save_job_input_stages_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    path = job_backend.save_job_input("abc123", b"RIFF....", ".wav")
    assert path.exists()
    assert path.read_bytes() == b"RIFF...."
    assert path.suffix == ".wav"


def test_run_transcription_formats_and_cleans_up(tmp_path, monkeypatch):
    audio = tmp_path / "job.m4a"
    audio.write_bytes(b"audio")

    # Run in-thread so the in-process fake is used; the default spawned
    # subprocess would re-import the real transcriber. Subprocess isolation is
    # covered by test_job_backend.py.
    monkeypatch.setenv("ISOLATE_TRANSCRIPTION", "0")

    def fake_transcribe_audio(**kwargs):
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "A"}]}

    monkeypatch.setattr(job_backend, "transcribe_audio", fake_transcribe_audio)

    params = {
        "job_id": "j1",
        "audio_path": str(audio),
        "source_name": "ep.m4a",
        "output_format": "txt",
    }
    result = job_backend.run_transcription(params)

    assert "Hi" in result.transcript
    assert result.output_format == "txt"
    assert result.source_name == "ep.m4a"
    # The staged input is removed after transcription.
    assert not audio.exists()


def test_build_store_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("JOB_BACKEND", raising=False)
    store = job_backend.build_store()
    assert store.__class__.__name__ == "JobStore"
    assert job_backend.get_backend_name() == "memory"
