"""Tests for the in-process job store and job backend."""

import time

import pytest

from podcast_transcriber import job_backend
from podcast_transcriber.jobs import JobStatus, JobStore, TranscriptionResult


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
    result = TranscriptionResult(transcript="hello\n", output_format="txt", source_name="a.mp3")

    job = store.submit(lambda: result, source_name="a.mp3", output_format="txt")
    done = _wait(store, job.id)

    assert done.status is JobStatus.DONE
    assert done.result.transcript == "hello\n"
    store.shutdown()


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
    assert j1.id in ids and j2.id in ids
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
