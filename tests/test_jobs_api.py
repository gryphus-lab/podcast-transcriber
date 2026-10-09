"""Tests for the async transcription job API (POST /jobs + poll + UI)."""

import time

import pytest
from fastapi.testclient import TestClient

import podcast_transcriber.api as api_module
import podcast_transcriber.job_backend as job_backend

client = TestClient(api_module.app)


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))


def _fake_transcribe(**kwargs):
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "A"}]}


def _poll(job_id, timeout=5.0):
    deadline = time.time() + timeout
    body = {}
    while time.time() < deadline:
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    return body


def _submit(output_format="txt"):
    return client.post(
        "/jobs",
        data={"output_format": output_format, "diarize": "false"},
        files={"file": ("episode.mp3", b"audio-bytes", "audio/mpeg")},
    )


def test_submit_returns_202_with_job_id(monkeypatch):
    monkeypatch.setattr(job_backend, "transcribe_audio", _fake_transcribe)
    resp = _submit()
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"]
    assert body["status_url"] == f"/jobs/{body['job_id']}"
    assert resp.headers["Location"] == body["status_url"]


def test_poll_returns_transcript_when_done(monkeypatch):
    monkeypatch.setattr(job_backend, "transcribe_audio", _fake_transcribe)
    job_id = _submit().json()["job_id"]
    body = _poll(job_id)

    assert body["status"] == "done"
    assert "Hello" in body["transcript"]
    assert body["result_url"] == f"/jobs/{job_id}/result"
    assert body["source_name"] == "episode.mp3"


def test_submit_rejects_bad_format():
    resp = client.post(
        "/jobs",
        data={"output_format": "docx"},
        files={"file": ("episode.mp3", b"audio", "audio/mpeg")},
    )
    assert resp.status_code == 400
    assert "Invalid format" in resp.json()["detail"]


def test_submit_rejects_bad_extension():
    resp = client.post(
        "/jobs",
        data={"output_format": "txt"},
        files={"file": ("notes.pdf", b"nope", "application/pdf")},
    )
    assert resp.status_code == 400
    assert "Unsupported audio format" in resp.json()["detail"]


def test_job_reports_error_status(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr(job_backend, "transcribe_audio", boom)
    job_id = _submit().json()["job_id"]
    body = _poll(job_id)

    assert body["status"] == "error"
    assert "whisper exploded" in body["error"]
    assert body["transcript"] is None


def test_unknown_job_returns_404():
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_jobs_list_includes_submitted(monkeypatch):
    monkeypatch.setattr(job_backend, "transcribe_audio", _fake_transcribe)
    job_id = _submit().json()["job_id"]
    _poll(job_id)

    jobs = client.get("/jobs").json()
    row = next(j for j in jobs if j["job_id"] == job_id)
    assert row["status"] == "done"
    assert row["result_url"] == f"/jobs/{job_id}/result"


def test_result_download_returns_transcript_file(monkeypatch):
    monkeypatch.setattr(job_backend, "transcribe_audio", _fake_transcribe)
    job_id = _submit().json()["job_id"]
    _poll(job_id)

    resp = client.get(f"/jobs/{job_id}/result")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "Hello" in resp.text


def test_result_download_unknown_job_404():
    assert client.get("/jobs/does-not-exist/result").status_code == 404


def test_index_served_as_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Podcast Transcriber" in resp.text


def test_api_banner_reports_backend():
    resp = client.get("/api")
    assert resp.status_code == 200
    assert resp.json()["service"] == "podcast-transcriber"
    assert "job_backend" in resp.json()
