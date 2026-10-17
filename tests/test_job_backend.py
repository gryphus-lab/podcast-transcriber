"""Tests for job_backend transcription isolation + OOM handling.

The default ``memory`` backend runs the heavy transcription in a short-lived
spawned subprocess so an out-of-memory kill fails just the job instead of
taking down the API container. These tests exercise that boundary without
loading any real ML models.
"""

import pytest

import podcast_transcriber.job_backend as job_backend


def _write_stub_audio(tmp_path):
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio-bytes")
    return audio


def test_isolation_enabled_by_default(monkeypatch):
    monkeypatch.delenv("ISOLATE_TRANSCRIPTION", raising=False)
    assert job_backend._isolation_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "NO", "False"])
def test_isolation_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("ISOLATE_TRANSCRIPTION", value)
    assert job_backend._isolation_enabled() is False


def test_run_transcription_in_thread_uses_fake_and_cleans_up(tmp_path, monkeypatch):
    """With isolation off, the patched transcriber runs in-process and the
    staged input file is removed afterwards."""
    monkeypatch.setenv("ISOLATE_TRANSCRIPTION", "0")
    audio = _write_stub_audio(tmp_path)

    monkeypatch.setattr(
        job_backend,
        "transcribe_audio",
        lambda **kw: {"segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}]},
    )

    params = {
        "job_id": "job1",
        "audio_path": str(audio),
        "source_name": "episode.mp3",
        "diarize": False,
        "output_format": "txt",
    }
    result = job_backend.run_transcription(params)

    assert "Hi" in result.transcript
    assert result.output_format == "txt"
    assert result.source_name == "episode.mp3"
    # Staged upload cleaned up.
    assert not audio.exists()


# --- OOM / killed-worker handling -----------------------------------------


def test_isolated_worker_killed_raises_memory_error(tmp_path, monkeypatch):
    """If the child process is killed (as the OOM-killer would), the parent
    translates the broken pool into a clear MemoryError instead of dying.

    Uses the ``_test_kill_worker`` seam so the real spawned subprocess SIGKILLs
    itself - a genuine broken-pool round-trip, not a mock.
    """
    monkeypatch.setenv("ISOLATE_TRANSCRIPTION", "1")
    audio = _write_stub_audio(tmp_path)

    params = {
        "job_id": "job2",
        "audio_path": str(audio),
        "output_format": "txt",
        "_test_kill_worker": True,
    }
    with pytest.raises(MemoryError, match="out of memory"):
        job_backend.run_transcription(params)

    # Even on failure, the staged input is cleaned up by run_transcription.
    assert not audio.exists()


# --- transcription timeout -------------------------------------------------


def test_transcription_timeout_default(monkeypatch):
    monkeypatch.delenv("TRANSCRIPTION_TIMEOUT", raising=False)
    assert job_backend._transcription_timeout() == 3600.0


def test_transcription_timeout_disabled_with_zero(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_TIMEOUT", "0")
    assert job_backend._transcription_timeout() is None


def test_transcription_timeout_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_TIMEOUT", "not-a-number")
    assert job_backend._transcription_timeout() == 3600.0


def test_transcription_timeout_custom(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_TIMEOUT", "42")
    assert job_backend._transcription_timeout() == 42.0


def test_isolated_worker_timeout_raises_memory_error(tmp_path, monkeypatch):
    """A worker that outlives the timeout (e.g. a VM-wide OOM that never raises
    BrokenProcessPool) is force-killed and the job fails cleanly rather than
    hanging on 'running' forever.

    The child blocks past the tiny timeout via the ``_test_sleep_seconds`` seam
    so the parent's ``result(timeout=...)`` raises, exercising the real
    subprocess + force-kill path.
    """
    monkeypatch.setenv("ISOLATE_TRANSCRIPTION", "1")
    monkeypatch.setenv("TRANSCRIPTION_TIMEOUT", "0.5")
    audio = _write_stub_audio(tmp_path)

    params = {
        "job_id": "job3",
        "audio_path": str(audio),
        "output_format": "txt",
        "_test_sleep_seconds": 30,
    }
    with pytest.raises(MemoryError, match="out of memory"):
        job_backend.run_transcription(params)

    # Staged input still cleaned up after a timeout failure.
    assert not audio.exists()
