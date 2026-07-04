"""Tests for the FastAPI transcription and conversion endpoints."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import podcast_transcriber.api as api_module


client = TestClient(api_module.app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_rejects_invalid_output_format():
    response = client.post(
        "/transcribe",
        data={"output_format": "docx"},
        files={"file": ("episode.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 400
    assert "Invalid format" in response.json()["detail"]


def test_transcribe_rejects_unsupported_upload_extension():
    response = client.post(
        "/transcribe",
        data={"output_format": "txt"},
        files={"file": ("notes.pdf", b"nope", "application/pdf")},
    )

    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_transcribe_returns_json_and_removes_temp_file(tmp_path):
    seen_tmp_path = None

    def fake_transcribe_audio(**kwargs):
        nonlocal seen_tmp_path
        seen_tmp_path = kwargs["audio_path"]
        assert seen_tmp_path.read_bytes() == b"audio bytes"
        assert kwargs["model_name"] == "tiny"
        assert kwargs["language"] == "de"
        assert kwargs["diarize"] is False
        assert kwargs["hf_token"] == api_module.HF_TOKEN
        return {
            "segments": [
                {"start": 0.123, "end": 1.987, "text": " Hallo ", "speaker": "Host"}
            ]
        }

    with (
        patch.object(api_module, "OUTPUT_DIR", tmp_path),
        patch.object(api_module, "transcribe_audio", side_effect=fake_transcribe_audio),
    ):
        response = client.post(
            "/transcribe",
            data={
                "model": "tiny",
                "language": "de",
                "diarize": "false",
                "output_format": "json",
            },
            files={"file": ("episode.mp3", b"audio bytes", "audio/mpeg")},
        )

    assert response.status_code == 200
    assert response.json() == [
        {"start": 0.12, "end": 1.99, "text": "Hallo", "speaker": "Host"}
    ]
    assert seen_tmp_path is not None
    assert not Path(seen_tmp_path).exists()


def test_transcribe_errors_are_reported_as_500(tmp_path):
    with (
        patch.object(api_module, "OUTPUT_DIR", tmp_path),
        patch.object(
            api_module, "transcribe_audio", side_effect=RuntimeError("model failed")
        ),
    ):
        response = client.post(
            "/transcribe",
            files={"file": ("episode.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "model failed"


def test_convert_rejects_invalid_output_format():
    response = client.post(
        "/convert",
        data={"output_format": "exe"},
        files={"file": ("episode.m4a", b"audio", "audio/mp4")},
    )

    assert response.status_code == 400
    assert "Invalid output format" in response.json()["detail"]


def test_convert_requires_ffmpeg():
    with patch.object(api_module.shutil, "which", return_value=None):
        response = client.post(
            "/convert",
            data={"output_format": "mp3"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "FFmpeg is not installed on this server."


def test_convert_success_builds_command_and_returns_file(tmp_path):
    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        output_path = Path(cmd[-1])
        output_path.write_bytes(b"converted")
        fake_create_subprocess_exec.cmd = cmd
        fake_create_subprocess_exec.kwargs = kwargs
        return FakeProcess()

    with (
        patch.object(api_module, "OUTPUT_DIR", tmp_path),
        patch.object(api_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
        patch.object(
            api_module.asyncio,
            "create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ),
    ):
        response = client.post(
            "/convert",
            data={"output_format": "mp3"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 200
    assert response.content == b"converted"
    assert response.headers["content-type"].startswith("audio/mpeg")
    cmd = fake_create_subprocess_exec.cmd
    assert cmd[:3] == ("ffmpeg", "-y", "-i")
    assert cmd[-5:] == ("-c:a", "libmp3lame", "-b:a", "192k", str(tmp_path / "episode.mp3"))


def test_convert_reports_ffmpeg_failure(tmp_path):
    class FakeProcess:
        returncode = 1

        async def communicate(self):
            return b"", b"bad codec"

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        return FakeProcess()

    with (
        patch.object(api_module, "OUTPUT_DIR", tmp_path),
        patch.object(api_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
        patch.object(
            api_module.asyncio,
            "create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ),
    ):
        response = client.post(
            "/convert",
            data={"output_format": "flac"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 500
    assert "FFmpeg conversion failed: bad codec" in response.json()["detail"]
