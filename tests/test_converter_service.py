"""Tests for the standalone converter service."""

import asyncio
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import podcast_transcriber.converter_service as converter_module


client = TestClient(converter_module.app)


def test_health_includes_ffmpeg_availability():
    with patch.object(converter_module.shutil, "which", return_value="/usr/bin/ffmpeg"):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ffmpeg": True}


def test_convert_rejects_invalid_format():
    response = client.post(
        "/convert",
        data={"output_format": "txt"},
        files={"file": ("episode.m4a", b"audio", "audio/mp4")},
    )

    assert response.status_code == 400
    assert "Invalid output format" in response.json()["detail"]


def test_convert_requires_ffmpeg():
    with patch.object(converter_module.shutil, "which", return_value=None):
        response = client.post(
            "/convert",
            data={"output_format": "aac"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "FFmpeg is not installed on this server."


def test_convert_success_uses_requested_bitrate_and_media_type(tmp_path):
    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"converted")
        fake_create_subprocess_exec.cmd = cmd
        return FakeProcess()

    with (
        patch.object(converter_module, "OUTPUT_DIR", tmp_path),
        patch.object(converter_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
        patch.object(
            converter_module.asyncio,
            "create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ),
    ):
        response = client.post(
            "/convert",
            data={"output_format": "aac", "audio_bitrate": "256k"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 200
    assert response.content == b"converted"
    assert response.headers["content-type"].startswith("audio/aac")
    assert fake_create_subprocess_exec.cmd[-5:] == (
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        str(tmp_path / "episode.aac"),
    )


def test_convert_reports_ffmpeg_stderr(tmp_path):
    class FakeProcess:
        returncode = 1

        async def communicate(self):
            return b"", b"cannot decode"

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        return FakeProcess()

    with (
        patch.object(converter_module, "OUTPUT_DIR", tmp_path),
        patch.object(converter_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
        patch.object(
            converter_module.asyncio,
            "create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ),
    ):
        response = client.post(
            "/convert",
            data={"output_format": "wav"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 500
    assert "FFmpeg conversion failed: cannot decode" in response.json()["detail"]


def test_convert_reports_timeout(tmp_path):
    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        return FakeProcess()

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    with (
        patch.object(converter_module, "OUTPUT_DIR", tmp_path),
        patch.object(converter_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
        patch.object(
            converter_module.asyncio,
            "create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ),
        patch.object(converter_module.asyncio, "wait_for", side_effect=fake_wait_for),
    ):
        response = client.post(
            "/convert",
            data={"output_format": "webm"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Conversion timed out (5 min)."
