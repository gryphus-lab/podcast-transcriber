"""Tests for the standalone converter service."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import podcast_transcriber.converter_service as converter_module
import podcast_transcriber.utils.converter as converter_util


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
            converter_util.asyncio,
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
    # Check that download filename is still original stem
    assert response.headers["content-disposition"].endswith("episode.aac\"")
    # Output file should have UUID to prevent concurrent overwrites
    output_path = Path(fake_create_subprocess_exec.cmd[-1])
    assert output_path.parent == tmp_path
    assert output_path.name.startswith("episode_")
    assert output_path.name.endswith(".aac")
    assert fake_create_subprocess_exec.cmd[-5:] == (
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        str(output_path),
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
            converter_util.asyncio,
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


def test_convert_prevents_concurrent_overwrites(tmp_path):
    """Verify that concurrent requests don't overwrite the same output file."""
    output_paths = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        output_path = Path(cmd[-1])
        output_path.write_bytes(b"converted")
        output_paths.append(str(output_path))
        return FakeProcess()

    with (
        patch.object(converter_module, "OUTPUT_DIR", tmp_path),
        patch.object(converter_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
        patch.object(
            converter_util.asyncio,
            "create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ),
    ):
        # Make two requests with same filename
        response1 = client.post(
            "/convert",
            data={"output_format": "wav"},
            files={"file": ("episode.m4a", b"audio1", "audio/mp4")},
        )
        response2 = client.post(
            "/convert",
            data={"output_format": "wav"},
            files={"file": ("episode.m4a", b"audio2", "audio/mp4")},
        )

    assert response1.status_code == 200
    assert response2.status_code == 200
    # Verify they used different output files
    assert output_paths[0] != output_paths[1]
    # But download filenames are the same
    assert response1.headers["content-disposition"].endswith("episode.wav\"")
    assert response2.headers["content-disposition"].endswith("episode.wav\"")


def test_convert_reports_timeout(tmp_path):
    async def fake_convert_audio(**kwargs):
        raise TimeoutError

    with (
        patch.object(converter_module, "OUTPUT_DIR", tmp_path),
        patch.object(converter_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
        patch.object(converter_module, "convert_audio", side_effect=fake_convert_audio),
    ):
        response = client.post(
            "/convert",
            data={"output_format": "webm"},
            files={"file": ("episode.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Conversion timed out."


def test_convert_rejects_oversized_upload():
    """Test that uploads exceeding MAX_UPLOAD_SIZE are rejected."""
    with (
        patch.object(converter_module, "MAX_UPLOAD_SIZE", 100),
        patch.object(converter_module.shutil, "which", return_value="/usr/bin/ffmpeg"),
    ):
        response = client.post(
            "/convert",
            data={"output_format": "mp3"},
            files={"file": ("episode.m4a", b"x" * 200, "audio/mp4")},
        )

    assert response.status_code == 413
    assert "exceeds maximum size" in response.json()["detail"]
