"""Tests for the CLI entry point in __main__.py."""

import argparse
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import podcast_transcriber.__main__ as main_module


class TestWarningAndLoggingSuppression:
    """Verify noisy warnings/loggers are silenced at import time."""

    def test_pythonwarnings_env_var_set(self):
        assert os.environ.get("PYTHONWARNINGS") == "ignore"

    def test_noisy_loggers_set_to_error(self):
        for name in [
            "lightning.pytorch.utilities.migration",
            "lightning",
            "whisperx",
            "pyannote",
            "pyannote.audio",
        ]:
            assert logging.getLogger(name).level == logging.ERROR


def _make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "audio_file": Path("podcast.m4a"),
        "model": "large-v3",
        "language": "en",
        "diarize": True,
        "no_diarize": False,
        "hf_token": "hf_test_token",
        "output_dir": Path("output"),
        "format": "txt",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestMainSuccessPath:
    """Verify the transcription pipeline runs without a spinner."""

    def test_returns_zero_and_writes_output(self, tmp_path):
        args = _make_args(output_dir=tmp_path)
        fake_result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
        output_path = tmp_path / "podcast.txt"

        with (
            patch.object(main_module, "parse_args", return_value=args),
            patch.object(main_module, "validate_inputs", return_value=True),
            patch.object(
                main_module, "transcribe_audio", return_value=fake_result
            ) as mock_transcribe,
            patch.object(
                main_module, "format_transcript", return_value="hi\n"
            ) as mock_format,
            patch.object(
                main_module, "write_transcript", return_value=output_path
            ) as mock_write,
            patch.object(main_module.console, "status") as mock_status,
        ):
            exit_code = main_module.main()

        assert exit_code == 0
        mock_transcribe.assert_called_once_with(
            audio_path=args.audio_file,
            model_name=args.model,
            language=args.language,
            diarize=True,
            hf_token=args.hf_token,
        )
        mock_format.assert_called_once_with(fake_result, output_format=args.format)
        mock_write.assert_called_once_with(
            "hi\n",
            audio_path=args.audio_file,
            output_dir=tmp_path,
            output_format=args.format,
        )
        # The spinner/status context manager must no longer be used.
        mock_status.assert_not_called()

    def test_creates_output_dir_if_missing(self, tmp_path):
        output_dir = tmp_path / "nested" / "output"
        args = _make_args(output_dir=output_dir)

        with (
            patch.object(main_module, "parse_args", return_value=args),
            patch.object(main_module, "validate_inputs", return_value=True),
            patch.object(
                main_module, "transcribe_audio", return_value={"segments": []}
            ),
            patch.object(main_module, "format_transcript", return_value=""),
            patch.object(
                main_module, "write_transcript", return_value=output_dir / "podcast.txt"
            ),
        ):
            exit_code = main_module.main()

        assert exit_code == 0
        assert output_dir.exists()


class TestMainValidationFailure:
    """Verify main() short-circuits when input validation fails."""

    def test_returns_one_when_validation_fails(self):
        args = _make_args()

        with (
            patch.object(main_module, "parse_args", return_value=args),
            patch.object(main_module, "validate_inputs", return_value=False),
            patch.object(main_module, "transcribe_audio") as mock_transcribe,
        ):
            exit_code = main_module.main()

        assert exit_code == 1
        mock_transcribe.assert_not_called()


class TestMainKeyboardInterrupt:
    """Verify Ctrl-C during transcription is handled gracefully."""

    def test_returns_130_on_keyboard_interrupt(self, tmp_path, capsys):
        args = _make_args(output_dir=tmp_path)

        with (
            patch.object(main_module, "parse_args", return_value=args),
            patch.object(main_module, "validate_inputs", return_value=True),
            patch.object(
                main_module, "transcribe_audio", side_effect=KeyboardInterrupt
            ),
            patch.object(main_module, "format_transcript") as mock_format,
            patch.object(main_module, "write_transcript") as mock_write,
        ):
            exit_code = main_module.main()

        assert exit_code == 130
        mock_format.assert_not_called()
        mock_write.assert_not_called()
        captured = capsys.readouterr()
        assert "Interrupted" in captured.out

    def test_keyboard_interrupt_from_format_transcript(self, tmp_path):
        args = _make_args(output_dir=tmp_path)

        with (
            patch.object(main_module, "parse_args", return_value=args),
            patch.object(main_module, "validate_inputs", return_value=True),
            patch.object(
                main_module, "transcribe_audio", return_value={"segments": []}
            ),
            patch.object(
                main_module, "format_transcript", side_effect=KeyboardInterrupt
            ),
            patch.object(main_module, "write_transcript") as mock_write,
        ):
            exit_code = main_module.main()

        assert exit_code == 130
        mock_write.assert_not_called()


class TestMainGenericException:
    """Verify unexpected errors are caught and reported without crashing."""

    def test_returns_one_and_prints_error(self, tmp_path, capsys):
        args = _make_args(output_dir=tmp_path)

        with (
            patch.object(main_module, "parse_args", return_value=args),
            patch.object(main_module, "validate_inputs", return_value=True),
            patch.object(
                main_module,
                "transcribe_audio",
                side_effect=ValueError("boom"),
            ),
        ):
            exit_code = main_module.main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "boom" in captured.out

    def test_error_during_write_is_caught(self, tmp_path):
        args = _make_args(output_dir=tmp_path)

        with (
            patch.object(main_module, "parse_args", return_value=args),
            patch.object(main_module, "validate_inputs", return_value=True),
            patch.object(
                main_module, "transcribe_audio", return_value={"segments": []}
            ),
            patch.object(main_module, "format_transcript", return_value=""),
            patch.object(
                main_module,
                "write_transcript",
                side_effect=OSError("disk full"),
            ),
        ):
            exit_code = main_module.main()

        assert exit_code == 1