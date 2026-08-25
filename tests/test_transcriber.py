"""Tests for WhisperX transcription orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_transcriber.utils import transcriber


def test_transcribe_without_diarization_aligns_segments():
    """Transcription should load audio, transcribe, and align timestamps."""
    audio = MagicMock()
    model = MagicMock()
    model.transcribe.return_value = {"segments": [{"start": 0, "text": "Hi"}]}
    aligned = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}

    with (
        patch.object(transcriber.whisperx, "load_model", return_value=model) as load_model,
        patch.object(transcriber.whisperx, "load_audio", return_value=audio) as load_audio,
        patch.object(
            transcriber.whisperx,
            "load_align_model",
            return_value=("align_model", "metadata"),
        ) as load_align_model,
        patch.object(transcriber.whisperx, "align", return_value=aligned) as align,
        patch.object(transcriber.whisperx, "assign_word_speakers") as assign_word_speakers,
    ):
        result = transcriber.transcribe_audio(
            Path("podcast.m4a"),
            model_name="medium",
            language="de",
            diarize=False,
            device="cuda",
            compute_type="float16",
        )

    assert result == aligned
    load_model.assert_called_once_with(
        "medium",
        device="cuda",
        compute_type="float16",
        language="de",
    )
    load_audio.assert_called_once_with("podcast.m4a")
    model.transcribe.assert_called_once_with(audio, batch_size=16)
    load_align_model.assert_called_once_with(language_code="de", device="cuda")
    align.assert_called_once_with(
        [{"start": 0, "text": "Hi"}],
        "align_model",
        "metadata",
        audio,
        "cuda",
        return_char_alignments=False,
    )
    assign_word_speakers.assert_not_called()


def test_transcribe_raises_when_diarization_pipeline_cannot_load():
    """A missing pyannote pipeline should produce a clear runtime error."""
    audio = MagicMock()
    model = MagicMock()
    model.transcribe.return_value = {"segments": []}

    with (
        patch.object(transcriber.whisperx, "load_model", return_value=model),
        patch.object(transcriber.whisperx, "load_audio", return_value=audio),
        patch.object(
            transcriber.whisperx,
            "load_align_model",
            return_value=("align_model", "metadata"),
        ),
        patch.object(transcriber.whisperx, "align", return_value={"segments": []}),
        patch("pyannote.audio.Pipeline.from_pretrained", return_value=None),
        pytest.raises(RuntimeError, match="Failed to load pyannote diarization model"),
    ):
        transcriber.transcribe_audio(
            Path("podcast.m4a"),
            diarize=True,
            hf_token="hf_token",
        )
