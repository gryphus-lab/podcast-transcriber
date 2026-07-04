"""Tests for the WhisperX transcription orchestration."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import podcast_transcriber.utils.transcriber as transcriber_module


def test_transcribe_audio_runs_whisperx_without_diarization(tmp_path, monkeypatch):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"audio")
    audio = MagicMock(name="audio")
    model = MagicMock()
    model.transcribe.return_value = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}
    aligned_result = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}

    monkeypatch.setattr(
        transcriber_module.whisperx, "load_model", MagicMock(return_value=model)
    )
    monkeypatch.setattr(
        transcriber_module.whisperx, "load_audio", MagicMock(return_value=audio)
    )
    monkeypatch.setattr(
        transcriber_module.whisperx,
        "load_align_model",
        MagicMock(return_value=("align-model", {"lang": "en"})),
    )
    monkeypatch.setattr(
        transcriber_module.whisperx, "align", MagicMock(return_value=aligned_result)
    )
    monkeypatch.setattr(
        transcriber_module.whisperx, "assign_word_speakers", MagicMock()
    )

    result = transcriber_module.transcribe_audio(
        audio_path,
        model_name="tiny",
        language="en",
        diarize=False,
        hf_token="",
        device="cpu",
        compute_type="int8",
    )

    assert result == aligned_result
    transcriber_module.whisperx.load_model.assert_called_once_with(
        "tiny", device="cpu", compute_type="int8", language="en"
    )
    transcriber_module.whisperx.load_audio.assert_called_once_with(str(audio_path))
    model.transcribe.assert_called_once_with(audio, batch_size=16)
    transcriber_module.whisperx.align.assert_called_once_with(
        [{"start": 0, "end": 1, "text": "Hi"}],
        "align-model",
        {"lang": "en"},
        audio,
        "cpu",
        return_char_alignments=False,
    )
    transcriber_module.whisperx.assign_word_speakers.assert_not_called()


def test_transcribe_audio_assigns_speakers_when_token_is_available(
    tmp_path, monkeypatch
):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"audio")
    audio = MagicMock(name="audio")
    waveform = MagicMock(name="waveform")
    diarization_result = SimpleNamespace(
        itertracks=lambda yield_label: [
            (SimpleNamespace(start=0.0, end=1.5), None, "SPEAKER_00")
        ]
    )
    pipeline = MagicMock()
    pipeline.to.return_value = pipeline
    pipeline.return_value = diarization_result
    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value = pipeline

    model = MagicMock()
    model.transcribe.return_value = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}
    aligned_result = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}
    speaker_result = {
        "segments": [
            {"start": 0, "end": 1, "text": "Hi", "speaker": "SPEAKER_00"}
        ]
    }

    monkeypatch.setattr(
        transcriber_module.whisperx, "load_model", MagicMock(return_value=model)
    )
    monkeypatch.setattr(
        transcriber_module.whisperx, "load_audio", MagicMock(return_value=audio)
    )
    monkeypatch.setattr(
        transcriber_module.whisperx,
        "load_align_model",
        MagicMock(return_value=("align-model", {})),
    )
    monkeypatch.setattr(
        transcriber_module.whisperx, "align", MagicMock(return_value=aligned_result)
    )
    monkeypatch.setattr(
        transcriber_module.whisperx,
        "assign_word_speakers",
        MagicMock(return_value=speaker_result),
    )
    monkeypatch.setattr(
        transcriber_module.torch,
        "from_numpy",
        MagicMock(return_value=SimpleNamespace(unsqueeze=lambda dim: waveform)),
    )
    monkeypatch.setattr(
        transcriber_module.torch, "device", MagicMock(return_value="cpu-device")
    )
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=pipeline_cls),
    )

    result = transcriber_module.transcribe_audio(
        audio_path,
        diarize=True,
        hf_token="hf_token",
        device="cpu",
    )

    assert result == speaker_result
    pipeline_cls.from_pretrained.assert_called_once_with(
        "pyannote/speaker-diarization-3.1", token="hf_token"
    )
    pipeline.to.assert_called_once_with("cpu-device")
    pipeline.assert_called_once_with({"waveform": waveform, "sample_rate": 16000})
    transcriber_module.whisperx.assign_word_speakers.assert_called_once()
    diarize_segments = transcriber_module.whisperx.assign_word_speakers.call_args.args[0]
    assert list(diarize_segments.to_dict("records")) == [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}
    ]


def test_transcribe_audio_raises_when_diarization_model_cannot_load(
    tmp_path, monkeypatch
):
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"audio")
    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value = None

    model = MagicMock()
    model.transcribe.return_value = {"segments": []}

    monkeypatch.setattr(
        transcriber_module.whisperx, "load_model", MagicMock(return_value=model)
    )
    monkeypatch.setattr(
        transcriber_module.whisperx, "load_audio", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(
        transcriber_module.whisperx,
        "load_align_model",
        MagicMock(return_value=("align-model", {})),
    )
    monkeypatch.setattr(
        transcriber_module.whisperx,
        "align",
        MagicMock(return_value={"segments": []}),
    )
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=pipeline_cls),
    )

    try:
        transcriber_module.transcribe_audio(
            audio_path,
            diarize=True,
            hf_token="hf_token",
            device="cpu",
        )
    except RuntimeError as exc:
        assert "Failed to load pyannote diarization model" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
