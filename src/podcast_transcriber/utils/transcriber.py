"""Audio transcription using WhisperX with optional speaker diarization."""

import logging
import warnings
from pathlib import Path

# Suppress noisy warnings from torchcodec/pyannote/lightning/whisperx
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("whisperx.vads.pyannote").setLevel(logging.ERROR)

import torch  # noqa: E402
import whisperx  # noqa: E402


def transcribe_audio(
    audio_path: Path,
    model_name: str = "large-v3",
    language: str = "en",
    diarize: bool = True,
    hf_token: str = "",
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict:
    """Transcribe an audio file using WhisperX.

    Args:
        audio_path: Path to the audio file.
        model_name: Whisper model size (tiny, base, small, medium, large-v3).
        language: ISO 639-1 language code.
        diarize: Whether to perform speaker diarization.
        hf_token: HuggingFace token for pyannote diarization models.
        device: Compute device ("cpu" or "cuda").
        compute_type: Compute precision ("int8", "float16", "float32").

    Returns:
        Dict with 'segments' list. Each segment has 'start', 'end', 'text',
        and optionally 'speaker' keys.
    """
    if device == "cpu" and torch.backends.mps.is_available():
        # WhisperX doesn't fully support MPS yet, stick with CPU
        device = "cpu"

    # Load model
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    # Load audio
    audio = whisperx.load_audio(str(audio_path))

    # Transcribe
    result = model.transcribe(audio, batch_size=16)

    # Align timestamps
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Speaker diarization
    if diarize and hf_token:
        from pyannote.audio import Pipeline

        diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        if diarize_pipeline is None:
            raise RuntimeError(
                "Failed to load pyannote diarization model. "
                "Check your HF_TOKEN and model access at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        diarize_pipeline = diarize_pipeline.to(torch.device(device))

        # Load audio as in-memory waveform to bypass torchcodec issues.
        # pyannote accepts {"waveform": Tensor, "sample_rate": int}.
        # Use the whisperx-loaded numpy audio (already decoded) converted to tensor,
        # since torchaudio may lack the backend for .m4a on some setups.

        # whisperx.load_audio returns float32 numpy at 16kHz mono
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        sample_rate = 16000

        diarize_segments = diarize_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

    return result
