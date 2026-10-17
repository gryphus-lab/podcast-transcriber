"""Audio transcription using WhisperX with optional speaker diarization.

Memory discipline
-----------------
On CPU, transcribing a full-length podcast (e.g. a ~45-minute, 100 MB MP3) with
``large-v3`` plus pyannote diarization is memory-bound, not just slow. The three
stages each hold large allocations:

* Whisper (faster-whisper / ctranslate2) keeps the acoustic model resident.
* Alignment loads a second (wav2vec2) model.
* pyannote feeds the *entire* decoded waveform through a sliding-window
  segmentation + embedding pipeline, whose intermediate tensors scale with the
  audio length.

If all three stay resident simultaneously, peak RSS for a 45-minute file reaches
~7.5 GiB, which overruns a default 8 GiB Docker Desktop VM and triggers a global
(VM-wide) OOM kill of the worker. With ``restart: unless-stopped`` the container
relaunches and the in-memory job is lost - observed as the transcriber "dying and
restarting" mid-run.

The functions below bound peak memory by releasing each model as soon as its
stage is done (so pyannote never runs alongside the whisper + alignment models)
and by capping native thread/allocator growth. This keeps the full pipeline well
under the VM ceiling on a long file instead of relying on an 8 GiB limit the VM
cannot actually honour.
"""

import gc
import os
from pathlib import Path

import torch  # noqa: E402
import whisperx  # noqa: E402

# Whisper decode batch size. Each batch item is a 30 s mel window held in RAM;
# 16 is whisperx's default but is tuned for GPU. On CPU it needlessly inflates
# peak memory, so default to a smaller batch and let callers/env override.
DEFAULT_BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))

# pyannote's diarization-3.1 pipeline runs at 16 kHz.
_DIARIZE_SAMPLE_RATE = 16000

# Cap intra-op CPU threads. Beyond a handful of threads the ML stack gives
# little speedup here but each worker thread grows its own allocator arenas,
# inflating peak RSS - a real risk on a memory-constrained container. Honour an
# explicit OMP_NUM_THREADS/TORCH_NUM_THREADS if the operator set one.
_env_threads = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
if _env_threads:
    try:
        torch.set_num_threads(max(1, int(_env_threads)))
    except (ValueError, RuntimeError):  # pragma: no cover - defensive
        pass
else:
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2))))


def _reclaim() -> None:
    """Force Python + torch to release freed allocations back to the OS.

    Deleting a model drops its Python references, but CPython won't necessarily
    hand the arena back promptly, and torch caches allocations. An explicit
    collect (plus emptying the CUDA cache when relevant) keeps the resident set
    from carrying a finished stage's memory into the next one.
    """
    gc.collect()
    if torch.cuda.is_available():  # pragma: no cover - CPU-only in CI/containers
        torch.cuda.empty_cache()


def transcribe_audio(
    audio_path: Path,
    model_name: str = "large-v3",
    language: str = "en",
    diarize: bool = True,
    hf_token: str = "",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = DEFAULT_BATCH_SIZE,
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
        batch_size: Whisper decode batch size. Smaller values lower peak memory
            on CPU at a modest throughput cost.

    Returns:
        Dict with 'segments' list. Each segment has 'start', 'end', 'text',
        and optionally 'speaker' keys.
    """
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
    result = model.transcribe(audio, batch_size=batch_size)

    # The whisper model is no longer needed once transcription is done. Release
    # it before loading the alignment model so the two never stack in RAM.
    del model
    _reclaim()

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

    # Alignment is done; free its model (and metadata) before diarization, whose
    # peak is the largest stage for long audio.
    del model_a, metadata
    _reclaim()

    # Speaker diarization
    if diarize and hf_token:
        result = _diarize(result, audio, hf_token, device)

    return result


def _diarize(result: dict, audio, hf_token: str, device: str) -> dict:
    """Run pyannote speaker diarization and merge speakers into ``result``.

    Split out from :func:`transcribe_audio` so the diarization pipeline, the
    waveform tensor, and the intermediate segments are all scoped to this frame
    and released as soon as speakers have been assigned - keeping diarization's
    peak from lingering after it completes.
    """
    from pyannote.audio import Pipeline

    # pyannote.audio 3.x names the auth parameter ``use_auth_token`` (not
    # ``token``); passing ``token=`` raises TypeError and fails every diarized
    # job.
    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if diarize_pipeline is None:
        raise RuntimeError(
            "Failed to load pyannote diarization model. "
            "Check your HF_TOKEN and model access at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    diarize_pipeline = diarize_pipeline.to(torch.device(device))

    # Reuse the whisperx-loaded audio (already decoded via ffmpeg) as a
    # waveform dict to bypass torchcodec/torchaudio backend issues.
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)

    diarize_output = diarize_pipeline(
        {"waveform": waveform, "sample_rate": _DIARIZE_SAMPLE_RATE}
    )

    # Convert pyannote Annotation to DataFrame for whisperx.
    # The pipeline returns an Annotation object directly.
    import pandas as pd

    diarize_segments = pd.DataFrame(
        [
            {"start": seg.start, "end": seg.end, "speaker": speaker}
            for seg, _, speaker in diarize_output.itertracks(yield_label=True)
        ]
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Drop the pipeline + large intermediates now that speakers are merged.
    del diarize_pipeline, waveform, diarize_output, diarize_segments
    _reclaim()

    return result
