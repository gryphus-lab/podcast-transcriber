# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CLI tool that transcribes podcast audio files with speaker diarization using WhisperX. It takes an audio file (`.m4a`, `.mp3`, `.wav`, etc.), runs it through Whisper for speech-to-text, aligns word timestamps, optionally performs speaker diarization via pyannote, and outputs a formatted transcript (plain text, SRT, or JSON).

## Commands

This project is driven by `uv` (Python 3.11, see `mise.toml` for tooling). `mise run <task>` wraps the common ones.

```bash
uv sync                                       # install deps (mise run install)
uv run transcribe audio.m4a                   # transcribe with diarization
uv run transcribe audio.m4a --no-diarize      # transcribe without speaker labels
uv run transcribe audio.m4a --format srt      # output as SRT subtitles
uv run pytest                                 # run tests (mise run test)
uv run ruff check . && uv run ruff format --check .  # lint (mise run lint)
uv run ruff format . && uv run ruff check --fix .    # format (mise run format)
```

## Architecture

Linear pipeline in `__main__.py:main()`:

1. **Parse & validate** - argparse CLI, check file exists and format is supported, validate HF token presence if diarization requested.
2. **Transcribe** - `utils/transcriber.py:transcribe_audio()`:
   - Load WhisperX model
   - Load audio via `whisperx.load_audio()`
   - Transcribe with batched inference
   - Align word-level timestamps via `whisperx.align()`
   - (Optional) Run pyannote diarization pipeline and assign speakers to segments
3. **Format** - `utils/formatter.py:format_transcript()` converts the segment list into txt/srt/json.
4. **Write** - `utils/formatter.py:write_transcript()` saves to `output/<filename>.<format>`.

## Key dependencies

- **whisperx** - core transcription + alignment + diarization orchestration
- **pyannote.audio** - speaker diarization models (requires HF token + model agreement)
- **torch** - ML backend (CPU on macOS, CUDA on Linux/Windows)
- **rich** - terminal output formatting

## Conventions & gotchas

- **Python 3.11 only.** WhisperX has known issues with Python 3.13 (torchcodec/PyTorch compatibility). The `mise.toml` and `pyproject.toml` enforce this.
- `HF_TOKEN` is required for diarization. Without it, use `--no-diarize`. The token must have `read` access and the user must accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1.
- First run downloads models (~3GB for large-v3). Subsequent runs use the cache.
- Speaker labels are generic (`SPEAKER_00`, `SPEAKER_01`) - they don't identify who is who by name.
- The tool runs on CPU by default (Apple Silicon MPS not yet fully supported by WhisperX).
- FFmpeg must be installed system-wide (`brew install ffmpeg`).
