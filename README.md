# Podcast Transcriber

A CLI tool for transcribing podcast audio files with automatic speaker diarization (labeling who said what) using [WhisperX](https://github.com/m-bain/whisperX).

## Features

- Transcribes audio files using OpenAI Whisper (via WhisperX)
- Speaker diarization - identifies and labels different speakers
- Multiple output formats: plain text, SRT subtitles, JSON
- Supports `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`, `.wma`, `.aac`
- Configurable model size (tiny to large-v3)
- Apple Silicon compatible (CPU mode)

## Requirements

- Python 3.11 (not 3.13 - WhisperX has compatibility issues with newer Python)
- FFmpeg installed (`brew install ffmpeg`)
- HuggingFace token (for speaker diarization)
- `uv` package manager

## Installation

1. Sync dependencies:

   ```bash
   uv sync
   ```

2. Copy and configure environment:

   ```bash
   cp .env.example .env
   ```

3. Set up your HuggingFace token in `.env`:
   - Create a token at: https://huggingface.co/settings/tokens
   - Accept pyannote model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Set `HF_TOKEN=hf_your_token_here`

## Project Structure

```text
podcast-transcriber/
├── src/podcast_transcriber/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # CLI entry point + argument parsing
│   ├── config.py                # Environment variable loading
│   └── utils/
│       ├── __init__.py
│       ├── transcriber.py       # WhisperX transcription + diarization
│       └── formatter.py         # Output formatting (txt, srt, json)
├── tests/
│   ├── test_config.py
│   └── test_formatter.py
├── output/                      # Default transcript output directory
├── docs/
├── .github/workflows/ci.yml
├── pyproject.toml
├── mise.toml
├── .env.example
└── README.md
```

## Usage

### Basic transcription with speaker diarization

```bash
uv run transcribe my_podcast.m4a
```

### Without diarization (faster, no HF token needed)

```bash
uv run transcribe my_podcast.m4a --no-diarize
```

### Specify model and output format

```bash
uv run transcribe episode.mp3 --model medium --format srt
```

### All options

```bash
uv run transcribe audio.m4a \
  --model large-v3 \
  --language en \
  --diarize \
  --hf-token hf_xxxxx \
  --output-dir ./transcripts \
  --format txt
```

### Using mise

```bash
mise run transcribe -- my_podcast.m4a --format srt
```

## Output Formats

### Plain text (`--format txt`)

```
[SPEAKER_00]
  Hello and welcome to the podcast.
  Today we have a special guest.

[SPEAKER_01]
  Thanks for having me.
```

### SRT subtitles (`--format srt`)

```
1
00:00:00,000 --> 00:00:03,500
[SPEAKER_00] Hello and welcome to the podcast.

2
00:00:03,500 --> 00:00:06,200
[SPEAKER_01] Thanks for having me.
```

### JSON (`--format json`)

```json
[
  {
    "start": 0.0,
    "end": 3.5,
    "text": "Hello and welcome.",
    "speaker": "SPEAKER_00"
  },
  {
    "start": 3.5,
    "end": 6.2,
    "text": "Thanks for having me.",
    "speaker": "SPEAKER_01"
  }
]
```

## Configuration

All settings can be configured via `.env` or CLI arguments (CLI takes precedence):

| Variable        | Default    | Description                       |
| --------------- | ---------- | --------------------------------- |
| `HF_TOKEN`      | (empty)    | HuggingFace token for diarization |
| `WHISPER_MODEL` | `large-v3` | Model size                        |
| `LANGUAGE`      | `en`       | Language code                     |
| `OUTPUT_DIR`    | `output`   | Transcript output directory       |

## Troubleshooting

### `Could not load libtorchcodec`

Use Python 3.11 (not 3.13). Create a clean environment:

```bash
conda create -n whisperx python=3.11 -y
conda activate whisperx
```

### FFmpeg not found

```bash
brew install ffmpeg
```

### HuggingFace model access denied

You must accept the model terms:

1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept the user agreement
3. Ensure your token has `read` access

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

# podcast-transcriber
