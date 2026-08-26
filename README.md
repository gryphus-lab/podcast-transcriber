# Podcast Transcriber

Transcribe podcast audio with automatic speaker diarization (labeling who said
what) using [WhisperX](https://github.com/m-bain/whisperX), plus FFmpeg-based
audio/video conversion. Available as a **CLI**, a **FastAPI service**, and
**Docker containers**.

## Features

- Transcribes audio using OpenAI Whisper (via WhisperX)
- Speaker diarization - identifies and labels different speakers
- Multiple output formats: plain text, SRT subtitles, JSON
- Audio/video conversion via FFmpeg (mp4, mp3, wav, flac, ogg, mkv, webm, aac)
- Supports `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`, `.wma`, `.aac`
- Configurable model size (tiny to large-v3)
- Three interfaces: CLI, REST API (FastAPI + Uvicorn), and Docker
- Apple Silicon compatible (CPU mode)

## Requirements

- Python 3.11 (not 3.13 - WhisperX has compatibility issues with newer Python)
- FFmpeg installed (`brew install ffmpeg`)
- HuggingFace token (for speaker diarization)
- `uv` package manager
- Docker (optional, for containerized services)

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
│   ├── api.py                   # FastAPI app: /transcribe and /convert
│   ├── converter_service.py     # Standalone converter FastAPI service
│   ├── config.py                # Environment variable loading
│   └── utils/
│       ├── __init__.py
│       ├── transcriber.py       # WhisperX transcription + diarization
│       ├── formatter.py         # Output formatting (txt, srt, json)
│       ├── converter.py         # Shared FFmpeg conversion helpers
│       ├── uploads.py           # Bounded upload streaming to temp files
│       └── warnings_filter.py   # Suppresses noisy third-party warnings
├── tests/
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_converter_service.py
│   ├── test_formatter.py
│   ├── test_main.py
│   └── test_transcriber.py
├── output/                      # Default output directory
├── Dockerfile                   # Multi-stage: transcriber + converter targets
├── docker-compose.yml           # Runs both services
├── pyproject.toml               # Transcriber deps
├── pyproject.converter.toml     # Minimal converter deps (no ML libs)
├── mise.toml
├── .env.example
└── .github/workflows/           # ci.yml (tests + SonarQube), trivy.yml (scan)
```

## Usage

### CLI

Basic transcription with speaker diarization:

```bash
uv run transcribe my_podcast.m4a
```

Without diarization (faster, no HF token needed):

```bash
uv run transcribe my_podcast.m4a --no-diarize
```

Specify model and output format:

```bash
uv run transcribe episode.mp3 --model medium --format srt
```

All options:

```bash
uv run transcribe audio.m4a \
  --model large-v3 \
  --language en \
  --diarize \
  --hf-token hf_xxxxx \
  --output-dir ./transcripts \
  --format json
```

Using mise:

```bash
mise run transcribe -- my_podcast.m4a --format srt
```

### REST API

Start the FastAPI server (exposes `/transcribe`, `/convert`, and `/health`):

```bash
uv run transcribe-api
# or
mise run api
```

By default it listens on `0.0.0.0:8000` (override with `HOST` / `PORT`).
Interactive docs are available at `http://localhost:8000/docs`.

Transcribe an upload:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@my_podcast.m4a" \
  -F "model=large-v3" \
  -F "language=en" \
  -F "diarize=true" \
  -F "output_format=srt" \
  -o transcript.srt
```

Convert an upload to MP4:

```bash
curl -X POST http://localhost:8000/convert \
  -F "file=@my_podcast.m4a" \
  -F "output_format=mp4" \
  -o my_podcast.mp4
```

### Docker

Two services are built from a single multi-stage `Dockerfile`:

| Service       | Port | Purpose                                        |
| ------------- | ---- | ---------------------------------------------- |
| `transcriber` | 8000 | WhisperX transcription + diarization + convert |
| `converter`   | 8001 | Lightweight FFmpeg-only conversion (no ML)     |

Start both with Docker Compose:

```bash
docker compose up --build -d
# or
mise run docker-up
```

Stop them:

```bash
docker compose down
# or
mise run docker-down
```

Set `HF_TOKEN` in your environment (or `.env`) so the transcriber can diarize.
The images are pinned to `linux/amd64` because some ML dependencies only ship
wheels for that platform.

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

| Variable          | Default    | Description                                  |
| ----------------- | ---------- | -------------------------------------------- |
| `HF_TOKEN`        | (empty)    | HuggingFace token for diarization            |
| `WHISPER_MODEL`   | `large-v3` | Model size                                   |
| `LANGUAGE`        | `en`       | Language code                                |
| `OUTPUT_DIR`      | `output`   | Transcript/conversion output directory       |
| `MAX_UPLOAD_SIZE` | `524288000` | Max API upload size in bytes (500 MB)       |
| `HOST`            | `0.0.0.0`  | API server bind address                      |
| `PORT`            | `8000`     | API server port                              |

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
# Run tests (with coverage)
uv run pytest
# or
mise run test

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

### CI

- `ci.yml` - runs the test suite and a SonarQube scan on push / PR to `main`.
- `trivy.yml` - builds both Docker targets and scans them for CRITICAL/HIGH
  vulnerabilities, uploading results to the GitHub Security tab.
