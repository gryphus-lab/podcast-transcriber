"""Configuration variables loaded from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the current working directory (project root when run via uv)
load_dotenv(Path.cwd() / ".env")

# HuggingFace token for pyannote speaker diarization models
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Whisper model size: tiny, base, small, medium, large-v3
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

# Default language (ISO 639-1)
LANGUAGE = os.environ.get("LANGUAGE", "en")

# Output directory
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))

# Maximum upload size (500MB default)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 500 * 1024 * 1024))

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".wma", ".aac"}
