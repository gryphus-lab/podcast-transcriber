FROM --platform=linux/amd64 python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create output directory and non-root user
RUN mkdir -p /app/output && useradd --create-home appuser

# ========== TRANSCRIBER TARGET ==========
FROM base AS transcriber

# Copy transcriber dependencies and source
COPY pyproject.toml .
COPY src/ src/

# Install with uv
RUN uv sync --no-dev

USER appuser

EXPOSE 8000

ENV BUILD_TIMESTAMP=2025-06-20
ENV HOST=0.0.0.0
ENV PORT=8000
ENV OUTPUT_DIR=/app/output
ENV WHISPER_MODEL=large-v3
ENV LANGUAGE=en

CMD [".venv/bin/uvicorn", "podcast_transcriber.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ========== CONVERTER TARGET ==========
FROM base AS converter

# Copy converter dependencies and service
COPY pyproject.converter.toml pyproject.toml
COPY src/podcast_transcriber/converter_service.py converter_service.py

# Install with uv
RUN uv sync --no-dev

USER appuser

EXPOSE 8001

ENV BUILD_TIMESTAMP=2025-06-20
ENV HOST=0.0.0.0
ENV PORT=8001
ENV OUTPUT_DIR=/app/output

CMD [".venv/bin/uvicorn", "converter_service:app", "--host", "0.0.0.0", "--port", "8001"]
