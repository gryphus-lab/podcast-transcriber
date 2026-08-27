FROM python:3.11-slim-trixie AS base

# Install system dependencies in a single RUN to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Create non-root user early
RUN useradd --create-home appuser

# ========== TRANSCRIBER TARGET ==========
FROM base AS transcriber

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=appuser:appuser src/ src/

# Install Python dependencies with uv (no dev deps)
RUN uv sync --frozen --no-dev && uv cache prune

# Clear executable stack flag on ctranslate2 for ARM64 compatibility
RUN find /app/.venv -name "libctranslate2*.so*" -exec patchelf --clear-execstack {} \; || true

# Create output directory for mounted volumes
RUN mkdir -p /app/output && chown appuser:appuser /app/output

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    OUTPUT_DIR=/app/output \
    WHISPER_MODEL=large-v3 \
    LANGUAGE=en

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "podcast_transcriber.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ========== CONVERTER TARGET ==========
FROM base AS converter

COPY --chown=appuser:appuser pyproject.converter.toml pyproject.toml
RUN uv sync --no-dev --no-install-project

COPY --chown=appuser:appuser src/ src/

# Install Python dependencies with uv (no dev deps)
RUN uv sync --no-dev && uv cache prune

# Clear executable stack flag on ctranslate2 for ARM64 compatibility
RUN find /app/.venv -name "libctranslate2*.so*" -exec patchelf --clear-execstack {} \; || true

# Create output directory for mounted volumes
RUN mkdir -p /app/output && chown appuser:appuser /app/output

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8001 \
    OUTPUT_DIR=/app/output

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "podcast_transcriber.converter_service:app", "--host", "0.0.0.0", "--port", "8001"]
