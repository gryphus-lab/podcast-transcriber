# Pinned to linux/amd64: torch/torchaudio/torchcodec only publish wheels for
# amd64 (no linux/arm64 wheels), so ARM hosts must build via emulation.
FROM --platform=linux/amd64 python:3.11-slim AS base

# Install system dependencies in a single RUN to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Create non-root user early; output directory created as needed
RUN useradd --create-home appuser

# ========== TRANSCRIBER TARGET ==========
FROM base AS transcriber

# execstack is needed to clear the executable-stack flag on the ctranslate2
# shared library (see the RUN step below).
RUN apt-get update && apt-get install -y --no-install-recommends execstack \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer) using the frozen lockfile,
# then copy source and install the project itself.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=appuser:appuser src/ src/
RUN uv sync --frozen --no-dev && uv cache prune

# ctranslate2's bundled .so requests an executable stack, which the loader
# refuses under emulated/hardened runtimes ("cannot enable executable stack").
# Clear the flag so `import ctranslate2` (via whisperx) works.
RUN find /app/.venv -name "libctranslate2*.so*" -exec execstack -c {} \; || true

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

# The converter pyproject declares only third-party deps (no local package
# build config), so install deps without the project and import the copied
# source via PYTHONPATH.
COPY --chown=appuser:appuser pyproject.converter.toml pyproject.toml
RUN uv sync --no-dev --no-install-project && uv cache prune

COPY --chown=appuser:appuser src/ src/

# Create output directory for mounted volumes
RUN mkdir -p /app/output && chown appuser:appuser /app/output

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8001 \
    OUTPUT_DIR=/app/output

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "podcast_transcriber.converter_service:app", "--host", "0.0.0.0", "--port", "8001"]
