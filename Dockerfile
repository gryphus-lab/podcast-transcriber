# Pull the official Python image from the AWS ECR Public mirror rather than
# Docker Hub: anonymous Docker Hub pulls are rate-limited (HTTP 429) in CI,
# which intermittently fails the image builds. ECR Public mirrors the same
# official images without anonymous pull limits.
FROM public.ecr.aws/docker/library/python:3.11-slim-trixie AS base

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

# Install third-party dependencies first (own cached layer, no project yet).
# Prune the cache, clear the ctranslate2 exec-stack flag (best-effort, scoped so
# it can't mask a failure) and create the output + jobs dirs in one layer
# (docker:S7031). Both dirs are backed by named volumes in docker-compose; create
# and chown them to appuser here so the non-root process can write to the mounts
# (a bind/named volume otherwise mounts root-owned and the app fails on write).
#
# docker:S8541 (--no-build) is intentionally not applied here: a required
# transitive dependency (antlr4-python3-runtime, via whisperx) is published as a
# source distribution only, so a source build is unavoidable. The build inputs
# come from the pinned lockfile, so this is safe.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    && uv cache prune \
    && { find /app/.venv -name "libctranslate2*.so*" -exec patchelf --clear-execstack {} \; || true; } \
    && mkdir -p /app/output /app/jobs \
    && chown appuser:appuser /app/output /app/jobs

COPY --chown=appuser:appuser src/ src/

# Install the first-party project so its console entrypoints (transcribe /
# transcribe-api, declared in pyproject [project.scripts]) are registered.
# Only the local, trusted package is built here; its dependencies were already
# installed above. docker:S8541 (--no-build) is not applied for the same
# first-party-build reason noted above.
RUN uv sync --frozen --no-dev

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    OUTPUT_DIR=/app/output \
    JOBS_DIR=/app/jobs \
    WHISPER_MODEL=large-v3 \
    LANGUAGE=en

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "podcast_transcriber.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ========== CONVERTER TARGET ==========
FROM base AS converter

# Install dependencies only (never the project); the converter runs via its
# module path (see CMD) and is imported from src/ on PYTHONPATH.
# --no-install-project keeps the first-party build out of the image and
# --no-build forbids dependency setup/build scripts (docker:S8541); deps ship as
# wheels. No lockfile exists for the converter pyproject, so --frozen is not
# used. Cache prune, exec-stack clear (best-effort, scoped) and output dir are
# merged into one layer (docker:S7031).
COPY --chown=appuser:appuser pyproject.converter.toml pyproject.toml
RUN uv sync --no-dev --no-install-project --no-build \
    && uv cache prune \
    && { find /app/.venv -name "libctranslate2*.so*" -exec patchelf --clear-execstack {} \; || true; } \
    && mkdir -p /app/output \
    && chown appuser:appuser /app/output

COPY --chown=appuser:appuser src/ src/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
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
