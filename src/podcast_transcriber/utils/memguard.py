"""Startup memory sanity check.

Diarized transcription of long audio with ``large-v3`` + pyannote peaks at
roughly 7.5-12 GB of RSS. If the memory the process can *actually* use is below
that, the run will be OOM-killed mid-way. This module surfaces the problem at
startup instead of letting it manifest as the opaque "transcriber dies and
restarts" symptom.

The subtle failure this guards against: a ``docker-compose`` ``mem_limit`` set
*higher* than the Docker Desktop VM's total RAM is unenforceable. Docker cannot
grant a container more memory than the VM has, so the declared limit is a
phantom - the real ceiling is the (smaller) VM size, and the OOM-killer runs
VM-wide and non-deterministically. Reading the live cgroup limit (what the
kernel will enforce) and clamping it to physical RAM reveals the true ceiling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Below this many GiB, long-audio large-v3 + diarization is likely to be
# OOM-killed. Chosen from observed peak RSS (~7.5 GiB) plus headroom.
SAFE_FLOOR_BYTES = 10 * 1024**3

_UNLIMITED = 1 << 62  # cgroup "max" sentinel threshold


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def cgroup_memory_limit_bytes() -> Optional[int]:
    """Return the effective cgroup memory limit in bytes, or None if unlimited.

    Supports cgroup v2 (``memory.max``) and v1
    (``memory.limit_in_bytes``). A value at/above the unlimited sentinel, or the
    literal ``max``, is reported as None (no enforced limit).
    """
    v2 = Path("/sys/fs/cgroup/memory.max")
    if v2.exists():
        raw = v2.read_text().strip() if v2.is_file() else ""
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return None if value >= _UNLIMITED else value

    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    value = _read_int(v1)
    if value is None or value >= _UNLIMITED:
        return None
    return value


def physical_memory_bytes() -> Optional[int]:
    """Total physical RAM visible to the OS (the VM, inside Docker), or None."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - exotic OS
        return None


def effective_memory_ceiling_bytes() -> Optional[int]:
    """The smaller of the cgroup limit and physical RAM - the true ceiling.

    A cgroup limit larger than physical RAM is unenforceable, so physical RAM
    wins; when there is no cgroup limit, physical RAM is the ceiling.
    """
    cgroup = cgroup_memory_limit_bytes()
    physical = physical_memory_bytes()
    candidates = [v for v in (cgroup, physical) if v is not None]
    return min(candidates) if candidates else None


def check_memory_ceiling(
    model: str = "", diarize: bool = True, floor_bytes: int = SAFE_FLOOR_BYTES
) -> Optional[str]:
    """Return a warning string if the effective ceiling is too low, else None.

    Logs the warning as a side effect so operators see it at startup. Only the
    memory-hungry configuration (large models with diarization) trips the floor;
    small models are left alone.
    """
    ceiling = effective_memory_ceiling_bytes()
    if ceiling is None:
        return None

    # Only large models + diarization realistically need the full floor.
    heavy = diarize and ("large" in model.lower() or "medium" in model.lower())
    if not heavy or ceiling >= floor_bytes:
        return None

    cgroup = cgroup_memory_limit_bytes()
    physical = physical_memory_bytes()
    gib = 1024**3
    detail = f"effective memory ceiling is only {ceiling / gib:.1f} GiB"
    if cgroup is not None and physical is not None and cgroup > physical:
        detail += (
            f" (container limit {cgroup / gib:.1f} GiB EXCEEDS the "
            f"{physical / gib:.1f} GiB VM, so it cannot be enforced - raise the "
            f"Docker Desktop VM memory, not just the container limit)"
        )
    message = (
        f"Low memory for transcription: {detail}. '{model or 'large-v3'}' with "
        f"diarization can peak ~7.5-12 GiB and will likely be OOM-killed. "
        f"Recommended: run the native 'transcribe' CLI on the host for long "
        f"files, or increase memory / use a smaller model / disable diarization."
    )
    logger.warning(message)
    return message
