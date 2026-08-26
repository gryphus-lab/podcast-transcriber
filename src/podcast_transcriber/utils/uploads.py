"""Shared helpers for handling FastAPI file uploads."""

import os
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

_CHUNK_SIZE = 1024 * 1024  # 1MB


async def stream_upload_to_file(
    file: UploadFile, tmp_path: Path, max_size: int
) -> None:
    """Stream an uploaded file to disk in bounded chunks.

    Args:
        file: The uploaded file.
        tmp_path: Path to write the file to.
        max_size: Maximum allowed upload size in bytes.

    Raises:
        HTTPException: If the upload exceeds ``max_size`` (413) or a write
            fails (400). The partial temp file is removed on any failure.
    """
    cumulative_size = 0
    try:
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                cumulative_size += len(chunk)
                if cumulative_size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum size of {max_size} bytes.",
                    )
                f.write(chunk)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except OSError as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}") from e


async def save_upload_to_temp(
    file: UploadFile, suffix: str, max_size: int
) -> Path:
    """Create a temp file and stream an upload into it.

    Returns the path to the written temp file.
    """
    fd, tmp_file = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # Close the descriptor; we write via the returned path.
    tmp_path = Path(tmp_file)
    await stream_upload_to_file(file, tmp_path, max_size)
    return tmp_path
