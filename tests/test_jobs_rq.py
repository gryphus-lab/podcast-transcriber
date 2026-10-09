"""Tests for the RQ-backed job store.

These require the optional ``rq`` extra; they are skipped automatically when it
is not installed (uv sync --extra rq). They exercise the mapping/serialization
surface without a live Redis by using RQ's fakeredis-free status mapping on
constructed objects.
"""

import pytest

pytest.importorskip("rq")
pytest.importorskip("redis")

from podcast_transcriber.jobs import JobStatus  # noqa: E402
from podcast_transcriber.jobs_rq import _aware, _STATUS_MAP  # noqa: E402


def test_status_map_covers_rq_states():
    # Every RQ terminal/active state maps to one of our four statuses.
    assert _STATUS_MAP["queued"] is JobStatus.PENDING
    assert _STATUS_MAP["started"] is JobStatus.RUNNING
    assert _STATUS_MAP["finished"] is JobStatus.DONE
    assert _STATUS_MAP["failed"] is JobStatus.ERROR


def test_aware_tags_naive_datetime_as_utc():
    from datetime import datetime, timezone

    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = _aware(naive)
    assert aware.tzinfo is timezone.utc
    # Already-aware datetimes pass through unchanged.
    assert _aware(aware) is aware
    assert _aware(None) is None
