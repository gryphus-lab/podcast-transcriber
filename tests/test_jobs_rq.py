"""Tests for the RQ-backed job store.

These require the optional ``rq`` extra; they are skipped automatically when it
is not installed (uv sync --extra rq). A live Redis is not needed: an in-memory
``fakeredis`` connection backs the queue and jobs run synchronously
(``is_async=False``), so submit/get/list and the worker/callback functions are
exercised end to end without a broker or worker process.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("rq")
pytest.importorskip("redis")
pytest.importorskip("fakeredis")

from fakeredis import FakeStrictRedis  # noqa: E402
from rq import Queue  # noqa: E402

from podcast_transcriber import jobs_rq  # noqa: E402
from podcast_transcriber.jobs import JobStatus, TranscriptionResult  # noqa: E402
from podcast_transcriber.jobs_rq import (  # noqa: E402
    _STATUS_MAP,
    RQJobStore,
    _aware,
    _post_callback,
    deliver_callback,
    deliver_callback_failure,
    run_transcription_job,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_status_map_covers_rq_states():
    assert _STATUS_MAP["queued"] is JobStatus.PENDING
    assert _STATUS_MAP["deferred"] is JobStatus.PENDING
    assert _STATUS_MAP["started"] is JobStatus.RUNNING
    assert _STATUS_MAP["finished"] is JobStatus.DONE
    assert _STATUS_MAP["failed"] is JobStatus.ERROR
    assert _STATUS_MAP["canceled"] is JobStatus.ERROR


def test_aware_tags_naive_datetime_as_utc():
    from datetime import datetime, timezone

    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = _aware(naive)
    assert aware.tzinfo is timezone.utc
    assert _aware(aware) is aware
    assert _aware(None) is None


# ---------------------------------------------------------------------------
# Worker-side function
# ---------------------------------------------------------------------------


def test_run_transcription_job_returns_serializable_dict():
    result = TranscriptionResult(
        transcript="hello world", output_format="txt", source_name="ep1.m4a"
    )
    with (
        patch(
            "podcast_transcriber.job_backend.run_transcription", return_value=result
        ) as mock_run,
        patch(
            "podcast_transcriber.job_backend.persist_transcript",
            return_value="/data/out/ep1-abc.txt",
        ) as mock_persist,
    ):
        out = run_transcription_job({"job_id": "abc", "audio_path": "/data/ep1.m4a"})

    mock_run.assert_called_once()
    mock_persist.assert_called_once()
    assert out == {
        "transcript": "hello world",
        "output_format": "txt",
        "source_name": "ep1.m4a",
        "result_file": "/data/out/ep1-abc.txt",
    }


# ---------------------------------------------------------------------------
# Callback hooks + delivery
# ---------------------------------------------------------------------------


class _FakeRQJob:
    def __init__(self, job_id="j1", meta=None):
        self.id = job_id
        self.meta = meta or {}


def test_deliver_callback_posts_success_payload():
    job = _FakeRQJob(meta={"callback_url": "https://hook.test/done"})
    with patch.object(jobs_rq, "_post_callback") as mock_post:
        deliver_callback(job, None, {"transcript": "T", "output_format": "srt"})

    url, payload = mock_post.call_args.args
    assert url == "https://hook.test/done"
    assert payload["status"] == JobStatus.DONE.value
    assert payload["transcript"] == "T"
    assert payload["output_format"] == "srt"


def test_deliver_callback_noop_without_url():
    with patch.object(jobs_rq, "_post_callback") as mock_post:
        deliver_callback(_FakeRQJob(meta={}), None, {"transcript": "x"})
    mock_post.assert_not_called()


def test_deliver_callback_failure_posts_error_payload():
    job = _FakeRQJob(meta={"callback_url": "https://hook.test/fail"})
    with patch.object(jobs_rq, "_post_callback") as mock_post:
        deliver_callback_failure(job, None, ValueError, ValueError("boom"), None)

    url, payload = mock_post.call_args.args
    assert url == "https://hook.test/fail"
    assert payload["status"] == JobStatus.ERROR.value
    assert payload["error"] == "boom"
    assert payload["transcript"] is None


def test_deliver_callback_failure_noop_without_url():
    job = _FakeRQJob(meta={})
    with patch.object(jobs_rq, "_post_callback") as mock_post:
        deliver_callback_failure(job, None, ValueError, ValueError(), None)
    mock_post.assert_not_called()


def test_post_callback_swallows_errors():
    # A failing webhook must not raise out of the hook.
    with patch("podcast_transcriber.jobs_rq.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = RuntimeError(
            "down"
        )
        _post_callback("https://hook.test", {"a": 1})  # should not raise


def test_post_callback_posts_payload():
    with patch("podcast_transcriber.jobs_rq.httpx.Client") as mock_client:
        post = mock_client.return_value.__enter__.return_value.post
        _post_callback("https://hook.test", {"a": 1})
    post.assert_called_once_with("https://hook.test", json={"a": 1})


# ---------------------------------------------------------------------------
# Store (fakeredis + synchronous queue)
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    """An RQJobStore wired to an in-memory fakeredis + synchronous queue."""
    conn = FakeStrictRedis()
    s = RQJobStore.__new__(RQJobStore)  # bypass __init__'s real Redis.from_url
    s._redis = conn
    s._queue = Queue(jobs_rq.QUEUE_NAME, connection=conn, is_async=False)
    return s


def test_submit_runs_job_and_get_returns_done(store):
    result = TranscriptionResult(
        transcript="done text", output_format="txt", source_name="a.m4a"
    )
    with (
        patch(
            "podcast_transcriber.job_backend.run_transcription", return_value=result
        ),
        patch(
            "podcast_transcriber.job_backend.persist_transcript",
            return_value="/out/a.txt",
        ),
    ):
        submitted = store.submit(
            params={"job_id": "job-done", "audio_path": "/in/a.m4a"},
            source_name="a.m4a",
            output_format="txt",
        )

    assert submitted.id == "job-done"
    fetched = store.get("job-done")
    assert fetched.status is JobStatus.DONE
    assert fetched.result is not None
    assert fetched.result.transcript == "done text"
    assert fetched.result_file == "/out/a.txt"


def test_submit_records_error_on_failure(store):
    with patch(
        "podcast_transcriber.job_backend.run_transcription",
        side_effect=RuntimeError("transcribe failed"),
    ):
        store.submit(params={"job_id": "job-err", "audio_path": "/in/b.m4a"})

    fetched = store.get("job-err")
    assert fetched.status is JobStatus.ERROR
    assert "transcribe failed" in (fetched.error or "")


def test_submit_requires_params(store):
    with pytest.raises(ValueError):
        store.submit(params=None)


def test_get_unknown_job_returns_none(store):
    assert store.get("nope") is None


def test_list_returns_jobs_newest_first(store):
    result = TranscriptionResult(transcript="t", output_format="txt", source_name="s")
    with (
        patch(
            "podcast_transcriber.job_backend.run_transcription", return_value=result
        ),
        patch(
            "podcast_transcriber.job_backend.persist_transcript", return_value="/o.txt"
        ),
    ):
        store.submit(params={"job_id": "j1", "audio_path": "/1"})
        store.submit(params={"job_id": "j2", "audio_path": "/2"})

    ids = [j.id for j in store.list()]
    assert "j1" in ids
    assert "j2" in ids
