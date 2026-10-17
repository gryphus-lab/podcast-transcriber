"""Tests for the startup memory sanity check (utils.memguard).

The guard warns when the memory the process can actually use is below what
long-audio large-v3 + diarization needs - including the subtle case where a
container memory limit is set higher than the Docker Desktop VM (a phantom
limit the kernel can't enforce).
"""

from podcast_transcriber.utils import memguard

GIB = 1024**3


def test_effective_ceiling_is_min_of_cgroup_and_physical(monkeypatch):
    monkeypatch.setattr(memguard, "cgroup_memory_limit_bytes", lambda: 12 * GIB)
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 8 * GIB)
    # Container limit (12G) exceeds the VM (8G): the real ceiling is the VM.
    assert memguard.effective_memory_ceiling_bytes() == 8 * GIB


def test_effective_ceiling_handles_no_cgroup_limit(monkeypatch):
    monkeypatch.setattr(memguard, "cgroup_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 16 * GIB)
    assert memguard.effective_memory_ceiling_bytes() == 16 * GIB


def test_check_warns_on_phantom_limit(monkeypatch):
    """A 12G container limit on an 8G VM must warn, and name the phantom-limit
    cause explicitly."""
    monkeypatch.setattr(memguard, "cgroup_memory_limit_bytes", lambda: 12 * GIB)
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 8 * GIB)

    msg = memguard.check_memory_ceiling(model="large-v3", diarize=True)
    assert msg is not None
    assert "8.0 GiB" in msg
    assert "EXCEEDS" in msg


def test_check_silent_when_ceiling_is_ample(monkeypatch):
    monkeypatch.setattr(memguard, "cgroup_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 32 * GIB)
    assert memguard.check_memory_ceiling(model="large-v3", diarize=True) is None


def test_check_silent_for_small_model(monkeypatch):
    """Small models don't need the full floor, so a low ceiling shouldn't warn."""
    monkeypatch.setattr(memguard, "cgroup_memory_limit_bytes", lambda: 4 * GIB)
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 4 * GIB)
    assert memguard.check_memory_ceiling(model="tiny", diarize=True) is None


def test_check_silent_when_diarization_off(monkeypatch):
    monkeypatch.setattr(memguard, "cgroup_memory_limit_bytes", lambda: 4 * GIB)
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: 4 * GIB)
    assert memguard.check_memory_ceiling(model="large-v3", diarize=False) is None


def test_check_handles_unknown_ceiling(monkeypatch):
    monkeypatch.setattr(memguard, "cgroup_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(memguard, "physical_memory_bytes", lambda: None)
    assert memguard.effective_memory_ceiling_bytes() is None
    assert memguard.check_memory_ceiling(model="large-v3", diarize=True) is None


def test_cgroup_limit_reads_a_real_value():
    """Smoke test: on any host this returns either an int limit or None
    (unlimited) - never raises, regardless of cgroup v1/v2/none."""
    result = memguard.cgroup_memory_limit_bytes()
    assert result is None or isinstance(result, int)
