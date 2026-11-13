"""pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock

import pytest

# Attributes the tests patch on the mocked heavy-ML modules. They must exist as
# *real* attributes on the mock (not lazily fabricated children), because
# pytest's monkeypatch.setattr(..., raising=True) checks existence with
# inspect.getattr_static, which does NOT trigger MagicMock's dynamic attribute
# creation. Pre-seeding them keeps `monkeypatch.setattr` working across tests
# regardless of whether the real libraries happen to be installed (pytest 9.2+
# tightened this check, which otherwise fails the second patch onward).
_WHISPERX_ATTRS = (
    "load_model",
    "load_audio",
    "load_align_model",
    "align",
    "assign_word_speakers",
    "DiarizationPipeline",
)

# torch attributes the tests patch on the mocked module (same getattr_static
# concern as whisperx).
_TORCH_ATTRS = (
    "from_numpy",
    "device",
    "cuda",
    "no_grad",
    "inference_mode",
)


def _make_whisperx_mock() -> MagicMock:
    mock = MagicMock(name="whisperx")
    for attr in _WHISPERX_ATTRS:
        setattr(mock, attr, MagicMock(name=f"whisperx.{attr}"))
    return mock


def pytest_configure(config):
    """Mock heavy ML dependencies before any tests are collected.

    These libraries (torch, whisperx, pyannote) are not needed for the unit
    tests and may or may not be installed in a given environment; mocking them
    in sys.modules makes the suite deterministic and import-light either way.
    """
    torch_mock = MagicMock(name="torch")
    # torch.backends.mps.is_available() -> False (keeps device selection on CPU).
    torch_mock.backends.mps.is_available = MagicMock(return_value=False)

    mock_modules = {
        "torch": torch_mock,
        "torch.backends": torch_mock.backends,
        "torch.backends.mps": torch_mock.backends.mps,
        "whisperx": _make_whisperx_mock(),
        "pyannote": MagicMock(name="pyannote"),
        "pyannote.audio": MagicMock(name="pyannote.audio"),
    }

    for module_name, mock_module in mock_modules.items():
        sys.modules[module_name] = mock_module


@pytest.fixture(autouse=True)
def _reset_whisperx_mock():
    """Re-seed whisperx mock attributes before each test.

    Tests patch attributes on the whisperx module object via monkeypatch, which
    removes them again on teardown. Because pytest 9.2+ checks attribute
    existence with inspect.getattr_static (which ignores MagicMock's dynamic
    attributes), the removed attributes must be restored as *real* attributes
    before the next test or its monkeypatch.setattr(..., raising=True) fails.

    Re-seed on the actual module object the code under test imported
    (transcriber_module.whisperx), falling back to sys.modules, so test order
    never leaks regardless of whether the real whisperx is installed.
    """
    import podcast_transcriber.utils.transcriber as transcriber_module

    whisperx_target = getattr(transcriber_module, "whisperx", None) or sys.modules.get(
        "whisperx"
    )
    if isinstance(whisperx_target, MagicMock):
        for attr in _WHISPERX_ATTRS:
            setattr(whisperx_target, attr, MagicMock(name=f"whisperx.{attr}"))

    torch_target = getattr(transcriber_module, "torch", None) or sys.modules.get(
        "torch"
    )
    if isinstance(torch_target, MagicMock):
        for attr in _TORCH_ATTRS:
            setattr(torch_target, attr, MagicMock(name=f"torch.{attr}"))
        # Keep the CPU-only device-selection behaviour the suite relies on.
        torch_target.backends.mps.is_available = MagicMock(return_value=False)
    yield
