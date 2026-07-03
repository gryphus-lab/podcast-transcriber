"""pytest configuration and fixtures."""

from unittest.mock import MagicMock
import sys
import pytest


def pytest_configure(config):
    """Mock heavy dependencies before any tests are collected."""
    # Create mock modules for libraries we can't install
    mock_modules = {
        "torch": MagicMock(),
        "torch.backends": MagicMock(),
        "torch.backends.mps": MagicMock(),
        "whisperx": MagicMock(),
        "pyannote": MagicMock(),
        "pyannote.audio": MagicMock(),
    }
    
    # Also mock torch.backends.mps.is_available to return False
    mock_modules["torch"].backends.mps.is_available = MagicMock(return_value=False)
    
    for module_name, mock_module in mock_modules.items():
        sys.modules[module_name] = mock_module
