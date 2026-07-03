"""Tests for config module."""

import os
from unittest.mock import patch


def test_defaults():
    """Config should have sensible defaults when env vars are missing."""
    env = {}
    with patch.dict(os.environ, env, clear=False):
        import importlib

        import podcast_transcriber.config as cfg

        importlib.reload(cfg)

        assert cfg.WHISPER_MODEL == "large-v3"
        assert cfg.LANGUAGE == "en"
        assert cfg.HF_TOKEN == "" or isinstance(cfg.HF_TOKEN, str)


def test_custom_env():
    """Config should respect environment overrides."""
    env = {
        "WHISPER_MODEL": "medium",
        "LANGUAGE": "de",
        "HF_TOKEN": "hf_test_token",
        "OUTPUT_DIR": "/tmp/transcripts",
    }
    with patch.dict(os.environ, env, clear=False):
        import importlib

        import podcast_transcriber.config as cfg

        importlib.reload(cfg)

        assert cfg.WHISPER_MODEL == "medium"
        assert cfg.LANGUAGE == "de"
        assert cfg.HF_TOKEN == "hf_test_token"
