"""Suppress noisy warnings and loggers from third-party libraries."""

import logging
import os


def suppress_noisy_output() -> None:
    """Suppress warnings and set noisy loggers to ERROR level.
    
    This must be called before importing libraries that generate
    noisy warnings (e.g., whisperx, lightning, pyannote).
    """
    # Suppress all warnings globally
    os.environ["PYTHONWARNINGS"] = "ignore"
    
    # Set noisy loggers to ERROR level to suppress info/debug output
    noisy_loggers = [
        "lightning.pytorch.utilities.migration",
        "lightning",
        "whisperx",
        "pyannote",
        "pyannote.audio",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
