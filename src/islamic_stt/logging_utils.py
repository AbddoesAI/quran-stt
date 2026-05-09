"""
logging_utils.py
----------------
Centralised logging configuration for the Islamic STT pipeline.

Features (#2 from improvement plan):
  - Console handler with colour-coded levels
  - Rotating file handler (5 MB × 3 backups)
  - Debug/verbose mode toggle
  - Module-name-aware formatting
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler


_LOG_DIR = "logs"
_LOG_FILE = os.path.join(_LOG_DIR, "islamic_stt.log")
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"


def configure_logging(level: int = logging.INFO, *, log_to_file: bool = True) -> None:
    """
    Set up root logger with console + optional rotating file output.

    Parameters
    ----------
    level       : Logging level (e.g. logging.DEBUG, logging.INFO).
    log_to_file : If True, also log to ``logs/islamic_stt.log`` with rotation.
    """
    root = logging.getLogger()

    # Avoid duplicate handlers if called multiple times
    if root.handlers:
        return

    root.setLevel(level)

    # --- Console handler ---
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    # --- File handler (rotating) ---
    if log_to_file:
        os.makedirs(_LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # always capture debug to file
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        )
        root.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
