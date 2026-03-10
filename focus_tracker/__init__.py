"""Focus Tracker — AI-powered focus monitoring for macOS."""

import logging
import os
from pathlib import Path

LOG_DIR = Path.home() / "Library" / "Logs" / "FocusTracker"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "focus_tracker.log"

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Console handler (INFO+)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_fmt)

# File handler (DEBUG+, rotates)
from logging.handlers import RotatingFileHandler
_file = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
_file.setLevel(logging.DEBUG)
_file.setFormatter(_fmt)

# Root package logger
logger = logging.getLogger("focus_tracker")
logger.setLevel(logging.DEBUG)
logger.addHandler(_console)
logger.addHandler(_file)