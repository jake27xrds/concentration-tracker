"""Focus Tracker — AI-powered focus monitoring for macOS."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path.home() / "Library" / "Logs" / "FocusTracker"
LOG_FILE = LOG_DIR / "focus_tracker.log"


def configure_logging(
    *,
    log_file: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure package logging once.

    This is intentionally explicit (not run at import time) so tests and
    restricted environments can import modules without file-system side effects.
    """
    logger = logging.getLogger("focus_tracker")
    if getattr(logger, "_focus_tracker_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    target = log_file or LOG_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(target, maxBytes=2_000_000, backupCount=3)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("File logging disabled: %s", exc)

    logger._focus_tracker_configured = True
    return logger
