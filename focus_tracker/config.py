"""
Configuration persistence module.
Saves and loads user settings to a JSON file in the user's
Application Support directory.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger("focus_tracker.config")

CONFIG_DIR = Path.home() / "Library" / "Application Support" / "FocusTracker"
CONFIG_FILE = CONFIG_DIR / "settings.json"

# Default settings
DEFAULTS = {
    "sound_enabled": True,
    "distraction_threshold_seconds": 30,
    "break_interval_minutes": 25,
    "productive_apps": [],  # empty = use built-in defaults
    "distracting_apps": [],
    "neutral_apps": [],
    "camera_index": 0,
    "score_weights": {
        "eye_engagement": 0.20,
        "gaze_stability": 0.20,
        "blink": 0.10,
        "activity": 0.25,
        "app_focus": 0.25,
    },
}


def load_config() -> dict:
    """Load settings from disk, falling back to defaults for missing keys."""
    config = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            # Merge saved values into defaults (so new keys have defaults)
            for key, value in saved.items():
                if key in config:
                    config[key] = value
            log.info("Loaded config from %s", CONFIG_FILE)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not load config (%s), using defaults", e)
    return config


def save_config(config: dict) -> None:
    """Persist settings to disk."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        log.debug("Saved config to %s", CONFIG_FILE)
    except OSError as e:
        log.error("Failed to save config: %s", e)
