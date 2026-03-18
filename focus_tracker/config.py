"""
Configuration persistence module.
Saves and loads user settings to a JSON file in the user's
Application Support directory.
"""

import json
import logging
import copy
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
    "goal_enabled": True,
    "goal_minutes_target": 45,
    "weekly_sessions_target": 5,
    "active_intent": "coding",
    "baseline_enabled": True,
    "posture_grace_seconds": 12,
    "distracted_confirm_seconds": 8,
    "overlay_enabled": False,
    "menubar_enabled": False,
    "auto_intent_enabled": True,
    "fatigue_break_enabled": True,
    "nudge_cooldowns_sec": {
        "prolonged_distraction": 45,
        "high_app_switching": 60,
        "long_idle_drift": 60,
        "break_due": 180,
        "fatigue_risk": 180,
    },
    "active_profile": "Coding",
    "profiles": {
        "Study": {
            "productive_apps": ["notion", "obsidian", "pages", "preview", "acrobat", "safari"],
            "neutral_apps": ["finder", "calendar", "mail"],
            "distracting_apps": ["youtube", "tiktok", "instagram", "netflix"],
            "productive_domains": ["docs.google.com", "scholar.google.com", "wikipedia.org"],
            "distracting_domains": ["youtube.com", "reddit.com", "x.com", "instagram.com", "tiktok.com"],
        },
        "Coding": {
            "productive_apps": ["code", "visual studio code", "xcode", "terminal", "iterm", "pycharm"],
            "neutral_apps": ["finder", "preview", "slack", "teams"],
            "distracting_apps": ["youtube", "tiktok", "instagram", "netflix", "steam"],
            "productive_domains": ["github.com", "stackoverflow.com", "docs.python.org"],
            "distracting_domains": ["youtube.com", "reddit.com", "x.com", "instagram.com", "tiktok.com"],
        },
        "Writing": {
            "productive_apps": ["pages", "microsoft word", "notion", "obsidian", "bear"],
            "neutral_apps": ["finder", "mail", "calendar"],
            "distracting_apps": ["youtube", "tiktok", "instagram", "netflix"],
            "productive_domains": ["docs.google.com", "grammarly.com"],
            "distracting_domains": ["youtube.com", "reddit.com", "x.com", "instagram.com", "tiktok.com"],
        },
    },
    "calibration_profile": {
        "neutral_attention_center": [0.0, 0.0],
        "neutral_attention_tolerance": [0.45, 0.40],
        "neutral_head_center": [0.0, 0.0],
        "neutral_gaze_center": [0.0, 0.0],
        "reading_baseline": {
            "confidence": 0.0,
            "gaze_horizontal_variance": 0.0,
            "gaze_vertical_variance": 0.0,
        },
        "distracted_baseline": {
            "attention_magnitude": 0.0,
        },
        "ear_baseline": 0.27,
        "blink_threshold": 0.22,
        "closed_threshold": 0.18,
        "calibrated_at": "",
    },
}


def load_config() -> dict:
    """Load settings from disk, falling back to defaults for missing keys."""
    config = copy.deepcopy(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            # Merge saved values into defaults (so new keys have defaults)
            for key, value in saved.items():
                if key not in config:
                    continue
                if isinstance(config[key], dict) and isinstance(value, dict):
                    merged = copy.deepcopy(config[key])
                    merged.update(value)
                    config[key] = merged
                else:
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
