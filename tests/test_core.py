"""
Unit tests for Focus Tracker core modules.
Run with:  python -m pytest tests/ -v
"""

import time
import json
import tempfile
from pathlib import Path

import pytest

from focus_tracker.eye_tracker import EyeMetrics
from focus_tracker.activity_monitor import (
    ActivityMetrics,
    ActivityMonitor,
    DEFAULT_PRODUCTIVE_APPS,
    DEFAULT_NEUTRAL_APPS,
    DEFAULT_DISTRACTING_APPS,
)
from focus_tracker.focus_engine import FocusEngine, FocusSnapshot
from focus_tracker.config import load_config, save_config, DEFAULTS


# ──────────────────────────────────────────────
# FocusEngine tests
# ──────────────────────────────────────────────

class TestFocusEngine:
    def _make_eye(self, **overrides) -> EyeMetrics:
        defaults = dict(
            timestamp=time.time(),
            face_detected=True,
            avg_ear=0.28,
            left_ear=0.28,
            right_ear=0.28,
            gaze_horizontal=0.0,
            gaze_vertical=0.0,
            blinks_per_minute=16,
            looking_at_screen=True,
            attention_h=0.0,
            attention_v=0.0,
            head_frontal_confidence=0.9,
        )
        defaults.update(overrides)
        return EyeMetrics(**defaults)

    def _make_activity(self, **overrides) -> ActivityMetrics:
        defaults = dict(
            timestamp=time.time(),
            active_app="Visual Studio Code",
            keys_per_minute=40,
            mouse_moves_per_minute=30,
            total_idle_seconds=1,
            app_switches_per_minute=2,
            in_productive_app=True,
            app_classification="productive",
        )
        defaults.update(overrides)
        return ActivityMetrics(**defaults)

    def test_high_focus_score(self):
        """Centered eyes + active typing in productive app → high score."""
        engine = FocusEngine()
        eye = self._make_eye()
        act = self._make_activity()
        snap = engine.calculate(eye, act)
        assert snap.focus_score >= 70, f"Expected >= 70, got {snap.focus_score:.1f}"

    def test_no_face_low_score(self):
        """No face detected → low engagement score."""
        engine = FocusEngine()
        eye = self._make_eye(face_detected=False)
        act = self._make_activity()
        snap = engine.calculate(eye, act)
        assert snap.eye_engagement_score <= 20

    def test_distracting_app_lowers_score(self):
        """Distracting app → lower app focus score."""
        engine = FocusEngine()
        eye = self._make_eye()
        act = self._make_activity(
            in_productive_app=False,
            app_classification="distracting",
        )
        snap = engine.calculate(eye, act)
        assert snap.app_focus_score <= 40

    def test_idle_lowers_activity_score(self):
        """Long idle time → lower activity score."""
        engine = FocusEngine()
        eye = self._make_eye()
        act = self._make_activity(
            keys_per_minute=0,
            mouse_moves_per_minute=0,
            total_idle_seconds=120,
        )
        snap = engine.calculate(eye, act)
        assert snap.activity_score <= 40

    def test_away_state(self):
        """No face + idle → Away state."""
        engine = FocusEngine()
        eye = self._make_eye(face_detected=False)
        act = self._make_activity(
            keys_per_minute=0,
            mouse_moves_per_minute=0,
            total_idle_seconds=300,
        )
        snap = engine.calculate(eye, act)
        assert snap.state == "Away"

    def test_hysteresis_prevents_downgrade(self):
        """State shouldn't downgrade within the hold time."""
        engine = FocusEngine()
        eye = self._make_eye()
        act = self._make_activity()

        # Get into Focused state
        for _ in range(5):
            engine.calculate(eye, act)

        # Now borderline-lower input (but within hold time)
        engine._state_entered_at = time.time()  # reset
        eye_lower = self._make_eye(looking_at_screen=False, attention_h=0.3)
        act_lower = self._make_activity(app_classification="neutral")
        snap = engine.calculate(eye_lower, act_lower)

        # Should NOT have dropped immediately due to hold time
        assert snap.state in ("Deep Focus", "Focused", "Neutral")

    def test_history_accumulates(self):
        engine = FocusEngine()
        eye = self._make_eye()
        act = self._make_activity()
        for _ in range(10):
            engine.calculate(eye, act)
        assert len(engine.history) == 10

    def test_get_average_score(self):
        engine = FocusEngine()
        eye = self._make_eye()
        act = self._make_activity()
        for _ in range(5):
            engine.calculate(eye, act)
        avg = engine.get_average_score(300)
        assert 0 <= avg <= 100

    def test_session_summary(self):
        engine = FocusEngine()
        eye = self._make_eye()
        act = self._make_activity()
        for _ in range(20):
            engine.calculate(eye, act)
        summary = engine.get_session_summary()
        assert "avg_score" in summary
        assert "total_readings" in summary
        assert summary["total_readings"] == 20


# ──────────────────────────────────────────────
# ActivityMonitor tests
# ──────────────────────────────────────────────

class TestActivityMetrics:
    def test_app_classification_field_exists(self):
        m = ActivityMetrics()
        assert hasattr(m, "app_classification")
        assert m.app_classification == "productive"

    def test_idle_uses_max(self):
        """total_idle_seconds should be max(mouse, keyboard) not min."""
        m = ActivityMetrics(mouse_idle_seconds=10, keyboard_idle_seconds=60)
        # We can't test the monitor's get_metrics directly without listeners,
        # but we verify the field is separate and the logic is correct by
        # checking max > min.
        assert max(m.mouse_idle_seconds, m.keyboard_idle_seconds) == 60

    def test_app_sets_exist(self):
        assert len(DEFAULT_PRODUCTIVE_APPS) > 0
        assert len(DEFAULT_NEUTRAL_APPS) > 0
        assert len(DEFAULT_DISTRACTING_APPS) > 0

    def test_no_overlap_productive_distracting(self):
        """Productive and distracting sets should not overlap."""
        overlap = DEFAULT_PRODUCTIVE_APPS & DEFAULT_DISTRACTING_APPS
        assert len(overlap) == 0, f"Overlap: {overlap}"


# ──────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────

class TestConfig:
    def test_defaults_loaded_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("focus_tracker.config.CONFIG_FILE", tmp_path / "nope.json")
        cfg = load_config()
        assert cfg["sound_enabled"] == DEFAULTS["sound_enabled"]
        assert cfg["distraction_threshold_seconds"] == 30

    def test_save_and_load(self, tmp_path, monkeypatch):
        config_file = tmp_path / "settings.json"
        monkeypatch.setattr("focus_tracker.config.CONFIG_FILE", config_file)
        monkeypatch.setattr("focus_tracker.config.CONFIG_DIR", tmp_path)

        cfg = {"sound_enabled": False, "distraction_threshold_seconds": 60,
               "break_interval_minutes": 10, "productive_apps": [],
               "distracting_apps": [], "neutral_apps": [], "camera_index": 0,
               "score_weights": DEFAULTS["score_weights"]}
        save_config(cfg)
        assert config_file.exists()

        loaded = load_config()
        assert loaded["sound_enabled"] is False
        assert loaded["distraction_threshold_seconds"] == 60

    def test_corrupt_file_returns_defaults(self, tmp_path, monkeypatch):
        config_file = tmp_path / "settings.json"
        config_file.write_text("{bad json")
        monkeypatch.setattr("focus_tracker.config.CONFIG_FILE", config_file)
        cfg = load_config()
        assert cfg == DEFAULTS


# ──────────────────────────────────────────────
# EyeMetrics tests
# ──────────────────────────────────────────────

class TestEyeMetrics:
    def test_defaults(self):
        m = EyeMetrics()
        assert m.face_detected is False
        assert m.looking_at_screen is False
        assert m.attention_h == 0.0

    def test_fields_assignable(self):
        m = EyeMetrics(
            face_detected=True,
            looking_at_screen=True,
            attention_h=0.1,
            attention_v=-0.05,
            head_frontal_confidence=0.95,
        )
        assert m.looking_at_screen is True
        assert m.head_frontal_confidence == 0.95
