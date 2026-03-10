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
from focus_tracker.focus_engine import FocusEngine, FocusSnapshot, _remap, _variance
from focus_tracker.alerts import AlertManager, AlertState
from focus_tracker.session_manager import SessionManager
from focus_tracker.config import load_config, save_config, DEFAULTS


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _make_eye(**overrides) -> EyeMetrics:
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


def _make_activity(**overrides) -> ActivityMetrics:
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


# ══════════════════════════════════════════════
# FocusEngine tests
# ══════════════════════════════════════════════

class TestFocusEngine:
    def test_high_focus_score(self):
        """Centered eyes + active typing in productive app → high score."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity()
        snap = engine.calculate(eye, act)
        assert snap.focus_score >= 70, f"Expected >= 70, got {snap.focus_score:.1f}"

    def test_no_face_low_score(self):
        """No face detected → low engagement score."""
        engine = FocusEngine()
        eye = _make_eye(face_detected=False)
        act = _make_activity()
        snap = engine.calculate(eye, act)
        assert snap.eye_engagement_score <= 20

    def test_distracting_app_lowers_score(self):
        """Distracting app → lower app focus score."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity(in_productive_app=False, app_classification="distracting")
        snap = engine.calculate(eye, act)
        assert snap.app_focus_score <= 40

    def test_neutral_app_mid_score(self):
        """Neutral app → mid-range app focus score."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity(app_classification="neutral")
        snap = engine.calculate(eye, act)
        assert 40 <= snap.app_focus_score <= 70

    def test_idle_lowers_activity_score(self):
        """Long idle time → lower activity score."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity(keys_per_minute=0, mouse_moves_per_minute=0, total_idle_seconds=120)
        snap = engine.calculate(eye, act)
        assert snap.activity_score <= 40

    def test_very_long_idle_very_low(self):
        """5+ min idle → near-zero activity score."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity(keys_per_minute=0, mouse_moves_per_minute=0, total_idle_seconds=400)
        snap = engine.calculate(eye, act)
        assert snap.activity_score <= 10

    def test_away_state(self):
        """No face + idle → Away state."""
        engine = FocusEngine()
        eye = _make_eye(face_detected=False)
        act = _make_activity(keys_per_minute=0, mouse_moves_per_minute=0, total_idle_seconds=300)
        snap = engine.calculate(eye, act)
        assert snap.state == "Away"

    def test_hysteresis_prevents_downgrade(self):
        """State shouldn't downgrade within the hold time."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity()

        for _ in range(5):
            engine.calculate(eye, act)

        engine._state_entered_at = time.time()
        eye_lower = _make_eye(looking_at_screen=False, attention_h=0.3)
        act_lower = _make_activity(app_classification="neutral")
        snap = engine.calculate(eye_lower, act_lower)
        assert snap.state in ("Deep Focus", "Focused", "Neutral")

    def test_hysteresis_allows_downgrade_after_hold(self):
        """State SHOULD downgrade after the hold time has passed."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity()

        for _ in range(5):
            engine.calculate(eye, act)

        # Force state entered long enough ago
        engine._state_entered_at = time.time() - 10
        low_eye = _make_eye(face_detected=False)
        low_act = _make_activity(keys_per_minute=0, mouse_moves_per_minute=0, total_idle_seconds=300)
        snap = engine.calculate(low_eye, low_act)
        assert snap.state == "Away"

    def test_hysteresis_away_immediate(self):
        """Away state always transitions immediately."""
        engine = FocusEngine()
        eye = _make_eye()
        act = _make_activity()
        engine.calculate(eye, act)

        away_eye = _make_eye(face_detected=False)
        away_act = _make_activity(total_idle_seconds=300, keys_per_minute=0, mouse_moves_per_minute=0)
        snap = engine.calculate(away_eye, away_act)
        assert snap.state == "Away"

    def test_history_accumulates(self):
        engine = FocusEngine()
        for _ in range(10):
            engine.calculate(_make_eye(), _make_activity())
        assert len(engine.history) == 10

    def test_history_maxlen(self):
        """History should be bounded by maxlen."""
        engine = FocusEngine(history_minutes=1)
        for _ in range(100):
            engine.calculate(_make_eye(), _make_activity())
        assert len(engine.history) <= 60

    def test_get_average_score(self):
        engine = FocusEngine()
        for _ in range(5):
            engine.calculate(_make_eye(), _make_activity())
        avg = engine.get_average_score(300)
        assert 0 <= avg <= 100

    def test_get_average_score_empty(self):
        engine = FocusEngine()
        assert engine.get_average_score(300) == 50.0

    def test_get_history_points(self):
        engine = FocusEngine()
        for _ in range(10):
            engine.calculate(_make_eye(), _make_activity())
        points = engine.get_history_points(300)
        assert len(points) == 10
        for t, score in points:
            assert t <= 0  # relative time should be <= 0 (in the past)
            assert 0 <= score <= 100

    def test_get_history_points_empty(self):
        engine = FocusEngine()
        assert engine.get_history_points(300) == []

    def test_session_summary(self):
        engine = FocusEngine()
        for _ in range(20):
            engine.calculate(_make_eye(), _make_activity())
        summary = engine.get_session_summary()
        assert "avg_score" in summary
        assert "total_readings" in summary
        assert summary["total_readings"] == 20
        assert 0 <= summary["avg_score"] <= 100
        assert 0 <= summary["time_focused_pct"] <= 100

    def test_session_summary_empty(self):
        engine = FocusEngine()
        summary = engine.get_session_summary()
        assert summary["avg_score"] == 0
        assert summary.get("total_readings", 0) == 0

    def test_score_clamped_0_100(self):
        """Focus score should always be in [0, 100]."""
        engine = FocusEngine()
        # Best-case scenario
        snap = engine.calculate(_make_eye(), _make_activity())
        assert 0 <= snap.focus_score <= 100
        # Worst-case scenario
        snap = engine.calculate(
            _make_eye(face_detected=False),
            _make_activity(keys_per_minute=0, mouse_moves_per_minute=0,
                           total_idle_seconds=300, app_classification="distracting",
                           in_productive_app=False),
        )
        assert 0 <= snap.focus_score <= 100

    def test_eyes_closed_lowers_engagement(self):
        """Prolonged eye closure → lower engagement."""
        engine = FocusEngine()
        open_eye = _make_eye(eyes_closed_duration=0.0)
        closed_eye = _make_eye(eyes_closed_duration=3.5)
        act = _make_activity()

        snap_open = engine.calculate(open_eye, act)
        engine2 = FocusEngine()
        snap_closed = engine2.calculate(closed_eye, act)
        assert snap_closed.eye_engagement_score < snap_open.eye_engagement_score

    def test_excessive_blinks_penalized(self):
        """Very high blink rate → lower blink score than healthy range."""
        engine1 = FocusEngine()
        engine2 = FocusEngine()
        act = _make_activity()

        healthy = engine1.calculate(_make_eye(blinks_per_minute=16), act)
        excessive = engine2.calculate(_make_eye(blinks_per_minute=50), act)
        assert excessive.blink_score < healthy.blink_score

    def test_app_switching_penalizes_activity(self):
        """Frequent app switching reduces activity score."""
        engine1 = FocusEngine()
        engine2 = FocusEngine()
        eye = _make_eye()

        calm = engine1.calculate(eye, _make_activity(app_switches_per_minute=1))
        frantic = engine2.calculate(eye, _make_activity(app_switches_per_minute=15))
        assert frantic.activity_score < calm.activity_score

    def test_reading_boosts_activity_when_idle(self):
        """Reading with low mouse/keyboard should NOT be penalized."""
        engine = FocusEngine()
        reading_eye = _make_eye(is_reading=True, reading_confidence=0.8)
        idle_act = _make_activity(keys_per_minute=0, mouse_moves_per_minute=0,
                                   total_idle_seconds=60)
        snap = engine.calculate(reading_eye, idle_act)
        # Reading confidence 0.8 → should give a decent activity score
        assert snap.activity_score >= 55

    def test_reading_doesnt_boost_without_confidence(self):
        """Low reading confidence shouldn't override idle penalty."""
        engine = FocusEngine()
        eye = _make_eye(is_reading=True, reading_confidence=0.1)
        idle_act = _make_activity(keys_per_minute=0, mouse_moves_per_minute=0,
                                   total_idle_seconds=120)
        snap = engine.calculate(eye, idle_act)
        assert snap.activity_score <= 40

    def test_reading_boosts_gaze_stability(self):
        """Reading saccades should not be penalized as instability."""
        engine = FocusEngine()
        reading_eye = _make_eye(is_reading=True, reading_confidence=0.9)
        act = _make_activity()
        # Feed enough frames to populate gaze history
        for _ in range(10):
            snap = engine.calculate(reading_eye, act)
        assert snap.gaze_stability_score >= 50

    def test_weights_sum_to_one(self):
        """Score weights must sum to 1.0."""
        total = sum(FocusEngine.WEIGHTS.values())
        assert abs(total - 1.0) < 0.001


# ══════════════════════════════════════════════
# Utility function tests
# ══════════════════════════════════════════════

class TestUtilities:
    def test_remap_midpoint(self):
        assert _remap(50, 0, 100, 0, 200) == 100.0

    def test_remap_clamps_low(self):
        assert _remap(-10, 0, 100, 0, 200) == 0.0

    def test_remap_clamps_high(self):
        assert _remap(150, 0, 100, 0, 200) == 200.0

    def test_remap_inverted_range(self):
        """Remapping to an inverted range should work."""
        assert _remap(100, 0, 100, 100, 0) == 0.0

    def test_remap_same_input_range(self):
        """Equal input min and max → midpoint of output."""
        assert _remap(5, 5, 5, 10, 20) == 15.0

    def test_variance_empty(self):
        assert _variance([]) == 0.0

    def test_variance_single(self):
        assert _variance([42]) == 0.0

    def test_variance_uniform(self):
        assert _variance([5, 5, 5, 5]) == 0.0

    def test_variance_known(self):
        # [1, 2, 3] → mean=2, var = ((1+0+1)/3) = 0.667
        v = _variance([1, 2, 3])
        assert abs(v - 2 / 3) < 0.001


# ══════════════════════════════════════════════
# ActivityMetrics / ActivityMonitor tests
# ══════════════════════════════════════════════

class TestActivityMetrics:
    def test_app_classification_field_exists(self):
        m = ActivityMetrics()
        assert hasattr(m, "app_classification")
        assert m.app_classification == "productive"

    def test_defaults(self):
        m = ActivityMetrics()
        assert m.total_idle_seconds == 0.0
        assert m.active_app == ""
        assert m.in_productive_app is True

    def test_idle_uses_max(self):
        """total_idle_seconds should be max(mouse, keyboard) not min."""
        m = ActivityMetrics(mouse_idle_seconds=10, keyboard_idle_seconds=60)
        assert max(m.mouse_idle_seconds, m.keyboard_idle_seconds) == 60

    def test_app_sets_exist(self):
        assert len(DEFAULT_PRODUCTIVE_APPS) > 0
        assert len(DEFAULT_NEUTRAL_APPS) > 0
        assert len(DEFAULT_DISTRACTING_APPS) > 0

    def test_no_overlap_productive_distracting(self):
        """Productive and distracting sets should not overlap."""
        overlap = DEFAULT_PRODUCTIVE_APPS & DEFAULT_DISTRACTING_APPS
        assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_no_overlap_neutral_distracting(self):
        overlap = DEFAULT_NEUTRAL_APPS & DEFAULT_DISTRACTING_APPS
        assert len(overlap) == 0, f"Overlap: {overlap}"


class TestActivityMonitorInit:
    def test_custom_app_lists(self):
        monitor = ActivityMonitor(
            productive_apps={"myapp"},
            distracting_apps={"badapp"},
            neutral_apps={"okapp"},
        )
        assert "myapp" in monitor.productive_apps
        assert "badapp" in monitor.distracting_apps
        assert "okapp" in monitor.neutral_apps

    def test_defaults_used(self):
        monitor = ActivityMonitor()
        assert monitor.productive_apps == DEFAULT_PRODUCTIVE_APPS


# ══════════════════════════════════════════════
# AlertManager tests
# ══════════════════════════════════════════════

class TestAlertManager:
    def test_no_alert_when_focused(self):
        am = AlertManager(distraction_threshold_sec=10)
        snap = FocusSnapshot(timestamp=time.time(), state="Focused", focus_score=75)
        state = am.update(snap)
        assert state.distraction_alert_active is False

    def test_distraction_alert_triggers(self):
        """Alert fires after being distracted for longer than threshold."""
        am = AlertManager(distraction_threshold_sec=5, sound_enabled=False)
        t = time.time()
        # Feed distracted snapshots spanning > 5 seconds
        for i in range(10):
            snap = FocusSnapshot(timestamp=t + i, state="Distracted", focus_score=25)
            state = am.update(snap)
        assert state.distraction_alert_active is True
        assert "distracted" in state.distraction_alert_message.lower() or "⚠" in state.distraction_alert_message

    def test_distraction_clears_on_focus(self):
        """Distraction alert clears when returning to focused state."""
        am = AlertManager(distraction_threshold_sec=2, sound_enabled=False)
        t = time.time()
        for i in range(5):
            am.update(FocusSnapshot(timestamp=t + i, state="Distracted", focus_score=25))
        # Now return to focused
        state = am.update(FocusSnapshot(timestamp=t + 6, state="Focused", focus_score=75))
        assert state.distraction_alert_active is False

    def test_break_reminder_triggers(self):
        """Break reminder fires after sustained focus exceeding interval."""
        am = AlertManager(break_interval_min=1, sound_enabled=False)  # 1 minute
        t = time.time()
        # Simulate 90 seconds of focus
        for i in range(90):
            snap = FocusSnapshot(timestamp=t + i, state="Deep Focus", focus_score=90)
            state = am.update(snap)
        assert state.break_reminder_active is True

    def test_acknowledge_break_resets(self):
        """Acknowledging a break clears the reminder."""
        am = AlertManager(break_interval_min=1, sound_enabled=False)
        t = time.time()
        for i in range(90):
            am.update(FocusSnapshot(timestamp=t + i, state="Focused", focus_score=75))
        am.acknowledge_break()
        assert am.state.break_reminder_active is False
        assert am.state.focused_since == 0

    def test_streak_tracking(self):
        """Focus streak increments during sustained focus."""
        am = AlertManager(sound_enabled=False)
        t = time.time()
        for i in range(120):
            am.update(FocusSnapshot(timestamp=t + i, state="Focused", focus_score=75))
        assert am.state.current_streak_minutes > 1.5

    def test_best_streak_persists(self):
        """Best streak recorded even after focus drops."""
        am = AlertManager(sound_enabled=False)
        t = time.time()
        # 60s of focus
        for i in range(60):
            am.update(FocusSnapshot(timestamp=t + i, state="Focused", focus_score=75))
        # Then distracted
        am.update(FocusSnapshot(timestamp=t + 61, state="Distracted", focus_score=25))
        assert am.state.best_streak_minutes >= 0.9

    def test_away_counts_as_break(self):
        """Being away long enough should count as a taken break."""
        am = AlertManager(break_duration_min=1, sound_enabled=False)
        t = time.time()
        # Be away for 2 minutes (exceeds break_duration of 1 min)
        for i in range(120):
            am.update(FocusSnapshot(timestamp=t + i, state="Away", focus_score=5))
        # Return
        state = am.update(FocusSnapshot(timestamp=t + 121, state="Focused", focus_score=75))
        assert am._break_taken_at > 0


# ══════════════════════════════════════════════
# SessionManager tests
# ══════════════════════════════════════════════

class TestSessionManager:
    def test_session_file_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr("focus_tracker.session_manager.DATA_DIR", str(tmp_path))
        sm = SessionManager()
        snapshots = [FocusSnapshot(timestamp=time.time(), focus_score=80, state="Focused")]
        summary = {"avg_score": 80, "total_readings": 1}
        path = sm.save_session(snapshots, summary)
        assert Path(path).exists()

    def test_session_json_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("focus_tracker.session_manager.DATA_DIR", str(tmp_path))
        sm = SessionManager()
        snapshots = [FocusSnapshot(timestamp=time.time(), focus_score=75, state="Focused")]
        summary = {"avg_score": 75, "total_readings": 1}
        sm.save_session(snapshots, summary)
        with open(sm.session_file) as f:
            data = json.load(f)
        assert "session_id" in data
        assert "summary" in data
        assert "snapshots" in data
        assert data["summary"]["avg_score"] == 75

    def test_csv_export(self, tmp_path, monkeypatch):
        monkeypatch.setattr("focus_tracker.session_manager.DATA_DIR", str(tmp_path))
        sm = SessionManager()
        snapshots = [
            FocusSnapshot(timestamp=time.time(), focus_score=80, state="Focused"),
            FocusSnapshot(timestamp=time.time() + 1, focus_score=70, state="Focused"),
        ]
        csv_path = sm.export_csv(snapshots)
        assert Path(csv_path).exists()
        lines = Path(csv_path).read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "focus_score" in lines[0]

    def test_should_autosave(self, tmp_path, monkeypatch):
        monkeypatch.setattr("focus_tracker.session_manager.DATA_DIR", str(tmp_path))
        sm = SessionManager()
        # Just initialized → should not autosave yet (though _last_save is 0)
        assert sm.should_autosave()
        sm._last_save = time.time()
        assert not sm.should_autosave()

    def test_list_past_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("focus_tracker.session_manager.DATA_DIR", str(tmp_path))
        sm = SessionManager()
        snapshots = [FocusSnapshot(timestamp=time.time(), focus_score=80, state="Focused")]
        summary = {"avg_score": 80, "total_readings": 1}
        sm.save_session(snapshots, summary)
        sessions = sm.list_past_sessions()
        assert len(sessions) >= 1
        assert sessions[0]["avg_score"] == 80

    def test_snapshot_sampling(self, tmp_path, monkeypatch):
        """Only every 5th snapshot should be saved."""
        monkeypatch.setattr("focus_tracker.session_manager.DATA_DIR", str(tmp_path))
        sm = SessionManager()
        snapshots = [FocusSnapshot(timestamp=time.time() + i, focus_score=50 + i)
                     for i in range(20)]
        summary = {"avg_score": 60, "total_readings": 20}
        sm.save_session(snapshots, summary)
        with open(sm.session_file) as f:
            data = json.load(f)
        assert len(data["snapshots"]) == 4  # 0, 5, 10, 15


# ══════════════════════════════════════════════
# Config tests
# ══════════════════════════════════════════════

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

    def test_partial_config_merges(self, tmp_path, monkeypatch):
        """A config with only some keys should merge with defaults."""
        config_file = tmp_path / "settings.json"
        config_file.write_text('{"sound_enabled": false}')
        monkeypatch.setattr("focus_tracker.config.CONFIG_FILE", config_file)
        cfg = load_config()
        assert cfg["sound_enabled"] is False
        assert cfg["distraction_threshold_seconds"] == DEFAULTS["distraction_threshold_seconds"]

    def test_unknown_keys_ignored(self, tmp_path, monkeypatch):
        """Unknown keys in saved config should not appear in loaded config."""
        config_file = tmp_path / "settings.json"
        config_file.write_text('{"sound_enabled": true, "unknown_key": 999}')
        monkeypatch.setattr("focus_tracker.config.CONFIG_FILE", config_file)
        cfg = load_config()
        assert "unknown_key" not in cfg


# ══════════════════════════════════════════════
# EyeMetrics tests
# ══════════════════════════════════════════════

class TestEyeMetrics:
    def test_defaults(self):
        m = EyeMetrics()
        assert m.face_detected is False
        assert m.looking_at_screen is False
        assert m.attention_h == 0.0
        assert m.is_reading is False
        assert m.reading_confidence == 0.0

    def test_fields_assignable(self):
        m = EyeMetrics(
            face_detected=True,
            looking_at_screen=True,
            attention_h=0.1,
            attention_v=-0.05,
            head_frontal_confidence=0.95,
            is_reading=True,
            reading_confidence=0.8,
        )
        assert m.looking_at_screen is True
        assert m.head_frontal_confidence == 0.95
        assert m.is_reading is True
        assert m.reading_confidence == 0.8


# ══════════════════════════════════════════════
# FocusSnapshot tests
# ══════════════════════════════════════════════

class TestFocusSnapshot:
    def test_defaults(self):
        s = FocusSnapshot()
        assert s.focus_score == 50.0
        assert s.state == "Neutral"

    def test_all_component_scores_have_defaults(self):
        s = FocusSnapshot()
        assert s.eye_engagement_score == 50.0
        assert s.gaze_stability_score == 50.0
        assert s.blink_score == 50.0
        assert s.activity_score == 50.0
        assert s.app_focus_score == 50.0
