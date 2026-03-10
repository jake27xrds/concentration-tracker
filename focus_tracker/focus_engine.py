"""
Focus Score Engine
Combines eye tracking metrics and computer activity metrics into a single
concentration / focus score from 0 (completely unfocused) to 100 (deep focus).
Also tracks focus history over time for trend analysis.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field

from focus_tracker.eye_tracker import EyeMetrics
from focus_tracker.activity_monitor import ActivityMetrics

log = logging.getLogger("focus_tracker.engine")


@dataclass
class FocusSnapshot:
    """A single focus score reading with its components."""
    timestamp: float = 0.0
    # Overall score (0-100)
    focus_score: float = 50.0
    # Component scores (each 0-100)
    eye_engagement_score: float = 50.0
    gaze_stability_score: float = 50.0
    blink_score: float = 50.0
    activity_score: float = 50.0
    app_focus_score: float = 50.0
    # Derived state
    state: str = "Neutral"  # "Deep Focus", "Focused", "Neutral", "Distracted", "Away"


class FocusEngine:
    """
    Calculates a focus score by combining signals:

    Eye Signals:
    - Eye openness (EAR) — open eyes = engaged
    - Gaze stability — looking at screen center = focused
    - Blink rate — normal 15-20/min is fine, too high = fatigue, too low = staring
    - Eye closure — prolonged = drowsy/distracted
    - Head pose — facing screen = engaged

    Computer Signals:
    - Keyboard/mouse activity — some activity = working
    - Idle time — long idle = possibly away or zoned out
    - App switching — frequent switching = distracted
    - Productive vs distracting apps
    """

    # Weights for final score
    WEIGHTS = {
        "eye_engagement": 0.20,
        "gaze_stability": 0.20,
        "blink": 0.10,
        "activity": 0.25,
        "app_focus": 0.25,
    }

    def __init__(self, history_minutes: int = 60):
        max_entries = history_minutes * 60  # ~1 per second
        self.history: deque[FocusSnapshot] = deque(maxlen=max_entries)
        self._gaze_history: deque = deque(maxlen=30)  # for stability calc
        self._gaze_variance_history: deque = deque(maxlen=20)  # sustained variance tracking

        # Hysteresis: prevent state jitter on score boundaries
        self._current_state = "Neutral"
        self._state_entered_at = time.time()
        self._HYSTERESIS_MARGIN = 3.0    # score must cross threshold by this margin
        self._MIN_STATE_HOLD = 2.0       # minimum seconds before transitioning down

    def calculate(self, eye: EyeMetrics, activity: ActivityMetrics) -> FocusSnapshot:
        """Compute focus score from the latest eye and activity metrics."""
        snap = FocusSnapshot(timestamp=time.time())

        # --- Eye Engagement (face + combined attention + sustained eye closure) ---
        if not eye.face_detected:
            snap.eye_engagement_score = 10.0  # face not visible = probably away
        else:
            # Base: face detected = engaged. Only penalize for sustained closure,
            # NOT brief blinks (blinks are handled by blink_score separately).
            if eye.eyes_closed_duration > 3.0:
                eye_open_score = 20.0  # very drowsy
            elif eye.eyes_closed_duration > 1.5:
                eye_open_score = 40.0
            elif eye.eyes_closed_duration > 0.5:
                eye_open_score = 65.0
            else:
                eye_open_score = 100.0  # eyes open or just a normal blink

            # Combined attention: fused head pose + eye gaze (more reliable than either alone)
            if eye.looking_at_screen:
                attention_score = 100.0
            else:
                # Magnitude of combined attention away from center
                away_mag = (abs(eye.attention_h) ** 2 + abs(eye.attention_v) ** 2) ** 0.5
                attention_score = _remap(away_mag, 0.3, 1.0, 85, 15)

            snap.eye_engagement_score = min(100, (eye_open_score * 0.4 + attention_score * 0.6))

            # Reading = actively processing screen content → strong engagement signal
            if eye.is_reading and eye.reading_confidence > 0.3:
                snap.eye_engagement_score = min(100, snap.eye_engagement_score + 15)

        # --- Gaze Stability (uses combined head+eye attention for reliability) ---
        # Two-tier: only penalize variance that persists (sustained drift),
        # not brief saccades which are normal during reading.
        self._gaze_history.append((eye.attention_h, eye.attention_v))
        if len(self._gaze_history) >= 5:
            h_vals = [g[0] for g in self._gaze_history]
            v_vals = [g[1] for g in self._gaze_history]
            instant_var = _variance(h_vals) + _variance(v_vals)
            self._gaze_variance_history.append(instant_var)

            # Use median of recent variance windows — filters brief spikes
            if len(self._gaze_variance_history) >= 3:
                sorted_vars = sorted(self._gaze_variance_history)
                sustained_var = sorted_vars[len(sorted_vars) // 2]  # median
            else:
                sustained_var = instant_var

            snap.gaze_stability_score = _remap(sustained_var, 0.0, 0.15, 100, 20)
        else:
            snap.gaze_stability_score = 50.0

        # Reading boost: saccades during reading are intentional, not instability
        if eye.is_reading:
            reading_boost = eye.reading_confidence * 25  # up to +25 pts
            snap.gaze_stability_score = min(100, snap.gaze_stability_score + reading_boost)

        # --- Blink Score ---
        # Healthy range: 12-20 blinks/min. Score is continuous, not stepped.
        bpm = eye.blinks_per_minute
        if bpm <= 20:
            # 0 bpm → 35, 12 bpm → 95, 20 bpm → 95 (peak in healthy range)
            snap.blink_score = _remap(bpm, 0, 12, 35, 95)
        else:
            # 20 bpm → 95, 40+ bpm → 30 (excessive = fatigue/stress)
            snap.blink_score = _remap(bpm, 20, 40, 95, 30)

        # --- Activity Score ---
        # Reading overrides idle penalty: reading IS the activity, even without
        # keyboard/mouse input. The eyes prove engagement.
        if eye.is_reading and eye.reading_confidence > 0.3:
            # Scale activity score with reading confidence (0.3-1.0 → 65-90)
            snap.activity_score = 55 + eye.reading_confidence * 35
        elif activity.total_idle_seconds > 300:
            snap.activity_score = 5.0  # away for 5+ minutes
        elif activity.total_idle_seconds > 120:
            snap.activity_score = 20.0
        elif activity.total_idle_seconds > 60:
            snap.activity_score = 40.0
        elif activity.total_idle_seconds > 30:
            snap.activity_score = 60.0
        else:
            # Active: base score from typing + mouse
            type_score = _remap(activity.keys_per_minute, 0, 100, 30, 100)
            mouse_score = _remap(activity.mouse_moves_per_minute, 0, 200, 30, 80)
            snap.activity_score = max(type_score, mouse_score)

        # Penalize excessive app switching (>8 switches/min = distracted)
        switch_penalty = max(0, (activity.app_switches_per_minute - 4) * 8)
        snap.activity_score = max(0, snap.activity_score - switch_penalty)

        # --- App Focus Score (3-tier: productive / neutral / distracting) ---
        if hasattr(activity, 'app_classification'):
            classification = activity.app_classification
        else:
            classification = "productive" if activity.in_productive_app else "distracting"

        if classification == "productive":
            snap.app_focus_score = 90.0
        elif classification == "neutral":
            snap.app_focus_score = 55.0
        else:
            snap.app_focus_score = 25.0

        # --- Weighted Final Score ---
        snap.focus_score = (
            snap.eye_engagement_score * self.WEIGHTS["eye_engagement"]
            + snap.gaze_stability_score * self.WEIGHTS["gaze_stability"]
            + snap.blink_score * self.WEIGHTS["blink"]
            + snap.activity_score * self.WEIGHTS["activity"]
            + snap.app_focus_score * self.WEIGHTS["app_focus"]
        )
        snap.focus_score = max(0, min(100, snap.focus_score))

        # --- Determine State (with hysteresis to prevent jitter) ---
        if not eye.face_detected and activity.total_idle_seconds > 60:
            raw_state = "Away"
        elif snap.focus_score >= 80:
            raw_state = "Deep Focus"
        elif snap.focus_score >= 60:
            raw_state = "Focused"
        elif snap.focus_score >= 40:
            raw_state = "Neutral"
        else:
            raw_state = "Distracted"

        snap.state = self._apply_hysteresis(raw_state, snap.focus_score)

        self.history.append(snap)
        return snap

    # State ranking for hysteresis: higher = better focus
    _STATE_RANK = {"Distracted": 0, "Neutral": 1, "Focused": 2, "Deep Focus": 3, "Away": -1}

    def _apply_hysteresis(self, raw_state: str, score: float) -> str:
        """
        Prevent state jitter by requiring:
        1. Score crosses threshold by a margin before upgrading/downgrading.
        2. Minimum hold time before transitioning to a worse state.
        """
        now = time.time()

        # "Away" always takes effect immediately
        if raw_state == "Away" or self._current_state == "Away":
            if raw_state != self._current_state:
                self._current_state = raw_state
                self._state_entered_at = now
            return raw_state

        raw_rank = self._STATE_RANK.get(raw_state, 1)
        cur_rank = self._STATE_RANK.get(self._current_state, 1)

        # Upgrading (improving focus) — apply immediately with small margin
        if raw_rank > cur_rank:
            # Check that we're clearly past the threshold (margin)
            thresholds = {"Deep Focus": 80, "Focused": 60, "Neutral": 40}
            threshold = thresholds.get(raw_state, 0)
            if score >= threshold + self._HYSTERESIS_MARGIN:
                self._current_state = raw_state
                self._state_entered_at = now

        # Downgrading (losing focus) — require margin + hold time
        elif raw_rank < cur_rank:
            thresholds = {"Focused": 80, "Neutral": 60, "Distracted": 40}
            threshold = thresholds.get(self._current_state, 100)
            time_in_state = now - self._state_entered_at
            if score < threshold - self._HYSTERESIS_MARGIN and time_in_state >= self._MIN_STATE_HOLD:
                self._current_state = raw_state
                self._state_entered_at = now

        return self._current_state

    def get_average_score(self, last_n_seconds: int = 300) -> float:
        """Average focus score over the last N seconds."""
        if not self.history:
            return 50.0
        cutoff = time.time() - last_n_seconds
        scores = [s.focus_score for s in self.history if s.timestamp >= cutoff]
        return sum(scores) / len(scores) if scores else 50.0

    def get_history_points(self, last_n_seconds: int = 300) -> list[tuple[float, float]]:
        """Return (relative_time_seconds, score) pairs for graphing."""
        if not self.history:
            return []
        now = time.time()
        cutoff = now - last_n_seconds
        return [
            (s.timestamp - now, s.focus_score)
            for s in self.history
            if s.timestamp >= cutoff
        ]

    def get_session_summary(self) -> dict:
        """Summary statistics for the entire session."""
        if not self.history:
            return {"avg_score": 0, "max_score": 0, "min_score": 0,
                    "time_focused_pct": 0, "time_distracted_pct": 0}

        scores = [s.focus_score for s in self.history]
        states = [s.state for s in self.history]
        total = len(states)

        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "time_focused_pct": (states.count("Deep Focus") + states.count("Focused")) / total * 100,
            "time_distracted_pct": states.count("Distracted") / total * 100,
            "time_away_pct": states.count("Away") / total * 100,
            "total_readings": total,
        }


def _remap(value: float, in_min: float, in_max: float,
           out_min: float, out_max: float) -> float:
    """Linearly remap a value from one range to another, clamped."""
    if in_max == in_min:
        return (out_min + out_max) / 2
    t = (value - in_min) / (in_max - in_min)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)


def _variance(values: list[float]) -> float:
    """Simple variance calculation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
