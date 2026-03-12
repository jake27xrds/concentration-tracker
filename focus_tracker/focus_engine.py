"""
Focus Score Engine
Combines eye tracking metrics and computer activity metrics into a single
concentration / focus score from 0 (completely unfocused) to 100 (deep focus).
Also tracks focus history over time for trend analysis.
"""

import math
import time
import logging
from collections import deque
from dataclasses import dataclass

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
    # Metadata for analytics/coaching
    active_app: str = ""
    app_classification: str = "neutral"
    profile_name: str = "Coding"
    active_domain: str = ""
    is_reading: bool = False
    intent_name: str = "Coding"
    baseline_adjustment: float = 0.0
    goal_progress_pct: float = 0.0
    nudge_reason: str = ""


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

    # Backward-compatible default weights
    WEIGHTS = {
        "eye_engagement": 0.20,
        "gaze_stability": 0.20,
        "blink": 0.10,
        "activity": 0.25,
        "app_focus": 0.25,
    }

    # Intent-specific weighting profiles (each sums to 1.0)
    INTENT_WEIGHTS = {
        "general": WEIGHTS,
        "coding": {
            "eye_engagement": 0.15,
            "gaze_stability": 0.15,
            "blink": 0.08,
            "activity": 0.32,
            "app_focus": 0.30,
        },
        "reading": {
            "eye_engagement": 0.28,
            "gaze_stability": 0.30,
            "blink": 0.12,
            "activity": 0.12,
            "app_focus": 0.18,
        },
        "studying": {
            "eye_engagement": 0.25,
            "gaze_stability": 0.25,
            "blink": 0.10,
            "activity": 0.18,
            "app_focus": 0.22,
        },
        "writing": {
            "eye_engagement": 0.18,
            "gaze_stability": 0.20,
            "blink": 0.10,
            "activity": 0.30,
            "app_focus": 0.22,
        },
    }

    def __init__(
        self,
        history_minutes: int = 60,
        goal_minutes_target: int = 45,
        goal_enabled: bool = True,
        intent_name: str = "coding",
        baseline_enabled: bool = True,
        weekly_sessions_target: int = 5,
    ):
        max_entries = history_minutes * 60  # ~1 per second
        self.history: deque[FocusSnapshot] = deque(maxlen=max_entries)
        self._gaze_history: deque = deque(maxlen=30)  # for stability calc
        self._gaze_variance_history: deque = deque(maxlen=20)  # sustained variance tracking

        self.goal_minutes_target = max(1, int(goal_minutes_target))
        self.goal_enabled = bool(goal_enabled)
        self.weekly_sessions_target = max(1, int(weekly_sessions_target))

        self.current_intent = self._normalize_intent(intent_name)
        self.baseline_enabled = bool(baseline_enabled)
        self._baseline_buffers: dict[str, deque[float]] = {}
        self._baseline_max_samples = 3600

        self._focused_seconds = 0.0
        self._last_timestamp: float | None = None

        # Hysteresis: prevent state jitter on score boundaries
        self._current_state = "Neutral"
        self._state_entered_at = time.time()
        self._HYSTERESIS_MARGIN = 3.0
        self._MIN_STATE_HOLD = 2.0

    def calculate(self, eye: EyeMetrics, activity: ActivityMetrics) -> FocusSnapshot:
        """Compute focus score from the latest eye and activity metrics."""
        snap = FocusSnapshot(timestamp=time.time())
        snap.active_app = getattr(activity, "active_app", "")
        snap.app_classification = getattr(activity, "app_classification", "neutral")
        snap.profile_name = getattr(activity, "profile_name", "Coding")
        snap.active_domain = getattr(activity, "active_domain", "")
        snap.is_reading = bool(eye.is_reading)
        snap.intent_name = self.current_intent.title()

        # --- Eye Engagement ---
        if not eye.face_detected:
            snap.eye_engagement_score = 10.0
        else:
            if eye.eyes_closed_duration > 3.0:
                eye_open_score = 20.0
            elif eye.eyes_closed_duration > 1.5:
                eye_open_score = 40.0
            elif eye.eyes_closed_duration > 0.5:
                eye_open_score = 65.0
            else:
                eye_open_score = 100.0

            if eye.looking_at_screen:
                attention_score = 100.0
            else:
                away_mag = (abs(eye.attention_h) ** 2 + abs(eye.attention_v) ** 2) ** 0.5
                attention_score = _remap(away_mag, 0.3, 1.0, 85, 15)

            snap.eye_engagement_score = min(100, (eye_open_score * 0.4 + attention_score * 0.6))
            if eye.is_reading and eye.reading_confidence > 0.3:
                snap.eye_engagement_score = min(100, snap.eye_engagement_score + 15)

        # --- Gaze Stability ---
        self._gaze_history.append((eye.attention_h, eye.attention_v))
        if len(self._gaze_history) >= 5:
            h_vals = [g[0] for g in self._gaze_history]
            v_vals = [g[1] for g in self._gaze_history]
            instant_var = _variance(h_vals) + _variance(v_vals)
            self._gaze_variance_history.append(instant_var)

            if len(self._gaze_variance_history) >= 3:
                sorted_vars = sorted(self._gaze_variance_history)
                sustained_var = sorted_vars[len(sorted_vars) // 2]
            else:
                sustained_var = instant_var

            snap.gaze_stability_score = _remap(sustained_var, 0.0, 0.15, 100, 20)
        else:
            snap.gaze_stability_score = 50.0

        if eye.is_reading:
            reading_boost = eye.reading_confidence * 25
            snap.gaze_stability_score = min(100, snap.gaze_stability_score + reading_boost)

        # --- Blink Score ---
        bpm = eye.blinks_per_minute
        if bpm <= 20:
            snap.blink_score = _remap(bpm, 0, 12, 35, 95)
        else:
            snap.blink_score = _remap(bpm, 20, 40, 95, 30)

        # --- Activity Score ---
        if eye.is_reading and eye.reading_confidence > 0.3:
            snap.activity_score = 55 + eye.reading_confidence * 35
        elif activity.total_idle_seconds > 300:
            snap.activity_score = 5.0
        elif activity.total_idle_seconds > 120:
            snap.activity_score = 20.0
        elif activity.total_idle_seconds > 60:
            snap.activity_score = 40.0
        elif activity.total_idle_seconds > 30:
            snap.activity_score = 60.0
        else:
            type_score = _remap(activity.keys_per_minute, 0, 100, 30, 100)
            mouse_score = _remap(activity.mouse_moves_per_minute, 0, 200, 30, 80)
            snap.activity_score = max(type_score, mouse_score)

        switch_penalty = max(0, (activity.app_switches_per_minute - 4) * 8)
        snap.activity_score = max(0, snap.activity_score - switch_penalty)

        # --- App Focus Score ---
        if hasattr(activity, "app_classification"):
            classification = activity.app_classification
        else:
            classification = "productive" if activity.in_productive_app else "distracting"

        if classification == "productive":
            snap.app_focus_score = 90.0
        elif classification == "neutral":
            snap.app_focus_score = 55.0
        else:
            snap.app_focus_score = 25.0

        weights = self._intent_weights(self.current_intent)
        raw_score = (
            snap.eye_engagement_score * weights["eye_engagement"]
            + snap.gaze_stability_score * weights["gaze_stability"]
            + snap.blink_score * weights["blink"]
            + snap.activity_score * weights["activity"]
            + snap.app_focus_score * weights["app_focus"]
        )

        # Small intent-aware shaping before baseline normalization.
        if self.current_intent == "reading" and eye.is_reading and eye.reading_confidence > 0.35:
            raw_score += 4.0
        elif self.current_intent == "coding" and classification == "productive":
            raw_score += 2.0
        elif self.current_intent == "writing" and activity.keys_per_minute >= 15:
            raw_score += 2.0

        raw_score = max(0, min(100, raw_score))
        adjusted_score, adjustment = self._apply_personal_baseline(raw_score, self.current_intent)
        snap.baseline_adjustment = adjustment
        snap.focus_score = max(0, min(100, adjusted_score))

        # --- Determine State ---
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
        self._update_goal_progress(snap)

        self.history.append(snap)
        self._record_baseline_sample(raw_score, self.current_intent)
        return snap

    _STATE_RANK = {"Distracted": 0, "Neutral": 1, "Focused": 2, "Deep Focus": 3, "Away": -1}

    def _apply_hysteresis(self, raw_state: str, score: float) -> str:
        now = time.time()

        if raw_state == "Away" or self._current_state == "Away":
            if raw_state != self._current_state:
                self._current_state = raw_state
                self._state_entered_at = now
            return raw_state

        raw_rank = self._STATE_RANK.get(raw_state, 1)
        cur_rank = self._STATE_RANK.get(self._current_state, 1)

        if raw_rank > cur_rank:
            thresholds = {"Deep Focus": 80, "Focused": 60, "Neutral": 40}
            threshold = thresholds.get(raw_state, 0)
            if score >= threshold + self._HYSTERESIS_MARGIN:
                self._current_state = raw_state
                self._state_entered_at = now

        elif raw_rank < cur_rank:
            thresholds = {"Focused": 80, "Neutral": 60, "Distracted": 40}
            threshold = thresholds.get(self._current_state, 100)
            time_in_state = now - self._state_entered_at
            if score < threshold - self._HYSTERESIS_MARGIN and time_in_state >= self._MIN_STATE_HOLD:
                self._current_state = raw_state
                self._state_entered_at = now

        return self._current_state

    def get_average_score(self, last_n_seconds: int = 300) -> float:
        if not self.history:
            return 50.0
        cutoff = time.time() - last_n_seconds
        scores = [s.focus_score for s in self.history if s.timestamp >= cutoff]
        return sum(scores) / len(scores) if scores else 50.0

    def get_history_points(self, last_n_seconds: int = 300) -> list[tuple[float, float]]:
        if not self.history:
            return []
        now = time.time()
        cutoff = now - last_n_seconds
        return [(s.timestamp - now, s.focus_score) for s in self.history if s.timestamp >= cutoff]

    def get_session_summary(self) -> dict:
        if not self.history:
            return {
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
                "time_focused_pct": 0,
                "time_distracted_pct": 0,
                "time_away_pct": 0,
                "total_readings": 0,
                "intent": self.current_intent,
                "goal_completion_pct": 0,
                "consistency_score": 0,
                "milestones_reached": [],
                "baseline_enabled": self.baseline_enabled,
            }

        scores = [s.focus_score for s in self.history]
        states = [s.state for s in self.history]
        total = len(states)
        goal = self.get_goal_progress()

        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "time_focused_pct": (states.count("Deep Focus") + states.count("Focused")) / total * 100,
            "time_distracted_pct": states.count("Distracted") / total * 100,
            "time_away_pct": states.count("Away") / total * 100,
            "total_readings": total,
            "intent": self.current_intent,
            "goal_completion_pct": goal["progress_pct"],
            "consistency_score": goal["consistency_score"],
            "milestones_reached": goal["milestones_reached"],
            "baseline_enabled": self.baseline_enabled,
        }

    def set_goal(self, minutes_target: int, enabled: bool = True) -> None:
        self.goal_minutes_target = max(1, int(minutes_target))
        self.goal_enabled = bool(enabled)

    def set_weekly_sessions_target(self, sessions_per_week: int) -> None:
        self.weekly_sessions_target = max(1, int(sessions_per_week))

    def set_intent(self, intent_name: str) -> None:
        self.current_intent = self._normalize_intent(intent_name)

    def set_baseline_enabled(self, enabled: bool) -> None:
        self.baseline_enabled = bool(enabled)

    def get_baseline_stats(self) -> dict:
        buf = self._baseline_buffers.get(self.current_intent)
        if not buf:
            return {
                "enabled": self.baseline_enabled,
                "intent": self.current_intent,
                "samples": 0,
                "avg": 50.0,
                "std": 0.0,
            }
        vals = list(buf)
        avg = sum(vals) / len(vals)
        std = math.sqrt(_variance(vals))
        return {
            "enabled": self.baseline_enabled,
            "intent": self.current_intent,
            "samples": len(vals),
            "avg": round(avg, 2),
            "std": round(std, 2),
        }

    def reset_goal_progress(self) -> None:
        self._focused_seconds = 0.0
        self._last_timestamp = None
        for snap in self.history:
            snap.goal_progress_pct = 0.0

    def get_goal_progress(self) -> dict:
        target_seconds = self.goal_minutes_target * 60
        focused_minutes = self._focused_seconds / 60
        pct = min(100.0, (self._focused_seconds / target_seconds) * 100) if target_seconds > 0 else 0.0

        session_seconds = 0.0
        if self.history:
            session_seconds = max(1.0, self.history[-1].timestamp - self.history[0].timestamp)

        pace = (self._focused_seconds / session_seconds) if session_seconds > 0 else 0.0
        on_track = pace >= 0.6 if self.goal_enabled else True
        eta_minutes = max(0.0, (target_seconds - self._focused_seconds) / 60) if pace > 0 else float("inf")

        consistency_score = 0.0
        if self.history:
            focused_points = sum(1 for s in self.history if s.state in ("Focused", "Deep Focus"))
            consistency_score = (focused_points / len(self.history)) * 100

        milestones = [25, 50, 75, 100]
        milestones_reached = [m for m in milestones if pct >= m]
        next_milestone = next((m for m in milestones if pct < m), None)

        return {
            "enabled": self.goal_enabled,
            "target_minutes": self.goal_minutes_target,
            "focused_minutes": focused_minutes,
            "progress_pct": pct,
            "on_track": on_track,
            "eta_minutes": eta_minutes,
            "consistency_score": consistency_score,
            "milestones_reached": milestones_reached,
            "next_milestone": next_milestone,
            "weekly_sessions_target": self.weekly_sessions_target,
        }

    def _update_goal_progress(self, snap: FocusSnapshot) -> None:
        now = snap.timestamp
        if self._last_timestamp is None:
            self._last_timestamp = now
            snap.goal_progress_pct = 0.0
            return

        dt = max(0.0, min(2.0, now - self._last_timestamp))
        self._last_timestamp = now
        if snap.state in ("Focused", "Deep Focus"):
            self._focused_seconds += dt

        target_seconds = self.goal_minutes_target * 60
        if self.goal_enabled and target_seconds > 0:
            snap.goal_progress_pct = min(100.0, (self._focused_seconds / target_seconds) * 100)
        else:
            snap.goal_progress_pct = 0.0

    def _intent_weights(self, intent_name: str) -> dict[str, float]:
        intent = self._normalize_intent(intent_name)
        return self.INTENT_WEIGHTS.get(intent, self.WEIGHTS)

    @staticmethod
    def _normalize_intent(intent_name: str) -> str:
        text = str(intent_name or "general").strip().lower()
        aliases = {
            "study": "studying",
            "read": "reading",
            "code": "coding",
            "write": "writing",
        }
        return aliases.get(text, text if text in FocusEngine.INTENT_WEIGHTS else "general")

    def _record_baseline_sample(self, raw_score: float, intent_name: str) -> None:
        intent = self._normalize_intent(intent_name)
        buf = self._baseline_buffers.get(intent)
        if buf is None:
            buf = deque(maxlen=self._baseline_max_samples)
            self._baseline_buffers[intent] = buf
        buf.append(raw_score)

    def _apply_personal_baseline(self, raw_score: float, intent_name: str) -> tuple[float, float]:
        if not self.baseline_enabled:
            return raw_score, 0.0

        intent = self._normalize_intent(intent_name)
        buf = self._baseline_buffers.get(intent)
        if not buf or len(buf) < 60:
            return raw_score, 0.0

        vals = list(buf)
        mean = sum(vals) / len(vals)
        std = max(5.0, math.sqrt(_variance(vals)))

        z = (raw_score - mean) / std
        normalized = 50.0 + z * 12.0
        adjusted = raw_score * 0.7 + normalized * 0.3
        adjusted = max(0.0, min(100.0, adjusted))
        return adjusted, adjusted - raw_score


def _remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
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
