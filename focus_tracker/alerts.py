"""
Alerts & Break Reminder System
Monitors focus state and triggers alerts/reminders.
"""

import time
import subprocess
import platform
from dataclasses import dataclass

from focus_tracker.focus_engine import FocusSnapshot


@dataclass
class AlertState:
    """Current state of alerts."""
    # Distraction alert
    distracted_since: float = 0.0
    distraction_alert_active: bool = False
    distraction_alert_message: str = ""
    # Break reminder
    focused_since: float = 0.0
    break_reminder_active: bool = False
    break_reminder_message: str = ""
    # Away detection
    away_since: float = 0.0
    # Streak tracking
    current_streak_minutes: float = 0.0
    best_streak_minutes: float = 0.0
    # Smart nudge payload
    nudge_active: bool = False
    nudge_type: str = ""
    nudge_message: str = ""
    nudge_reason: str = ""
    nudge_last_fired_at: float = 0.0


class AlertManager:
    """
    Manages focus alerts and break reminders.

    Alerts:
    - Distraction alert: triggers after being distracted for N seconds
    - Break reminder: suggests a break after sustained focus for N minutes
    - Away alert: notes when user has been away

    Uses macOS system sounds for audio feedback.
    """

    def __init__(
        self,
        distraction_threshold_sec: int = 30,
        break_interval_min: int = 25,
        break_duration_min: int = 5,
        sound_enabled: bool = True,
        nudge_cooldowns_sec: dict | None = None,
        fatigue_break_enabled: bool = True,
    ):
        self.distraction_threshold = distraction_threshold_sec
        self.break_interval = break_interval_min * 60  # convert to seconds
        self.break_duration = break_duration_min * 60
        self.sound_enabled = sound_enabled
        self.fatigue_break_enabled = fatigue_break_enabled

        self.state = AlertState()
        self._last_sound_time = 0.0
        self._sound_cooldown = 30.0  # don't spam sounds
        self._break_taken_at = 0.0
        self._in_break = False
        self._nudge_cooldowns = nudge_cooldowns_sec or {
            "prolonged_distraction": 45,
            "high_app_switching": 60,
            "long_idle_drift": 60,
            "break_due": 180,
            "fatigue_risk": 180,
        }
        self._last_nudge_at: dict[str, float] = {}

        # Focus streak tracking
        self._streak_start: float | None = None
        self._best_streak = 0.0

    def update(self, snapshot: FocusSnapshot, eye_metrics=None, activity_metrics=None, goal_progress: dict | None = None) -> AlertState:
        """Update alert state based on the latest focus snapshot."""
        now = snapshot.timestamp
        state = snapshot.state

        # --- Distraction Alert ---
        if state == "Distracted":
            if self.state.distracted_since == 0:
                self.state.distracted_since = now
            elapsed = now - self.state.distracted_since

            if elapsed >= self.distraction_threshold:
                self.state.distraction_alert_active = True
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                if minutes > 0:
                    self.state.distraction_alert_message = (
                        f"⚠️ You've been distracted for {minutes}m {seconds}s — "
                        f"try refocusing on your task!"
                    )
                else:
                    self.state.distraction_alert_message = (
                        f"⚠️ Distracted for {seconds}s — get back on track!"
                    )
                self._play_alert_sound()
                self._maybe_fire_nudge(
                    now,
                    "prolonged_distraction",
                    self.state.distraction_alert_message,
                    "Sustained distracted state exceeded threshold",
                )
            else:
                self.state.distraction_alert_active = False
                self.state.distraction_alert_message = ""
        else:
            self.state.distracted_since = 0
            self.state.distraction_alert_active = False
            self.state.distraction_alert_message = ""

        # --- Break Reminder ---
        if state in ("Deep Focus", "Focused"):
            if self.state.focused_since == 0:
                self.state.focused_since = now
            elapsed = now - self.state.focused_since

            # Don't remind if they just took a break
            time_since_break = now - self._break_taken_at
            if elapsed >= self.break_interval and time_since_break > self.break_interval:
                self.state.break_reminder_active = True
                minutes = int(elapsed // 60)
                self.state.break_reminder_message = (
                    f"🧘 Great focus for {minutes} minutes! "
                    f"Consider a {self.break_duration // 60}-minute break."
                )
                self._play_break_sound()
                self._maybe_fire_nudge(
                    now,
                    "break_due",
                    self.state.break_reminder_message,
                    "Sustained focus reached break interval",
                )
            else:
                self.state.break_reminder_active = False
                self.state.break_reminder_message = ""
        else:
            if self.state.focused_since > 0:
                # Track the focus streak that just ended
                streak = now - self.state.focused_since
                if streak > self._best_streak:
                    self._best_streak = streak
            self.state.focused_since = 0
            self.state.break_reminder_active = False
            self.state.break_reminder_message = ""

        # --- Away Detection ---
        if state == "Away":
            if self.state.away_since == 0:
                self.state.away_since = now
        else:
            if self.state.away_since > 0:
                away_duration = now - self.state.away_since
                if away_duration >= self.break_duration:
                    # Count as a break taken
                    self._break_taken_at = now
                self.state.away_since = 0

        # --- Streak tracking ---
        if state in ("Deep Focus", "Focused"):
            if self._streak_start is None:
                self._streak_start = now
            self.state.current_streak_minutes = (now - self._streak_start) / 60
        else:
            if self._streak_start is not None:
                streak = (now - self._streak_start) / 60
                if streak > self.state.best_streak_minutes:
                    self.state.best_streak_minutes = streak
                self._streak_start = None
            self.state.current_streak_minutes = 0

        self.state.best_streak_minutes = max(
            self.state.best_streak_minutes, self.state.current_streak_minutes
        )

        self._evaluate_contextual_nudges(
            now=now,
            snapshot=snapshot,
            eye_metrics=eye_metrics,
            activity_metrics=activity_metrics,
            goal_progress=goal_progress or {},
        )

        return self.state

    def acknowledge_break(self):
        """User acknowledges the break reminder — reset timer."""
        self._break_taken_at = time.time()
        self.state.focused_since = 0
        self.state.break_reminder_active = False
        self.state.break_reminder_message = ""
        self.state.nudge_active = False
        self.state.nudge_message = ""
        self.state.nudge_reason = ""
        self.state.nudge_type = ""

    def _play_alert_sound(self):
        """Play a gentle alert sound on macOS."""
        if not self.sound_enabled:
            return
        now = time.time()
        if now - self._last_sound_time < self._sound_cooldown:
            return
        self._last_sound_time = now
        if platform.system() == "Darwin":
            # Use macOS system sounds
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Basso.aiff"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _play_break_sound(self):
        """Play a pleasant break reminder sound."""
        if not self.sound_enabled:
            return
        now = time.time()
        if now - self._last_sound_time < self._sound_cooldown * 2:
            return
        self._last_sound_time = now
        if platform.system() == "Darwin":
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _maybe_fire_nudge(self, now: float, nudge_type: str, message: str, reason: str) -> bool:
        cooldown = float(self._nudge_cooldowns.get(nudge_type, 60))
        last = self._last_nudge_at.get(nudge_type, 0.0)
        if now - last < cooldown:
            return False
        self._last_nudge_at[nudge_type] = now
        self.state.nudge_active = True
        self.state.nudge_type = nudge_type
        self.state.nudge_message = message
        self.state.nudge_reason = reason
        self.state.nudge_last_fired_at = now
        return True

    def _evaluate_contextual_nudges(
        self,
        now: float,
        snapshot: FocusSnapshot,
        eye_metrics=None,
        activity_metrics=None,
        goal_progress: dict | None = None,
    ) -> None:
        # Reset transient nudge if stale.
        if self.state.nudge_active and (now - self.state.nudge_last_fired_at > 20):
            self.state.nudge_active = False
            self.state.nudge_type = ""
            self.state.nudge_message = ""
            self.state.nudge_reason = ""

        if activity_metrics is not None:
            if getattr(activity_metrics, "app_switches_per_minute", 0) >= 10:
                self._maybe_fire_nudge(
                    now,
                    "high_app_switching",
                    "🔁 Rapid app switching detected. Try committing to one task for 10 minutes.",
                    "High app switching indicates context fragmentation",
                )
            elif getattr(activity_metrics, "total_idle_seconds", 0) >= 120 and snapshot.state != "Away":
                self._maybe_fire_nudge(
                    now,
                    "long_idle_drift",
                    "⏳ Idle drift detected. Reset with a tiny next action and resume.",
                    "Long idle interval without Away state",
                )

        fatigue_risk = False
        if eye_metrics is not None and self.fatigue_break_enabled:
            if getattr(eye_metrics, "eyes_closed_duration", 0.0) >= 1.2:
                fatigue_risk = True
            if getattr(eye_metrics, "blinks_per_minute", 0.0) >= 32:
                fatigue_risk = True

        if fatigue_risk:
            self._maybe_fire_nudge(
                now,
                "fatigue_risk",
                "😴 Fatigue signs detected. A 3–5 minute break can preserve focus quality.",
                "Eye fatigue indicators exceeded threshold",
            )
            # Adaptive break coach: fatigue can activate reminder earlier.
            if not self.state.break_reminder_active:
                streak_min = self.state.current_streak_minutes
                if streak_min >= 15:
                    self.state.break_reminder_active = True
                    self.state.break_reminder_message = (
                        "🧘 You are showing fatigue signals after a long streak. "
                        "Consider a short recovery break."
                    )

        # Goal-aware encouragement / warning.
        if goal_progress and goal_progress.get("enabled", False):
            if not goal_progress.get("on_track", True) and snapshot.state == "Distracted":
                self._maybe_fire_nudge(
                    now,
                    "prolonged_distraction",
                    "🎯 Goal pace is slipping. Try one 5-minute focused sprint now.",
                    "Goal pace is behind while distracted",
                )
