"""
Alerts & Break Reminder System
Monitors focus state and triggers alerts/reminders.
"""

import time
import subprocess
import platform
from collections import deque
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
    ):
        self.distraction_threshold = distraction_threshold_sec
        self.break_interval = break_interval_min * 60  # convert to seconds
        self.break_duration = break_duration_min * 60
        self.sound_enabled = sound_enabled

        self.state = AlertState()
        self._last_sound_time = 0.0
        self._sound_cooldown = 30.0  # don't spam sounds
        self._break_taken_at = 0.0
        self._in_break = False

        # Focus streak tracking
        self._streak_start: float | None = None
        self._best_streak = 0.0

    def update(self, snapshot: FocusSnapshot) -> AlertState:
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

        return self.state

    def acknowledge_break(self):
        """User acknowledges the break reminder — reset timer."""
        self._break_taken_at = time.time()
        self.state.focused_since = 0
        self.state.break_reminder_active = False
        self.state.break_reminder_message = ""

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
