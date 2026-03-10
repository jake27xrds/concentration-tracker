"""
Computer Activity Monitor
Tracks active window, mouse movement, and keyboard activity on macOS
to help assess whether the user is focused on their work.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field

from pynput import mouse, keyboard

# macOS-specific active window detection
try:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
    )
    from AppKit import NSWorkspace
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False


@dataclass
class ActivityMetrics:
    """Snapshot of computer activity at a moment in time."""
    timestamp: float = 0.0
    # Active window info
    active_app: str = ""
    active_window_title: str = ""
    # Mouse activity
    mouse_moves_per_minute: float = 0.0
    mouse_clicks_per_minute: float = 0.0
    mouse_idle_seconds: float = 0.0
    # Keyboard activity
    keys_per_minute: float = 0.0
    keyboard_idle_seconds: float = 0.0
    # Combined
    total_idle_seconds: float = 0.0
    # App switching frequency (high = distracted)
    app_switches_per_minute: float = 0.0
    # Whether the user is in a "productive" app (configurable)
    in_productive_app: bool = True


# Default set of apps considered "productive" — users can customize
DEFAULT_PRODUCTIVE_APPS = {
    "code", "visual studio code", "xcode", "terminal", "iterm",
    "sublime text", "intellij", "pycharm", "webstorm",
    "safari", "google chrome", "firefox", "arc",  # browsers can be productive
    "finder", "preview", "notes", "pages", "keynote", "numbers",
    "microsoft word", "microsoft excel", "microsoft powerpoint",
    "notion", "obsidian", "bear", "craft",
    "slack", "microsoft teams", "zoom", "discord",
    "figma", "sketch", "affinity",
}

# Apps that are almost always distracting
DEFAULT_DISTRACTING_APPS = {
    "tiktok", "instagram", "facebook", "twitter",
    "netflix", "youtube",  # youtube CAN be productive though
    "steam", "epic games",
    "messages", "imessage",
}


class ActivityMonitor:
    """Monitors mouse, keyboard, and active window to gauge computer activity."""

    def __init__(self, productive_apps: set[str] | None = None,
                 distracting_apps: set[str] | None = None):
        self.productive_apps = productive_apps or DEFAULT_PRODUCTIVE_APPS
        self.distracting_apps = distracting_apps or DEFAULT_DISTRACTING_APPS

        # Mouse tracking
        self._mouse_moves: deque = deque(maxlen=500)
        self._mouse_clicks: deque = deque(maxlen=200)
        self._last_mouse_time = time.time()

        # Keyboard tracking
        self._key_presses: deque = deque(maxlen=500)
        self._last_key_time = time.time()

        # App switching
        self._app_switches: deque = deque(maxlen=200)
        self._last_active_app = ""

        # Listeners
        self._mouse_listener = None
        self._key_listener = None
        self._running = False

        self.latest_metrics = ActivityMetrics()

    def start(self):
        """Start monitoring mouse and keyboard in background threads."""
        self._running = True
        self._last_mouse_time = time.time()
        self._last_key_time = time.time()

        self._mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
        )
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        self._key_listener = keyboard.Listener(
            on_press=self._on_key_press,
        )
        self._key_listener.daemon = True
        self._key_listener.start()

    def stop(self):
        """Stop all listeners."""
        self._running = False
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._key_listener:
            self._key_listener.stop()

    def get_metrics(self) -> ActivityMetrics:
        """Collect current activity metrics."""
        now = time.time()
        cutoff = now - 60  # look at last 60 seconds

        metrics = ActivityMetrics(timestamp=now)

        # --- Active Window ---
        app_name, window_title = self._get_active_window()
        metrics.active_app = app_name
        metrics.active_window_title = window_title

        # Check for app switch
        if app_name and app_name != self._last_active_app:
            if self._last_active_app:  # don't count the first detection
                self._app_switches.append(now)
            self._last_active_app = app_name

        # App switches per minute
        while self._app_switches and self._app_switches[0] < cutoff:
            self._app_switches.popleft()
        metrics.app_switches_per_minute = len(self._app_switches)

        # Productive app check
        app_lower = app_name.lower()
        if any(d in app_lower for d in self.distracting_apps):
            metrics.in_productive_app = False
        elif any(p in app_lower for p in self.productive_apps):
            metrics.in_productive_app = True
        else:
            metrics.in_productive_app = True  # default: benefit of the doubt

        # --- Mouse ---
        while self._mouse_moves and self._mouse_moves[0] < cutoff:
            self._mouse_moves.popleft()
        while self._mouse_clicks and self._mouse_clicks[0] < cutoff:
            self._mouse_clicks.popleft()

        metrics.mouse_moves_per_minute = len(self._mouse_moves)
        metrics.mouse_clicks_per_minute = len(self._mouse_clicks)
        metrics.mouse_idle_seconds = now - self._last_mouse_time

        # --- Keyboard ---
        while self._key_presses and self._key_presses[0] < cutoff:
            self._key_presses.popleft()

        metrics.keys_per_minute = len(self._key_presses)
        metrics.keyboard_idle_seconds = now - self._last_key_time

        # --- Combined idle ---
        metrics.total_idle_seconds = min(metrics.mouse_idle_seconds,
                                          metrics.keyboard_idle_seconds)

        self.latest_metrics = metrics
        return metrics

    # ---- Listener callbacks ----

    def _on_mouse_move(self, x, y):
        now = time.time()
        # Throttle: only record moves every 0.1s
        if now - self._last_mouse_time > 0.1:
            self._mouse_moves.append(now)
            self._last_mouse_time = now

    def _on_mouse_click(self, x, y, button, pressed):
        if pressed:
            now = time.time()
            self._mouse_clicks.append(now)
            self._last_mouse_time = now

    def _on_key_press(self, key):
        now = time.time()
        self._key_presses.append(now)
        self._last_key_time = now

    # ---- Active window detection ----

    @staticmethod
    def _get_active_window() -> tuple[str, str]:
        """Get the currently active application name and window title on macOS."""
        if not HAS_QUARTZ:
            return ("Unknown", "")

        try:
            workspace = NSWorkspace.sharedWorkspace()
            active_app_info = workspace.activeApplication()
            if active_app_info:
                app_name = active_app_info.get("NSApplicationName", "Unknown")
            else:
                app_name = "Unknown"

            # Try to get window title
            window_title = ""
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
                kCGNullWindowID
            )
            if window_list:
                for window in window_list:
                    owner = window.get("kCGWindowOwnerName", "")
                    if owner == app_name:
                        title = window.get("kCGWindowName", "")
                        if title:
                            window_title = title
                            break

            return (app_name, window_title)
        except Exception:
            return ("Unknown", "")
