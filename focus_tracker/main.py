"""
Focus Tracker — Main Entry Point
Launches the eye tracker, activity monitor, focus engine, and dashboard.
"""

import sys
import signal
import logging

from focus_tracker.eye_tracker import EyeTracker
from focus_tracker.activity_monitor import ActivityMonitor
from focus_tracker.focus_engine import FocusEngine
from focus_tracker.dashboard import FocusDashboard

log = logging.getLogger("focus_tracker.main")


def main():
    log.info("Focus Tracker — Starting up")

    # --- Initialize components ---
    eye_tracker = EyeTracker(camera_index=0)
    activity_monitor = ActivityMonitor()
    focus_engine = FocusEngine(history_minutes=60)

    # --- Start eye tracker ---
    camera_available = False
    log.info("Opening webcam & loading face model...")
    try:
        eye_tracker.start()
        camera_available = True
        log.info("Webcam ready & model loaded")
    except RuntimeError as e:
        log.warning("Camera unavailable: %s", e)
        log.info("Running in activity-only mode (keyboard/mouse/app tracking)")

    # --- Start activity monitor ---
    log.info("Starting activity monitor...")
    try:
        activity_monitor.start()
        log.info("Mouse & keyboard tracking active")
    except Exception as e:
        log.warning("Activity monitor error: %s — continuing without it", e)

    log.info("Launching dashboard...")

    # --- Launch dashboard ---
    dashboard = FocusDashboard(
        eye_tracker, activity_monitor, focus_engine,
        camera_available=camera_available,
    )

    def shutdown(*_):
        log.info("Shutting down...")
        eye_tracker.stop()
        activity_monitor.stop()
        summary = focus_engine.get_session_summary()
        if summary["total_readings"] > 0:
            log.info(
                "Session summary — Avg: %.0f | Focused: %.0f%% | "
                "Distracted: %.0f%% | Away: %.0f%%",
                summary["avg_score"],
                summary["time_focused_pct"],
                summary["time_distracted_pct"],
                summary.get("time_away_pct", 0),
            )

    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))

    try:
        dashboard.start()  # blocks until window is closed
    finally:
        shutdown()


if __name__ == "__main__":
    main()
