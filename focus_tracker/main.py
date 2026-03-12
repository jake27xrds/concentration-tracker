"""
Focus Tracker — Main Entry Point
Launches the eye tracker, activity monitor, focus engine, and dashboard.
"""

import signal
import logging
import argparse

from focus_tracker import configure_logging
from focus_tracker.eye_tracker import EyeTracker
from focus_tracker.activity_monitor import ActivityMonitor
from focus_tracker.focus_engine import FocusEngine
from focus_tracker.dashboard import FocusDashboard

log = logging.getLogger("focus_tracker.main")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for runtime configuration."""
    parser = argparse.ArgumentParser(description="Run the Focus Tracker dashboard.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to open (default: 0).",
    )
    parser.add_argument(
        "--history-minutes",
        type=int,
        default=60,
        help="Focus history window size in minutes (default: 60).",
    )
    parser.add_argument(
        "--intent",
        type=str,
        default="coding",
        choices=["coding", "reading", "studying", "writing", "general"],
        help="Session intent profile to tune scoring (default: coding).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Disable personal baseline adaptation for this run.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    history_minutes = max(1, args.history_minutes)
    if history_minutes != args.history_minutes:
        log.warning("history_minutes must be >= 1; using %d", history_minutes)

    log.info(
        "Focus Tracker — Starting up (camera_index=%d, history_minutes=%d, intent=%s, baseline=%s)",
        args.camera_index,
        history_minutes,
        args.intent,
        "on" if not args.no_baseline else "off",
    )

    # --- Initialize components ---
    eye_tracker = EyeTracker(camera_index=args.camera_index)
    activity_monitor = ActivityMonitor()
    focus_engine = FocusEngine(
        history_minutes=history_minutes,
        intent_name=args.intent,
        baseline_enabled=not args.no_baseline,
    )

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
    except Exception:
        log.exception("Unexpected eye tracker startup error; continuing without camera")

    # --- Start activity monitor ---
    log.info("Starting activity monitor...")
    try:
        activity_monitor.start()
        log.info("Mouse & keyboard tracking active")
    except Exception:
        log.exception("Activity monitor error; continuing without it")

    log.info("Launching dashboard...")

    # --- Launch dashboard ---
    dashboard = FocusDashboard(
        eye_tracker, activity_monitor, focus_engine,
        camera_available=camera_available,
    )

    shutting_down = False

    def shutdown(*_) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        log.info("Shutting down...")
        try:
            eye_tracker.stop()
        except Exception:
            log.exception("Error while stopping eye tracker")
        try:
            activity_monitor.stop()
        except Exception:
            log.exception("Error while stopping activity monitor")
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

    def _signal_handler(signum, _frame):
        log.info("Received signal %s", signum)
        shutdown()
        if getattr(dashboard, "root", None) is not None:
            try:
                dashboard.root.after(0, dashboard.root.destroy)
            except Exception:
                log.exception("Failed to close dashboard after signal")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        dashboard.start()  # blocks until window is closed
    finally:
        shutdown()


if __name__ == "__main__":
    main()
