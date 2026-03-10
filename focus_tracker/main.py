"""
Focus Tracker — Main Entry Point
Launches the eye tracker, activity monitor, focus engine, and dashboard.
"""

import sys
import signal

from focus_tracker.eye_tracker import EyeTracker
from focus_tracker.activity_monitor import ActivityMonitor
from focus_tracker.focus_engine import FocusEngine
from focus_tracker.dashboard import FocusDashboard


def main():
    print("🎯 Focus Tracker — Starting up...")
    print()

    # --- Initialize components ---
    eye_tracker = EyeTracker(camera_index=0)
    activity_monitor = ActivityMonitor()
    focus_engine = FocusEngine(history_minutes=60)

    # --- Start eye tracker ---
    print("📷 Opening webcam & loading face model...")
    try:
        eye_tracker.start()
        print("   ✓ Webcam ready & model loaded")
    except RuntimeError as e:
        print(f"   ✗ {e}")
        print()
        print("The app requires a webcam. Please connect one and try again.")
        print("On macOS, you may also need to grant camera access in:")
        print("  System Settings → Privacy & Security → Camera")
        sys.exit(1)

    # --- Start activity monitor ---
    print("💻 Starting activity monitor...")
    try:
        activity_monitor.start()
        print("   ✓ Mouse & keyboard tracking active")
    except Exception as e:
        print(f"   ⚠ Activity monitor error: {e}")
        print("   Continuing without activity monitoring...")

    print("🖥  Launching dashboard...")
    print()
    print("Press Ctrl+C or close the window to quit.")
    print()

    # --- Launch dashboard ---
    dashboard = FocusDashboard(eye_tracker, activity_monitor, focus_engine)

    def shutdown(*_):
        print("\n🛑 Shutting down...")
        eye_tracker.stop()
        activity_monitor.stop()
        # Print session summary
        summary = focus_engine.get_session_summary()
        if summary["total_readings"] > 0:
            print()
            print("📊 Session Summary:")
            print(f"   Average Focus Score: {summary['avg_score']:.0f}/100")
            print(f"   Time Focused:        {summary['time_focused_pct']:.0f}%")
            print(f"   Time Distracted:     {summary['time_distracted_pct']:.0f}%")
            print(f"   Time Away:           {summary.get('time_away_pct', 0):.0f}%")
            print(f"   Best Score:          {summary['max_score']:.0f}")
            print(f"   Worst Score:         {summary['min_score']:.0f}")
        print()
        print("Goodbye! 👋")

    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))

    try:
        dashboard.start()  # blocks until window is closed
    finally:
        shutdown()


if __name__ == "__main__":
    main()
