"""
Session Manager
Saves focus session data to JSON files and can export to CSV.
Sessions are auto-saved periodically and on shutdown.
"""

import json
import csv
import os
import time
from datetime import datetime
from pathlib import Path

from focus_tracker.focus_engine import FocusSnapshot


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")


class SessionManager:
    """Handles saving and loading focus sessions."""

    AUTOSAVE_INTERVAL = 60  # seconds between autosaves

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._last_save = 0.0

    @property
    def session_file(self) -> str:
        return os.path.join(DATA_DIR, f"session_{self.session_id}.json")

    def should_autosave(self) -> bool:
        return time.time() - self._last_save > self.AUTOSAVE_INTERVAL

    def save_session(self, snapshots: list[FocusSnapshot], summary: dict):
        """Save the current session to a JSON file."""
        data = {
            "session_id": self.session_id,
            "start_time": self.session_start,
            "end_time": time.time(),
            "duration_minutes": (time.time() - self.session_start) / 60,
            "summary": summary,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "focus_score": round(s.focus_score, 1),
                    "state": s.state,
                    "eye_engagement": round(s.eye_engagement_score, 1),
                    "gaze_stability": round(s.gaze_stability_score, 1),
                    "blink": round(s.blink_score, 1),
                    "activity": round(s.activity_score, 1),
                    "app_focus": round(s.app_focus_score, 1),
                }
                # Sample every 5th snapshot to keep file size reasonable
                for i, s in enumerate(snapshots) if i % 5 == 0
            ],
        }

        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=2)

        self._last_save = time.time()
        return self.session_file

    def export_csv(self, snapshots: list[FocusSnapshot]) -> str:
        """Export session data to CSV. Returns the file path."""
        csv_path = os.path.join(DATA_DIR, f"session_{self.session_id}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "time_elapsed_s", "focus_score", "state",
                "eye_engagement", "gaze_stability", "blink",
                "activity", "app_focus"
            ])
            for s in snapshots:
                writer.writerow([
                    datetime.fromtimestamp(s.timestamp).isoformat(),
                    round(s.timestamp - self.session_start, 1),
                    round(s.focus_score, 1),
                    s.state,
                    round(s.eye_engagement_score, 1),
                    round(s.gaze_stability_score, 1),
                    round(s.blink_score, 1),
                    round(s.activity_score, 1),
                    round(s.app_focus_score, 1),
                ])
        return csv_path

    def list_past_sessions(self) -> list[dict]:
        """List all saved sessions with basic info."""
        sessions = []
        for f in sorted(Path(DATA_DIR).glob("session_*.json"), reverse=True):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                sessions.append({
                    "file": str(f),
                    "session_id": data.get("session_id", ""),
                    "duration_minutes": round(data.get("duration_minutes", 0), 1),
                    "avg_score": round(data.get("summary", {}).get("avg_score", 0), 1),
                    "date": data.get("session_id", "").replace("_", " ", 1).replace("-", ":", 2),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions
