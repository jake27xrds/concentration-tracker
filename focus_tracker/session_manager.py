"""
Session Manager
Saves focus session data to JSON files and can export to CSV.
Sessions are auto-saved periodically and on shutdown.
"""

import json
import csv
import os
import time
from collections import defaultdict
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
                    "active_app": s.active_app,
                    "app_classification": s.app_classification,
                    "profile_name": s.profile_name,
                    "active_domain": s.active_domain,
                    "is_reading": s.is_reading,
                    "goal_progress_pct": round(s.goal_progress_pct, 1),
                    "nudge_reason": s.nudge_reason,
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
                , "active_app", "app_classification", "profile_name", "active_domain",
                "is_reading", "goal_progress_pct", "nudge_reason"
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
                    s.active_app,
                    s.app_classification,
                    s.profile_name,
                    s.active_domain,
                    int(bool(s.is_reading)),
                    round(s.goal_progress_pct, 1),
                    s.nudge_reason,
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

    def load_recent_sessions(self, max_sessions: int = 30) -> list[dict]:
        """Load full session payloads, skipping corrupt files."""
        loaded = []
        for f in sorted(Path(DATA_DIR).glob("session_*.json"), reverse=True)[:max_sessions]:
            try:
                with open(f) as fh:
                    loaded.append(json.load(fh))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        return loaded

    def aggregate_hourly_focus(self, max_sessions: int = 30) -> list[dict]:
        """Average focus by hour of day across recent sessions."""
        buckets: dict[int, list[float]] = defaultdict(list)
        for sess in self.load_recent_sessions(max_sessions=max_sessions):
            for snap in sess.get("snapshots", []):
                ts = snap.get("timestamp")
                score = snap.get("focus_score")
                if ts is None or score is None:
                    continue
                hour = datetime.fromtimestamp(ts).hour
                buckets[hour].append(float(score))

        out = []
        for hour in sorted(buckets):
            vals = buckets[hour]
            out.append({
                "hour": hour,
                "avg_score": round(sum(vals) / len(vals), 2),
                "samples": len(vals),
            })
        return out

    def aggregate_app_impact(self, max_sessions: int = 30, top_n: int = 10) -> dict:
        """Compute app/profile/classification impact aggregates."""
        by_app: dict[str, list[float]] = defaultdict(list)
        by_class: dict[str, list[float]] = defaultdict(list)
        by_profile: dict[str, list[float]] = defaultdict(list)

        for sess in self.load_recent_sessions(max_sessions=max_sessions):
            for snap in sess.get("snapshots", []):
                score = snap.get("focus_score")
                if score is None:
                    continue
                score = float(score)
                app = (snap.get("active_app") or "Unknown").strip()
                cls = (snap.get("app_classification") or "neutral").strip()
                profile = (snap.get("profile_name") or "Coding").strip()
                by_app[app].append(score)
                by_class[cls].append(score)
                by_profile[profile].append(score)

        def summarize_map(source: dict[str, list[float]], limit: int | None = None) -> list[dict]:
            rows = []
            for key, vals in source.items():
                if not vals:
                    continue
                rows.append({
                    "key": key,
                    "avg_score": round(sum(vals) / len(vals), 2),
                    "samples": len(vals),
                })
            rows.sort(key=lambda r: r["samples"], reverse=True)
            return rows[:limit] if limit is not None else rows

        return {
            "top_apps": summarize_map(by_app, limit=top_n),
            "by_classification": summarize_map(by_class),
            "by_profile": summarize_map(by_profile),
        }

    def detect_distraction_windows(
        self,
        max_sessions: int = 30,
        score_threshold: float = 45.0,
        min_points: int = 3,
    ) -> list[dict]:
        """Find repeated low-focus windows and nudge clusters."""
        windows = []
        for sess in self.load_recent_sessions(max_sessions=max_sessions):
            snaps = sess.get("snapshots", [])
            current = []
            for snap in snaps:
                score = float(snap.get("focus_score", 50))
                nudged = bool(snap.get("nudge_reason"))
                low = score <= score_threshold or nudged
                if low:
                    current.append(snap)
                else:
                    if len(current) >= min_points:
                        windows.append(self._window_from_points(sess, current))
                    current = []
            if len(current) >= min_points:
                windows.append(self._window_from_points(sess, current))
        return sorted(windows, key=lambda w: w["duration_sec"], reverse=True)

    @staticmethod
    def _window_from_points(session: dict, points: list[dict]) -> dict:
        start_ts = points[0].get("timestamp", 0.0)
        end_ts = points[-1].get("timestamp", start_ts)
        avg = sum(float(p.get("focus_score", 50)) for p in points) / len(points)
        reasons = [p.get("nudge_reason", "") for p in points if p.get("nudge_reason")]
        return {
            "session_id": session.get("session_id", ""),
            "start": datetime.fromtimestamp(start_ts).isoformat() if start_ts else "",
            "end": datetime.fromtimestamp(end_ts).isoformat() if end_ts else "",
            "duration_sec": round(max(0.0, end_ts - start_ts), 1),
            "avg_score": round(avg, 2),
            "nudge_reasons": sorted(set(reasons)),
        }
