"""Session analysis utility for Focus Tracker saved JSON sessions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnalysisResult:
    session_count: int
    total_minutes: float
    avg_session_score: float
    state_pct: dict[str, float]
    false_distracted_count: int
    false_distracted_pct: float
    by_app_counter: Counter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Focus Tracker session files.")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sessions",
        help="Directory containing session_*.json files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only analyze the most recent N session files (0 = all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser.parse_args()


def _load_session_files(sessions_dir: Path, limit: int) -> list[Path]:
    files = sorted(sessions_dir.glob("session_*.json"))
    if limit > 0:
        return files[-limit:]
    return files


def _is_false_distracted(snapshot: dict) -> bool:
    if snapshot.get("state") != "Distracted":
        return False

    if bool(snapshot.get("is_reading")):
        return True

    activity = float(snapshot.get("activity", 0.0))
    app_class = str(snapshot.get("app_classification", "neutral"))
    if activity >= 55:
        return True
    if app_class in ("productive", "neutral") and activity >= 45:
        return True
    return False


def analyze_sessions(session_files: list[Path]) -> AnalysisResult:
    if not session_files:
        return AnalysisResult(0, 0.0, 0.0, {}, 0, 0.0, Counter())

    session_summaries = []
    all_snapshots = []
    by_app_counter: Counter[str] = Counter()

    for file_path in session_files:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        summary = payload.get("summary", {})
        session_summaries.append(summary)

        snapshots = payload.get("snapshots", [])
        all_snapshots.extend(snapshots)

        for snap in snapshots:
            if snap.get("state") == "Distracted":
                app = str(snap.get("active_app", "Unknown") or "Unknown")
                by_app_counter[app] += 1

    total_minutes = sum(float(s.get("duration_minutes", 0.0)) for s in _iter_session_meta(session_files))
    if total_minutes <= 0:
        total_minutes = sum(float(s.get("duration_minutes", 0.0)) for s in session_summaries)

    avg_session_score = sum(float(s.get("avg_score", 0.0)) for s in session_summaries) / len(session_summaries)

    state_counter = Counter(str(snap.get("state", "Unknown")) for snap in all_snapshots)
    total_snapshots = max(1, len(all_snapshots))
    state_pct = {
        state: (count / total_snapshots) * 100.0
        for state, count in state_counter.items()
    }

    false_distracted_count = sum(1 for snap in all_snapshots if _is_false_distracted(snap))
    distracted_total = max(1, state_counter.get("Distracted", 0))
    false_distracted_pct = (false_distracted_count / distracted_total) * 100.0

    return AnalysisResult(
        session_count=len(session_files),
        total_minutes=total_minutes,
        avg_session_score=avg_session_score,
        state_pct=state_pct,
        false_distracted_count=false_distracted_count,
        false_distracted_pct=false_distracted_pct,
        by_app_counter=by_app_counter,
    )


def _iter_session_meta(session_files: list[Path]) -> list[dict]:
    data = []
    for file_path in session_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            data.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return data


def _to_dict(result: AnalysisResult) -> dict:
    top_apps = [{"app": app, "distracted_count": count} for app, count in result.by_app_counter.most_common(5)]
    return {
        "session_count": result.session_count,
        "total_minutes": round(result.total_minutes, 2),
        "avg_session_score": round(result.avg_session_score, 2),
        "state_pct": {k: round(v, 2) for k, v in sorted(result.state_pct.items())},
        "false_distracted_count": result.false_distracted_count,
        "false_distracted_pct": round(result.false_distracted_pct, 2),
        "top_distracted_apps": top_apps,
    }


def _print_human(result: AnalysisResult) -> None:
    payload = _to_dict(result)
    print(f"Sessions analyzed: {payload['session_count']}")
    print(f"Total tracked minutes: {payload['total_minutes']}")
    print(f"Average session score: {payload['avg_session_score']}")

    print("State distribution:")
    if not payload["state_pct"]:
        print("  (no snapshots)")
    else:
        for state, pct in payload["state_pct"].items():
            print(f"  - {state}: {pct}%")

    print(
        "Possible false Distracted events: "
        f"{payload['false_distracted_count']} "
        f"({payload['false_distracted_pct']}% of Distracted snapshots)"
    )

    print("Top apps during Distracted:")
    if not payload["top_distracted_apps"]:
        print("  (none)")
    else:
        for row in payload["top_distracted_apps"]:
            print(f"  - {row['app']}: {row['distracted_count']}")


def main() -> None:
    args = _parse_args()
    session_files = _load_session_files(args.sessions_dir, args.limit)
    result = analyze_sessions(session_files)

    if args.json:
        print(json.dumps(_to_dict(result), indent=2))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()