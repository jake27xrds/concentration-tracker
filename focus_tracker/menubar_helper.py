"""
macOS Menu Bar Helper
Runs as a separate subprocess launched by the dashboard.
Reads a shared JSON status file and displays the focus score in the menu bar.

Usage (standalone):
    python -m focus_tracker.menubar_helper
"""

import json
import os
import sys

STATUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_menubar_status.json")

try:
    import rumps
    _RUMPS_AVAILABLE = True
except ImportError:
    _RUMPS_AVAILABLE = False


# State abbreviations and unicode indicators (no emoji)
_STATE_ABBR = {
    "Deep Focus": "DF",
    "Focused":    "F",
    "Neutral":    "N",
    "Distracted": "D",
    "Away":       "—",
}

# Unicode filled/hollow dots used as a subtle level indicator
_STATE_DOT = {
    "Deep Focus": "◉",
    "Focused":    "◉",
    "Neutral":    "◎",
    "Distracted": "○",
    "Away":       "○",
}


def _bar(score: float, width: int = 10) -> str:
    """Render a compact ASCII progress bar, e.g. '████████░░'."""
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


if _RUMPS_AVAILABLE:
    class FocusMenuBarApp(rumps.App):
        def __init__(self):
            # quit_button=None removes rumps' auto-added Quit item so we control it
            super().__init__("Focus", title="◎ --", quit_button=None)
            self._score = 0.0
            self._state = "Neutral"
            self._intent = "general"
            self._streak = 0.0

            self.menu = [
                rumps.MenuItem("Focus Tracker", callback=None),
                None,
                rumps.MenuItem("score_row"),
                rumps.MenuItem("state_row"),
                rumps.MenuItem("intent_row"),
                rumps.MenuItem("streak_row"),
                None,
                rumps.MenuItem("Quit Focus Tracker", callback=self._quit),
            ]
            # Set initial placeholder text
            self.menu["score_row"].title   = "Score    —"
            self.menu["state_row"].title   = "State    —"
            self.menu["intent_row"].title  = "Intent   —"
            self.menu["streak_row"].title  = "Streak   —"

        @rumps.timer(2)
        def _refresh(self, _sender):
            """Read shared status file every 2 seconds."""
            try:
                if not os.path.exists(STATUS_FILE):
                    return
                with open(STATUS_FILE) as f:
                    data = json.load(f)

                self._score  = float(data.get("score", self._score))
                self._state  = data.get("state", self._state)
                self._intent = data.get("intent", self._intent)
                self._streak = float(data.get("streak_min", self._streak))

                abbr = _STATE_ABBR.get(self._state, "?")
                dot  = _STATE_DOT.get(self._state, "○")

                # Compact menu bar title: dot + score + state abbr
                self.title = f"{dot} {self._score:.0f}  {abbr}"

                bar = _bar(self._score)
                self.menu["score_row"].title  = f"Score    {self._score:.0f}/100  {bar}"
                self.menu["state_row"].title  = f"State    {self._state}"
                self.menu["intent_row"].title = f"Intent   {self._intent.capitalize()}"
                streak_str = f"{self._streak:.0f} min" if self._streak > 0 else "none"
                self.menu["streak_row"].title = f"Streak   {streak_str}"

            except (json.JSONDecodeError, OSError, KeyError):
                pass

        def _quit(self, _sender):
            try:
                if os.path.exists(STATUS_FILE):
                    os.remove(STATUS_FILE)
            except OSError:
                pass
            rumps.quit_application()

    def main():
        FocusMenuBarApp().run()

else:
    def main():
        print("rumps not available — menu bar helper cannot run.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
