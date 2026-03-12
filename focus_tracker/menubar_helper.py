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
import tempfile

STATUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_menubar_status.json")

# Hide this subprocess from the macOS Dock before AppKit/rumps initialises.
# Setting LSUIElement in the in-memory bundle info marks the process as a
# "background-only" UI agent, exactly like a proper menu-bar-only app bundle.
try:
    from Foundation import NSBundle
    _bundle_info = NSBundle.mainBundle().infoDictionary()
    if _bundle_info is not None:
        _bundle_info["LSUIElement"] = "1"
except Exception:
    pass

try:
    import rumps
    _RUMPS_AVAILABLE = True
except ImportError:
    _RUMPS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# RGBA colors per focus state
_STATE_COLOR = {
    "Deep Focus": (16, 185, 129, 255),   # emerald green
    "Focused":    (59, 130, 246, 255),   # blue
    "Neutral":    (245, 158, 11, 255),   # amber
    "Distracted": (239, 68, 68, 255),    # red
    "Away":       (107, 114, 128, 255),  # gray
}

_TRACK_COLOR = (80, 80, 80, 60)   # subtle dark-gray track ring
_ICON_SIZE   = 44                  # 44px → crisp on Retina (22pt logical)
_RING_WIDTH  = 5
_MARGIN      = 4


def _make_ring_icon(score: float, state: str, prev_path: str | None) -> str:
    """
    Render a circular progress ring as a transparent PNG.
    Returns the path to the (new) temp file.
    The caller is responsible for deleting `prev_path` after setting the icon.
    """
    size = _ICON_SIZE
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    color = _STATE_COLOR.get(state, _STATE_COLOR["Neutral"])
    bbox  = [_MARGIN, _MARGIN, size - _MARGIN, size - _MARGIN]

    # Background track (full circle)
    draw.arc(bbox, start=0, end=360, fill=_TRACK_COLOR, width=_RING_WIDTH)

    # Foreground arc proportional to score (starts at 12 o'clock = -90°)
    if score > 0:
        sweep = (score / 100.0) * 360.0
        draw.arc(bbox, start=-90, end=-90 + sweep, fill=color, width=_RING_WIDTH)

    # Write to a new temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()

    # Clean up old temp file
    if prev_path and os.path.exists(prev_path):
        try:
            os.remove(prev_path)
        except OSError:
            pass

    return tmp.name


_STATE_SYMBOL = {
    "Deep Focus": "●",
    "Focused":    "●",
    "Neutral":    "◐",
    "Distracted": "○",
    "Away":       "○",
}


def _bar(score: float, width: int = 14) -> str:
    """Thin segmented progress bar using parallelogram blocks."""
    filled = round(score / 100 * width)
    return "▰" * filled + "▱" * (width - filled)


if _RUMPS_AVAILABLE:
    class FocusMenuBarApp(rumps.App):
        def __init__(self):
            # quit_button=None removes rumps' auto-added Quit item so we control it
            super().__init__("Focus", title="--", quit_button=None)
            self._score  = 0.0
            self._state  = "Neutral"
            self._intent = "general"
            self._streak = 0.0
            self._icon_path: str | None = None

            self.menu = [
                rumps.MenuItem("Focus Tracker", callback=None),
                None,
                rumps.MenuItem("score_row"),
                rumps.MenuItem("bar_row"),
                None,
                rumps.MenuItem("state_row"),
                rumps.MenuItem("intent_row"),
                rumps.MenuItem("streak_row"),
                None,
                rumps.MenuItem("Quit Focus Tracker", callback=self._quit),
            ]
            self._set_menu_placeholders()

            # Draw initial neutral ring
            self._update_icon()

        def _set_menu_placeholders(self):
            self.menu["score_row"].title  = "  —  /  100"
            self.menu["bar_row"].title    = "  " + "▱" * 14
            self.menu["state_row"].title  = "  ○  —"
            self.menu["intent_row"].title = "  ⌘  —"
            self.menu["streak_row"].title = "  ⏱  —"

        def _update_icon(self):
            """Regenerate the ring icon and apply it to the menu bar."""
            if not _PIL_AVAILABLE:
                self.title = f"{self._score:.0f}"
                return

            new_path = _make_ring_icon(self._score, self._state, self._icon_path)
            self._icon_path = new_path
            self.icon  = new_path
            self.title = f"{self._score:.0f}"

        @rumps.timer(2)
        def _refresh(self, _sender):
            """Read shared status file every 2 seconds."""
            try:
                if not os.path.exists(STATUS_FILE):
                    return
                with open(STATUS_FILE) as f:
                    data = json.load(f)

                prev_score = self._score
                prev_state = self._state

                self._score  = float(data.get("score", self._score))
                self._state  = data.get("state", self._state)
                self._intent = data.get("intent", self._intent)
                self._streak = float(data.get("streak_min", self._streak))

                # Redraw ring icon only when something meaningful changed
                if abs(self._score - prev_score) >= 1 or self._state != prev_state:
                    self._update_icon()

                sym      = _STATE_SYMBOL.get(self._state, "○")
                bar      = _bar(self._score)
                streak_s = f"{self._streak:.0f} min" if self._streak > 0 else "—"

                self.menu["score_row"].title  = f"  {self._score:.0f} / 100"
                self.menu["bar_row"].title    = f"  {bar}"
                self.menu["state_row"].title  = f"  {sym}  {self._state}"
                self.menu["intent_row"].title = f"  ⌘  {self._intent.capitalize()}"
                self.menu["streak_row"].title = f"  ⏱  {streak_s}"

            except (json.JSONDecodeError, OSError, KeyError):
                pass

        def _quit(self, _sender):
            try:
                if os.path.exists(STATUS_FILE):
                    os.remove(STATUS_FILE)
            except OSError:
                pass
            # Clean up icon temp file
            if self._icon_path and os.path.exists(self._icon_path):
                try:
                    os.remove(self._icon_path)
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
