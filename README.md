# 🎯 Focus Tracker — Eye Tracking Concentration Monitor

A real-time concentration monitoring app that uses your **webcam** and **computer activity** to calculate how focused you are.

## What It Does

| Signal | How It's Used |
|---|---|
| **Eye openness** | Open eyes = engaged, closed = drowsy |
| **Blink rate** | Normal ~15-20/min is healthy; extremes = fatigue |
| **Gaze direction** | Looking at screen center = focused |
| **Gaze stability** | Steady gaze = deep focus; darting eyes = distracted |
| **Head pose** | Facing the screen = engaged; turned away = distracted |
| **Keyboard activity** | Typing = working |
| **Mouse activity** | Clicking/moving = interacting |
| **Idle time** | Long idle = away or zoned out |
| **Active app** | Productive apps score higher than distracting ones |
| **App switching** | Rapid switching between apps = distracted |

All these signals are weighted and combined into a **Focus Score (0–100)** with states:
- 🟢 **Deep Focus** (80–100)
- 🟡 **Focused** (60–79)
- 🟠 **Neutral** (40–59)
- 🔴 **Distracted** (0–39)
- ⬛ **Away** (no face + computer idle)

## Requirements

- **Python 3.10+**
- **macOS** (uses macOS APIs for active window detection)
- **Webcam** (built-in or external)

## Setup

```bash
# 1. Navigate to the project folder
cd "eye tracking software"

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### macOS Permissions

You'll need to grant these permissions (macOS will prompt you):
- **Camera** — for eye tracking
- **Accessibility** — for keyboard/mouse monitoring (System Settings → Privacy & Security → Accessibility → add Terminal / VS Code)
- **Screen Recording** — for active window title detection (optional)

## Running

```bash
# Make sure your venv is activated
source venv/bin/activate

# Run the app
python -m focus_tracker
```

## How It Works

### Architecture

```
┌─────────────────┐     ┌──────────────────────┐
│   EyeTracker    │     │   ActivityMonitor     │
│  (webcam +      │     │  (mouse, keyboard,    │
│   MediaPipe)    │     │   active window)      │
└────────┬────────┘     └──────────┬────────────┘
         │                         │
         └──────────┬──────────────┘
                    │
            ┌───────▼────────┐
            │  FocusEngine   │
            │  (scoring &    │
            │   analysis)    │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │   Dashboard    │
            │  (live UI)     │
            └────────────────┘
```

### Files

| File | Purpose |
|---|---|
| `focus_tracker/eye_tracker.py` | Webcam eye tracking via MediaPipe Face Mesh (head + eye fusion) |
| `focus_tracker/activity_monitor.py` | Mouse, keyboard, and active window monitoring (3-tier app classification) |
| `focus_tracker/focus_engine.py` | Combines all signals into a 0–100 focus score with hysteresis |
| `focus_tracker/dashboard.py` | Real-time CustomTkinter dashboard UI with score ring |
| `focus_tracker/config.py` | Settings persistence (saved to ~/Library/Application Support/FocusTracker/) |
| `focus_tracker/alerts.py` | Distraction alerts and break reminders |
| `focus_tracker/session_manager.py` | JSON session persistence and CSV export |
| `focus_tracker/model_downloader.py` | Auto-downloads MediaPipe face model on first run |
| `focus_tracker/main.py` | Entry point — wires everything together |
| `tests/test_core.py` | Unit tests (18 tests covering engine, config, metrics) |

## Customization

### App Classification (3 Tiers)

Edit the app lists in `focus_tracker/activity_monitor.py`:

```python
DEFAULT_PRODUCTIVE_APPS = {"code", "terminal", "xcode", ...}
DEFAULT_NEUTRAL_APPS = {"slack", "safari", "finder", ...}
DEFAULT_DISTRACTING_APPS = {"tiktok", "netflix", ...}
```

Settings are also persisted to `~/Library/Application Support/FocusTracker/settings.json`.

### Focus Score Weights

Adjust how much each signal matters in `focus_tracker/focus_engine.py`:

```python
WEIGHTS = {
    "eye_engagement": 0.20,
    "gaze_stability": 0.20,
    "blink": 0.10,
    "activity": 0.25,
    "app_focus": 0.25,
}
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot open camera" | Check webcam connection and macOS Camera permissions |
| No keyboard/mouse data | Grant Accessibility permission to your terminal app |
| Active window shows "Unknown" | Grant Screen Recording permission |
| Says "looking away" when you're not | Wait 2 seconds for auto-baseline to calibrate, or run manual calibration |
| Score flickers between states | Hysteresis should handle this — if not, check lighting and camera angle |

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```
| Low FPS | Close other camera-using apps; check lighting |
| Face not detected | Ensure good lighting and that your face is visible to the webcam |
