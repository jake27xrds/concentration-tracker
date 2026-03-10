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
python -m focus_tracker.main
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
| `focus_tracker/eye_tracker.py` | Webcam eye tracking via MediaPipe Face Mesh |
| `focus_tracker/activity_monitor.py` | Mouse, keyboard, and active window monitoring |
| `focus_tracker/focus_engine.py` | Combines all signals into a 0–100 focus score |
| `focus_tracker/dashboard.py` | Real-time CustomTkinter dashboard UI |
| `focus_tracker/main.py` | Entry point — wires everything together |

## Customization

### Productive / Distracting Apps

Edit the app lists in `focus_tracker/activity_monitor.py`:

```python
DEFAULT_PRODUCTIVE_APPS = {"code", "terminal", "safari", ...}
DEFAULT_DISTRACTING_APPS = {"tiktok", "netflix", ...}
```

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
| Low FPS | Close other camera-using apps; check lighting |
| Face not detected | Ensure good lighting and that your face is visible to the webcam |
