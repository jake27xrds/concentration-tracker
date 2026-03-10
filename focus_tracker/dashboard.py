"""
Dashboard UI
A CustomTkinter-based live dashboard with tabs for:
- Live monitoring (camera feed, score, components)
- Focus timeline graph (canvas-drawn scrolling chart)
- Session history & settings
"""

import time
import threading
import subprocess
import logging
import math
import tkinter as tk
from collections import deque
from datetime import datetime

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np

from focus_tracker.eye_tracker import EyeTracker, EyeMetrics
from focus_tracker.activity_monitor import ActivityMonitor, ActivityMetrics
from focus_tracker.focus_engine import FocusEngine, FocusSnapshot
from focus_tracker.alerts import AlertManager, AlertState
from focus_tracker.session_manager import SessionManager
from focus_tracker.config import load_config, save_config

log = logging.getLogger("focus_tracker.dashboard")


# Color scheme
COLORS = {
    "deep_focus": "#10b981",   # emerald
    "focused": "#34d399",      # light emerald
    "neutral": "#f59e0b",      # amber
    "distracted": "#f97316",   # orange
    "away": "#ef4444",         # red
    "bg_dark": "#0f0f0f",      # near-black
    "bg_card": "#181818",      # dark card
    "bg_card_alt": "#202020",  # card hover
    "text": "#f5f5f5",
    "text_dim": "#737373",
    "accent": "#3b82f6",       # clean blue
    "accent_light": "#93c5fd",
    "graph_bg": "#0a0a0a",
    "graph_grid": "#1e1e1e",
    "alert_bg": "#3b0000",
    "break_bg": "#001a33",
}


def state_color(state: str) -> str:
    return {
        "Deep Focus": COLORS["deep_focus"],
        "Focused": COLORS["focused"],
        "Neutral": COLORS["neutral"],
        "Distracted": COLORS["distracted"],
        "Away": COLORS["away"],
    }.get(state, COLORS["text_dim"])


class FocusDashboard:
    """Main dashboard window with tabbed interface."""

    UPDATE_INTERVAL_MS = 100  # ~10 FPS

    def __init__(self, eye_tracker: EyeTracker, activity_monitor: ActivityMonitor,
                 focus_engine: FocusEngine, camera_available: bool = True):
        self.eye_tracker = eye_tracker
        self.activity_monitor = activity_monitor
        self.focus_engine = focus_engine
        self.alert_manager = AlertManager()
        self.session_manager = SessionManager()
        self.camera_available = camera_available

        # Load persisted settings
        self._config = load_config()

        self._latest_snapshot = FocusSnapshot()
        self._alert_state = AlertState()
        self._running = False
        self._calibrating = False
        self._calibration_stage_results: dict[str, dict] = {}

        # Graph data buffer
        self._graph_scores: deque = deque(maxlen=600)  # 10 min at 1/sec
        self._graph_states: deque = deque(maxlen=600)
        self._last_graph_draw = 0.0  # throttle graph redraws to 1/sec

        # --- Build Window ---
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Focus")
        self.root.geometry("1280x820")
        self.root.minsize(1060, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._apply_config()

    def _build_layout(self):
        """Create the tabbed UI."""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # ---- Top bar: score + state + alerts ----
        self._build_top_bar()

        # ---- Tabview ----
        self.tabview = ctk.CTkTabview(self.root, fg_color=COLORS["bg_dark"],
                                       segmented_button_fg_color=COLORS["bg_card"],
                                       segmented_button_selected_color=COLORS["accent"],
                                       segmented_button_unselected_color=COLORS["bg_card_alt"],
                                       segmented_button_selected_hover_color=COLORS["accent"],
                                       segmented_button_unselected_hover_color=COLORS["bg_card_alt"])
        self.tabview.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        self.tab_live = self.tabview.add("  Live  ")
        self.tab_graph = self.tabview.add("  Timeline  ")
        self.tab_history = self.tabview.add("  History  ")
        self.tab_settings = self.tabview.add("  Settings  ")
        self.tab_analytics = self.tabview.add("  Analytics  ")

        self._build_live_tab()
        self._build_graph_tab()
        self._build_history_tab()
        self._build_settings_tab()
        self._build_analytics_tab()

    # ---- TOP BAR ----

    def _build_top_bar(self):
        top = ctk.CTkFrame(self.root, fg_color=COLORS["bg_card"], height=80, corner_radius=0)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(2, weight=1)

        # Score ring
        score_frame = ctk.CTkFrame(top, fg_color="transparent")
        score_frame.grid(row=0, column=0, padx=(20, 12), pady=10)

        self.score_canvas = tk.Canvas(
            score_frame, width=56, height=56,
            bg=COLORS["bg_card"], highlightthickness=0
        )
        self.score_canvas.grid(row=0, column=0)

        # State + stats side by side
        stats_frame = ctk.CTkFrame(top, fg_color="transparent")
        stats_frame.grid(row=0, column=1, padx=0, pady=10, sticky="w")

        self.top_state_label = ctk.CTkLabel(
            stats_frame, text="Neutral",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["neutral"]
        )
        self.top_state_label.grid(row=0, column=0, sticky="w")

        meta_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        meta_frame.grid(row=1, column=0, sticky="w")

        self.avg_label = ctk.CTkLabel(meta_frame, text="avg --",
                                       font=ctk.CTkFont(size=12), text_color=COLORS["text_dim"])
        self.avg_label.grid(row=0, column=0, padx=(0, 16), sticky="w")

        self.streak_label = ctk.CTkLabel(meta_frame, text="streak 0m",
                                          font=ctk.CTkFont(size=12), text_color=COLORS["text_dim"])
        self.streak_label.grid(row=0, column=1, padx=(0, 16), sticky="w")

        self.session_time_label = ctk.CTkLabel(meta_frame, text="0:00",
                                                font=ctk.CTkFont(size=12),
                                                text_color=COLORS["text_dim"])
        self.session_time_label.grid(row=0, column=2, padx=(0, 16), sticky="w")

        self.reading_label = ctk.CTkLabel(meta_frame, text="",
                                           font=ctk.CTkFont(size=12),
                                           text_color=COLORS["accent_light"])
        self.reading_label.grid(row=0, column=3, sticky="w")

        # Goal progress (compact)
        goal_frame = ctk.CTkFrame(top, fg_color="transparent")
        goal_frame.grid(row=0, column=3, padx=20, pady=10, sticky="e")

        self.goal_label = ctk.CTkLabel(goal_frame, text="Goal: --",
                                       font=ctk.CTkFont(size=11),
                                       text_color=COLORS["text_dim"])
        self.goal_label.grid(row=0, column=0, sticky="w")

        self.goal_progress = ctk.CTkProgressBar(goal_frame, width=140, height=4,
                                                progress_color=COLORS["deep_focus"],
                                                fg_color=COLORS["bg_card_alt"])
        self.goal_progress.grid(row=1, column=0, pady=(3, 0), sticky="w")
        self.goal_progress.set(0)

        # Alert banner
        self.alert_frame = ctk.CTkFrame(top, fg_color=COLORS["alert_bg"],
                                         corner_radius=6, height=36)
        self.alert_frame.grid(row=0, column=2, padx=16, pady=20, sticky="ew")
        self.alert_frame.grid_remove()

        self.alert_label = ctk.CTkLabel(
            self.alert_frame, text="",
            font=ctk.CTkFont(size=12),
            text_color="#fca5a5"
        )
        self.alert_label.grid(row=0, column=0, padx=10, pady=4, sticky="w")

        self.alert_dismiss_btn = ctk.CTkButton(
            self.alert_frame, text="Dismiss", width=64, height=24,
            font=ctk.CTkFont(size=11), fg_color="#6b0f0f",
            hover_color="#7f1d1d",
            command=self._dismiss_alert
        )
        self.alert_dismiss_btn.grid(row=0, column=1, padx=6, pady=4)

        # Break banner
        self.break_frame = ctk.CTkFrame(top, fg_color=COLORS["break_bg"],
                                         corner_radius=6, height=36)
        self.break_frame.grid(row=0, column=2, padx=16, pady=20, sticky="ew")
        self.break_frame.grid_remove()

        self.break_label = ctk.CTkLabel(
            self.break_frame, text="",
            font=ctk.CTkFont(size=12),
            text_color="#7dd3fc"
        )
        self.break_label.grid(row=0, column=0, padx=10, pady=4, sticky="w")

        self.break_btn = ctk.CTkButton(
            self.break_frame, text="Take Break", width=80, height=24,
            font=ctk.CTkFont(size=11), fg_color="#0c3d5e",
            hover_color="#0e4f78",
            command=self._take_break
        )
        self.break_btn.grid(row=0, column=1, padx=6, pady=4)

    # ---- LIVE TAB ----

    def _build_live_tab(self):
        tab = self.tab_live
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Left: Camera
        left = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        left.grid(row=0, column=0, padx=(6, 4), pady=6, sticky="nsew")
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        cam_title = ctk.CTkLabel(left, text="Eye Tracking",
                                  font=ctk.CTkFont(size=13, weight="bold"),
                                  text_color=COLORS["text_dim"])
        cam_title.grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        self.camera_label = ctk.CTkLabel(left, text="Starting camera...")
        self.camera_label.grid(row=1, column=0, padx=8, pady=4, sticky="nsew")

        eye_info = ctk.CTkFrame(left, fg_color=COLORS["bg_dark"], corner_radius=6)
        eye_info.grid(row=2, column=0, padx=10, pady=(4, 10), sticky="ew")

        self.eye_info_label = ctk.CTkLabel(
            eye_info, text="Waiting for data...",
            font=ctk.CTkFont(family="Menlo", size=11),
            text_color=COLORS["text_dim"], justify="left", anchor="w"
        )
        self.eye_info_label.grid(row=0, column=0, padx=10, pady=8, sticky="w")

        # Right: Components + Activity
        right = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        right.grid(row=0, column=1, padx=(4, 6), pady=6, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Component bars
        comp_title = ctk.CTkLabel(right, text="Score Breakdown",
                                   font=ctk.CTkFont(size=13, weight="bold"),
                                   text_color=COLORS["text_dim"])
        comp_title.grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")

        comp_frame = ctk.CTkFrame(right, fg_color=COLORS["bg_dark"], corner_radius=6)
        comp_frame.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")
        comp_frame.grid_columnconfigure(1, weight=1)

        self.component_bars = {}
        component_names = [
            ("eye_engagement", "Eye Engagement"),
            ("gaze_stability", "Gaze Stability"),
            ("blink", "Blink Pattern"),
            ("activity", "Activity"),
            ("app_focus", "App Focus"),
        ]
        for i, (key, label) in enumerate(component_names):
            lbl = ctk.CTkLabel(comp_frame, text=label, font=ctk.CTkFont(size=11),
                               text_color=COLORS["text_dim"])
            lbl.grid(row=i, column=0, padx=(12, 8), pady=5, sticky="w")

            bar = ctk.CTkProgressBar(comp_frame, height=6, corner_radius=3,
                                      progress_color=COLORS["accent"],
                                      fg_color=COLORS["bg_card_alt"])
            bar.grid(row=i, column=1, padx=(0, 6), pady=5, sticky="ew")
            bar.set(0.5)

            val = ctk.CTkLabel(comp_frame, text="50",
                               font=ctk.CTkFont(family="Menlo", size=11),
                               text_color=COLORS["text"], width=32)
            val.grid(row=i, column=2, padx=(0, 12), pady=5)

            self.component_bars[key] = (bar, val)

        # Activity info
        act_frame = ctk.CTkFrame(right, fg_color=COLORS["bg_dark"], corner_radius=6)
        act_frame.grid(row=2, column=0, padx=10, pady=6, sticky="nsew")
        act_frame.grid_columnconfigure(0, weight=1)

        act_title = ctk.CTkLabel(act_frame, text="Activity",
                                  font=ctk.CTkFont(size=13, weight="bold"),
                                  text_color=COLORS["text_dim"])
        act_title.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        self.activity_label = ctk.CTkLabel(
            act_frame, text="Waiting...",
            font=ctk.CTkFont(family="Menlo", size=11),
            text_color=COLORS["text_dim"], justify="left", anchor="nw"
        )
        self.activity_label.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="nw")

        # Session summary bar
        summary_frame = ctk.CTkFrame(right, fg_color=COLORS["bg_dark"], corner_radius=6)
        summary_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.summary_label = ctk.CTkLabel(
            summary_frame, text="Session starting...",
            font=ctk.CTkFont(size=11), text_color=COLORS["text_dim"],
            justify="left", anchor="w"
        )
        self.summary_label.grid(row=0, column=0, padx=12, pady=8, sticky="w")

    # ---- GRAPH TAB ----

    def _build_graph_tab(self):
        tab = self.tab_graph
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Controls
        ctrl = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=6, pady=(6, 3), sticky="ew")

        ctk.CTkLabel(ctrl, text="Timeline",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text_dim"]).grid(row=0, column=0, padx=14, pady=8)

        self.graph_range_var = ctk.StringVar(value="5 min")
        graph_range = ctk.CTkSegmentedButton(
            ctrl, values=["1 min", "5 min", "10 min", "30 min"],
            variable=self.graph_range_var,
            font=ctk.CTkFont(size=11),
        )
        graph_range.grid(row=0, column=1, padx=10, pady=8)

        # Canvas for graph
        graph_frame = ctk.CTkFrame(tab, fg_color=COLORS["graph_bg"], corner_radius=8)
        graph_frame.grid(row=1, column=0, padx=6, pady=3, sticky="nsew")
        graph_frame.grid_columnconfigure(0, weight=1)
        graph_frame.grid_rowconfigure(0, weight=1)

        self.graph_canvas = tk.Canvas(
            graph_frame, bg=COLORS["graph_bg"], highlightthickness=0
        )
        self.graph_canvas.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")

        # Legend
        legend = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=8)
        legend.grid(row=2, column=0, padx=6, pady=(3, 6), sticky="ew")

        for i, (state, clr) in enumerate([
            ("Deep Focus", COLORS["deep_focus"]),
            ("Focused", COLORS["focused"]),
            ("Neutral", COLORS["neutral"]),
            ("Distracted", COLORS["distracted"]),
            ("Away", COLORS["away"]),
        ]):
            ctk.CTkLabel(legend, text="─", text_color=clr,
                         font=ctk.CTkFont(size=14)).grid(row=0, column=i * 2, padx=(12, 2), pady=6)
            ctk.CTkLabel(legend, text=state, text_color=COLORS["text_dim"],
                         font=ctk.CTkFont(size=11)).grid(row=0, column=i * 2 + 1, padx=(0, 10), pady=6)

    # ---- HISTORY TAB ----

    def _build_history_tab(self):
        tab = self.tab_history
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=6, pady=(6, 3), sticky="ew")

        ctk.CTkLabel(ctrl, text="History",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text_dim"]).grid(row=0, column=0, padx=14, pady=8)

        export_btn = ctk.CTkButton(ctrl, text="Export CSV", width=110,
                                    height=30, font=ctk.CTkFont(size=12),
                                    command=self._export_csv)
        export_btn.grid(row=0, column=1, padx=8, pady=8)

        refresh_btn = ctk.CTkButton(ctrl, text="Refresh", width=80,
                                     height=30, font=ctk.CTkFont(size=12),
                                     command=self._refresh_history)
        refresh_btn.grid(row=0, column=2, padx=4, pady=8)

        # Scrollable history list
        self.history_scroll = ctk.CTkScrollableFrame(
            tab, fg_color=COLORS["bg_dark"], corner_radius=8
        )
        self.history_scroll.grid(row=1, column=0, padx=6, pady=(3, 6), sticky="nsew")
        self.history_scroll.grid_columnconfigure(0, weight=1)

        self.history_status_label = ctk.CTkLabel(
            self.history_scroll, text="No past sessions yet. Sessions auto-save every minute.",
            font=ctk.CTkFont(size=13), text_color=COLORS["text_dim"]
        )
        self.history_status_label.grid(row=0, column=0, padx=10, pady=20)

    # ---- SETTINGS TAB ----

    def _build_settings_tab(self):
        tab = self.tab_settings
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.settings_scroll = ctk.CTkScrollableFrame(
            tab, fg_color=COLORS["bg_dark"], corner_radius=8
        )
        self.settings_scroll.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.settings_scroll.grid_columnconfigure(0, weight=1)
        tab = self.settings_scroll

        title = ctk.CTkLabel(tab, text="Settings",
                             font=ctk.CTkFont(size=16, weight="bold"),
                             text_color=COLORS["text"])
        title.grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")

        # Alert settings
        alert_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        alert_frame.grid(row=1, column=0, padx=12, pady=5, sticky="ew")
        alert_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(alert_frame, text="Alerts & Reminders",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2,
                                                      padx=14, pady=(12, 6), sticky="w")

        # Sound toggle
        ctk.CTkLabel(alert_frame, text="Alert sounds",
                     text_color=COLORS["text_dim"]).grid(row=1, column=0, padx=12, pady=4, sticky="w")
        self.sound_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(alert_frame, text="", variable=self.sound_var,
                       command=self._update_sound_setting).grid(row=1, column=1, padx=12, pady=4, sticky="w")

        # Distraction threshold
        ctk.CTkLabel(alert_frame, text="Distraction alert after (seconds)",
                     text_color=COLORS["text_dim"]).grid(row=2, column=0, padx=12, pady=4, sticky="w")
        self.distraction_slider = ctk.CTkSlider(alert_frame, from_=10, to=120,
                                                 number_of_steps=11)
        self.distraction_slider.set(30)
        self.distraction_slider.grid(row=2, column=1, padx=12, pady=4, sticky="ew")
        self.distraction_val = ctk.CTkLabel(alert_frame, text="30s",
                                             text_color=COLORS["text"], width=40)
        self.distraction_val.grid(row=2, column=2, padx=(0, 12), pady=4)

        # Break interval
        ctk.CTkLabel(alert_frame, text="Break reminder every (minutes)",
                     text_color=COLORS["text_dim"]).grid(row=3, column=0, padx=12, pady=4, sticky="w")
        self.break_slider = ctk.CTkSlider(alert_frame, from_=10, to=60,
                                           number_of_steps=10)
        self.break_slider.set(25)
        self.break_slider.grid(row=3, column=1, padx=12, pady=4, sticky="ew")
        self.break_val = ctk.CTkLabel(alert_frame, text="25m",
                                       text_color=COLORS["text"], width=40)
        self.break_val.grid(row=3, column=2, padx=(0, 12), pady=(4, 12))

        # Goal settings
        goal_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        goal_frame.grid(row=2, column=0, padx=12, pady=5, sticky="ew")
        goal_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(goal_frame, text="Session Goal",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=3,
                                                      padx=14, pady=(12, 6), sticky="w")

        self.goal_enabled_var = ctk.BooleanVar(value=True)
        ctk.CTkLabel(goal_frame, text="Enable goal",
                     text_color=COLORS["text_dim"]).grid(row=1, column=0, padx=12, pady=4, sticky="w")
        ctk.CTkSwitch(goal_frame, text="", variable=self.goal_enabled_var).grid(
            row=1, column=1, padx=12, pady=4, sticky="w"
        )

        ctk.CTkLabel(goal_frame, text="Focused minutes target",
                     text_color=COLORS["text_dim"]).grid(row=2, column=0, padx=12, pady=4, sticky="w")
        self.goal_slider = ctk.CTkSlider(goal_frame, from_=15, to=180, number_of_steps=33)
        self.goal_slider.set(45)
        self.goal_slider.grid(row=2, column=1, padx=12, pady=4, sticky="ew")
        self.goal_val = ctk.CTkLabel(goal_frame, text="45m",
                                     text_color=COLORS["text"], width=50)
        self.goal_val.grid(row=2, column=2, padx=(0, 12), pady=4)

        ctk.CTkButton(goal_frame, text="Reset Goal Progress", width=150,
                       command=self._reset_goal_progress).grid(
            row=3, column=0, padx=12, pady=(4, 12), sticky="w"
        )

        # Profile settings
        profile_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        profile_frame.grid(row=3, column=0, padx=12, pady=5, sticky="ew")
        profile_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(profile_frame, text="Focus Profiles",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2,
                                                      padx=14, pady=(12, 6), sticky="w")

        ctk.CTkLabel(profile_frame, text="Active profile",
                     text_color=COLORS["text_dim"]).grid(row=1, column=0, padx=12, pady=4, sticky="w")
        self.profile_var = ctk.StringVar(value="Coding")
        self.profile_menu = ctk.CTkOptionMenu(
            profile_frame,
            values=["Study", "Coding", "Writing"],
            variable=self.profile_var,
            command=lambda _: self._on_profile_change(),
        )
        self.profile_menu.grid(row=1, column=1, padx=12, pady=4, sticky="w")

        ctk.CTkLabel(profile_frame, text="Productive apps (comma-separated)",
                     text_color=COLORS["text_dim"]).grid(row=2, column=0, padx=12, pady=(4, 2), sticky="w")
        self.profile_prod_entry = ctk.CTkEntry(profile_frame)
        self.profile_prod_entry.grid(row=2, column=1, padx=12, pady=(4, 2), sticky="ew")

        ctk.CTkLabel(profile_frame, text="Neutral apps (comma-separated)",
                     text_color=COLORS["text_dim"]).grid(row=3, column=0, padx=12, pady=2, sticky="w")
        self.profile_neut_entry = ctk.CTkEntry(profile_frame)
        self.profile_neut_entry.grid(row=3, column=1, padx=12, pady=2, sticky="ew")

        ctk.CTkLabel(profile_frame, text="Distracting apps (comma-separated)",
                     text_color=COLORS["text_dim"]).grid(row=4, column=0, padx=12, pady=2, sticky="w")
        self.profile_dist_entry = ctk.CTkEntry(profile_frame)
        self.profile_dist_entry.grid(row=4, column=1, padx=12, pady=2, sticky="ew")

        ctk.CTkLabel(profile_frame, text="Productive domains",
                     text_color=COLORS["text_dim"]).grid(row=5, column=0, padx=12, pady=2, sticky="w")
        self.profile_prod_domain_entry = ctk.CTkEntry(profile_frame)
        self.profile_prod_domain_entry.grid(row=5, column=1, padx=12, pady=2, sticky="ew")

        ctk.CTkLabel(profile_frame, text="Distracting domains",
                     text_color=COLORS["text_dim"]).grid(row=6, column=0, padx=12, pady=2, sticky="w")
        self.profile_dist_domain_entry = ctk.CTkEntry(profile_frame)
        self.profile_dist_domain_entry.grid(row=6, column=1, padx=12, pady=2, sticky="ew")

        ctk.CTkButton(profile_frame, text="Apply Profile Lists", width=160,
                      command=self._apply_profile_entries).grid(
            row=7, column=0, padx=12, pady=(6, 12), sticky="w"
        )

        # Calibration
        cal_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        cal_frame.grid(row=4, column=0, padx=12, pady=5, sticky="ew")
        cal_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cal_frame, text="Eye Calibration",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2,
                                                      padx=14, pady=(12, 6), sticky="w")

        self.cal_status = ctk.CTkLabel(cal_frame, text="Not calibrated — using defaults",
                                        text_color=COLORS["text_dim"])
        self.cal_status.grid(row=1, column=0, padx=12, pady=4, sticky="w")

        self.cal_btn = ctk.CTkButton(cal_frame, text="Calibrate Now (Quick)",
                                      width=140, command=self._start_calibration)
        self.cal_btn.grid(row=1, column=1, padx=12, pady=(4, 12), sticky="e")

        self.cal_wizard_btn = ctk.CTkButton(
            cal_frame, text="Start Guided Wizard", width=170, command=self._start_guided_calibration
        )
        self.cal_wizard_btn.grid(row=2, column=1, padx=12, pady=(0, 12), sticky="e")

        # Permissions
        perm_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        perm_frame.grid(row=5, column=0, padx=12, pady=5, sticky="ew")
        perm_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(perm_frame, text="Permissions",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=3,
                                                      padx=14, pady=(12, 6), sticky="w")

        ctk.CTkLabel(perm_frame,
                     text="This app needs Camera and Accessibility access to work fully.",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_dim"]).grid(
            row=1, column=0, columnspan=3, padx=12, pady=(0, 6), sticky="w")

        # Camera permission
        self.cam_status = ctk.CTkLabel(perm_frame, text="● Camera",
                                        font=ctk.CTkFont(size=13),
                                        text_color=COLORS["text"])
        self.cam_status.grid(row=2, column=0, padx=12, pady=4, sticky="w")
        self.cam_status_detail = ctk.CTkLabel(perm_frame, text="Checking...",
                                              font=ctk.CTkFont(size=11),
                                              text_color=COLORS["text_dim"])
        self.cam_status_detail.grid(row=2, column=1, padx=4, pady=4, sticky="w")
        ctk.CTkButton(perm_frame, text="Open Camera Settings", width=170,
                       height=28, font=ctk.CTkFont(size=11),
                       command=lambda: subprocess.Popen(
                           ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"]
                       )).grid(row=2, column=2, padx=(4, 12), pady=4, sticky="e")

        # Accessibility permission
        self.acc_status = ctk.CTkLabel(perm_frame, text="● Accessibility",
                                        font=ctk.CTkFont(size=13),
                                        text_color=COLORS["text"])
        self.acc_status.grid(row=3, column=0, padx=12, pady=4, sticky="w")
        self.acc_status_detail = ctk.CTkLabel(perm_frame, text="Checking...",
                                              font=ctk.CTkFont(size=11),
                                              text_color=COLORS["text_dim"])
        self.acc_status_detail.grid(row=3, column=1, padx=4, pady=4, sticky="w")
        ctk.CTkButton(perm_frame, text="Open Accessibility Settings", width=190,
                       height=28, font=ctk.CTkFont(size=11),
                       command=lambda: subprocess.Popen(
                           ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"]
                       )).grid(row=3, column=2, padx=(4, 12), pady=4, sticky="e")

        # Screen Recording permission (needed for active window title)
        self.screen_status = ctk.CTkLabel(perm_frame, text="● Screen Recording",
                                           font=ctk.CTkFont(size=13),
                                           text_color=COLORS["text"])
        self.screen_status.grid(row=4, column=0, padx=12, pady=4, sticky="w")
        self.screen_status_detail = ctk.CTkLabel(perm_frame, text="Checking...",
                                                  font=ctk.CTkFont(size=11),
                                                  text_color=COLORS["text_dim"])
        self.screen_status_detail.grid(row=4, column=1, padx=4, pady=4, sticky="w")
        ctk.CTkButton(perm_frame, text="Open Screen Recording Settings", width=210,
                       height=28, font=ctk.CTkFont(size=11),
                       command=lambda: subprocess.Popen(
                           ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"]
                       )).grid(row=4, column=2, padx=(4, 12), pady=4, sticky="e")

        # Refresh button
        ctk.CTkButton(perm_frame, text="↻ Refresh Status", width=130,
                       height=28, font=ctk.CTkFont(size=11),
                       fg_color=COLORS["bg_card_alt"],
                       command=self._check_permissions).grid(
            row=5, column=0, columnspan=3, padx=12, pady=(4, 12), sticky="w")

        # Run initial check
        self.root.after(500, self._check_permissions)

        # Score weights info
        weights_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        weights_frame.grid(row=6, column=0, padx=12, pady=5, sticky="ew")

        ctk.CTkLabel(weights_frame, text="Score Weights",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")

        weights_text = (
            "Eye Engagement: 20%  |  Gaze Stability: 20%  |  "
            "Blink Pattern: 10%  |  Computer Activity: 25%  |  App Focus: 25%"
        )
        ctk.CTkLabel(weights_frame, text=weights_text,
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_dim"],
                     wraplength=700).grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")

    # ---- ACTIONS ----

    def _apply_config(self):
        """Apply loaded config to UI widgets and managers."""
        cfg = self._config
        self.sound_var.set(cfg.get("sound_enabled", True))
        self.alert_manager.sound_enabled = self.sound_var.get()
        self.distraction_slider.set(cfg.get("distraction_threshold_seconds", 30))
        self.break_slider.set(cfg.get("break_interval_minutes", 25))
        self.goal_enabled_var.set(cfg.get("goal_enabled", True))
        self.goal_slider.set(cfg.get("goal_minutes_target", 45))
        self.alert_manager.fatigue_break_enabled = cfg.get("fatigue_break_enabled", True)
        self.alert_manager._nudge_cooldowns = dict(cfg.get("nudge_cooldowns_sec", {}))

        active_profile = cfg.get("active_profile", "Coding")
        profiles = cfg.get("profiles", {})
        self.activity_monitor.set_profiles(profiles, active_profile=active_profile)
        self.profile_menu.configure(values=list(self.activity_monitor.profiles.keys()))
        self.profile_var.set(self.activity_monitor.active_profile)
        self._populate_profile_entries(self.activity_monitor.active_profile)

        self.focus_engine.set_goal(
            minutes_target=int(cfg.get("goal_minutes_target", 45)),
            enabled=bool(cfg.get("goal_enabled", True)),
        )
        self.eye_tracker.apply_calibration_profile(cfg.get("calibration_profile", {}))
        if self.eye_tracker.calibrated:
            self.cal_status.configure(text="✓ Calibration profile loaded",
                                      text_color=COLORS["deep_focus"])

    def _save_current_config(self):
        """Persist current UI settings to disk."""
        self._config["sound_enabled"] = self.sound_var.get()
        self._config["distraction_threshold_seconds"] = int(self.distraction_slider.get())
        self._config["break_interval_minutes"] = int(self.break_slider.get())
        self._config["goal_enabled"] = self.goal_enabled_var.get()
        self._config["goal_minutes_target"] = int(self.goal_slider.get())
        self._config["fatigue_break_enabled"] = self.alert_manager.fatigue_break_enabled
        self._config["nudge_cooldowns_sec"] = dict(self.alert_manager._nudge_cooldowns)
        self._config["active_profile"] = self.activity_monitor.active_profile
        self._config["profiles"] = self._profiles_to_config_dict()
        self._config["calibration_profile"] = dict(self.eye_tracker.calibration_profile or {})
        save_config(self._config)

    def _dismiss_alert(self):
        self.alert_frame.grid_remove()

    def _take_break(self):
        self.alert_manager.acknowledge_break()
        self.break_frame.grid_remove()

    def _update_sound_setting(self):
        self.alert_manager.sound_enabled = self.sound_var.get()

    def _reset_goal_progress(self):
        self.focus_engine.reset_goal_progress()

    def _profiles_to_config_dict(self) -> dict:
        out = {}
        for name, prof in self.activity_monitor.profiles.items():
            out[name] = {
                "productive_apps": sorted(prof.get("productive_apps", [])),
                "neutral_apps": sorted(prof.get("neutral_apps", [])),
                "distracting_apps": sorted(prof.get("distracting_apps", [])),
                "productive_domains": sorted(prof.get("productive_domains", [])),
                "distracting_domains": sorted(prof.get("distracting_domains", [])),
            }
        return out

    @staticmethod
    def _parse_csv_list(text: str) -> list[str]:
        return [x.strip().lower() for x in text.split(",") if x.strip()]

    def _populate_profile_entries(self, profile_name: str) -> None:
        profile = self.activity_monitor.profiles.get(profile_name, {})
        self.profile_prod_entry.delete(0, "end")
        self.profile_prod_entry.insert(0, ", ".join(sorted(profile.get("productive_apps", []))))
        self.profile_neut_entry.delete(0, "end")
        self.profile_neut_entry.insert(0, ", ".join(sorted(profile.get("neutral_apps", []))))
        self.profile_dist_entry.delete(0, "end")
        self.profile_dist_entry.insert(0, ", ".join(sorted(profile.get("distracting_apps", []))))
        self.profile_prod_domain_entry.delete(0, "end")
        self.profile_prod_domain_entry.insert(0, ", ".join(sorted(profile.get("productive_domains", []))))
        self.profile_dist_domain_entry.delete(0, "end")
        self.profile_dist_domain_entry.insert(0, ", ".join(sorted(profile.get("distracting_domains", []))))

    def _on_profile_change(self):
        name = self.profile_var.get()
        self.activity_monitor.set_active_profile(name)
        self._populate_profile_entries(name)

    def _apply_profile_entries(self):
        name = self.profile_var.get()
        profiles = self.activity_monitor.profiles
        profiles[name] = {
            "productive_apps": set(self._parse_csv_list(self.profile_prod_entry.get())),
            "neutral_apps": set(self._parse_csv_list(self.profile_neut_entry.get())),
            "distracting_apps": set(self._parse_csv_list(self.profile_dist_entry.get())),
            "productive_domains": set(self._parse_csv_list(self.profile_prod_domain_entry.get())),
            "distracting_domains": set(self._parse_csv_list(self.profile_dist_domain_entry.get())),
        }
        self.activity_monitor.set_profiles(profiles, active_profile=name)
        self._save_current_config()

    def _check_permissions(self):
        """Check macOS permission status and update indicators."""
        act = self.activity_monitor.latest_metrics
        running_time = time.time() - self.session_manager.session_start

        # Camera
        if not self.camera_available:
            self.cam_status.configure(text="● Camera", text_color=COLORS["distracted"])
            self.cam_status_detail.configure(
                text="Unavailable — check connection or permission",
                text_color=COLORS["distracted"],
            )
        elif self.eye_tracker.annotated_frame is not None:
            self.cam_status.configure(text="● Camera", text_color=COLORS["deep_focus"])
            self.cam_status_detail.configure(text="Granted ✓", text_color=COLORS["deep_focus"])
        else:
            self.cam_status.configure(text="● Camera", text_color=COLORS["neutral"])
            self.cam_status_detail.configure(text="No frames yet — may need permission",
                                             text_color=COLORS["neutral"])

        # Accessibility — use AXIsProcessTrusted() for reliable detection
        try:
            from ApplicationServices import AXIsProcessTrusted
            ax_trusted = AXIsProcessTrusted()
        except ImportError:
            ax_trusted = None

        if ax_trusted is True:
            self.acc_status.configure(text="● Accessibility", text_color=COLORS["deep_focus"])
            self.acc_status_detail.configure(text="Granted ✓", text_color=COLORS["deep_focus"])
        elif ax_trusted is False:
            self.acc_status.configure(text="● Accessibility", text_color=COLORS["distracted"])
            self.acc_status_detail.configure(
                text="Not granted — keyboard/mouse tracking disabled",
                text_color=COLORS["distracted"],
            )
        else:
            # Fallback: heuristic check
            if running_time > 5 and act.keys_per_minute == 0 and act.mouse_moves_per_minute == 0:
                self.acc_status.configure(text="● Accessibility", text_color=COLORS["distracted"])
                self.acc_status_detail.configure(
                    text="Not granted — keyboard/mouse tracking disabled",
                    text_color=COLORS["distracted"],
                )
            elif running_time > 5:
                self.acc_status.configure(text="● Accessibility", text_color=COLORS["deep_focus"])
                self.acc_status_detail.configure(text="Granted ✓", text_color=COLORS["deep_focus"])
            else:
                self.acc_status.configure(text="● Accessibility", text_color=COLORS["neutral"])
                self.acc_status_detail.configure(text="Waiting for data...",
                                                 text_color=COLORS["text_dim"])

        # Screen Recording — check if we can read window titles
        if running_time > 5 and act.active_window_title and act.active_window_title != "Unknown":
            self.screen_status.configure(text="● Screen Recording", text_color=COLORS["deep_focus"])
            self.screen_status_detail.configure(text="Granted ✓", text_color=COLORS["deep_focus"])
        elif running_time > 5:
            self.screen_status.configure(text="● Screen Recording", text_color=COLORS["distracted"])
            self.screen_status_detail.configure(
                text="Not granted — window titles unavailable",
                text_color=COLORS["distracted"]
            )
        else:
            self.screen_status.configure(text="● Screen Recording", text_color=COLORS["neutral"])
            self.screen_status_detail.configure(text="Waiting for data...",
                                                 text_color=COLORS["text_dim"])

    def _start_calibration(self):
        if self._calibrating:
            return
        self._calibrating = True
        self.cal_btn.configure(state="disabled", text="Calibrating...")
        self.cal_status.configure(text="Look at the screen normally...")

        def do_calibrate():
            baseline = self.eye_tracker.calibrate(duration=3.0)
            self._calibrating = False
            self.root.after(0, lambda: self._calibration_done(baseline))

        threading.Thread(target=do_calibrate, daemon=True).start()

    def _calibration_done(self, baseline: float):
        self.cal_btn.configure(state="normal", text="Recalibrate (3s)")
        if self.eye_tracker.calibrated:
            self.cal_status.configure(
                text=f"✓ Calibrated — baseline EAR: {baseline:.3f}  "
                     f"(blink threshold: {self.eye_tracker.EAR_BLINK_THRESHOLD:.3f})",
                text_color=COLORS["deep_focus"]
            )
        else:
            self.cal_status.configure(
                text="✗ Calibration failed — no face detected. Try again.",
                text_color=COLORS["distracted"]
            )

    def _start_guided_calibration(self):
        if self._calibrating:
            return
        self._calibrating = True
        self.cal_btn.configure(state="disabled")
        self.cal_wizard_btn.configure(state="disabled", text="Wizard Running...")
        self.cal_status.configure(
            text="Guided calibration started: Neutral (30s) → Reading (30s) → Away (20s)"
        )
        self._calibration_stage_results = {}

        def run_wizard():
            try:
                self.eye_tracker.start_calibration_session()
                stages = [("neutral", 30.0), ("reading", 30.0), ("distracted", 20.0)]
                for stage, dur in stages:
                    self._calibration_stage_results[stage] = self.eye_tracker.collect_calibration_sample(
                        stage, duration_s=dur
                    )
                profile = self.eye_tracker.finalize_calibration()
                self.eye_tracker.apply_calibration_profile(profile)
                self._config["calibration_profile"] = profile
                save_config(self._config)
                baseline = self.eye_tracker.baseline_ear
                self.root.after(0, lambda: self._calibration_done(baseline))
            finally:
                self._calibrating = False
                self.root.after(0, lambda: self.cal_btn.configure(state="normal"))
                self.root.after(0, lambda: self.cal_wizard_btn.configure(
                    state="normal", text="Start Guided Wizard"
                ))

        threading.Thread(target=run_wizard, daemon=True).start()

    def _export_csv(self):
        snapshots = list(self.focus_engine.history)
        if not snapshots:
            return
        path = self.session_manager.export_csv(snapshots)
        # Show brief confirmation
        self.history_status_label.configure(text=f"✓ Exported to: {path}")

    def _refresh_history(self):
        # Clear existing items
        for widget in self.history_scroll.winfo_children():
            widget.destroy()

        sessions = self.session_manager.list_past_sessions()
        if not sessions:
            lbl = ctk.CTkLabel(self.history_scroll,
                               text="No past sessions yet.",
                               font=ctk.CTkFont(size=13), text_color=COLORS["text_dim"])
            lbl.grid(row=0, column=0, padx=10, pady=20)
            return

        for i, s in enumerate(sessions[:20]):  # show last 20
            row = ctk.CTkFrame(self.history_scroll, fg_color=COLORS["bg_card"],
                               corner_radius=6, height=40)
            row.grid(row=i, column=0, padx=4, pady=2, sticky="ew")
            row.grid_columnconfigure(1, weight=1)

            score = s.get("avg_score", 0)
            score_color = COLORS["deep_focus"] if score >= 70 else (
                COLORS["neutral"] if score >= 50 else COLORS["distracted"]
            )
            ctk.CTkLabel(row, text=f"{score:.0f}",
                         font=ctk.CTkFont(size=18, weight="bold"),
                         text_color=score_color, width=50).grid(row=0, column=0, padx=8, pady=6)

            date_str = s.get("session_id", "Unknown")
            dur = s.get("duration_minutes", 0)
            ctk.CTkLabel(row, text=f"{date_str}  •  {dur:.0f} min",
                         font=ctk.CTkFont(size=12),
                         text_color=COLORS["text"]).grid(row=0, column=1, padx=4, pady=6, sticky="w")

    def _refresh_analytics(self):
        for widget in self.analytics_scroll.winfo_children():
            widget.destroy()

        hourly = self.session_manager.aggregate_hourly_focus(max_sessions=40)
        impact = self.session_manager.aggregate_app_impact(max_sessions=40, top_n=8)
        windows = self.session_manager.detect_distraction_windows(max_sessions=40)

        if not hourly and not impact.get("top_apps") and not windows:
            ctk.CTkLabel(
                self.analytics_scroll,
                text="Not enough historical data yet. Keep the app running and save sessions.",
                font=ctk.CTkFont(size=13),
                text_color=COLORS["text_dim"],
            ).grid(row=0, column=0, padx=10, pady=20, sticky="w")
            return

        row = 0
        ctk.CTkLabel(self.analytics_scroll, text="Focus By Hour",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).grid(row=row, column=0, padx=10, pady=(8, 3), sticky="w")
        row += 1
        for item in hourly[:24]:
            heat = "█" * max(1, min(10, int(item["avg_score"] // 10)))
            ctk.CTkLabel(
                self.analytics_scroll,
                text=f"{item['hour']:02d}:00  {heat}  avg={item['avg_score']:.1f}  n={item['samples']}",
                font=ctk.CTkFont(family="Menlo", size=11),
                text_color=COLORS["text_dim"],
            ).grid(row=row, column=0, padx=12, pady=1, sticky="w")
            row += 1

        row += 1
        ctk.CTkLabel(self.analytics_scroll, text="App/Profile Impact",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).grid(row=row, column=0, padx=10, pady=(8, 3), sticky="w")
        row += 1
        for app in impact.get("top_apps", [])[:8]:
            ctk.CTkLabel(
                self.analytics_scroll,
                text=f"{app['key'][:38]:<38}  avg={app['avg_score']:.1f}  n={app['samples']}",
                font=ctk.CTkFont(family="Menlo", size=11),
                text_color=COLORS["text_dim"],
            ).grid(row=row, column=0, padx=12, pady=1, sticky="w")
            row += 1

        row += 1
        ctk.CTkLabel(self.analytics_scroll, text="Top Distraction Windows",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).grid(row=row, column=0, padx=10, pady=(8, 3), sticky="w")
        row += 1
        for win in windows[:8]:
            reasons = ", ".join(win.get("nudge_reasons", [])) or "low-score cluster"
            ctk.CTkLabel(
                self.analytics_scroll,
                text=f"{win['session_id']}  {win['duration_sec']:.0f}s  avg={win['avg_score']:.1f}  ({reasons})",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_dim"],
                wraplength=900,
                justify="left",
            ).grid(row=row, column=0, padx=12, pady=1, sticky="w")
            row += 1

    def _build_analytics_tab(self):
        tab = self.tab_analytics
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=6, pady=(6, 3), sticky="ew")
        ctk.CTkLabel(ctrl, text="Analytics",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text_dim"]).grid(row=0, column=0, padx=14, pady=8, sticky="w")
        ctk.CTkButton(ctrl, text="Refresh", width=90, height=30,
                      font=ctk.CTkFont(size=12),
                      command=self._refresh_analytics).grid(row=0, column=1, padx=10, pady=8)

        self.analytics_scroll = ctk.CTkScrollableFrame(
            tab, fg_color=COLORS["bg_dark"], corner_radius=8
        )
        self.analytics_scroll.grid(row=1, column=0, padx=5, pady=(2, 5), sticky="nsew")
        self.analytics_scroll.grid_columnconfigure(0, weight=1)

        self.analytics_status = ctk.CTkLabel(
            self.analytics_scroll,
            text="Analytics will appear after enough session data is collected.",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_dim"],
        )
        self.analytics_status.grid(row=0, column=0, padx=10, pady=20, sticky="w")

    # ---- SCORE RING ----

    def _draw_score_ring(self, score: float, color: str):
        """Draw a circular progress ring with score text."""
        c = self.score_canvas
        c.delete("all")
        cx, cy, r = 28, 28, 22
        width = 5

        # Track ring background
        c.create_oval(cx - r, cy - r, cx + r, cy + r,
                       outline=COLORS["bg_card_alt"], width=width + 1)

        # Main arc
        extent = -3.6 * score
        if abs(extent) > 1:
            c.create_arc(cx - r, cy - r, cx + r, cy + r,
                         start=90, extent=extent,
                         outline=color, width=width, style="arc")

        # Trend arrow
        trend = self._get_trend()
        arrow = "↑" if trend > 2 else "↓" if trend < -2 else ""
        arrow_color = COLORS["deep_focus"] if trend > 2 else COLORS["distracted"] if trend < -2 else color

        # Score text
        c.create_text(cx, cy, text=f"{score:.0f}",
                       fill=color, font=("Helvetica", 16, "bold"))
        if arrow:
            c.create_text(cx + 14, cy - 10, text=arrow,
                           fill=arrow_color, font=("Helvetica", 10, "bold"))

    def _get_trend(self) -> float:
        """Return score trend: positive = improving, negative = declining."""
        history = self.focus_engine.history
        if len(history) < 30:
            return 0.0
        recent = [s.focus_score for s in list(history)[-15:]]
        older = [s.focus_score for s in list(history)[-30:-15]]
        return (sum(recent) / len(recent)) - (sum(older) / len(older))

    # ---- GRAPH DRAWING ----

    def _draw_graph(self):
        """Draw the focus timeline graph on canvas."""
        canvas = self.graph_canvas
        canvas.delete("all")

        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 50 or h < 50:
            return

        # Determine time range
        range_str = self.graph_range_var.get()
        range_map = {"1 min": 60, "5 min": 300, "10 min": 600, "30 min": 1800}
        time_range = range_map.get(range_str, 300)

        # Get data
        points = self.focus_engine.get_history_points(time_range)
        if len(points) < 2:
            canvas.create_text(w // 2, h // 2, text="Collecting data...",
                               fill=COLORS["text_dim"], font=("Menlo", 14))
            return

        margin_l, margin_r, margin_t, margin_b = 45, 15, 15, 30
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b

        # Grid lines + labels
        for score_val in [0, 20, 40, 60, 80, 100]:
            y = margin_t + plot_h * (1 - score_val / 100)
            canvas.create_line(margin_l, y, w - margin_r, y,
                               fill=COLORS["graph_grid"], width=1)
            canvas.create_text(margin_l - 5, y, text=str(score_val),
                               fill=COLORS["text_dim"], font=("Menlo", 9), anchor="e")

        # Zone fills
        zones = [
            (80, 100, COLORS["deep_focus"]),
            (60, 80, COLORS["focused"]),
            (40, 60, COLORS["neutral"]),
            (0, 40, COLORS["distracted"]),
        ]
        for lo, hi, color in zones:
            y_top = margin_t + plot_h * (1 - hi / 100)
            y_bot = margin_t + plot_h * (1 - lo / 100)
            canvas.create_rectangle(margin_l, y_top, w - margin_r, y_bot,
                                     fill=color, stipple="gray12", outline="")

        # Time labels
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            x = margin_l + plot_w * frac
            t = -time_range * (1 - frac)
            if abs(t) >= 60:
                label = f"{int(t // 60)}m"
            else:
                label = f"{int(t)}s"
            canvas.create_text(x, h - 8, text=label,
                               fill=COLORS["text_dim"], font=("Menlo", 9))

        # Plot line
        if points:
            min_t = points[0][0]
            max_t = points[-1][0]
            t_span = max_t - min_t if max_t != min_t else 1

            coords = []
            for t, score in points:
                x = margin_l + ((t - min_t) / t_span) * plot_w
                y = margin_t + plot_h * (1 - score / 100)
                coords.append((x, y))

            # Gradient fill under curve (layered strips for depth)
            if len(coords) >= 2:
                bottom = margin_t + plot_h
                fill_coords = [(coords[0][0], bottom)]
                fill_coords.extend(coords)
                fill_coords.append((coords[-1][0], bottom))
                flat = [c for pt in fill_coords for c in pt]
                canvas.create_polygon(flat, fill=COLORS["accent"], stipple="gray25", outline="")
                # Second lighter layer (upper half only)
                mid_y = margin_t + plot_h * 0.5
                upper_coords = [(x, max(y, mid_y)) for x, y in coords]
                upper_fill = [(upper_coords[0][0], bottom)]
                upper_fill.extend(upper_coords)
                upper_fill.append((upper_coords[-1][0], bottom))
                flat_upper = [c for pt in upper_fill for c in pt]
                canvas.create_polygon(flat_upper, fill=COLORS["accent"], stipple="gray50", outline="")

            # Line with color-coded segments
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                score = points[i][1]
                color = (COLORS["deep_focus"] if score >= 80 else
                         COLORS["focused"] if score >= 60 else
                         COLORS["neutral"] if score >= 40 else
                         COLORS["distracted"])
                canvas.create_line(x1, y1, x2, y2, fill=color, width=2.5, smooth=True)

            # Current score dot with glow
            if coords:
                lx, ly = coords[-1]
                cur_color = state_color(self._latest_snapshot.state)
                canvas.create_oval(lx - 8, ly - 8, lx + 8, ly + 8,
                                    fill="", outline=cur_color, width=1)
                canvas.create_oval(lx - 5, ly - 5, lx + 5, ly + 5,
                                    fill=cur_color, outline="white", width=2)

                # Score label next to dot
                cur_score = self._latest_snapshot.focus_score
                label_y = ly - 14 if ly > margin_t + 20 else ly + 14
                canvas.create_text(lx, label_y, text=f"{cur_score:.0f}",
                                    fill="white", font=("Menlo", 10, "bold"))

    # ---- MAIN LOOP ----

    def start(self):
        """Begin processing and show window."""
        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        self._refresh_analytics()
        self.root.after(self.UPDATE_INTERVAL_MS, self._update_ui)
        self.root.mainloop()

    def _on_close(self):
        """Save session and settings, then close."""
        self._running = False
        # Save settings
        self._save_current_config()
        # Final save
        snapshots = list(self.focus_engine.history)
        if snapshots:
            summary = self.focus_engine.get_session_summary()
            self.session_manager.save_session(snapshots, summary)
        self.root.destroy()

    def _process_loop(self):
        consecutive_errors = 0
        while self._running:
            try:
                if self._calibrating:
                    time.sleep(0.1)
                    continue

                if self.camera_available:
                    eye_metrics = self.eye_tracker.process_frame()
                else:
                    eye_metrics = EyeMetrics()

                activity_metrics = self.activity_monitor.get_metrics()
                snapshot = self.focus_engine.calculate(eye_metrics, activity_metrics)
                self._latest_snapshot = snapshot

                # Update alerts
                goal_progress = self.focus_engine.get_goal_progress()
                self._alert_state = self.alert_manager.update(
                    snapshot,
                    eye_metrics=eye_metrics,
                    activity_metrics=activity_metrics,
                    goal_progress=goal_progress,
                )
                snapshot.nudge_reason = self._alert_state.nudge_type or ""

                # Update settings sliders → alert manager
                self.alert_manager.distraction_threshold = int(self.distraction_slider.get())
                self.alert_manager.break_interval = int(self.break_slider.get()) * 60
                self.focus_engine.set_goal(
                    minutes_target=int(self.goal_slider.get()),
                    enabled=bool(self.goal_enabled_var.get()),
                )
                self.activity_monitor.set_active_profile(self.profile_var.get())

                # Autosave
                if self.session_manager.should_autosave():
                    snapshots = list(self.focus_engine.history)
                    summary = self.focus_engine.get_session_summary()
                    self.session_manager.save_session(snapshots, summary)

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                log.error("Process loop error (#%d): %s", consecutive_errors, e)
                if consecutive_errors >= 10 and self.camera_available:
                    # Camera may have disconnected — try to reconnect
                    log.warning("Too many errors, attempting webcam reconnect...")
                    try:
                        self.eye_tracker.stop()
                        time.sleep(2)
                        self.eye_tracker.start()
                        self.camera_available = True
                        consecutive_errors = 0
                        log.info("Webcam reconnected successfully")
                    except Exception:
                        log.warning("Webcam reconnect failed, switching to activity-only mode")
                        self.camera_available = False
                        consecutive_errors = 0
            time.sleep(0.05)

    def _update_ui(self):
        if not self._running:
            return

        try:
            self._do_update_ui()
        except Exception as e:
            log.error("UI update error: %s", e)

        self.root.after(self.UPDATE_INTERVAL_MS, self._update_ui)

    def _do_update_ui(self):
        snap = self._latest_snapshot
        eye = self.eye_tracker.latest_metrics
        act = self.activity_monitor.latest_metrics
        alert = self._alert_state

        # --- Top bar ---
        color = state_color(snap.state)
        self._draw_score_ring(snap.focus_score, color)
        self.top_state_label.configure(text=snap.state, text_color=color)

        avg_5 = self.focus_engine.get_average_score(300)
        self.avg_label.configure(text=f"avg {avg_5:.0f}")

        streak = alert.current_streak_minutes
        best = alert.best_streak_minutes
        self.streak_label.configure(text=f"streak {streak:.0f}m")

        goal = self.focus_engine.get_goal_progress()
        if goal["enabled"]:
            self.goal_label.configure(
                text=f"Goal  {goal['focused_minutes']:.0f} / {goal['target_minutes']} min"
            )
            self.goal_progress.set(max(0.0, min(1.0, goal["progress_pct"] / 100)))
        else:
            self.goal_label.configure(text="Goal: off")
            self.goal_progress.set(0.0)

        elapsed = time.time() - self.session_manager.session_start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.session_time_label.configure(text=f"{mins}:{secs:02d}")

        # Reading mode indicator in top bar
        if eye.is_reading and eye.reading_confidence > 0.3:
            self.reading_label.configure(text=f"reading {eye.reading_confidence:.0%}")
        else:
            self.reading_label.configure(text="")

        # Settings slider readouts
        self.distraction_val.configure(text=f"{int(self.distraction_slider.get())}s")
        self.break_val.configure(text=f"{int(self.break_slider.get())}m")
        self.goal_val.configure(text=f"{int(self.goal_slider.get())}m")

        # --- Alerts ---
        if alert.nudge_active and alert.nudge_message:
            self.alert_label.configure(text=alert.nudge_message)
            self.alert_frame.grid()
        elif alert.distraction_alert_active:
            self.alert_label.configure(text=alert.distraction_alert_message)
            self.alert_frame.grid()
        else:
            self.alert_frame.grid_remove()

        if alert.break_reminder_active:
            self.break_label.configure(text=alert.break_reminder_message)
            self.break_frame.grid()
        else:
            self.break_frame.grid_remove()

        # --- Camera ---
        if not self.camera_available:
            self.camera_label.configure(
                text="📷 Camera unavailable — activity-only mode",
                image=None,
            )
        elif (frame := self.eye_tracker.annotated_frame) is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((480, 360), Image.LANCZOS)
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(480, 360))
            self.camera_label.configure(image=ctk_image, text="")
            self.camera_label._ctk_image = ctk_image

        # --- Eye info ---
        if eye.face_detected:
            cal_str = "✓ Calibrated" if self.eye_tracker.calibrated else "Default"
            screen_str = "👁 Screen" if eye.looking_at_screen else "👀 Away"
            reading_str = f"  📖 Reading ({eye.reading_confidence:.0%})" if eye.is_reading else ""
            eye_text = (
                f"EAR: {eye.avg_ear:.3f}  ({cal_str})  |  Blinks/min: {eye.blinks_per_minute:.0f}\n"
                f"Gaze: H={eye.gaze_horizontal:+.2f}  V={eye.gaze_vertical:+.2f}  |  "
                f"Head: Yaw={eye.head_yaw:+.2f}  Pitch={eye.head_pitch:+.2f}\n"
                f"Attention: {eye.attention_h:+.2f},{eye.attention_v:+.2f}  {screen_str}{reading_str}"
            )
            if eye.eyes_closed_duration > 0.5:
                eye_text += f"\n⚠ Eyes closed: {eye.eyes_closed_duration:.1f}s"
        else:
            eye_text = "⚠ No face detected"
        self.eye_info_label.configure(text=eye_text)

        # --- Components ---
        scores = {
            "eye_engagement": snap.eye_engagement_score,
            "gaze_stability": snap.gaze_stability_score,
            "blink": snap.blink_score,
            "activity": snap.activity_score,
            "app_focus": snap.app_focus_score,
        }
        for key, (bar, val_lbl) in self.component_bars.items():
            score = scores.get(key, 50)
            bar.set(score / 100.0)
            val_lbl.configure(text=f"{score:.0f}")
            if score >= 70:
                bar.configure(progress_color=COLORS["deep_focus"])
            elif score >= 50:
                bar.configure(progress_color=COLORS["neutral"])
            else:
                bar.configure(progress_color=COLORS["distracted"])

        # --- Activity ---
        class_icon = {"productive": "✅", "neutral": "➖", "distracting": "❌"}.get(
            getattr(act, "app_classification", "neutral"), "➖"
        )
        act_text = (
            f"Profile: {getattr(act, 'profile_name', 'Coding')}  |  "
            f"App: {act.active_app}  {class_icon} ({getattr(act, 'app_classification', 'neutral')})\n"
            f"Window: {act.active_window_title[:55]}\n"
            f"Domain: {getattr(act, 'active_domain', '') or '--'}\n"
            f"Keys/min: {act.keys_per_minute:.0f}  |  Mouse/min: {act.mouse_moves_per_minute:.0f}\n"
            f"Idle: {act.total_idle_seconds:.0f}s  |  Switches/min: {act.app_switches_per_minute:.0f}"
        )
        self.activity_label.configure(text=act_text)

        # --- Session ---
        summary = self.focus_engine.get_session_summary()
        if summary["total_readings"] > 10:
            self.summary_label.configure(
                text=(
                    f"Avg: {summary['avg_score']:.0f}  |  "
                    f"Focused: {summary['time_focused_pct']:.0f}%  |  "
                    f"Distracted: {summary['time_distracted_pct']:.0f}%  |  "
                    f"Away: {summary.get('time_away_pct', 0):.0f}%"
                )
            )

        # --- Graph (throttled to 1/sec) ---
        now = time.time()
        if now - self._last_graph_draw >= 1.0:
            self._draw_graph()
            self._last_graph_draw = now
