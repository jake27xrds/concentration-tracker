"""
Eye Tracking Module
Uses MediaPipe FaceLandmarker (Tasks API) to track eye movements, blink rate,
and gaze direction via the webcam in real time.
"""

import time
import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
import numpy as np

from focus_tracker.model_downloader import ensure_model


@dataclass
class EyeMetrics:
    """Snapshot of eye-tracking data at a moment in time."""
    timestamp: float = 0.0
    # Eye aspect ratios (low = closed, high = open)
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    # Gaze direction (-1 to 1 range, 0 = center)
    gaze_horizontal: float = 0.0
    gaze_vertical: float = 0.0
    # Blink detection
    is_blinking: bool = False
    blinks_per_minute: float = 0.0
    # Eye closure (prolonged closure = drowsiness / distraction)
    eyes_closed_duration: float = 0.0
    # Whether a face is detected at all
    face_detected: bool = False
    # Head pose (looking away = distraction)
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    # Combined attention direction (fuses head pose + eye gaze, -1 to 1)
    attention_h: float = 0.0
    attention_v: float = 0.0
    # Whether the combined signal indicates looking at the screen
    looking_at_screen: bool = False
    # Confidence that head is facing camera (0-1, higher = more frontal)
    head_frontal_confidence: float = 0.0
    # Reading detection
    is_reading: bool = False
    reading_confidence: float = 0.0


# MediaPipe Face Mesh landmark indices for eyes
# Left eye — specific vertical pairs for EAR calculation
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EAR_PAIRS = [(159, 145), (160, 144), (161, 163)]  # upper, lower pairs
LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LEFT_IRIS = [469, 470, 471, 472]

# Right eye — specific vertical pairs for EAR calculation
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EAR_PAIRS = [(386, 374), (387, 373), (388, 390)]  # upper, lower pairs
RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_IRIS = [474, 475, 476, 477]

# Nose tip and face contour for head pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

# Face oval contour landmarks (for drawing face outline)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]

# Nose bridge for face tracking visualization
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]


class ReadingDetector:
    """
    Detects reading behaviour from gaze patterns.

    Reading produces distinctive eye movements:
    - Small horizontal saccades (jumps) progressing left-to-right
    - Brief fixation pauses between saccades (~200-400ms)
    - Low vertical drift within a line
    - Periodic line-return sweeps (large right-to-left jump)

    We analyse a rolling window of raw gaze samples and score how
    closely the pattern matches reading.
    """

    def __init__(self):
        # Rolling window of (timestamp, gaze_h, gaze_v)
        self._gaze_samples: deque = deque(maxlen=90)  # ~3 seconds at 30fps
        self._reading_confidence = 0.0
        # Smoothed output (reading state shouldn't flicker)
        self._confidence_buffer: deque = deque(maxlen=15)
        # Saccade detection
        self._last_h = 0.0
        self._saccade_directions: deque = deque(maxlen=40)  # +1=right, -1=left
        self._fixation_count = 0
        self._fixation_start: float | None = None
        self._fixation_durations: deque = deque(maxlen=30)
        self._FIXATION_THRESHOLD = 0.015  # gaze movement below this = fixation
        self._SACCADE_THRESHOLD = 0.03    # gaze jump above this = saccade

    def update(self, timestamp: float, gaze_h: float, gaze_v: float,
               looking_at_screen: bool) -> tuple[bool, float]:
        """Feed a new gaze sample. Returns (is_reading, confidence 0-1)."""
        self._gaze_samples.append((timestamp, gaze_h, gaze_v))

        if not looking_at_screen or len(self._gaze_samples) < 15:
            self._confidence_buffer.append(0.0)
            return self._smoothed_result()

        # Detect saccades vs fixations from frame-to-frame horizontal delta
        dh = gaze_h - self._last_h
        self._last_h = gaze_h
        abs_dh = abs(dh)

        if abs_dh > self._SACCADE_THRESHOLD:
            # Saccade detected
            direction = 1.0 if dh > 0 else -1.0
            self._saccade_directions.append(direction)
            self._fixation_start = None
        elif abs_dh < self._FIXATION_THRESHOLD:
            # Fixation (eyes relatively still)
            if self._fixation_start is None:
                self._fixation_start = timestamp
            dur = timestamp - self._fixation_start
            if dur > 0.15:  # fixation must last at least 150ms
                if len(self._fixation_durations) == 0 or self._fixation_durations[-1] != dur:
                    self._fixation_durations.append(dur)

        # Score the pattern
        score = self._score_pattern()
        self._confidence_buffer.append(score)
        return self._smoothed_result()

    def _score_pattern(self) -> float:
        """Score how reading-like the recent gaze pattern is (0-1)."""
        score = 0.0

        # 1. Saccade directionality: reading = mostly forward (right) saccades
        #    with occasional line-return (left) sweeps
        if len(self._saccade_directions) >= 5:
            rights = sum(1 for d in self._saccade_directions if d > 0)
            total = len(self._saccade_directions)
            forward_ratio = rights / total
            # Reading typically has 70-85% forward saccades
            if 0.55 <= forward_ratio <= 0.92:
                score += 0.35
            elif 0.45 <= forward_ratio <= 0.95:
                score += 0.15

        # 2. Fixation count: reading produces regular fixations
        recent_fixations = len(self._fixation_durations)
        if recent_fixations >= 3:
            score += min(0.25, recent_fixations * 0.04)

        # 3. Fixation duration regularity: reading fixations are ~200-400ms
        if len(self._fixation_durations) >= 3:
            durations = list(self._fixation_durations)
            avg_dur = sum(durations) / len(durations)
            if 0.15 <= avg_dur <= 0.6:
                score += 0.20
            elif 0.10 <= avg_dur <= 1.0:
                score += 0.08

        # 4. Low vertical drift: reading stays on ~same vertical line
        if len(self._gaze_samples) >= 15:
            recent = list(self._gaze_samples)[-15:]
            v_vals = [s[2] for s in recent]
            v_range = max(v_vals) - min(v_vals)
            if v_range < 0.08:
                score += 0.20
            elif v_range < 0.15:
                score += 0.10

        return min(1.0, score)

    def _smoothed_result(self) -> tuple[bool, float]:
        if not self._confidence_buffer:
            return False, 0.0
        avg = sum(self._confidence_buffer) / len(self._confidence_buffer)
        is_reading = avg >= 0.35
        return is_reading, round(avg, 3)


class EyeTracker:
    """Tracks eyes using webcam + MediaPipe and produces focus-related metrics."""

    # Default thresholds (will be adjusted by calibration)
    EAR_BLINK_THRESHOLD = 0.22
    EAR_CLOSED_THRESHOLD = 0.18
    BLINK_CONSEC_FRAMES = 1  # single frame dip counts (blinks are fast)

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.landmarker: FaceLandmarker | None = None

        # Blink tracking — uses raw (unsmoothed) EAR
        self._blink_counter = 0
        self._blink_total = 0
        self._blink_timestamps: deque = deque(maxlen=200)
        self._was_below_threshold = False  # for edge-based blink detection

        # Eye closure tracking
        self._eyes_closed_start: float | None = None

        # Smoothing buffers (small — 3 frames keeps responsiveness)
        self._ear_buffer: deque = deque(maxlen=3)
        self._gaze_h_buffer: deque = deque(maxlen=5)
        self._gaze_v_buffer: deque = deque(maxlen=5)
        self._attention_h_buffer: deque = deque(maxlen=5)
        self._attention_v_buffer: deque = deque(maxlen=5)
        self._head_yaw_buffer: deque = deque(maxlen=3)
        self._head_pitch_buffer: deque = deque(maxlen=3)

        # Frame counter for video mode timestamps
        self._frame_counter = 0

        # Calibration data
        self.calibrated = False
        self.baseline_ear = 0.27  # default, updated by calibration

        # Auto-baseline for head pose & gaze (learns neutral "looking at screen" offset)
        self._baseline_samples: list[tuple[float, float, float, float]] = []  # (yaw, pitch, gaze_h, gaze_v)
        self._baseline_frames_needed = 60  # ~2 seconds at 30fps
        self._baseline_yaw = 0.0
        self._baseline_pitch = 0.0
        self._baseline_gaze_h = 0.0
        self._baseline_gaze_v = 0.0
        self._baseline_locked = False

        # Latest frame + annotated frame
        self.latest_frame = None
        self.annotated_frame = None
        self.latest_metrics = EyeMetrics()

        # Face loss recovery: keep last-known metrics briefly when face disappears
        self._face_lost_at: float | None = None
        self._FACE_GRACE_PERIOD = 2.0  # seconds to keep last metrics after face loss

        # Reading detector
        self._reading_detector = ReadingDetector()

        self._running = False

    def start(self):
        """Initialize camera and MediaPipe FaceLandmarker."""
        # Download model if needed
        model_path = ensure_model()

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Check that your webcam is connected and permissions are granted."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self._frame_counter = 0
        self._running = True

    def calibrate(self, duration: float = 3.0) -> float:
        """
        Run a quick calibration: capture EAR samples for `duration` seconds
        while user looks at the screen with eyes open. Returns baseline EAR.
        """
        if not self._running:
            return self.baseline_ear

        samples = []
        start = time.time()
        while time.time() - start < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_counter += 1
            timestamp_ms = int(self._frame_counter * (1000 / 30))
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                h, w, _ = frame.shape
                pts = [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks]
                left_ear = self._ear_from_pairs(pts, LEFT_EAR_PAIRS, LEFT_EYE_OUTER, LEFT_EYE_INNER)
                right_ear = self._ear_from_pairs(pts, RIGHT_EAR_PAIRS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
                avg = (left_ear + right_ear) / 2
                if avg > 0.1:  # filter out blinks during calibration
                    samples.append(avg)

            # Show calibration frame
            cv2.putText(frame, "CALIBRATING - Look at screen normally",
                        (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            progress = min(1.0, (time.time() - start) / duration)
            bar_w = int(540 * progress)
            cv2.rectangle(frame, (50, 60), (50 + bar_w, 75), (0, 255, 0), -1)
            cv2.rectangle(frame, (50, 60), (590, 75), (255, 255, 255), 1)
            self.annotated_frame = frame
            time.sleep(0.03)

        if len(samples) >= 10:
            self.baseline_ear = sum(samples) / len(samples)
            # Set thresholds relative to the user's baseline
            self.EAR_BLINK_THRESHOLD = self.baseline_ear * 0.75
            self.EAR_CLOSED_THRESHOLD = self.baseline_ear * 0.65
            self.calibrated = True

        return self.baseline_ear

    def stop(self):
        """Release resources."""
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def process_frame(self) -> EyeMetrics:
        """Capture one frame, analyse it, and return metrics."""
        if not self._running or self.cap is None:
            return EyeMetrics()

        ret, frame = self.cap.read()
        if not ret:
            return self.latest_metrics

        frame = cv2.flip(frame, 1)  # mirror
        self.latest_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image and run landmarker
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_counter += 1
        timestamp_ms = int(self._frame_counter * (1000 / 30))  # assume ~30fps

        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        metrics = EyeMetrics(timestamp=time.time())

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            h, w, _ = frame.shape
            pts = [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks]

            metrics.face_detected = True

            # --- Eye Aspect Ratio (using specific landmark pairs) ---
            left_ear = self._ear_from_pairs(pts, LEFT_EAR_PAIRS, LEFT_EYE_OUTER, LEFT_EYE_INNER)
            right_ear = self._ear_from_pairs(pts, RIGHT_EAR_PAIRS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
            raw_ear = (left_ear + right_ear) / 2

            # Smoothed EAR for display / gaze metrics
            self._ear_buffer.append(raw_ear)
            avg_ear = sum(self._ear_buffer) / len(self._ear_buffer)

            metrics.left_ear = left_ear
            metrics.right_ear = right_ear
            metrics.avg_ear = avg_ear

            # --- Blink Detection (uses RAW ear, not smoothed) ---
            is_below = raw_ear < self.EAR_BLINK_THRESHOLD
            if is_below:
                self._blink_counter += 1
            else:
                # Blink = eyes were below threshold, now opened back up
                if self._blink_counter >= self.BLINK_CONSEC_FRAMES:
                    self._blink_total += 1
                    self._blink_timestamps.append(time.time())
                self._blink_counter = 0

            metrics.is_blinking = is_below

            # Blinks per minute
            now = time.time()
            cutoff = now - 60
            while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
                self._blink_timestamps.popleft()
            metrics.blinks_per_minute = len(self._blink_timestamps)

            # --- Eye Closure Duration ---
            if avg_ear < self.EAR_CLOSED_THRESHOLD:
                if self._eyes_closed_start is None:
                    self._eyes_closed_start = now
                metrics.eyes_closed_duration = now - self._eyes_closed_start
            else:
                self._eyes_closed_start = None
                metrics.eyes_closed_duration = 0.0

            # --- Gaze Direction (using iris landmarks) ---
            # Only try iris if we have enough landmarks (478 = with iris)
            if len(pts) > max(max(LEFT_IRIS), max(RIGHT_IRIS)):
                gaze_h, gaze_v = self._compute_gaze(pts)
            else:
                gaze_h, gaze_v = 0.0, 0.0
            self._gaze_h_buffer.append(gaze_h)
            self._gaze_v_buffer.append(gaze_v)
            metrics.gaze_horizontal = sum(self._gaze_h_buffer) / len(self._gaze_h_buffer)
            metrics.gaze_vertical = sum(self._gaze_v_buffer) / len(self._gaze_v_buffer)

            # --- Head Pose (smoothed) ---
            raw_yaw, raw_pitch = self._estimate_head_pose(pts, w, h)
            self._head_yaw_buffer.append(raw_yaw)
            self._head_pitch_buffer.append(raw_pitch)
            metrics.head_yaw = sum(self._head_yaw_buffer) / len(self._head_yaw_buffer)
            metrics.head_pitch = sum(self._head_pitch_buffer) / len(self._head_pitch_buffer)

            # --- Combined Attention (head pose + eye gaze fusion) ---
            self._compute_combined_attention(metrics)

            # --- Reading Detection ---
            is_reading, reading_conf = self._reading_detector.update(
                metrics.timestamp, metrics.gaze_horizontal, metrics.gaze_vertical,
                metrics.looking_at_screen
            )
            metrics.is_reading = is_reading
            metrics.reading_confidence = reading_conf

            # --- Annotate frame ---
            self._annotate(frame, pts, metrics)

            # Face recovered — reset grace period
            self._face_lost_at = None
        else:
            # Face not detected — apply grace period
            now = time.time()
            if self._face_lost_at is None:
                self._face_lost_at = now

            elapsed = now - self._face_lost_at
            if elapsed < self._FACE_GRACE_PERIOD and self.latest_metrics.face_detected:
                # Within grace period: carry forward last-known metrics
                metrics = EyeMetrics(
                    timestamp=now,
                    face_detected=True,
                    left_ear=self.latest_metrics.left_ear,
                    right_ear=self.latest_metrics.right_ear,
                    avg_ear=self.latest_metrics.avg_ear,
                    gaze_horizontal=self.latest_metrics.gaze_horizontal,
                    gaze_vertical=self.latest_metrics.gaze_vertical,
                    blinks_per_minute=self.latest_metrics.blinks_per_minute,
                    head_yaw=self.latest_metrics.head_yaw,
                    head_pitch=self.latest_metrics.head_pitch,
                    attention_h=self.latest_metrics.attention_h,
                    attention_v=self.latest_metrics.attention_v,
                    looking_at_screen=self.latest_metrics.looking_at_screen,
                    head_frontal_confidence=self.latest_metrics.head_frontal_confidence,
                )
            else:
                metrics.face_detected = False

        self.annotated_frame = frame
        self.latest_metrics = metrics
        return metrics

    # ---- Internal helpers ----

    @staticmethod
    def _ear_from_pairs(pts, vertical_pairs, outer_id, inner_id) -> float:
        """
        Compute Eye Aspect Ratio using specific landmark pairs.
        EAR = mean(vertical_distances) / horizontal_distance
        Uses 3 vertical pairs (upper/lower eyelid) and eye corners for horizontal.
        """
        v_sum = 0.0
        for upper_id, lower_id in vertical_pairs:
            v_sum += math.dist(pts[upper_id][:2], pts[lower_id][:2])
        v_mean = v_sum / len(vertical_pairs)

        h_dist = math.dist(pts[outer_id][:2], pts[inner_id][:2])
        if h_dist == 0:
            return 0.0
        return v_mean / h_dist

    @staticmethod
    def _compute_gaze(pts) -> tuple[float, float]:
        """
        Estimate gaze direction from iris position relative to eye corners.
        Returns (horizontal, vertical) each in roughly -1..1 range.
        """
        def iris_center(indices):
            xs = [pts[i][0] for i in indices]
            ys = [pts[i][1] for i in indices]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        l_iris = iris_center(LEFT_IRIS)
        r_iris = iris_center(RIGHT_IRIS)

        # Left eye horizontal
        l_inner = pts[LEFT_EYE_INNER][:2]
        l_outer = pts[LEFT_EYE_OUTER][:2]
        l_w = math.dist(l_inner, l_outer)
        l_h_ratio = (l_iris[0] - l_outer[0]) / l_w if l_w > 0 else 0.5

        # Right eye horizontal
        r_inner = pts[RIGHT_EYE_INNER][:2]
        r_outer = pts[RIGHT_EYE_OUTER][:2]
        r_w = math.dist(r_inner, r_outer)
        r_h_ratio = (r_iris[0] - r_outer[0]) / r_w if r_w > 0 else 0.5

        # Average and normalize to -1..1 (0.5 = center)
        h = ((l_h_ratio + r_h_ratio) / 2 - 0.5) * 2

        # Vertical: use left eye upper/lower midpoints
        l_upper_mid_y = (pts[LEFT_EYE_UPPER[0]][1] + pts[LEFT_EYE_UPPER[-1]][1]) / 2
        l_lower_mid_y = (pts[LEFT_EYE_LOWER[0]][1] + pts[LEFT_EYE_LOWER[-1]][1]) / 2
        eye_h = l_lower_mid_y - l_upper_mid_y
        v = ((l_iris[1] - l_upper_mid_y) / eye_h - 0.5) * 2 if eye_h > 0 else 0.0

        return float(np.clip(h, -1, 1)), float(np.clip(v, -1, 1))

    @staticmethod
    def _estimate_head_pose(pts, img_w, img_h) -> tuple[float, float]:
        """Rough head pose estimation (yaw, pitch) using key face landmarks."""
        nose = pts[NOSE_TIP]
        left = pts[LEFT_CHEEK]
        right = pts[RIGHT_CHEEK]
        chin = pts[CHIN]

        # Yaw: ratio of nose-to-left vs nose-to-right
        d_left = nose[0] - left[0]
        d_right = right[0] - nose[0]
        total = d_left + d_right
        if total > 0:
            yaw = (d_left / total - 0.5) * 2  # -1 (left) to 1 (right)
        else:
            yaw = 0.0

        # Pitch: nose Y relative to midpoint of eye corners
        eye_mid_y = (pts[LEFT_EYE_CORNER][1] + pts[RIGHT_EYE_CORNER][1]) / 2
        chin_y = chin[1]
        face_h = chin_y - eye_mid_y
        if face_h > 0:
            nose_ratio = (nose[1] - eye_mid_y) / face_h
            pitch = (nose_ratio - 0.45) * 4  # rough normalization
        else:
            pitch = 0.0

        return float(np.clip(yaw, -1, 1)), float(np.clip(pitch, -1, 1))

    def _update_attention_baseline(self, metrics: EyeMetrics):
        """Auto-learn the user's neutral head/gaze offset from the first ~2s of face data."""
        if self._baseline_locked:
            return
        self._baseline_samples.append((
            metrics.head_yaw, metrics.head_pitch,
            metrics.gaze_horizontal, metrics.gaze_vertical,
        ))
        if len(self._baseline_samples) >= self._baseline_frames_needed:
            yaws = [s[0] for s in self._baseline_samples]
            pitches = [s[1] for s in self._baseline_samples]
            gaze_hs = [s[2] for s in self._baseline_samples]
            gaze_vs = [s[3] for s in self._baseline_samples]
            # Use median (robust to blinks/glances during startup)
            yaws.sort(); pitches.sort(); gaze_hs.sort(); gaze_vs.sort()
            mid = len(yaws) // 2
            self._baseline_yaw = yaws[mid]
            self._baseline_pitch = pitches[mid]
            self._baseline_gaze_h = gaze_hs[mid]
            self._baseline_gaze_v = gaze_vs[mid]
            self._baseline_locked = True

    def _compute_combined_attention(self, metrics: EyeMetrics):
        """
        Fuse head pose and eye gaze into a single attention direction.

        Strategy:
        - Auto-baseline subtracted so the user's natural resting position = center.
        - When head faces the camera (frontal), iris gaze is reliable for fine
          detail, so weight eye gaze more heavily.
        - When head is turned, iris landmarks become unreliable (foreshortening),
          so head pose dominates.
        - If head and eyes both point away in the same direction, that's a strong
          "looking away" signal. If the head is slightly turned but eyes
          compensate back toward center, the user is still watching the screen.
        """
        # Auto-learn neutral offset
        self._update_attention_baseline(metrics)

        # Subtract baseline so neutral position → 0,0
        yaw = metrics.head_yaw - self._baseline_yaw
        pitch = metrics.head_pitch - self._baseline_pitch
        gaze_h = metrics.gaze_horizontal - self._baseline_gaze_h
        gaze_v = metrics.gaze_vertical - self._baseline_gaze_v

        # How frontal is the head? 1.0 = directly facing camera, 0.0 = turned far away
        frontal = 1.0 - min(1.0, (abs(yaw) + abs(pitch) * 0.5))
        frontal = max(0.0, frontal)
        metrics.head_frontal_confidence = frontal

        # Adaptive weighting: frontal head → trust eyes more, turned head → trust head
        eye_weight = 0.3 + 0.3 * frontal   # 0.3 (turned) to 0.6 (frontal)
        head_weight = 1.0 - eye_weight      # 0.4 (frontal) to 0.7 (turned)

        raw_ah = yaw * head_weight + gaze_h * eye_weight
        raw_av = pitch * head_weight + gaze_v * eye_weight

        # Smooth the combined signal
        self._attention_h_buffer.append(raw_ah)
        self._attention_v_buffer.append(raw_av)
        metrics.attention_h = float(np.clip(
            sum(self._attention_h_buffer) / len(self._attention_h_buffer), -1, 1))
        metrics.attention_v = float(np.clip(
            sum(self._attention_v_buffer) / len(self._attention_v_buffer), -1, 1))

        # "Looking at screen" = combined attention roughly centered
        screen_h_thresh = 0.40
        screen_v_thresh = 0.35
        metrics.looking_at_screen = (
            abs(metrics.attention_h) < screen_h_thresh
            and abs(metrics.attention_v) < screen_v_thresh
        )

    @staticmethod
    def _annotate(frame, pts, metrics: EyeMetrics):
        """Draw face tracking visualization on the frame."""
        h, w, _ = frame.shape

        # --- Face oval outline ---
        if len(pts) > max(FACE_OVAL):
            oval_pts = [(int(pts[i][0]), int(pts[i][1])) for i in FACE_OVAL]
            for i in range(len(oval_pts) - 1):
                cv2.line(frame, oval_pts[i], oval_pts[i + 1], (180, 220, 255), 2)

        # --- Nose bridge ---
        if len(pts) > max(NOSE_BRIDGE):
            nose_pts = [(int(pts[i][0]), int(pts[i][1])) for i in NOSE_BRIDGE]
            for i in range(len(nose_pts) - 1):
                cv2.line(frame, nose_pts[i], nose_pts[i + 1], (200, 200, 200), 1)

        # --- Eye contours (thicker) ---
        for eye_ids in [LEFT_EYE_UPPER + LEFT_EYE_LOWER, RIGHT_EYE_UPPER + RIGHT_EYE_LOWER]:
            points = [(int(pts[i][0]), int(pts[i][1])) for i in eye_ids]
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 220, 220), 2)

        # --- Iris circles (larger, with gaze line) ---
        max_iris_idx = max(max(LEFT_IRIS), max(RIGHT_IRIS))
        if len(pts) > max_iris_idx:
            for iris_ids, outer_id, inner_id in [
                (LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER),
                (RIGHT_IRIS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER),
            ]:
                cx = int(sum(pts[i][0] for i in iris_ids) / len(iris_ids))
                cy = int(sum(pts[i][1] for i in iris_ids) / len(iris_ids))
                # Iris dot
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), 1)

        # --- Attention direction indicator (arrow from nose) ---
        nose_x = int(pts[NOSE_TIP][0])
        nose_y = int(pts[NOSE_TIP][1])
        arrow_len = 50
        ax = int(nose_x + metrics.attention_h * arrow_len)
        ay = int(nose_y + metrics.attention_v * arrow_len)
        arrow_color = (0, 255, 100) if metrics.looking_at_screen else (0, 80, 255)
        cv2.arrowedLine(frame, (nose_x, nose_y), (ax, ay), arrow_color, 2, tipLength=0.3)

        # --- Screen/Away badge ---
        if metrics.looking_at_screen:
            badge_text = "SCREEN"
            badge_color = (0, 180, 0)
        else:
            badge_text = "AWAY"
            badge_color = (0, 0, 220)
        badge_x = w - 110
        cv2.rectangle(frame, (badge_x, 8), (badge_x + 100, 32), badge_color, -1)
        cv2.putText(frame, badge_text, (badge_x + 8, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Reading badge ---
        if metrics.is_reading:
            rd_x = w - 110
            cv2.rectangle(frame, (rd_x, 36), (rd_x + 100, 58), (180, 120, 0), -1)
            conf_pct = int(metrics.reading_confidence * 100)
            cv2.putText(frame, f"READING {conf_pct}%", (rd_x + 4, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Status text ---
        color = (0, 255, 0) if metrics.face_detected else (0, 0, 255)
        y_off = 25
        texts = [
            f"EAR: {metrics.avg_ear:.2f}",
            f"Blinks/min: {metrics.blinks_per_minute:.0f}",
            f"Gaze H: {metrics.gaze_horizontal:+.2f}  V: {metrics.gaze_vertical:+.2f}",
            f"Head Yaw: {metrics.head_yaw:+.2f}  Pitch: {metrics.head_pitch:+.2f}",
            f"Attention: {metrics.attention_h:+.2f},{metrics.attention_v:+.2f}",
        ]
        if metrics.eyes_closed_duration > 0.5:
            texts.append(f"EYES CLOSED: {metrics.eyes_closed_duration:.1f}s")
        if metrics.is_blinking:
            texts.append("* BLINK *")

        for i, txt in enumerate(texts):
            cv2.putText(frame, txt, (10, y_off + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
