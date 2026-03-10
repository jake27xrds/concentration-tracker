"""Tests for reading detection from eye movement patterns."""
import time
import random

from focus_tracker.eye_tracker import ReadingDetector


class TestReadingDetector:
    def test_reading_pattern_detected(self):
        """Simulated reading (forward saccades + fixations) triggers reading."""
        rd = ReadingDetector()
        t = time.time()
        gaze_h = -0.3

        for i in range(60):
            t += 0.033
            if i == 45:
                gaze_h = -0.3
                rd.update(t, gaze_h, 0.08, True)
                continue
            if i % 10 < 8:
                is_r, conf = rd.update(t, gaze_h + 0.002, 0.05, True)
            else:
                gaze_h += 0.05
                is_r, conf = rd.update(t, gaze_h, 0.05, True)

        assert is_r is True
        assert conf > 0.2

    def test_random_gaze_not_reading(self):
        """Random eye movements should not trigger reading."""
        rd = ReadingDetector()
        t = time.time()
        random.seed(42)

        for i in range(60):
            t += 0.033
            h = random.uniform(-0.5, 0.5)
            v = random.uniform(-0.3, 0.3)
            is_r, conf = rd.update(t, h, v, True)

        assert is_r is False
        assert conf < 0.35

    def test_looking_away_not_reading(self):
        """When not looking at screen, reading should not be detected."""
        rd = ReadingDetector()
        t = time.time()

        for i in range(30):
            t += 0.033
            is_r, conf = rd.update(t, 0.01, 0.01, False)

        assert is_r is False
        assert conf == 0.0
