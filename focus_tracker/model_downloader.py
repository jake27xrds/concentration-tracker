"""
Model Downloader
Downloads the MediaPipe FaceLandmarker model on first run.
"""

import logging
import os
import ssl
import urllib.request

log = logging.getLogger("focus_tracker.model")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "face_landmarker.task"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Official MediaPipe model URL
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model() -> str:
    """Download the FaceLandmarker model if it doesn't exist. Returns the path."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    log.info("Downloading FaceLandmarker model from %s", MODEL_URL)

    # Create SSL context — try default first, fall back to certifi or unverified
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    try:
        _download(MODEL_URL, MODEL_PATH, ctx)
    except (ssl.SSLCertVerificationError, urllib.error.URLError):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        log.warning("SSL verification failed, retrying without verification")
        _download(MODEL_URL, MODEL_PATH, ctx)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    log.info("Model downloaded (%.1f MB)", size_mb)

    return MODEL_PATH


def _download(url: str, dest: str, ctx: ssl.SSLContext):
    """Download a URL to a file with progress logging."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -1
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded / total * 100)
                    # Log every 10%
                    if pct // 10 > last_pct // 10:
                        log.info("   Download progress: %d%%", pct)
                        last_pct = pct
