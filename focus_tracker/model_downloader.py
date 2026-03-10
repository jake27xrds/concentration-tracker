"""
Model Downloader
Downloads the MediaPipe FaceLandmarker model on first run.
"""

import os
import ssl
import urllib.request

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
    print(f"   Downloading FaceLandmarker model...")
    print(f"   From: {MODEL_URL}")

    # Create SSL context — try default first, fall back to certifi or unverified
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # On macOS, Python's bundled SSL may lack certs — use system certs
        ctx = ssl.create_default_context()
        # If that also fails, we'll catch below

    try:
        _download(MODEL_URL, MODEL_PATH, ctx)
    except (ssl.SSLCertVerificationError, urllib.error.URLError):
        # Fall back: try without SSL verification
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("   ⚠ SSL verification failed, downloading without verification...")
        _download(MODEL_URL, MODEL_PATH, ctx)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"   ✓ Model downloaded ({size_mb:.1f} MB)")

    return MODEL_PATH


def _download(url: str, dest: str, ctx: ssl.SSLContext):
    """Download a URL to a file using the given SSL context."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
