"""
Centralized configuration for the Chest X-Ray Pneumonia XAI Web UI.
Override any value via environment variable (uppercase prefix).
"""

import os

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
GOOGLE_DRIVE_MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/drive/folders/1miVWZ-5JvCTSHO5NgGdeBOF_iGz89U0e?usp=sharing",
)
# Optional SHA-256 pin for downloaded weights. Set this in production.
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "").strip().lower()
MODEL_SETUP_TOKEN = os.environ.get("MODEL_SETUP_TOKEN", "").strip()

# ── Model ───────────────────────────────────────────────────────────────
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))

# ── XAI Settings ────────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "features.norm5"          # DenseNet-121 last BN
SHAP_BACKGROUND_SIZE = int(os.environ.get("SHAP_BACKGROUND_SIZE", "10"))
SHAP_NSAMPLES = int(os.environ.get("SHAP_NSAMPLES", "30"))
LIME_NUM_SAMPLES = int(os.environ.get("LIME_NUM_SAMPLES", "500"))
LIME_NUM_FEATURES = int(os.environ.get("LIME_NUM_FEATURES", "10"))
LIME_NUM_RUNS = int(os.environ.get("LIME_NUM_RUNS", "1"))
IG_STEPS = int(os.environ.get("IG_STEPS", "25"))

# ── Server ──────────────────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))
MIN_IMAGE_DIMENSION = int(os.environ.get("MIN_IMAGE_DIMENSION", 100))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 25_000_000))
MAX_XAI_METHODS = int(os.environ.get("MAX_XAI_METHODS", "3"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
ALLOWED_XAI_METHODS = {"gradcam", "gradcam_pp", "lime", "shap", "ig"}
