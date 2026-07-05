"""
Centralized configuration for the Chest X-Ray Pneumonia XAI Web UI.
Override any value via environment variable (uppercase prefix).
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
GOOGLE_DRIVE_MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/drive/folders/1miVWZ-5JvCTSHO5NgGdeBOF_iGz89U0e?usp=sharing",
)
# ── Model ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ── XAI Settings ─────────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "features.norm5"          # DenseNet-121 last BN
SHAP_BACKGROUND_SIZE = 10                         # small to save RAM
SHAP_NSAMPLES = 30                               # reduced for speed
LIME_NUM_SAMPLES = 500
LIME_NUM_FEATURES = 10
LIME_NUM_RUNS = 1
IG_STEPS = 25

# ── Server ───────────────────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))