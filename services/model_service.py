"""
Model service — loads DenseNet-121, preprocesses images, runs inference.
"""

import os, logging, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import gdown

from config import (
    MODEL_PATH, MODEL_DIR, GOOGLE_DRIVE_MODEL_URL,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)

logger = logging.getLogger(__name__)

_model_cache = None


# ── DenseNet inplace-ReLU patch ────────────────────────────────────────
_original_densenet_forward = None


def _patched_densenet_forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=False)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out


def _apply_patch():
    global _original_densenet_forward
    import torchvision.models.densenet as _dn
    if _original_densenet_forward is None:
        _original_densenet_forward = _dn.DenseNet.forward
        _dn.DenseNet.forward = _patched_densenet_forward
        logger.info("Patched DenseNet.forward: inplace=True -> inplace=False")


# ── Model builder ──────────────────────────────────────────────────────
def _build_model() -> nn.Module:
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_f = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_f, 1))
    return model


# ── Download from Google Drive ─────────────────────────────────────────
def download_model_from_drive(url: str = GOOGLE_DRIVE_MODEL_URL) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        logger.info("Model already exists at %s", MODEL_PATH)
        return MODEL_PATH
    logger.info("Downloading model from Google Drive …")
    # gdown handles folders by finding .pt files inside
    gdown.download_folder(url, output=MODEL_DIR, quiet=False)
    # find the .pt/.pth file — prioritize files with "best" in the name
    candidates = [f for f in os.listdir(MODEL_DIR)
                  if f.endswith((".pt", ".pth")) and f != "best.pt"]
    candidates.sort(key=lambda f: ("best" not in f.lower(), f))
    for f in candidates:
        os.replace(os.path.join(MODEL_DIR, f), MODEL_PATH)
        logger.info("Renamed %s → best.pt", f)
        break
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No .pt/.pth found in {MODEL_DIR} after download"
        )
    logger.info("Model saved to %s", MODEL_PATH)
    return MODEL_PATH


# ── Load / cache ───────────────────────────────────────────────────────
def load_model(device: torch.device = None) -> nn.Module:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    _apply_patch()
    model = _build_model()

    ckpt_path = MODEL_PATH
    if not os.path.exists(ckpt_path):
        ckpt_path = download_model_from_drive()

    ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    _model_cache = model
    logger.info("Model loaded from %s", ckpt_path)
    return model


# ── Preprocessing ──────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(__import__("io").BytesIO(file_bytes)).convert("RGB")
    # Handle grayscale: replicate to 3 channels
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _transform(img).unsqueeze(0)  # (1, 3, H, W)


# ── Predict ────────────────────────────────────────────────────────────
def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device):
    tensor = tensor.to(device)
    with torch.no_grad():
        logit = model(tensor).item()
    prob = 1.0 / (1.0 + np.exp(-logit))
    label_idx = int(prob >= 0.5)
    return {"label": label_idx, "label_name": ["NORMAL", "PNEUMONIA"][label_idx],
            "probability": float(prob), "logit": float(logit)}