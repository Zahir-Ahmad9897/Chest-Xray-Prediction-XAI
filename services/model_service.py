"""
Model service — loads DenseNet-121, preprocesses images, runs inference.
"""

import hashlib
import logging
import os
import io

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.models import DenseNet121_Weights

from config import (
    CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MODEL_DIR,
    MODEL_PATH, MODEL_SHA256, GOOGLE_DRIVE_MODEL_URL, PREDICTION_THRESHOLD,
)

logger = logging.getLogger(__name__)
_model_cache = None
_original_densenet_forward = None


def _patched_densenet_forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=False)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    return self.classifier(out)


def _apply_patch():
    global _original_densenet_forward
    import torchvision.models.densenet as _dn
    if _original_densenet_forward is None:
        _original_densenet_forward = _dn.DenseNet.forward
        _dn.DenseNet.forward = _patched_densenet_forward
        logger.info("Patched DenseNet.forward: inplace=True -> inplace=False")


def _build_model() -> nn.Module:
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_f = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_f, 1))
    return model


def _verify_checksum(path: str) -> None:
    if not MODEL_SHA256:
        logger.warning("MODEL_SHA256 is not configured; downloaded weights are not checksum-pinned")
        return
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != MODEL_SHA256:
        raise RuntimeError("Downloaded model checksum does not match MODEL_SHA256")


def download_model_from_drive(url: str = GOOGLE_DRIVE_MODEL_URL) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        _verify_checksum(MODEL_PATH)
        return MODEL_PATH

    logger.info("Downloading model from Google Drive …")
    gdown.download_folder(url, output=MODEL_DIR, quiet=False)
    candidates = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith((".pt", ".pth")) and f != "best.pt"
    ]
    candidates.sort(key=lambda f: ("best" not in f.lower(), f))
    for filename in candidates:
        os.replace(os.path.join(MODEL_DIR, filename), MODEL_PATH)
        break
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No .pt/.pth found in {MODEL_DIR} after download")
    _verify_checksum(MODEL_PATH)
    return MODEL_PATH


def load_model(device: torch.device = None) -> nn.Module:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    _apply_patch()
    model = _build_model()
    ckpt_path = MODEL_PATH if os.path.exists(MODEL_PATH) else download_model_from_drive()
    _verify_checksum(ckpt_path)

    # weights_only=True avoids arbitrary-code execution from untrusted checkpoints.
    try:
        ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=True)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError("Model checkpoint must be a tensor state dict") from exc
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise RuntimeError("Model checkpoint has an unsupported format")
    model.load_state_dict(state_dict)
    model.eval()
    _model_cache = model
    logger.info("Model loaded from %s", ckpt_path)
    return model


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(file_bytes)) as image:
        image = image.convert("RGB")
        return _transform(image).unsqueeze(0)


def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device):
    tensor = tensor.to(device)
    with torch.inference_mode():
        logit = float(model(tensor).reshape(-1)[0].item())
    # Stable sigmoid avoids overflow for extreme logits.
    probability = float(torch.sigmoid(torch.tensor(logit)).item())
    label_idx = int(probability >= PREDICTION_THRESHOLD)
    return {
        "label": label_idx,
        "label_name": CLASS_NAMES[label_idx],
        "probability": probability,
        "threshold": PREDICTION_THRESHOLD,
        "logit": logit,
    }
