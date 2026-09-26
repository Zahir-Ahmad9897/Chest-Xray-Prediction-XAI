"""
XAI service — Grad-CAM, Grad-CAM++, LIME, SHAP, Integrated Gradients.
"""

import base64
import gc
import io
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    CLASS_NAMES, GRADCAM_TARGET_LAYER, IG_STEPS, IMAGENET_MEAN, IMAGENET_STD,
    IMG_SIZE, LIME_NUM_FEATURES, LIME_NUM_RUNS, LIME_NUM_SAMPLES,
    SHAP_NSAMPLES,
)

logger = logging.getLogger(__name__)


def _overlay_heatmap(image_np, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    h, w = image_np.shape[:2]
    hm = cv2.resize(np.nan_to_num(heatmap, nan=0.0), (w, h))
    hm8 = np.ascontiguousarray(np.clip(hm * 255, 0, 255).astype(np.uint8))
    color = cv2.cvtColor(cv2.applyColorMap(hm8, colormap), cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * image_np + alpha * color, 0, 1)


def _denormalize(tensor):
    m = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return tensor * s + m


def _tensor_to_b64(arr):
    image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _normalize(array):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = array.min(), array.max()
    return (array - low) / (high - low) if high > low else np.zeros_like(array)


class _GradCAMBase:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def _compute_cam(self, input_tensor, target_class, device):
        self.model.eval()
        activation_holder = {}

        def _capture(module, inp, out):
            activation_holder["val"] = out

        handle = self.target_layer.register_forward_hook(_capture)
        try:
            output = self.model(input_tensor.to(device))
            one_hot = torch.ones_like(output) if target_class == 1 else -torch.ones_like(output)
            grads = torch.autograd.grad(output, activation_holder["val"], grad_outputs=one_hot)[0]
        finally:
            handle.remove()
        return grads, activation_holder["val"].detach()

    @staticmethod
    def _normalize(cam_np):
        return _normalize(cam_np)


class GradCAM(_GradCAMBase):
    def generate(self, input_tensor, target_class, device):
        grads, acts = self._compute_cam(input_tensor, target_class, device)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        return self._normalize(F.relu((weights * acts).sum(dim=1, keepdim=True)).squeeze().cpu().numpy())


class GradCAMPlusPlus(_GradCAMBase):
    def generate(self, input_tensor, target_class, device):
        grads, acts = self._compute_cam(input_tensor, target_class, device)
        g2, g3 = grads ** 2, grads ** 3
        alpha = g2 / (2.0 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True) + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        return self._normalize(F.relu((weights * acts).sum(dim=1, keepdim=True)).squeeze().cpu().numpy())


def _run_lime(model, image_np_uint8, device):
    from lime import lime_image
    from skimage.segmentation import slic, mark_boundaries

    def predict_fn(images):
        t = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        with torch.inference_mode():
            p = torch.sigmoid(model((t / 255.0 - mean) / std)).cpu().numpy().reshape(-1)
        return np.column_stack([1 - p, p])

    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(
        image_np_uint8, predict_fn, top_labels=2, hide_color=0,
        num_samples=LIME_NUM_SAMPLES,
        segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10, sigma=1),
    )
    top = exp.top_labels[0]
    temp, mask = exp.get_image_and_mask(top, positive_only=True, num_features=LIME_NUM_FEATURES, hide_rest=False)
    return np.clip(mark_boundaries(temp / 255.0, mask), 0, 1)


def _run_shap(shap_explainer, input_tensor, device):
    values = shap_explainer.shap_values(input_tensor.to(device), nsamples=SHAP_NSAMPLES)
    values = values[0] if isinstance(values, list) else values
    return _normalize(np.abs(np.asarray(values[0])).mean(axis=0).squeeze())


def _run_ig(ig, input_tensor, target_class, device):
    attr = ig.attribute(input_tensor.to(device), target=0, n_steps=IG_STEPS)
    if target_class == 0:
        attr = -attr
    return _normalize(attr.detach().cpu().numpy()[0].mean(axis=0))


def run_xai(model, input_tensor, methods: list, device, shap_explainer=None, ig_instance=None, train_dataset=None):
    img_np = np.clip(_denormalize(input_tensor.squeeze().cpu()).permute(1, 2, 0).numpy(), 0, 1)
    image_uint8 = (img_np * 255).astype(np.uint8)
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(input_tensor.to(device)).reshape(-1)[0]).item())
    target_class = int(probability >= 0.5)

    target_layer = model
    for part in GRADCAM_TARGET_LAYER.split("."):
        target_layer = getattr(target_layer, part)

    results = {}
    for method in methods:
        try:
            if method == "gradcam":
                heatmap = GradCAM(model, target_layer).generate(input_tensor, target_class, device)
                results[method] = _tensor_to_b64(_overlay_heatmap(img_np, heatmap))
            elif method == "gradcam_pp":
                heatmap = GradCAMPlusPlus(model, target_layer).generate(input_tensor, target_class, device)
                results[method] = _tensor_to_b64(_overlay_heatmap(img_np, heatmap, colormap=cv2.COLORMAP_VIRIDIS))
            elif method == "lime":
                results[method] = _tensor_to_b64(_run_lime(model, image_uint8, device))
            elif method == "shap":
                if shap_explainer is None:
                    raise ValueError("SHAP explainer not initialized")
                results[method] = _tensor_to_b64(_overlay_heatmap(img_np, _run_shap(shap_explainer, input_tensor, device), colormap=cv2.COLORMAP_BONE))
            elif method == "ig":
                if ig_instance is None:
                    raise ValueError("IG instance not initialized")
                results[method] = _tensor_to_b64(_overlay_heatmap(img_np, _run_ig(ig_instance, input_tensor, target_class, device), colormap=cv2.COLORMAP_VIRIDIS))
        except Exception:
            logger.exception("XAI %s failed", method)
            results[method] = None

    del input_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results
