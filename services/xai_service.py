"""
XAI service — Grad-CAM, Grad-CAM++, LIME, SHAP, Integrated Gradients.

Grad-CAM uses torch.autograd.grad() instead of register_full_backward_hook
to avoid PyTorch 2.x BackwardHookFunctionBackward caching bugs.
"""

import logging, gc, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2

from config import (
    GRADCAM_TARGET_LAYER, SHAP_BACKGROUND_SIZE, SHAP_NSAMPLES,
    LIME_NUM_SAMPLES, LIME_NUM_FEATURES, LIME_NUM_RUNS,
    IG_STEPS, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
)

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────

def _overlay_heatmap(image_np, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    h, w = image_np.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm8 = np.ascontiguousarray(np.clip(hm * 255, 0, 255).astype(np.uint8))
    color = cv2.applyColorMap(hm8, colormap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * image_np + alpha * color, 0, 1)


def _denormalize(tensor):
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    return tensor * s + m


def _tensor_to_b64(arr):
    """uint8 RGB numpy array → base64 JPEG string."""
    import base64, io
    from PIL import Image
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ── Grad-CAM (autograd.grad — no backward hooks) ──────────────────────

class _GradCAMBase:
    """
    Base class for Grad-CAM / Grad-CAM++.
    Uses torch.autograd.grad() instead of register_full_backward_hook.
    Each call to generate() is fully self-contained.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def _compute_cam(self, input_tensor, target_class, device):
        self.model.eval()
        activation_holder = {}

        def _capture(module, inp, out):
            activation_holder["val"] = out  # NOT .detach()

        handle = self.target_layer.register_forward_hook(_capture)
        try:
            output = self.model(input_tensor.to(device))
            if target_class is None:
                target_class = int(torch.sigmoid(output).item() >= 0.5)
            one_hot = torch.zeros_like(output)
            one_hot[0, 0] = 1.0 if target_class == 1 else -1.0
            grads = torch.autograd.grad(
                outputs=output,
                inputs=activation_holder["val"],
                grad_outputs=one_hot,
            )[0]
        finally:
            handle.remove()

        acts = activation_holder["val"].detach()
        return grads, acts, target_class

    def _normalize(self, cam_np):
        if cam_np.max() > cam_np.min():
            return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        return np.zeros_like(cam_np)


class GradCAM(_GradCAMBase):
    def generate(self, input_tensor, target_class=None, device=None):
        device = device or next(self.model.parameters()).device
        grads, acts, tc = self._compute_cam(input_tensor, target_class, device)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_np = cam.squeeze().cpu().numpy()
        return self._normalize(cam_np)


class GradCAMPlusPlus(_GradCAMBase):
    def generate(self, input_tensor, target_class=None, device=None):
        device = device or next(self.model.parameters()).device
        grads, acts, tc = self._compute_cam(input_tensor, target_class, device)
        g2, g3 = grads ** 2, grads ** 3
        alpha = g2 / (2.0 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True) + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_np = cam.squeeze().cpu().numpy()
        return self._normalize(cam_np)


# ── LIME ───────────────────────────────────────────────────────────────

def _run_lime(model, input_tensor, image_np_uint8, device, n_samples=None,
              n_features=None, n_runs=None):
    from lime import lime_image
    from skimage.segmentation import slic, mark_boundaries

    n_samples = n_samples or LIME_NUM_SAMPLES
    n_features = n_features or LIME_NUM_FEATURES
    n_runs = n_runs or LIME_NUM_RUNS

    def _predict_fn(images):
        t = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
        m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
        t = (t / 255.0 - m) / s
        with torch.no_grad():
            p = torch.sigmoid(model(t)).cpu().numpy().flatten()
        return np.column_stack([1 - p, p])

    explainer = lime_image.LimeImageExplainer()
    seg_fn = lambda x: slic(x, n_segments=50, compactness=10, sigma=1)

    exp = explainer.explain_instance(
        image_np_uint8, _predict_fn,
        top_labels=2, hide_color=0, num_samples=n_samples,
        segmentation_fn=seg_fn,
    )
    top = exp.top_labels[0]
    temp, mask = exp.get_image_and_mask(top, positive_only=True,
                                        num_features=n_features, hide_rest=False)
    overlay = mark_boundaries(temp / 255.0, mask)
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


# ── SHAP ───────────────────────────────────────────────────────────────

def _run_shap(shap_explainer, input_tensor, device):
    sv = shap_explainer.shap_values(input_tensor.to(device), nsamples=SHAP_NSAMPLES)
    if isinstance(sv, list):
        sv_np = np.abs(sv[0][0]).mean(axis=0)
    else:
        sv_np = np.abs(sv[0]).mean(axis=0)
    sv_np = np.squeeze(sv_np)
    if sv_np.max() > sv_np.min():
        sv_np = (sv_np - sv_np.min()) / (sv_np.max() - sv_np.min())
    return sv_np


# ── Integrated Gradients (Captum) ─────────────────────────────────────

def _run_ig(ig, input_tensor, target_class, device):
    input_tensor = input_tensor.to(device)
    
    # The model has a single output node, so target index must always be 0
    attr = ig.attribute(input_tensor, target=0, n_steps=IG_STEPS)
    
    # If the prediction is Normal (0), invert attributions so the heatmap
    # highlights areas pushing the prediction towards Normal instead of Pneumonia.
    if target_class == 0:
        attr = -attr

    attr_np = np.array(attr.detach().cpu(), dtype=np.float64)
    # Captum returns (1, 3, H, W) → reduce to (H, W)
    if attr_np.ndim == 4:
        attr_np = attr_np[0].mean(axis=0)
    elif attr_np.ndim == 3:
        if attr_np.shape[0] == 3:
            attr_np = attr_np.mean(axis=0)
        elif attr_np.shape[2] == 3:
            attr_np = attr_np.mean(axis=2)
    attr_np = np.squeeze(attr_np)
    if attr_np.max() > attr_np.min():
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
    return attr_np


# ── Dispatcher ─────────────────────────────────────────────────────────

def run_xai(model, input_tensor, methods: list, device, shap_explainer=None,
             ig_instance=None, train_dataset=None):
    """
    Run selected XAI methods. Returns dict of {method_name: base64_image}.

    Args:
        model: DenseNet-121 model (eval mode).
        input_tensor: (1, 3, H, W) preprocessed tensor.
        methods: list like ["gradcam", "gradcam_pp", "lime", "shap", "ig"].
        device: torch.device.
        shap_explainer: pre-built shap.GradientExplainer (or None to skip SHAP).
        ig_instance: pre-built Captum IntegratedGradients (or None to skip IG).
        train_dataset: needed to build SHAP explainer on the fly.
    """
    # Denormalized image for overlay
    img_np = _denormalize(input_tensor.squeeze().cpu()).permute(1, 2, 0).numpy()
    image_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    # Determine target class from model prediction
    with torch.no_grad():
        logit = model(input_tensor.to(device)).item()
    prob = 1.0 / (1.0 + np.exp(-logit))
    target_class = int(prob >= 0.5)

    # Get target layer
    parts = GRADCAM_TARGET_LAYER.split(".")
    target_layer = model
    for p in parts:
        target_layer = getattr(target_layer, p)

    results = {}

    for method in methods:
        try:
            if method == "gradcam":
                gcam = GradCAM(model, target_layer)
                heatmap = gcam.generate(input_tensor, target_class, device)
                overlay = _overlay_heatmap(img_np, heatmap)
                results["gradcam"] = _tensor_to_b64(overlay)

            elif method == "gradcam_pp":
                gcpp = GradCAMPlusPlus(model, target_layer)
                heatmap = gcpp.generate(input_tensor, target_class, device)
                overlay = _overlay_heatmap(img_np, heatmap, colormap=cv2.COLORMAP_VIRIDIS)
                results["gradcam_pp"] = _tensor_to_b64(overlay)

            elif method == "lime":
                lime_img = _run_lime(model, input_tensor, image_uint8, device)
                results["lime"] = _tensor_to_b64(lime_img)

            elif method == "shap":
                if shap_explainer is None:
                    raise ValueError("SHAP explainer not initialized")
                shap_map = _run_shap(shap_explainer, input_tensor, device)
                overlay = _overlay_heatmap(img_np, shap_map, colormap=cv2.COLORMAP_BONE)
                results["shap"] = _tensor_to_b64(overlay)

            elif method == "ig":
                if ig_instance is None:
                    raise ValueError("IG instance not initialized")
                ig_map = _run_ig(ig_instance, input_tensor, target_class, device)
                overlay = _overlay_heatmap(img_np, ig_map, colormap=cv2.COLORMAP_VIRIDIS)
                results["ig"] = _tensor_to_b64(overlay)

        except Exception as e:
            logger.error("XAI %s failed: %s", method, e)
            results[method] = None

    # Free GPU memory
    del input_tensor
    torch.cuda.empty_cache()
    gc.collect()

    return results