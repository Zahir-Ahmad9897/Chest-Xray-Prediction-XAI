"""
FastAPI application for Chest X-Ray Pneumonia XAI Dashboard.
"""

import asyncio
import io
import logging
import os
from typing import Optional

import torch
from captum.attr import IntegratedGradients
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFile, UnidentifiedImageError

from config import (
    ALLOWED_IMAGE_TYPES, ALLOWED_XAI_METHODS, MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS, MAX_XAI_METHODS, MIN_IMAGE_DIMENSION,
    MODEL_SETUP_TOKEN, SHAP_BACKGROUND_SIZE, SHAP_NSAMPLES,
)
from services import model_service, xai_service

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chest X-Ray Pneumonia XAI", version="2.0.0")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
shap_explainer = None
ig_instance = None
inference_lock = asyncio.Lock()


def _validate_image(contents: bytes, content_type: Optional[str]) -> None:
    if not contents:
        raise HTTPException(400, "The uploaded file is empty")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the maximum allowed size")
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, "Unsupported image type; upload a JPEG or PNG")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
        with Image.open(io.BytesIO(contents)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(415, "The file content is not a JPEG or PNG")
            if image.width < MIN_IMAGE_DIMENSION or image.height < MIN_IMAGE_DIMENSION:
                raise HTTPException(400, "Image dimensions are too small")
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
        logger.info("Rejected invalid image upload: %s", exc)
        raise HTTPException(400, "The uploaded file is not a valid JPEG or PNG image") from exc


async def _read_upload(file: UploadFile) -> bytes:
    if file.size is not None and file.size > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the maximum allowed size")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    _validate_image(contents, file.content_type)
    return contents


def _validate_methods(methods: str) -> list[str]:
    method_list = list(dict.fromkeys(m.strip().lower() for m in methods.split(",") if m.strip()))
    unknown = sorted(set(method_list) - ALLOWED_XAI_METHODS)
    if unknown:
        raise HTTPException(400, f"Unsupported XAI method(s): {', '.join(unknown)}")
    if not method_list:
        raise HTTPException(400, "At least one XAI method is required")
    if len(method_list) > MAX_XAI_METHODS:
        raise HTTPException(400, f"Select at most {MAX_XAI_METHODS} XAI methods per request")
    return method_list


def _check_setup_token(token: Optional[str]) -> None:
    if not MODEL_SETUP_TOKEN:
        raise HTTPException(503, "Model download is disabled until MODEL_SETUP_TOKEN is configured")
    if token != MODEL_SETUP_TOKEN:
        raise HTTPException(401, "Invalid model setup token")


@app.on_event("startup")
async def startup():
    global model, ig_instance
    logger.info("Loading model on %s …", DEVICE)
    model = model_service.load_model(DEVICE).to(DEVICE)
    ig_instance = IntegratedGradients(model)


def _build_shap_explainer():
    global shap_explainer
    if shap_explainer is None:
        import shap
        bg = torch.zeros(SHAP_BACKGROUND_SIZE, 3, 224, 224, device=DEVICE)
        bg[:, 0] = 0.485; bg[:, 1] = 0.456; bg[:, 2] = 0.406
        shap_explainer = shap.GradientExplainer(model, bg)
    return shap_explainer


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    contents = await _read_upload(file)
    async with inference_lock:
        return JSONResponse(model_service.predict(model, model_service.preprocess_image(contents), DEVICE))


@app.post("/api/explain")
async def explain(file: UploadFile = File(...), methods: str = Query("gradcam,gradcam_pp,ig")):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    contents = await _read_upload(file)
    method_list = _validate_methods(methods)
    async with inference_lock:
        tensor = model_service.preprocess_image(contents)
        if "shap" in method_list:
            _build_shap_explainer()
        pred = model_service.predict(model, tensor, DEVICE)
        xai_results = xai_service.run_xai(model, tensor, method_list, DEVICE, shap_explainer, ig_instance)
    return JSONResponse({"prediction": pred, "explanations": xai_results})


@app.post("/api/setup/download-model")
async def download_model(x_setup_token: Optional[str] = Header(default=None)):
    _check_setup_token(x_setup_token)
    try:
        global model, ig_instance, shap_explainer
        async with inference_lock:
            path = model_service.download_model_from_drive()
            model_service._model_cache = None
            model = model_service.load_model(DEVICE).to(DEVICE)
            ig_instance = IntegratedGradients(model)
            shap_explainer = None
        return {"status": "ok", "model_path": path}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Model download failed")
        raise HTTPException(500, "Model download failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
