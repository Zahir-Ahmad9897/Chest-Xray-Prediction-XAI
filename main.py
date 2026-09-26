"""
FastAPI application for Chest X-Ray Pneumonia XAI Dashboard.
"""

import os, io, logging, gc, torch
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFile, UnidentifiedImageError

from config import CLASS_NAMES, SHAP_BACKGROUND_SIZE, SHAP_NSAMPLES
from services import model_service, xai_service
from captum.attr import IntegratedGradients

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chest X-Ray Pneumonia XAI", version="2.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
shap_explainer = None
ig_instance = None

# Input guardrails. These prevent accidental uploads and unbounded memory use.
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))
MIN_IMAGE_DIMENSION = int(os.environ.get("MIN_IMAGE_DIMENSION", 100))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}


def _validate_image(contents: bytes, content_type: Optional[str]) -> None:
    """Validate an uploaded image before it reaches the model pipeline."""
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large; maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB",
        )

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type; upload a JPEG or PNG chest X-ray",
        )

    try:
        # verify() checks the file structure without decoding the entire image.
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()

        # Reopen after verify(); verify() invalidates the image object.
        with Image.open(io.BytesIO(contents)) as image:
            if image.width < MIN_IMAGE_DIMENSION or image.height < MIN_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image dimensions must be at least {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels",
                )
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.info("Rejected invalid image upload: %s", exc)
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid JPEG or PNG image") from exc


# ── Startup ───────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global model, shap_explainer, ig_instance
    logger.info("Loading model on %s …", DEVICE)
    model = model_service.load_model(DEVICE)
    model.to(DEVICE)

    # Captum IG
    ig_instance = IntegratedGradients(model)

    logger.info("Ready. SHAP explainer will be built on first /api/explain call "
                "using uploaded background samples.")


def _build_shap_explainer(train_loader=None):
    """Build SHAP GradientExplainer. Called lazily or after model download."""
    global shap_explainer
    if shap_explainer is not None:
        return shap_explainer
    import shap
    # Use a blurred background as fallback
    bg = torch.zeros(1, 3, 224, 224, device=DEVICE)
    bg[0, 0] = 0.485; bg[0, 1] = 0.456; bg[0, 2] = 0.406
    shap_explainer = shap.GradientExplainer(model, bg)
    return shap_explainer


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    contents = await file.read()
    _validate_image(contents, file.content_type)
    tensor = model_service.preprocess_image(contents)
    result = model_service.predict(model, tensor, DEVICE)
    return JSONResponse(result)


@app.post("/api/explain")
async def explain(
    file: UploadFile = File(...),
    methods: str = Query("gradcam,gradcam_pp,lime,shap,ig"),
):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    contents = await file.read()
    _validate_image(contents, file.content_type)
    tensor = model_service.preprocess_image(contents)

    # Parse requested methods
    method_list = [m.strip() for m in methods.split(",") if m.strip()]

    # Ensure SHAP explainer exists
    if "shap" in method_list and shap_explainer is None:
        _build_shap_explainer()

    # Get prediction
    pred = model_service.predict(model, tensor, DEVICE)

    # Run XAI
    xai_results = xai_service.run_xai(
        model=model,
        input_tensor=tensor,
        methods=method_list,
        device=DEVICE,
        shap_explainer=shap_explainer,
        ig_instance=ig_instance,
    )

    return JSONResponse({
        "prediction": pred,
        "explanations": xai_results,
    })


@app.post("/api/setup/download-model")
async def download_model():
    """Download model from Google Drive."""
    try:
        path = model_service.download_model_from_drive()
        # Reload
        global model, ig_instance
        model = model_service.load_model(DEVICE)
        model.to(DEVICE)
        ig_instance = IntegratedGradients(model)
        return {"status": "ok", "model_path": path}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
