"""
main.py — ECG Potassium Analyzer — FastAPI Backend  v3.1
=========================================================
Model: best_efficientnetb4.pth  (PyTorch — same directory as this file)
Architecture: timm tf_efficientnet_b4_ns + 2-layer custom head

Endpoints:
  GET  /health
  POST /predict                  — Clean ECG mode
  POST /predict-12lead-single    — 12-lead mode (single image, all 3 leads)
  POST /reprocess                — Live threshold preview (clean / red-grid ECG)
  POST /reprocess-12lead-single  — Live threshold preview (single 12-lead image)
  POST /explain             ✨   — Explainable AI: Grad-CAM + ECG wave features

  [Legacy — kept for backward compat]
  POST /predict-12lead           — Old 3-separate-lead endpoint
  POST /reprocess-12lead         — Old 3-separate-lead reprocess

Run:
  pip install fastapi uvicorn[standard] pillow opencv-python-headless numpy torch torchvision timm
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
import logging
import traceback
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_utils import (
    load_model,
    full_preprocess,
    predict,
    bytes_to_pil,
    enhance_ecg_waveform,
    remove_ecg_grid,
    stack_and_resize_leads,
    detect_red_grid,
    detect_12lead,
    gradcam_explain,
    get_ecg_features,
    TARGET_W,
    TARGET_H,
    WAVEFORM_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="ECG Potassium Analyzer API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Server starting — loading model...")
    model = load_model()
    if model is None:
        logger.warning(
            "⚠️  Model not found — running in DEMO MODE.\n"
            "  • Place best_efficientnetb4.pth in the same folder as this file\n"
            "  • pip install timm  (required for EfficientNetB4ECG architecture)"
        )
    else:
        logger.info("✅  EfficientNetB4ECG (timm tf_efficientnet_b4_ns) ready.")


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _read_upload(upload: UploadFile, raw_bytes: bytes) -> Image.Image:
    allowed = {
        "image/jpeg", "image/jpg", "image/png",
        "image/bmp", "image/tiff", "image/webp",
    }
    ct = (upload.content_type or "").lower()
    if ct and ct not in allowed:
        raise HTTPException(400, f"Unsupported type '{ct}'. Please send PNG/JPEG/BMP/TIFF.")
    if not raw_bytes:
        raise HTTPException(400, "Empty file received.")
    try:
        return bytes_to_pil(raw_bytes)
    except Exception as exc:
        raise HTTPException(400, f"Could not open image: {exc}")


def _clamp_thresh(value: int) -> int:
    return max(40, min(130, value))


def _build_result_response(result: dict, **extra) -> dict:
    return {
        "verdict":         result["verdict"],
        "confidence_pct":  result["confidence_pct"],
        "prob_normal":     result["prob_normal"],
        "prob_hyper":      result["prob_hyper"],
        "prob_hypo":       result["prob_hypo"],
        "device":          result["device"],
        "entropy_pct":     result.get("entropy_pct", 0.0),
        **extra,
    }


def _preprocess_single_12lead(img: Image.Image, thresh: int):
    """
    Shared pipeline for single-image 12-lead endpoints.
    Returns: (inp_array, cleaned_pil)
    """
    enhanced    = enhance_ecg_waveform(img)
    cleaned_pil = remove_ecg_grid(enhanced, thresh)
    inp_array   = stack_and_resize_leads([cleaned_pil])
    return inp_array, cleaned_pil


# ─────────────────────────────────────────────────────────────────────────────
#  /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    model = load_model()
    return {
        "status":  "ok",
        "model":   "loaded" if model is not None else "demo",
        "backend": "EfficientNetB4ECG — timm tf_efficientnet_b4_ns (PyTorch)",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  /predict  — Clean ECG mode
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    waveform_threshold: int = Form(WAVEFORM_THRESHOLD),
):
    logger.info(f"/predict | threshold={waveform_threshold}")

    try:
        raw_bytes = await file.read()
    except Exception:
        raise HTTPException(400, "Failed to read uploaded file.")

    img    = _read_upload(file, raw_bytes)
    thresh = _clamp_thresh(waveform_threshold)

    try:
        inp_array, is_12lead, has_red_grid, is_blurry = full_preprocess(
            img, waveform_threshold=thresh
        )
    except Exception:
        logger.error(f"/predict preprocess:\n{traceback.format_exc()}")
        raise HTTPException(500, "Preprocessing failed. Please try with a cleaner ECG image.")

    processed_b64 = None
    try:
        if has_red_grid or is_12lead:
            enhanced  = enhance_ecg_waveform(img)
            proc_pil  = remove_ecg_grid(enhanced, thresh)
        else:
            proc_pil  = img
        processed_b64 = _pil_to_b64(proc_pil)
    except Exception:
        logger.warning(f"Preview build failed:\n{traceback.format_exc()}")

    try:
        result = predict(load_model(), inp_array)
    except Exception:
        logger.error(f"/predict inference:\n{traceback.format_exc()}")
        raise HTTPException(500, "Model inference failed.")

    logger.info(f"verdict={result['verdict']} conf={result['confidence_pct']}%")
    return _build_result_response(
        result,
        is_12lead       = is_12lead,
        has_red_grid    = has_red_grid,
        is_blurry       = is_blurry,
        processed_image = processed_b64,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  /predict-12lead-single
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict-12lead-single")
async def predict_12lead_single_endpoint(
    file: UploadFile = File(...),
    waveform_threshold: int = Form(WAVEFORM_THRESHOLD),
):
    logger.info(f"/predict-12lead-single | threshold={waveform_threshold}")

    try:
        raw_bytes = await file.read()
    except Exception:
        raise HTTPException(400, "Failed to read uploaded file.")

    img    = _read_upload(file, raw_bytes)
    thresh = _clamp_thresh(waveform_threshold)
    logger.info(f"  Image size: {img.size}")

    try:
        inp_array, cleaned_pil = _preprocess_single_12lead(img, thresh)
    except Exception:
        logger.error(f"/predict-12lead-single preprocess:\n{traceback.format_exc()}")
        raise HTTPException(500, "12-lead preprocessing failed.")

    processed_b64 = None
    try:
        recon_pil     = Image.fromarray(inp_array[0].astype(np.uint8))
        processed_b64 = _pil_to_b64(recon_pil)
    except Exception:
        logger.warning(f"12lead-single preview failed:\n{traceback.format_exc()}")

    try:
        result = predict(load_model(), inp_array)
    except Exception:
        logger.error(f"/predict-12lead-single inference:\n{traceback.format_exc()}")
        raise HTTPException(500, "Model inference failed.")

    logger.info(f"12lead-single verdict={result['verdict']} conf={result['confidence_pct']}%")
    return _build_result_response(
        result,
        is_12lead       = True,
        has_red_grid    = True,
        is_blurry       = False,
        processed_image = processed_b64,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  /reprocess  — Live threshold preview (Clean ECG)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/reprocess")
async def reprocess_endpoint(
    file: UploadFile = File(...),
    waveform_threshold: int = Form(WAVEFORM_THRESHOLD),
):
    logger.info(f"/reprocess | threshold={waveform_threshold}")

    try:
        raw_bytes = await file.read()
    except Exception:
        raise HTTPException(400, "Failed to read uploaded file.")

    img    = _read_upload(file, raw_bytes)
    thresh = _clamp_thresh(waveform_threshold)

    try:
        has_red_grid = detect_red_grid(img)
        is_12lead    = detect_12lead(img)

        if has_red_grid or is_12lead:
            enhanced  = enhance_ecg_waveform(img)
            proc_pil  = remove_ecg_grid(enhanced, thresh)
        else:
            proc_pil = img

        processed_b64 = _pil_to_b64(proc_pil)
    except Exception:
        logger.error(f"/reprocess:\n{traceback.format_exc()}")
        raise HTTPException(500, "Reprocessing failed.")

    return {"processed_image": processed_b64}


# ─────────────────────────────────────────────────────────────────────────────
#  /reprocess-12lead-single  — Live threshold preview (12-lead)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/reprocess-12lead-single")
async def reprocess_12lead_single_endpoint(
    file: UploadFile = File(...),
    waveform_threshold: int = Form(WAVEFORM_THRESHOLD),
):
    logger.info(f"/reprocess-12lead-single | threshold={waveform_threshold}")

    try:
        raw_bytes = await file.read()
    except Exception:
        raise HTTPException(400, "Failed to read uploaded file.")

    img    = _read_upload(file, raw_bytes)
    thresh = _clamp_thresh(waveform_threshold)

    try:
        inp_array, _  = _preprocess_single_12lead(img, thresh)
        recon_pil     = Image.fromarray(inp_array[0].astype(np.uint8))
        processed_b64 = _pil_to_b64(recon_pil)
    except Exception:
        logger.error(f"/reprocess-12lead-single:\n{traceback.format_exc()}")
        raise HTTPException(500, "12-lead single reprocessing failed.")

    return {"processed_image": processed_b64}


# ─────────────────────────────────────────────────────────────────────────────
#  /explain  ✨ NEW (v3.1) — Explainable AI: Grad-CAM + ECG wave features
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/explain")
async def explain_endpoint(
    file: UploadFile = File(...),
    waveform_threshold: int = Form(WAVEFORM_THRESHOLD),
    verdict: str = Form("Normal"),
    is_12lead: bool = Form(False),
):
    """
    Explainable AI endpoint. Call AFTER /predict or /predict-12lead-single
    with the same image file and threshold.

    Returns:
      heatmap_image — base64 PNG of Grad-CAM overlay on preprocessed ECG
                      (None in demo mode or when Grad-CAM fails)
      features      — list of ECG wave findings per verdict:
                      [{wave, finding, severity, detail}, ...]
      verdict       — echoed back for reference
    """
    logger.info(
        f"/explain | verdict={verdict!r} thresh={waveform_threshold} is_12lead={is_12lead}"
    )

    try:
        raw_bytes = await file.read()
    except Exception:
        raise HTTPException(400, "Failed to read uploaded file.")

    img    = _read_upload(file, raw_bytes)
    thresh = _clamp_thresh(waveform_threshold)

    try:
        if is_12lead:
            inp_array, _ = _preprocess_single_12lead(img, thresh)
        else:
            inp_array, _, _, _ = full_preprocess(img, waveform_threshold=thresh)

        overlay_pil, features = gradcam_explain(
            load_model(), inp_array, verdict=verdict
        )

        heatmap_b64 = _pil_to_b64(overlay_pil) if overlay_pil is not None else None

        return {
            "heatmap_image": heatmap_b64,
            "features":      features,
            "verdict":       verdict,
        }

    except Exception:
        logger.error(f"/explain:\n{traceback.format_exc()}")
        # Graceful fallback — always return rule-based features even on error
        return {
            "heatmap_image": None,
            "features":      get_ecg_features(verdict),
            "verdict":       verdict,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  LEGACY ENDPOINTS — kept for backward compat (v2.8 and earlier)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict-12lead")
async def predict_12lead_endpoint(
    lead1: UploadFile = File(...),
    lead2: UploadFile = File(...),
    lead3: UploadFile = File(...),
    waveform_threshold: int = Form(WAVEFORM_THRESHOLD),
):
    """Legacy: 3 separate lead images. Use /predict-12lead-single instead."""
    logger.info(f"/predict-12lead (legacy) | threshold={waveform_threshold}")
    thresh = _clamp_thresh(waveform_threshold)

    lead_imgs = []
    for idx, lead_file in enumerate([lead1, lead2, lead3], start=1):
        try:
            raw = await lead_file.read()
        except Exception:
            raise HTTPException(400, f"Failed to read lead {idx}.")
        lead_imgs.append(_read_upload(lead_file, raw))

    try:
        cleaned_leads = [remove_ecg_grid(enhance_ecg_waveform(p), thresh) for p in lead_imgs]
        inp_array     = stack_and_resize_leads(cleaned_leads)
    except Exception:
        logger.error(f"/predict-12lead preprocess:\n{traceback.format_exc()}")
        raise HTTPException(500, "Lead preprocessing failed.")

    processed_b64 = None
    try:
        recon_pil     = Image.fromarray(inp_array[0].astype(np.uint8))
        processed_b64 = _pil_to_b64(recon_pil)
    except Exception:
        pass

    try:
        result = predict(load_model(), inp_array)
    except Exception:
        logger.error(f"/predict-12lead inference:\n{traceback.format_exc()}")
        raise HTTPException(500, "Model inference failed.")

    return _build_result_response(
        result,
        is_12lead=True, has_red_grid=True, is_blurry=False,
        processed_image=processed_b64,
    )


@app.post("/reprocess-12lead")
async def reprocess_12lead_endpoint(
    lead1: UploadFile = File(...),
    lead2: UploadFile = File(...),
    lead3: UploadFile = File(...),
    waveform_threshold: int = Form(WAVEFORM_THRESHOLD),
):
    """Legacy: 3 separate lead images. Use /reprocess-12lead-single instead."""
    logger.info(f"/reprocess-12lead (legacy) | threshold={waveform_threshold}")
    thresh = _clamp_thresh(waveform_threshold)

    lead_imgs = []
    for idx, lead_file in enumerate([lead1, lead2, lead3], start=1):
        try:
            raw = await lead_file.read()
        except Exception:
            raise HTTPException(400, f"Failed to read lead {idx}.")
        lead_imgs.append(_read_upload(lead_file, raw))

    try:
        cleaned = [remove_ecg_grid(enhance_ecg_waveform(p), thresh) for p in lead_imgs]
        inp     = stack_and_resize_leads(cleaned)
        recon   = Image.fromarray(inp[0].astype(np.uint8))
        return {"processed_image": _pil_to_b64(recon)}
    except Exception:
        logger.error(f"/reprocess-12lead:\n{traceback.format_exc()}")
        raise HTTPException(500, "12-lead reprocessing failed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)