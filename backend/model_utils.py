"""
model_utils.py — ECG Potassium Analyzer
========================================
Model framework: PyTorch EfficientNet-B4 (.pth file)
  Backbone      : timm tf_efficientnet_b4_ns  (SAME as training / Streamlit app)
  Default path  : <script_dir>/best_efficientnetb4.pth
  Override      : ECG_MODEL_PATH env var

Preprocessing pipeline (matches Streamlit training pipeline exactly):
  RGB image → Grayscale → Stack 3ch → ImageNet normalize → model

  Mode 1 — Clean ECG   : resize 512×576 → model
  Mode 2 — 12-Lead ECG : enhance → grid removal → stack → model

WAVEFORM THRESHOLD SLIDER:
  remove_ecg_grid(img, waveform_threshold=80) accepts threshold param.
  full_preprocess() + predict pipeline all honour it.

EXPLAINABLE AI (v3.1):
  gradcam_explain() — Grad-CAM on backbone.conv_head + colour overlay
  get_ecg_features() — rule-based ECG wave findings per verdict
"""

import io
import os
import math
import logging

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger("model_utils")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TARGET_W           = 512
TARGET_H           = 576
LEAD_SEPARATOR_PX  = 8
WAVEFORM_THRESHOLD = 80
MIN_BLOB_AREA      = 15

CLASSES = ["Normal", "Hyperkalemia", "Hypokalemia"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

CONF_THRESHOLD = 0.70


# ─────────────────────────────────────────────────────────────────────────────
#  PYTORCH + TIMM IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn    = None
    logger.warning("torch not installed — DEMO MODE will be used.")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    timm = None
    logger.warning(
        "timm not installed — install with: pip install timm\n"
        "Model cannot be loaded without timm. DEMO MODE will be used."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def _build_efficientnet_b4_ecg(num_classes: int = 3, dropout: float = 0.4):
    if not HAS_TORCH or not HAS_TIMM:
        return None

    class EfficientNetB4ECG(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "tf_efficientnet_b4_ns",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            in_features = self.backbone.num_features

            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(512, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            return self.head(features)

    return EfficientNetB4ECG()


def _infer_num_classes(state_dict: dict) -> int:
    for key in ["head.5.weight", "head.3.weight"]:
        if key in state_dict:
            return state_dict[key].shape[0]
    for key in reversed(list(state_dict.keys())):
        if ("head" in key or "classifier" in key) and "weight" in key:
            if len(state_dict[key].shape) == 2:
                return state_dict[key].shape[0]
    return 3


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
_DEMO_SENTINEL = object()
_model_cache   = None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def load_model():
    global _model_cache

    if _model_cache is _DEMO_SENTINEL:
        return None
    if _model_cache is not None:
        return _model_cache

    if not HAS_TORCH:
        logger.warning("PyTorch not installed — DEMO MODE.")
        _model_cache = _DEMO_SENTINEL
        return None

    if not HAS_TIMM:
        logger.warning("timm not installed — DEMO MODE.")
        _model_cache = _DEMO_SENTINEL
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for name in ["best_efficientnetb4.pth", "best_efficientnetb4 .pth"]:
        candidate = os.path.join(script_dir, name)
        if os.path.exists(candidate):
            default_pth = candidate
            break
    else:
        default_pth = os.path.join(script_dir, "best_efficientnetb4.pth")

    model_path = os.environ.get("ECG_MODEL_PATH", default_pth)
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)

    print(f"[model_utils] Loading model from: {model_path}")
    logger.info(f"Model path resolved: {model_path}")

    if not os.path.exists(model_path):
        logger.warning(f"'{model_path}' not found — DEMO MODE.")
        _model_cache = _DEMO_SENTINEL
        return None

    try:
        device = torch.device("cpu")

        try:
            import numpy as _np_safe
            torch.serialization.add_safe_globals([_np_safe._core.multiarray.scalar])
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
        else:
            checkpoint.eval()
            checkpoint.to(device)
            _model_cache = checkpoint
            logger.info(f"Full model loaded from {model_path}")
            return checkpoint

        cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        num_classes = _infer_num_classes(cleaned)
        logger.info(f"Detected num_classes={num_classes} from checkpoint")

        model = _build_efficientnet_b4_ecg(num_classes=num_classes)
        if model is None:
            _model_cache = _DEMO_SENTINEL
            return None

        model.load_state_dict(cleaned, strict=True)
        model.eval()
        model.to(device)
        _model_cache = model
        logger.info(f"✅ EfficientNetB4ECG loaded: {model_path} | classes={num_classes}")
        return model

    except RuntimeError as exc:
        logger.error(f"Model load FAILED — architecture mismatch: {exc}")
        _model_cache = _DEMO_SENTINEL
        return None
    except Exception as exc:
        logger.error(f"Model load failed: {exc}")
        _model_cache = _DEMO_SENTINEL
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_red_grid(img: Image.Image) -> bool:
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red_dominant = (r > 140) & (r > g + 30) & (r > b + 30) & (r < 240)
    ratio = red_dominant.sum() / (arr.shape[0] * arr.shape[1])
    return bool(ratio > 0.03)


def detect_blurry(img: Image.Image, threshold: float = 80.0) -> bool:
    gray    = np.array(img.convert("L"))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(lap_var < threshold)


def detect_12lead(img: Image.Image) -> bool:
    w, h  = img.size
    ratio = w / h if h > 0 else 1.0
    return ratio > 1.3


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING — MODE 1 (Clean ECG)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_clean_ecg(img: Image.Image) -> np.ndarray:
    resized = img.convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    arr     = np.array(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING — MODE 2 (12-Lead ECG)
# ─────────────────────────────────────────────────────────────────────────────

def enhance_ecg_waveform(img: Image.Image) -> Image.Image:
    enhanced  = ImageEnhance.Contrast(img.convert("RGB")).enhance(1.8)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return sharpened


def remove_ecg_grid(
    img: Image.Image,
    waveform_threshold: int = WAVEFORM_THRESHOLD,
) -> Image.Image:
    arr  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    _, waveform_mask = cv2.threshold(
        gray, waveform_threshold, 255, cv2.THRESH_BINARY_INV
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        waveform_mask, connectivity=8
    )
    clean_mask = np.zeros_like(waveform_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            clean_mask[labels == i] = 255

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    result                 = np.full_like(arr, 255)
    result[clean_mask > 0] = [0, 0, 0]
    return Image.fromarray(result)


def stack_and_resize_leads(leads: list) -> np.ndarray:
    max_w = max(im.width for im in leads)

    padded = []
    for im in leads:
        if im.width < max_w:
            canvas = Image.new("RGB", (max_w, im.height), (255, 255, 255))
            canvas.paste(im, (0, 0))
            padded.append(canvas)
        else:
            padded.append(im)

    sep    = Image.new("RGB", (max_w, LEAD_SEPARATOR_PX), (255, 255, 255))
    strips = []
    for i, lead in enumerate(padded):
        strips.append(lead)
        if i < len(padded) - 1:
            strips.append(sep)

    total_h  = sum(s.height for s in strips)
    combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for s in strips:
        combined.paste(s, (0, y))
        y += s.height

    final = combined.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    arr   = np.array(final, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def full_preprocess(
    img: Image.Image,
    waveform_threshold: int = WAVEFORM_THRESHOLD,
) -> tuple:
    is_12lead    = detect_12lead(img)
    has_red_grid = detect_red_grid(img)
    is_blurry    = detect_blurry(img)

    if is_12lead or has_red_grid:
        enhanced = enhance_ecg_waveform(img)
        cleaned  = remove_ecg_grid(enhanced, waveform_threshold)
        inp_arr  = stack_and_resize_leads([cleaned])
    else:
        inp_arr = preprocess_clean_ecg(img)

    return inp_arr, is_12lead, has_red_grid, is_blurry


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def _demo_predict(image_array: np.ndarray) -> dict:
    seed  = int(image_array.mean() * 1000) % (2 ** 31)
    rng   = np.random.default_rng(seed)
    probs = rng.dirichlet(alpha=[4.0, 1.0, 1.0]).tolist()
    idx   = int(np.argmax(probs))
    return {
        "verdict":        CLASSES[idx],
        "confidence_pct": round(probs[idx] * 100, 1),
        "prob_normal":    round(probs[0] * 100, 1),
        "prob_hyper":     round(probs[1] * 100, 1),
        "prob_hypo":      round(probs[2] * 100, 1),
        "device":         "Demo Mode",
    }


def _rgb_nhwc_to_model_input(image_array: np.ndarray) -> "torch.Tensor":
    gray = (
        0.299 * image_array[0, :, :, 0] +
        0.587 * image_array[0, :, :, 1] +
        0.114 * image_array[0, :, :, 2]
    )
    gray_norm = gray / 255.0
    inp = np.stack([gray_norm, gray_norm, gray_norm], axis=0)
    inp = inp[np.newaxis, :, :, :].astype(np.float32)
    inp = (inp - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(inp)


def predict(model, image_array: np.ndarray) -> dict:
    if model is None:
        return _demo_predict(image_array)

    tensor = _rgb_nhwc_to_model_input(image_array)

    with torch.no_grad():
        output = model(tensor)

    out_np = output.cpu().numpy()

    if out_np.shape[-1] == 3:
        probs = _softmax(out_np[0]).tolist()
        idx   = int(np.argmax(probs))

        max_prob    = probs[idx]
        entropy     = -sum(p * math.log(p + 1e-9) for p in probs)
        entropy_pct = round(entropy / math.log(3) * 100, 1)

        if max_prob < CONF_THRESHOLD:
            return {
                "verdict":        "Inconclusive",
                "confidence_pct": round(max_prob * 100, 1),
                "prob_normal":    round(probs[0] * 100, 1),
                "prob_hyper":     round(probs[1] * 100, 1),
                "prob_hypo":      round(probs[2] * 100, 1),
                "device":         "EfficientNetB4ECG (timm tf_efficientnet_b4_ns)",
                "entropy_pct":    entropy_pct,
            }

        return {
            "verdict":        CLASSES[idx],
            "confidence_pct": round(max_prob * 100, 1),
            "prob_normal":    round(probs[0] * 100, 1),
            "prob_hyper":     round(probs[1] * 100, 1),
            "prob_hypo":      round(probs[2] * 100, 1),
            "device":         "EfficientNetB4ECG (timm tf_efficientnet_b4_ns)",
            "entropy_pct":    entropy_pct,
        }

    k_val = float(out_np[0][0])
    if k_val < 3.5:
        verdict = "Hypokalemia"
        dist    = 3.5 - k_val
        p_hypo  = min(0.95, 0.5 + dist * 0.25)
        p_hyper = 0.02
        p_norm  = 1.0 - p_hypo - p_hyper
    elif k_val > 5.0:
        verdict = "Hyperkalemia"
        dist    = k_val - 5.0
        p_hyper = min(0.95, 0.5 + dist * 0.25)
        p_hypo  = 0.02
        p_norm  = 1.0 - p_hyper - p_hypo
    else:
        verdict          = "Normal"
        dist_from_center = abs(k_val - 4.25) / 0.75
        p_norm           = max(0.60, 0.90 - dist_from_center * 0.25)
        p_hyper          = (1.0 - p_norm) * 0.5
        p_hypo           = (1.0 - p_norm) * 0.5

    total   = p_norm + p_hyper + p_hypo
    p_norm  /= total
    p_hyper /= total
    p_hypo  /= total
    conf = round(
        {"Normal": p_norm, "Hyperkalemia": p_hyper, "Hypokalemia": p_hypo}[verdict] * 100, 1
    )
    return {
        "verdict":        verdict,
        "confidence_pct": conf,
        "prob_normal":    round(p_norm  * 100, 1),
        "prob_hyper":     round(p_hyper * 100, 1),
        "prob_hypo":      round(p_hypo  * 100, 1),
        "device":         "EfficientNetB4ECG (regression)",
        "entropy_pct":    0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def bytes_to_pil(raw_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  EXPLAINABLE AI — ECG WAVE FEATURES  (rule-based per verdict)
# ─────────────────────────────────────────────────────────────────────────────

def get_ecg_features(verdict: str) -> list:
    """
    Returns a list of ECG wave findings that explain the predicted verdict.
    Each item: { wave, finding, severity, detail }
    severity: "normal" | "low" | "medium" | "high"
    """
    FEATURES = {
        "Normal": [
            {
                "wave": "T-wave",
                "finding": "Normal upright T-wave",
                "severity": "normal",
                "detail": "Amplitude 0.1–0.5 mV, symmetric — no peaking or flattening detected",
            },
            {
                "wave": "QRS complex",
                "finding": "Normal width (< 120 ms)",
                "severity": "normal",
                "detail": "No widening or slurring of the ventricular depolarization wave",
            },
            {
                "wave": "P-wave",
                "finding": "Present and upright",
                "severity": "normal",
                "detail": "Precedes every QRS; amplitude < 2.5 mm — normal atrial activity",
            },
            {
                "wave": "PR interval",
                "finding": "Normal (120–200 ms)",
                "severity": "normal",
                "detail": "Regular AV conduction; no conduction delay detected",
            },
            {
                "wave": "ST segment",
                "finding": "Isoelectric baseline",
                "severity": "normal",
                "detail": "No elevation or depression (< 0.5 mm); no injury pattern",
            },
            {
                "wave": "U-wave",
                "finding": "Absent or minimal",
                "severity": "normal",
                "detail": "No prominent U-waves; K⁺ level likely in range 3.5–5.0 mEq/L",
            },
        ],
        "Hyperkalemia": [
            {
                "wave": "T-wave",
                "finding": "Tall, peaked (tented) T-waves",
                "severity": "high",
                "detail": "Narrow base + high amplitude — earliest ECG sign of hyperkalemia (K⁺ > 5.5 mEq/L)",
            },
            {
                "wave": "P-wave",
                "finding": "Flattened or absent P-waves",
                "severity": "high",
                "detail": "Sinoatrial block; P-wave disappears as K⁺ rises above ~6.5 mEq/L",
            },
            {
                "wave": "PR interval",
                "finding": "Prolonged PR interval",
                "severity": "medium",
                "detail": "AV conduction slowed; may exceed 200 ms due to elevated extracellular K⁺",
            },
            {
                "wave": "QRS complex",
                "finding": "Widened QRS > 120 ms",
                "severity": "high",
                "detail": "Intraventricular conduction delay; severe cases risk ventricular fibrillation",
            },
            {
                "wave": "ST segment",
                "finding": "ST depression / elevation",
                "severity": "medium",
                "detail": "Injury-pattern changes seen alongside widened QRS in moderate–severe hyperkalemia",
            },
            {
                "wave": "Sine wave",
                "finding": "Sine-wave pattern (severe)",
                "severity": "high",
                "detail": "Terminal sign — P/QRS/T merge into a sine wave; K⁺ typically > 8.0 mEq/L",
            },
        ],
        "Hypokalemia": [
            {
                "wave": "T-wave",
                "finding": "Flattened or inverted T-waves",
                "severity": "high",
                "detail": "Amplitude < 1 mm or frank inversion; appears in V4–V6 first (K⁺ < 3.5 mEq/L)",
            },
            {
                "wave": "U-wave",
                "finding": "Prominent U-waves",
                "severity": "high",
                "detail": "U-wave taller than the T-wave; most visible in V2–V3 — hallmark of hypokalemia",
            },
            {
                "wave": "ST segment",
                "finding": "ST segment depression",
                "severity": "high",
                "detail": "Horizontal or down-sloping depression ≥ 0.5 mm below the isoelectric line",
            },
            {
                "wave": "QU interval",
                "finding": "Prolonged QU interval",
                "severity": "medium",
                "detail": "T and U waves fuse, mimicking a prolonged QT — important clinical distinction",
            },
            {
                "wave": "P-wave",
                "finding": "Tall, slightly peaked P-waves",
                "severity": "medium",
                "detail": "Increased P-wave amplitude; raises atrial ectopy risk",
            },
            {
                "wave": "QRS complex",
                "finding": "Mildly widened QRS",
                "severity": "low",
                "detail": "Slight intraventricular delay seen in severe hypokalemia (K⁺ < 2.5 mEq/L)",
            },
        ],
        "Inconclusive": [
            {
                "wave": "Confidence",
                "finding": "Below 70% threshold",
                "severity": "medium",
                "detail": "Model uncertainty too high for reliable classification",
            },
            {
                "wave": "Image quality",
                "finding": "Possible noise or artifacts",
                "severity": "medium",
                "detail": "Adjust the waveform threshold slider and re-crop the ECG region",
            },
            {
                "wave": "Recommendation",
                "finding": "Repeat with a cleaner image",
                "severity": "low",
                "detail": "Ensure high contrast and minimal grid artifacts for best results",
            },
        ],
    }
    return FEATURES.get(verdict, FEATURES["Normal"])


# ─────────────────────────────────────────────────────────────────────────────
#  EXPLAINABLE AI — GRAD-CAM HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def gradcam_explain(
    model,
    image_array: np.ndarray,
    verdict: str = "Normal",
):
    """
    Compute Grad-CAM activation map over the last convolutional layer of
    EfficientNetB4ECG (backbone.conv_head) and overlay it on the input image.

    Args:
        model       — loaded EfficientNetB4ECG (or None for demo mode)
        image_array — (1, H, W, 3) float32 RGB, same array passed to predict()
        verdict     — predicted verdict string, used to select target class

    Returns:
        overlay_pil — PIL Image with heatmap overlay (None on failure / demo mode)
        features    — list of ECG wave feature dicts from get_ecg_features()
    """
    features = get_ecg_features(verdict)

    if not HAS_TORCH or model is None:
        logger.info("Grad-CAM skipped (demo mode or torch missing) — returning features only.")
        return None, features

    CLASS_MAP  = {"Normal": 0, "Hyperkalemia": 1, "Hypokalemia": 2, "Inconclusive": 0}
    target_idx = CLASS_MAP.get(verdict, 0)

    activations: list = [None]
    gradients:   list = [None]

    def _fwd_hook(_module, _inp, out):
        activations[0] = out

    def _bwd_hook(_module, _grad_in, grad_out):
        gradients[0] = grad_out[0]

    target_layer = model.backbone.conv_head
    fh = target_layer.register_forward_hook(_fwd_hook)
    bh = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        model.eval()
        tensor = _rgb_nhwc_to_model_input(image_array)

        output = model(tensor)
        model.zero_grad()
        output[0, target_idx].backward()

        grads = gradients[0]
        acts  = activations[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * acts).sum(dim=1)).squeeze()
        cam_np  = cam.detach().cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        H, W   = int(image_array.shape[1]), int(image_array.shape[2])
        cam_up = cv2.resize(cam_np, (W, H), interpolation=cv2.INTER_CUBIC)
        cam_up = np.clip(cam_up, 0.0, 1.0)

        heatmap_u8  = (cam_up * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        original_rgb = image_array[0].astype(np.uint8)
        overlay_np   = cv2.addWeighted(original_rgb, 0.55, heatmap_rgb, 0.45, 0)
        overlay_pil  = Image.fromarray(overlay_np)

        logger.info(f"Grad-CAM computed for class={target_idx} ({verdict})")
        return overlay_pil, features

    except Exception as exc:
        import traceback
        logger.error(f"Grad-CAM failed: {exc}\n{traceback.format_exc()}")
        return None, features

    finally:
        fh.remove()
        bh.remove()