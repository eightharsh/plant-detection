"""
Plant Disease Detection — v3 (M1 Pro Optimized)
================================================
New in v3:
  • Multi-scale TTA: inference at 512 + 768 per pass
  • Horizontal flip TTA per scale
  • Gray-world white balance before inference
  • GrabCut + HSV leaf mask — background set to YOLO pad color (114,114,114)
  • Annotation drawn from merged list — NO second inference pass
  • Image size guard: warns at 5 MB, hard cap 10 MB, resize >4096px
  • Leaf content sanity check (green pixel ratio)
  • Class-specific thresholds from model/class_thresholds.json
  • Robust label parser: snake_case, CamelCase, PlantVillage, custom
  • Differential diagnosis: all detected diseases ranked
  • Wikipedia quality gate: suppresses off-topic genus articles
  • Session history: last 5 uploads with thumbnail + result
  • Normalized bbox in JSON export (stable across preprocessing)
  • Time-to-action per severity level
  • Full MPS→CPU fallback covering ALL TTA passes
"""

import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
WIKIPEDIA_UA       = "PlantDiseaseDetection/3.0 (Streamlit; hrajnir@gmail.com)"
DEFAULT_MODEL_PATH = "model/best.pt"
CLASS_THRESH_PATH  = "model/class_thresholds.json"
MAX_FILE_BYTES     = 10 * 1024 * 1024
WARN_FILE_BYTES    = 5  * 1024 * 1024
MAX_DIM_PX         = 4096
MIN_GREEN_RATIO    = 0.08
SESSION_HISTORY_N  = 5
TTA_SIZES          = [512, 768]   # M1 Pro handles both in ~1-2s total on MPS

SEVERITY_THRESHOLDS = {"mild": 0.28, "moderate": 0.52, "severe": 0.76}

SEVERITY_ACTION: Dict[str, str] = {
    "none":     "No action needed. Re-inspect in 1-2 weeks.",
    "mild":     "Monitor daily. Apply preventive treatment within 5-7 days if spreading.",
    "moderate": "Intervene within 24-48 hours. Begin treatment and remove infected tissue.",
    "severe":   "Act immediately. Remove and bag infected plants. Apply treatment today.",
    "critical": "Quarantine area now. Consider full removal to protect neighboring plants.",
}

SEVERITY_COLORS: Dict[str, Tuple[str, str]] = {
    "none":     ("success", "No disease detected"),
    "mild":     ("warning", "Mild infection"),
    "moderate": ("warning", "Moderate infection"),
    "severe":   ("error",   "Severe infection"),
    "critical": ("error",   "Critical — crop loss risk"),
}

# ══════════════════════════════════════════════════════════════════════════════
# Disease care sheet
# ══════════════════════════════════════════════════════════════════════════════
DiseaseInfo = Dict[str, Union[str, List[str]]]

DISEASE_CARE_SHEET: Dict[str, DiseaseInfo] = {
    "late blight": {
        "type": "Oomycete — Phytophthora infestans",
        "severity_note": "Rapidly destructive; can destroy a field in days.",
        "symptoms": [
            "Large dark brown/black lesions, water-soaked at margins",
            "White fuzzy sporulation on leaf undersides in humid conditions",
            "Stems and tubers also affected; dark firm rot",
        ],
        "management": [
            "Remove and bag infected tissue immediately; do not compost",
            "Switch to drip irrigation; avoid wetting foliage",
            "Apply copper-based or mancozeb fungicide preventively",
            "Plant resistant varieties; rotate crops >= 3 years",
        ],
    },
    "early blight": {
        "type": "Fungal — Alternaria solani",
        "severity_note": "Usually manageable; causes premature defoliation if ignored.",
        "symptoms": [
            "Brown lesions with concentric rings (target board appearance)",
            "Yellow halo around lesions; lower/older leaves first",
            "Defoliation exposes fruit to sunscald",
        ],
        "management": [
            "Remove infected lower leaves; sanitize tools with 10% bleach",
            "Mulch soil surface to prevent spore splash",
            "Chlorothalonil or copper sprays at first sign",
            "Rotate crops; avoid solanaceous crops in same bed",
        ],
    },
    "septoria leaf spot": {
        "type": "Fungal — Septoria lycopersici",
        "severity_note": "Primary cause of premature tomato defoliation in wet seasons.",
        "symptoms": [
            "Small circular spots (3-5 mm) with dark border, pale tan center",
            "Tiny black pycnidia visible in spot centers",
            "Starts on oldest leaves; moves upward through canopy",
        ],
        "management": [
            "Remove infected leaves; bag and dispose (not compost)",
            "Use drip irrigation; water in morning so foliage dries quickly",
            "Copper or mancozeb fungicide at first symptoms",
            "2-3 year rotation; clean up all debris post-harvest",
        ],
    },
    "bacterial spot": {
        "type": "Bacterial — Xanthomonas campestris pv. vesicatoria",
        "severity_note": "Spreads rapidly in warm wet weather.",
        "symptoms": [
            "Small water-soaked spots becoming dark, angular, greasy-looking",
            "Spots dry, turn brown, may drop out (shot-hole)",
            "Fruit develops raised, scabby lesions",
        ],
        "management": [
            "Use certified pathogen-free seed; avoid overhead irrigation",
            "Copper bactericides at 7-10 day intervals in high pressure",
            "Remove infected parts; do not work plants when wet",
        ],
    },
    "mosaic virus": {
        "type": "Viral — TMV / CMV (multiple strains)",
        "severity_note": "No chemical cure; infected plants are permanent reservoirs.",
        "symptoms": [
            "Mottled light/dark-green mosaic pattern on leaves",
            "Leaf distortion, puckering, or shoe-string narrowing",
            "Stunted growth; fruit may be discolored",
        ],
        "management": [
            "Remove and destroy infected plants immediately",
            "Control aphid vectors with reflective mulches or insecticidal soap",
            "Disinfect tools; use certified virus-free transplants",
        ],
    },
    "yellow virus": {
        "type": "Viral — TYLCV / Begomovirus (whitefly-transmitted)",
        "severity_note": "Can devastate entire crops when whitefly pressure is high.",
        "symptoms": [
            "Chlorotic leaf margins and interveinal yellowing",
            "Leaf curling upward; stunted shoot growth",
            "Blossom drop; minimal fruit set",
        ],
        "management": [
            "Remove infected plants immediately",
            "Install yellow sticky traps; use insect-proof mesh",
            "Plant TYLCV-resistant varieties",
        ],
    },
    "powdery mildew": {
        "type": "Fungal (obligate biotrophic — multiple genera)",
        "severity_note": "Favors dry heat + high humidity at night.",
        "symptoms": [
            "White powdery coating on upper leaf surfaces",
            "Yellowing and drying; premature defoliation",
        ],
        "management": [
            "Improve airflow; avoid high-nitrogen fertilization",
            "Apply potassium bicarbonate, neem oil, or sulfur at first sign",
            "Remove heavily infected leaves; use resistant varieties",
        ],
    },
    "downy mildew": {
        "type": "Oomycete — Peronospora / Plasmopara spp.",
        "severity_note": "Requires cool wet conditions; often confused with powdery mildew.",
        "symptoms": [
            "Yellow to pale green angular lesions on upper leaf surface",
            "Gray-purple downy growth on leaf underside (key diagnostic)",
            "Rapid collapse in humid cool weather",
        ],
        "management": [
            "Avoid overhead watering; improve drainage and spacing",
            "Apply copper or phosphonate fungicides preventively",
            "Remove infected tissue; rotate crops",
        ],
    },
    "black rot": {
        "type": "Fungal — Guignardia bidwellii (grapes) / Alternaria (brassicas)",
        "severity_note": "Major grape disease; entire clusters can be lost.",
        "symptoms": [
            "Small tan lesions with dark margins on leaves",
            "Berries shrivel into hard black mummies",
        ],
        "management": [
            "Remove mummified berries and infected canes before bud break",
            "Prune for maximum airflow",
            "Apply fungicides from bud break through veraison",
        ],
    },
    "rust": {
        "type": "Fungal — Puccinia / Phakopsora spp.",
        "severity_note": "Spreads rapidly via airborne spores.",
        "symptoms": [
            "Orange/yellow/reddish-brown pustules on leaf undersides",
            "Yellow flecks on upper surface; premature defoliation",
        ],
        "management": [
            "Remove infected leaves immediately; do not compost",
            "Apply triazole or strobilurin fungicides early",
            "Resistant varieties strongly preferred",
        ],
    },
    "gray leaf spot": {
        "type": "Fungal — Cercospora zeae-maydis",
        "severity_note": "Major yield-limiting disease of corn in humid regions.",
        "symptoms": [
            "Long narrow rectangular tan-gray lesions aligned between veins",
            "Lesions coalesce under humid conditions",
        ],
        "management": [
            "Rotate crops; till to break down infected residue",
            "Plant resistant hybrids",
            "Foliar fungicide at VT/R1 in high-risk situations",
        ],
    },
    "northern corn leaf blight": {
        "type": "Fungal — Exserohilum turcicum",
        "severity_note": "Can cause 50%+ yield loss in severe epidemics.",
        "symptoms": [
            "Large (5-15 cm) cigar-shaped grayish-green lesions",
            "Lesions turn tan; dark sporulation visible",
        ],
        "management": [
            "Resistant hybrids (Ht1, Ht2, Htn1 genes)",
            "Fungicide at tasseling in high-pressure years",
            "Crop rotation and residue management",
        ],
    },
    "botrytis gray mold": {
        "type": "Fungal — Botrytis cinerea",
        "severity_note": "Especially severe under cool humid conditions.",
        "symptoms": [
            "Fluffy gray sporulation on infected tissue (key diagnostic)",
            "Brown water-soaked lesions on stems, flowers, fruit",
        ],
        "management": [
            "Remove spent flowers and debris immediately",
            "Keep relative humidity <85%; ventilate greenhouses",
            "Apply iprodione, fenhexamid, or Bacillus subtilis biofungicide",
        ],
    },
    "anthracnose": {
        "type": "Fungal — Colletotrichum spp.",
        "severity_note": "Warm wet conditions trigger epidemics.",
        "symptoms": [
            "Sunken circular dark lesions on fruit; salmon-pink spore masses",
            "Leaf spots with dark borders; shoot dieback",
        ],
        "management": [
            "Avoid overhead irrigation; harvest and cool promptly",
            "Remove and destroy infected fruit",
            "Copper or azoxystrobin-based fungicides preventively",
        ],
    },
    "fire blight": {
        "type": "Bacterial — Erwinia amylovora",
        "severity_note": "Most destructive bacterial disease of apple and pear.",
        "symptoms": [
            "Rapid browning and wilting of shoot tips (shepherd's crook)",
            "Infected tissue looks fire-scorched",
            "Amber bacterial ooze from cankers",
        ],
        "management": [
            "Prune 30 cm below visible symptoms; sterilize tools between cuts",
            "Apply copper bactericide at silver tip",
            "Plant resistant rootstocks (Geneva series)",
        ],
    },
    "fusarium wilt": {
        "type": "Fungal — Fusarium oxysporum f. sp.",
        "severity_note": "Soil-borne; persists indefinitely once established.",
        "symptoms": [
            "One-sided yellowing, starting with lower leaves",
            "Brown vascular tissue (diagnostic cut-stem test)",
            "Wilting during heat even with adequate soil moisture",
        ],
        "management": [
            "Plant Fusarium-resistant varieties",
            "Solarize soil; improve drainage; maintain pH 6.5-7.0",
            "Use grafted transplants on resistant rootstocks",
        ],
    },
    "scab": {
        "type": "Fungal — Venturia inaequalis (apple) / Cladosporium cucumerinum",
        "severity_note": "Most economically important disease of apples globally.",
        "symptoms": [
            "Olive-green velvety lesions on leaves and fruit",
            "Fruit lesions become dark, corky, cracked",
        ],
        "management": [
            "Apply sulfur or captan from green tip through petal fall",
            "Rake and destroy fallen leaves",
            "Plant scab-resistant varieties (Enterprise, Liberty, Redfree)",
        ],
    },
    "two spotted spider mites": {
        "type": "Pest — Tetranychus urticae",
        "severity_note": "Worst in hot dry conditions; populations explode in 1-2 weeks.",
        "symptoms": [
            "Fine stippling/bronzing on upper leaf surface",
            "Yellowing, browning, and leaf drop in severe cases",
            "Fine silk webbing on undersides",
        ],
        "management": [
            "Rinse plants with strong water spray",
            "Release predatory mites (Phytoseiidae)",
            "Apply insecticidal soap or neem oil",
        ],
    },
    "sooty mold": {
        "type": "Fungal (secondary — grows on insect honeydew)",
        "severity_note": "Indicates active pest infestation; not directly pathogenic.",
        "symptoms": [
            "Black crusty coating on leaf upper surfaces",
            "Traces to aphids, scale, or whiteflies above",
        ],
        "management": [
            "Control the honeydew-producing insect (primary action)",
            "Wash affected leaves with mild soapy water",
            "Horticultural oil smothers insects and softens mold",
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Streamlit page config + session state
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════════════════════════════════════════
# Model + config loading
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed.")
    return YOLO(weights_path)

@st.cache_data(show_spinner=False)
def load_class_thresholds(path: str) -> Dict[str, float]:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# ══════════════════════════════════════════════════════════════════════════════
# Image validation + sizing
# ══════════════════════════════════════════════════════════════════════════════
def check_file_size(uploaded_file) -> Optional[str]:
    """Returns 'error:msg' or 'warn:msg' or None."""
    sz = uploaded_file.size
    if sz > MAX_FILE_BYTES:
        return f"error:File is {sz/1e6:.1f} MB — max allowed is {MAX_FILE_BYTES/1e6:.0f} MB."
    if sz > WARN_FILE_BYTES:
        return f"warn:Large file ({sz/1e6:.1f} MB) — processing may be slower."
    return None

def to_bgr_uint8(pil_img: Image.Image) -> np.ndarray:
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cap_image_size(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIM_PX:
        return bgr
    scale = MAX_DIM_PX / longest
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def check_leaf_content(bgr: np.ndarray) -> float:
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 35, 35), (90, 255, 255))
    return float(mask.sum() / 255) / float(bgr.shape[0] * bgr.shape[1])

# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Scale each BGR channel so its mean equals overall mean."""
    f   = bgr.astype(np.float32)
    b, g, r = cv2.split(f)
    mu  = (b.mean() + g.mean() + r.mean()) / 3.0
    return cv2.merge([
        np.clip(b * (mu / (b.mean() + 1e-6)), 0, 255).astype(np.uint8),
        np.clip(g * (mu / (g.mean() + 1e-6)), 0, 255).astype(np.uint8),
        np.clip(r * (mu / (r.mean() + 1e-6)), 0, 255).astype(np.uint8),
    ])

def apply_clahe(bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def sharpen_image(bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (0, 0), 3)
    return cv2.addWeighted(bgr, 1 + strength, blurred, -strength, 0)

def build_leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """
    HSV green threshold + morphology + GrabCut.
    Returns binary mask (255 = leaf, 0 = background).
    """
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Wide range: includes yellowing diseased leaves
    green = cv2.inRange(hsv, (18, 18, 25), (105, 255, 255))
    kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kern, iterations=3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kern, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(bgr.shape[:2], 255, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    pad = 20
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(bgr.shape[1], x + w + pad);  y2 = min(bgr.shape[0], y + h + pad)

    try:
        gc  = np.zeros(bgr.shape[:2], np.uint8)
        bgm = np.zeros((1, 65), np.float64)
        fgm = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, gc, (x1, y1, x2 - x1, y2 - y1), bgm, fgm, 3, cv2.GC_INIT_WITH_RECT)
        final = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if final.sum() / 255 < 200:
            final = clean
    except Exception:
        final = clean

    return final

def apply_leaf_mask_to_image(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace background with YOLO's default padding color (114, 114, 114)."""
    out = bgr.copy()
    bg  = np.full_like(bgr, 114)
    inv = cv2.bitwise_not(mask)
    out[inv > 0] = bg[inv > 0]
    return out

def assess_quality(bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness":  float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast":   float(gray.std()),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Inference — multi-scale TTA, full MPS fallback
# ══════════════════════════════════════════════════════════════════════════════
def safe_cls_name(names: Dict[int, str], cls_id: int) -> str:
    try:
        return names[cls_id]
    except (KeyError, AttributeError):
        return str(cls_id)

def _infer_one(model, bgr: np.ndarray, imgsz: int, conf: float, device: str) -> List[Dict]:
    results = model.predict(bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)
    dets = []
    if not results or not results[0].boxes:
        return dets
    for b in results[0].boxes:
        try:
            cls_id = int(b.cls[0])
            conf_v = float(b.conf[0])
            label  = safe_cls_name(model.names, cls_id)
            xyxy   = b.xyxy[0].tolist()
            xyn    = b.xyxyn[0].tolist()   # normalized — use in JSON export
            dets.append({
                "label": label, "cls_id": cls_id,
                "confidence": round(conf_v, 5),
                "bbox_xyxy": xyxy,
                "bbox_norm": xyn,
            })
        except Exception:
            continue
    return dets

def _run_tta(model, bgr: np.ndarray, sizes: List[int],
             conf: float, device: str, use_flip: bool) -> List[Dict]:
    """
    Multi-scale TTA. All passes use the same device.
    Raises on failure so caller can switch device.
    """
    h, w  = bgr.shape[:2]
    all_d: List[Dict] = []

    for sz in sizes:
        all_d.extend(_infer_one(model, bgr, sz, conf, device))
        if use_flip:
            flipped   = cv2.flip(bgr, 1)
            flip_dets = _infer_one(model, flipped, sz, conf, device)
            for d in flip_dets:
                if d.get("bbox_norm"):
                    x1n, y1n, x2n, y2n = d["bbox_norm"]
                    d["bbox_norm"]  = [1 - x2n, y1n, 1 - x1n, y2n]
                if d.get("bbox_xyxy"):
                    x1, y1, x2, y2 = d["bbox_xyxy"]
                    d["bbox_xyxy"] = [w - x2, y1, w - x1, y2]
            all_d.extend(flip_dets)

    return all_d

def run_inference(model, bgr: np.ndarray, sizes: List[int],
                  conf: float, device: str, use_flip: bool) -> Tuple[List[Dict], str]:
    """
    Returns (detections, actual_device_used).
    Automatically falls back MPS → CPU on any error.
    """
    try:
        return _run_tta(model, bgr, sizes, conf, device, use_flip), device
    except Exception:
        if device == "mps":
            st.warning("MPS failed — retrying on CPU…")
            try:
                return _run_tta(model, bgr, sizes, conf, "cpu", use_flip), "cpu"
            except Exception as e2:
                raise RuntimeError(f"Inference failed on both MPS and CPU: {e2}") from e2
        raise

# ══════════════════════════════════════════════════════════════════════════════
# Confidence thresholding
# ══════════════════════════════════════════════════════════════════════════════
def get_class_threshold(label: str, global_thresh: float, cls_thresholds: Dict[str, float]) -> float:
    if not cls_thresholds:
        return global_thresh
    low = label.lower()
    for k, v in cls_thresholds.items():
        if k.lower() == low or k.lower() in low or low in k.lower():
            return float(v)
    return global_thresh

def auto_threshold(dets: List[Dict], base: float, quality: Dict[str, float]) -> float:
    if not dets:
        return base
    confs = [float(d["confidence"]) for d in dets]
    q75   = float(np.percentile(confs, 75))
    sharp = quality.get("sharpness", 200)
    qf    = max(0.45, min(1.0, sharp / 280.0))
    any_diseased = any(
        parse_label_fields(normalize_label(d["label"]))["status"] == "diseased"
        for d in dets
    )
    if any_diseased:
        return round(max(0.12, q75 * 0.48 * qf), 4)
    return round(max(0.30, q75 * 0.72 * qf), 4)

# ══════════════════════════════════════════════════════════════════════════════
# Label parsing
# ══════════════════════════════════════════════════════════════════════════════
DISEASE_KW = frozenset([
    "blight", "mildew", "rust", "spot", "rot", "scab", "virus", "mosaic", "wilt",
    "canker", "anthracnose", "septoria", "bacterial", "fungal", "powdery", "downy",
    "mold", "mould", "mites", "spider", "yellow", "gray", "grey", "black", "curl",
    "gall", "edema", "sooty", "cercospora", "alternaria", "botrytis", "fusarium",
    "verticillium", "fire", "crown", "clubroot", "smut", "damping", "necrosis",
    "chlorosis", "lesion", "infected", "blossom", "end",
])
HEALTHY_KW = frozenset([
    "healthy", "normal", "no_disease", "no disease", "clean", "uninfected", "good",
])

def normalize_label(label: str) -> str:
    """
    Convert any naming convention to space-separated words.
    PlantVillage: Tomato___Late_blight → Tomato | Late blight
    snake_case:   tomato_late_blight   → tomato late blight
    CamelCase:    TomatoLateBlight     → Tomato Late Blight
    """
    if not label:
        return ""
    s = label.replace("___", " | ").replace("__", " ").replace("_", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return " ".join(s.split()).strip()

def parse_label_fields(norm: str) -> Dict[str, str]:
    """Parse normalized label into {crop, disease, status}."""
    s   = norm.strip()
    if not s:
        return {"crop": "", "disease": "", "status": "unknown"}
    low = s.lower()
    tokens   = s.split()
    low_tok  = [t.lower() for t in tokens]

    # Explicit healthy check (no disease keyword present)
    if any(h in low for h in HEALTHY_KW) and not any(d in low for d in DISEASE_KW):
        crop = " ".join(t for t in tokens if t.lower() not in HEALTHY_KW).strip()
        return {"crop": crop or s, "disease": "", "status": "healthy"}

    is_diseased = any(d in low for d in DISEASE_KW)

    # PlantVillage pipe separator: "Tomato | Late blight"
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        if len(parts) == 2:
            cp, dp = parts[0], parts[1]
            if any(d in dp.lower() for d in DISEASE_KW):
                return {"crop": cp, "disease": dp, "status": "diseased"}
            if any(h in dp.lower() for h in HEALTHY_KW):
                return {"crop": cp, "disease": "", "status": "healthy"}

    # "leaf" anchor
    if "leaf" in low_tok:
        idx    = low_tok.index("leaf")
        before = tokens[:idx]
        after  = tokens[idx + 1:]
        if is_diseased:
            if after:
                return {"crop": " ".join(before), "disease": " ".join(after), "status": "diseased"}
            elif len(before) >= 2:
                return {"crop": before[0], "disease": " ".join(before[1:]), "status": "diseased"}
            else:
                return {"crop": " ".join(before), "disease": "Diseased", "status": "diseased"}
        else:
            return {"crop": " ".join(before), "disease": "", "status": "healthy"}

    if is_diseased:
        return {
            "crop":    tokens[0] if tokens else "Unknown",
            "disease": " ".join(tokens[1:]) if len(tokens) > 1 else "Diseased",
            "status":  "diseased",
        }

    return {"crop": s, "disease": "", "status": "unknown"}

# ══════════════════════════════════════════════════════════════════════════════
# NMS + aggregation + primary selection
# ══════════════════════════════════════════════════════════════════════════════
def _area(xyxy: Optional[List[float]]) -> float:
    if not xyxy or len(xyxy) != 4:
        return 0.0
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _iou(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return float(inter / union) if union > 0 else 0.0

def weighted_nms(rows: List[Dict], iou_thres: float) -> List[Dict]:
    """
    Status-aware NMS: diseased boxes only compete with other diseased boxes.
    Prevents a confident 'healthy' box from suppressing a real disease detection.
    """
    if not rows:
        return []
    groups: Dict[str, List] = {}
    for r in rows:
        groups.setdefault(r.get("status", "unknown"), []).append(r)

    out: List[Dict] = []
    for grp in groups.values():
        grp.sort(key=lambda x: -float(x.get("confidence", 0)))
        used = [False] * len(grp)
        for i, ri in enumerate(grp):
            if used[i]:
                continue
            used[i] = True
            m = dict(ri)
            mc = 1
            for j in range(i + 1, len(grp)):
                if not used[j] and _iou(m.get("bbox_xyxy"), grp[j].get("bbox_xyxy")) >= iou_thres:
                    used[j] = True
                    mc += 1
            m["_merged"] = mc
            out.append(m)

    out.sort(key=lambda x: (x.get("status") != "diseased", -float(x.get("confidence", 0))))
    return out

def aggregate_by_label(rows: List[Dict]) -> List[Dict]:
    by: Dict[str, Dict] = {}
    for r in rows:
        lbl = r.get("label", "")
        if not lbl:
            continue
        conf = float(r.get("confidence", 0))
        if lbl not in by:
            by[lbl] = {
                "label": lbl, "crop": r.get("crop", ""), "disease": r.get("disease", ""),
                "status": r.get("status", "unknown"), "count": 0, "max_conf": 0.0, "sum_conf": 0.0,
            }
        by[lbl]["count"]   += 1
        by[lbl]["sum_conf"] += conf
        by[lbl]["max_conf"]  = max(by[lbl]["max_conf"], conf)

    out = []
    for v in by.values():
        c = v["count"]
        out.append({
            "label": v["label"], "crop": v["crop"], "disease": v["disease"],
            "status": v["status"], "count": c,
            "max_conf": round(v["max_conf"], 4),
            "avg_conf": round(v["sum_conf"] / c, 4) if c else 0.0,
        })
    out.sort(key=lambda x: -x["max_conf"])
    return out

def choose_primary(rows: List[Dict], mode: str) -> Optional[Dict]:
    if not rows:
        return None
    def score(r):
        c = float(r.get("confidence", 0))
        return c * max(1.0, _area(r.get("bbox_xyxy"))) if mode == "confidence x area" else c
    diseased = [r for r in rows if r.get("status") == "diseased"]
    healthy  = [r for r in rows if r.get("status") == "healthy"]
    if diseased:
        return max(diseased, key=score)
    if healthy:
        return max(healthy, key=score)
    return max(rows, key=score)

# ══════════════════════════════════════════════════════════════════════════════
# Severity
# ══════════════════════════════════════════════════════════════════════════════
def estimate_severity(rows: List[Dict], img_area: int) -> Tuple[str, float]:
    diseased = [r for r in rows if r.get("status") == "diseased"]
    if not diseased:
        return "none", 0.0
    avg_conf  = float(np.mean([float(d.get("confidence", 0)) for d in diseased]))
    box_area  = sum(_area(d.get("bbox_xyxy")) for d in diseased)
    coverage  = min(1.0, box_area / max(img_area, 1))
    score     = round(float(0.55 * avg_conf + 0.45 * coverage), 3)
    if score < SEVERITY_THRESHOLDS["mild"]:     label = "mild"
    elif score < SEVERITY_THRESHOLDS["moderate"]: label = "moderate"
    elif score < SEVERITY_THRESHOLDS["severe"]:   label = "severe"
    else:                                           label = "critical"
    return label, score

# ══════════════════════════════════════════════════════════════════════════════
# Annotation — drawn from merged list (no second model.predict call)
# ══════════════════════════════════════════════════════════════════════════════
_STATUS_BGR = {
    "diseased": (0, 50, 220),
    "healthy":  (20, 180, 20),
    "unknown":  (160, 100, 0),
}

def draw_detections(bgr: np.ndarray, rows: List[Dict]) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    fscale = max(0.38, min(0.75, w / 1300))
    thick  = max(1, int(w / 650))
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for r in rows:
        bb = r.get("bbox_xyxy")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color  = _STATUS_BGR.get(r.get("status", "unknown"), (160, 100, 0))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)

        dname = r.get("disease") or r.get("label", "")
        text  = f"{dname} {float(r.get('confidence', 0)):.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
        ly1 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, ly1), (x1 + tw + 6, ly1 + th + 6), color, -1)
        cv2.putText(out, text, (x1 + 3, ly1 + th + 2),
                    font, fscale, (255, 255, 255), thick, cv2.LINE_AA)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Care sheet lookup (fuzzy)
# ══════════════════════════════════════════════════════════════════════════════
def lookup_care(disease: str) -> Optional[DiseaseInfo]:
    if not disease:
        return None
    low = disease.lower().strip()
    if low in DISEASE_CARE_SHEET:
        return DISEASE_CARE_SHEET[low]
    for k, v in DISEASE_CARE_SHEET.items():
        if k in low or low in k:
            return v
    dw = set(low.split())
    best_sc, best_v = 0, None
    for k, v in DISEASE_CARE_SHEET.items():
        sc = len(dw & set(k.split()))
        if sc > best_sc:
            best_sc, best_v = sc, v
    return best_v if best_sc >= 1 else None

# ══════════════════════════════════════════════════════════════════════════════
# Wikipedia (with quality gate)
# ══════════════════════════════════════════════════════════════════════════════
def build_wiki_query(primary: Dict, mode: str) -> str:
    crop    = primary.get("crop", "")
    disease = primary.get("disease", "")
    status  = primary.get("status", "unknown")
    if status != "diseased":
        return crop
    if mode == "disease-only":
        return disease or crop
    return f"{crop} {disease}".strip() if crop and disease else (disease or crop)

def _wiki_from_title(title: str, lang: str) -> Dict[str, Any]:
    headers = {"User-Agent": WIKIPEDIA_UA, "Accept": "application/json"}
    try:
        r = requests.get(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
            headers=headers, timeout=8,
        )
        r.raise_for_status()
        rj      = r.json() or {}
        extract = rj.get("extract", "")
        url     = ((rj.get("content_urls") or {}).get("desktop", {}).get("page")
                   or f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}")
        if not extract:
            return {"error": f"Empty extract for '{title}'"}
        return {"title": title, "extract": extract, "url": url}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False, ttl=86400)
def _wiki_search(query: str, lang: str) -> Dict[str, Any]:
    headers = {"User-Agent": WIKIPEDIA_UA, "Accept": "application/json"}
    try:
        sr = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
            headers=headers, timeout=8,
        )
        sr.raise_for_status()
        hits = sr.json().get("query", {}).get("search", [])
        if hits:
            return _wiki_from_title(hits[0]["title"], lang)
        return {"error": f"No results for '{query}'"}
    except Exception as e:
        return {"error": str(e)}

def _wiki_quality_ok(result: Dict, query: str) -> bool:
    if "error" in result:
        return False
    title_words = set(result.get("title", "").lower().split())
    query_words = set(query.lower().split())
    stopwords   = {"the", "a", "an", "of", "and", "in", "on", "plant", "leaf"}
    return bool((title_words & query_words) - stopwords)

def fetch_wiki(query: str, lang: str) -> Dict[str, Any]:
    for attempt in [query, f"{query} plant disease", f"{query} disease"]:
        res = _wiki_search(attempt, lang)
        if _wiki_quality_ok(res, query):
            return res
    return {"error": f"No relevant Wikipedia article found for '{query}'"}

# ══════════════════════════════════════════════════════════════════════════════
# Session history
# ══════════════════════════════════════════════════════════════════════════════
def update_history(rgb: np.ndarray, primary: Optional[Dict], severity: str) -> None:
    thumb = cv2.resize(rgb, (80, 60))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 70])
    entry = {
        "thumb":    buf.tobytes() if ok else b"",
        "label":    primary["label"] if primary else "—",
        "conf":     f"{float(primary['confidence'])*100:.1f}%" if primary else "—",
        "severity": severity,
        "ts":       time.strftime("%H:%M:%S"),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:SESSION_HISTORY_N]

def render_history() -> None:
    if not st.session_state.history:
        st.caption("No uploads yet this session.")
        return
    icons = {"none": "✅", "mild": "⚠️", "moderate": "⚠️", "severe": "🚨", "critical": "🚨"}
    for i, h in enumerate(st.session_state.history):
        cols = st.columns([1, 3])
        with cols[0]:
            if h["thumb"]:
                st.image(h["thumb"], width=80)
        with cols[1]:
            st.write(f"**{h['label']}** {icons.get(h['severity'], '')}")
            st.caption(f"{h['conf']} · {h['ts']}")
        if i < len(st.session_state.history) - 1:
            st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.title("🌱 Plant Disease Detection")
    st.caption("Upload a plant leaf image to detect and diagnose disease.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        model_path = st.text_input("Model weights path", DEFAULT_MODEL_PATH)
        device     = st.selectbox("Device", ["mps", "cpu"], index=0,
                                  help="mps = Apple Silicon GPU — ~3-5x faster than CPU on M1 Pro")
        st.info("M1 Pro tip: MPS is recommended. CoreML export (`yolo export format=coreml`) can give extra speedup.", icon="🍎")

        st.divider()
        st.subheader("Inference")
        use_multiscale = st.checkbox("Multi-scale TTA (512 + 768)", True,
                                     help="Two resolutions per pass. Better at small lesions.")
        use_flip       = st.checkbox("Horizontal flip TTA", True,
                                     help="Each scale also run flipped.")
        iou_thres      = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)

        st.divider()
        st.subheader("Confidence")
        auto_conf   = st.checkbox("Auto-threshold", True)
        manual_conf = st.slider("Baseline / manual threshold", 0.05, 0.95, 0.12, 0.01)

        st.divider()
        st.subheader("Preprocessing")
        white_bal     = st.checkbox("Gray-world white balance", True)
        preproc       = st.selectbox("Contrast enhancement",
                                     ["None", "CLAHE", "CLAHE + Sharpen", "Autocontrast"], index=1)
        leaf_mask_on  = st.checkbox("Leaf background masking", True,
                                    help="Isolates leaf from background; suppresses false positives.")

        st.divider()
        st.subheader("Display")
        show_boxes = st.checkbox("Show bounding boxes", True)
        show_table = st.checkbox("Show predictions table", True)
        show_debug = st.checkbox("Debug columns", False)
        score_mode = st.selectbox("Primary scoring", ["confidence", "confidence x area"], index=1)

        st.divider()
        st.subheader("Diagnosis")
        show_diag    = st.checkbox("Show diagnosis", True)
        show_sev     = st.checkbox("Show severity + action", True)
        show_diff_dx = st.checkbox("Differential diagnosis", True)
        wiki_lang    = st.text_input("Wikipedia language", "en")
        wiki_mode    = st.selectbox("Wiki query mode", ["disease-only", "crop + disease"], index=1)
        wiki_chars   = st.slider("Wiki excerpt length", 200, 2000, 900, 50)

        st.divider()
        export_json = st.checkbox("Enable JSON export", True)

        st.divider()
        st.subheader("Session History")
        render_history()

    # ── Guards ────────────────────────────────────────────────────────────────
    if not YOLO_AVAILABLE:
        st.error("ultralytics is not installed. Run: `pip install ultralytics`")
        return

    if not os.path.exists(model_path):
        st.error(f"Model not found at `{model_path}`.")
        return

    with st.spinner("Loading model…"):
        try:
            model        = load_model(model_path)
            cls_thresholds = load_class_thresholds(CLASS_THRESH_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    if cls_thresholds:
        st.sidebar.success(f"Per-class thresholds: {len(cls_thresholds)} classes")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    )
    if not uploaded:
        st.info("Upload a leaf image to get started.")
        return

    sz_check = check_file_size(uploaded)
    if sz_check:
        kind, msg = sz_check.split(":", 1)
        if kind == "error":
            st.error(msg); return
        st.warning(msg)

    try:
        pil_img = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not open image: {e}"); return

    # ── Prepare image ─────────────────────────────────────────────────────────
    img_bgr_orig = to_bgr_uint8(pil_img)
    img_bgr_orig = cap_image_size(img_bgr_orig)
    orig_h, orig_w = img_bgr_orig.shape[:2]
    img_area = orig_h * orig_w

    green_ratio = check_leaf_content(img_bgr_orig)
    if green_ratio < MIN_GREEN_RATIO:
        st.warning(
            f"Only {green_ratio*100:.1f}% green pixels detected. "
            "This may not be a leaf — results could be unreliable."
        )

    # Preprocessing pipeline
    img_bgr = img_bgr_orig.copy()
    if white_bal:
        img_bgr = gray_world_white_balance(img_bgr)
    if preproc == "CLAHE":
        img_bgr = apply_clahe(img_bgr)
    elif preproc == "CLAHE + Sharpen":
        img_bgr = apply_clahe(img_bgr)
        img_bgr = sharpen_image(img_bgr)
    elif preproc == "Autocontrast":
        pil_tmp = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil_tmp = ImageOps.autocontrast(pil_tmp)
        img_bgr = cv2.cvtColor(np.array(pil_tmp), cv2.COLOR_BGR2RGB)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    leaf_mask = None
    if leaf_mask_on:
        with st.spinner("Building leaf mask…"):
            leaf_mask     = build_leaf_mask(img_bgr)
            img_bgr_infer = apply_leaf_mask_to_image(img_bgr, leaf_mask)
    else:
        img_bgr_infer = img_bgr

    quality = assess_quality(img_bgr)

    # ── Layout ────────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Input")
        st.image(cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB), use_container_width=True)
        with st.expander("Image metrics"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpness", f"{quality['sharpness']:.0f}")
            c2.metric("Brightness", f"{quality['brightness']:.0f}")
            c3.metric("Green %", f"{green_ratio*100:.1f}%")
            if quality["sharpness"] < 60:
                st.warning("Blurry image — results may be less reliable.")
            if not (40 < quality["brightness"] < 220):
                st.warning("Poor exposure — consider retaking.")
        if leaf_mask is not None:
            with st.expander("Leaf mask"):
                st.image(cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2RGB),
                         caption="White = leaf region used for inference",
                         use_container_width=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    sizes     = TTA_SIZES if use_multiscale else [TTA_SIZES[-1]]
    n_passes  = len(sizes) * (2 if use_flip else 1)
    est_s     = n_passes * (0.7 if device == "mps" else 2.2)

    with st.spinner(f"Running inference ({n_passes} passes, ~{est_s:.0f}s on {device.upper()})…"):
        t0 = time.time()
        try:
            raw_dets, used_device = run_inference(
                model, img_bgr_infer, sizes, float(manual_conf), device, use_flip
            )
        except RuntimeError as e:
            st.error(str(e)); return
        elapsed = time.time() - t0

    st.sidebar.caption(f"Inference: {elapsed:.2f}s on {used_device.upper()}")

    # ── Threshold + filter ────────────────────────────────────────────────────
    global_thresh = auto_threshold(raw_dets, float(manual_conf), quality) if auto_conf else float(manual_conf)
    if auto_conf:
        st.sidebar.info(f"Auto-threshold: {global_thresh:.4f}")

    filtered: List[Dict] = []
    for d in raw_dets:
        t = get_class_threshold(d.get("label", ""), global_thresh, cls_thresholds)
        if float(d.get("confidence", 0)) >= t:
            filtered.append(d)

    # ── Enrich + NMS + aggregate ──────────────────────────────────────────────
    enriched: List[Dict] = []
    for d in filtered:
        norm   = normalize_label(d.get("label", ""))
        fields = parse_label_fields(norm)
        enriched.append({**d, "label_norm": norm,
                         "crop": fields["crop"], "disease": fields["disease"],
                         "status": fields["status"],
                         "bbox_norm_orig": d.get("bbox_norm")})

    merged  = weighted_nms(enriched, iou_thres)
    agg     = aggregate_by_label(merged)
    primary = choose_primary(merged, score_mode)

    sev_label, sev_score = estimate_severity(merged, img_area)
    update_history(cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB), primary, sev_label)

    # ── Right column ──────────────────────────────────────────────────────────
    with right_col:
        # Annotated image — NO second model call
        if show_boxes and merged:
            annotated     = draw_detections(img_bgr, merged)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.subheader("Detections")
            st.image(annotated_rgb, use_container_width=True)
            ok, buf = cv2.imencode(".png", annotated)
            if ok:
                st.download_button("Download annotated image",
                                   buf.tobytes(), "detections.png", "image/png")
        elif not merged:
            st.info("No detections above threshold.")

        # Table
        if show_table and merged:
            st.subheader("Predictions")
            cols = ["label", "crop", "disease", "status", "confidence"]
            if show_debug:
                cols += ["label_norm", "_merged"]
            st.dataframe([{k: r.get(k) for k in cols if k in r} for r in merged],
                         hide_index=True)
            with st.expander("Summary by label"):
                st.dataframe(agg, hide_index=True)

        if not merged:
            if export_json:
                _export([], [], None, sev_label, sev_score, quality,
                        global_thresh, sizes, use_flip, preproc, orig_w, orig_h)
            return

        if primary:
            pct = float(primary["confidence"]) * 100
            st.caption(f"Primary: **{primary['label']}** — {pct:.1f}%")

        # Severity banner
        if show_sev and primary:
            sc, sm = SEVERITY_COLORS.get(sev_label, ("info", sev_label))
            getattr(st, sc)(
                f"**{sm}** (score: {sev_score:.2f})  \n"
                f"**Action:** {SEVERITY_ACTION.get(sev_label, '')}"
            )

        # Diagnosis
        if show_diag and primary:
            st.subheader("Diagnosis")
            pct = float(primary["confidence"]) * 100

            if primary["status"] == "healthy":
                st.success(
                    f"**Healthy:** {primary['crop']} ({pct:.1f}%)  \n"
                    "No disease markers found. Re-inspect in 1-2 weeks."
                )
            elif primary["status"] == "diseased":
                st.error(
                    f"**Disease:** {primary['disease']}  \n"
                    f"**Crop:** {primary['crop']}  \n"
                    f"**Confidence:** {pct:.1f}%"
                )
                sheet = lookup_care(primary["disease"])
                if sheet:
                    st.markdown("### Care Sheet")
                    st.write(f"**Pathogen type:** {sheet['type']}")
                    if "severity_note" in sheet:
                        st.info(sheet["severity_note"])
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Symptoms**")
                        for s in sheet["symptoms"]:
                            st.write(f"• {s}")
                    with c2:
                        st.write("**Management**")
                        for m in sheet["management"]:
                            st.write(f"• {m}")
                else:
                    st.info("No local care sheet found. See Wikipedia below.")
            else:
                st.info(f"**Result:** {primary['label']} ({pct:.1f}%)")

            # Differential diagnosis
            if show_diff_dx:
                all_dis = [r for r in merged if r.get("status") == "diseased"]
                if len(all_dis) > 1:
                    st.markdown("### Differential Diagnosis")
                    st.caption(f"{len(all_dis)} disease candidates:")
                    for i, r in enumerate(all_dis[:5]):
                        pct_d = float(r["confidence"]) * 100
                        bar   = "█" * int(pct_d / 10) + "░" * (10 - int(pct_d / 10))
                        st.write(
                            f"**{i+1}. {r.get('disease') or r['label']}** "
                            f"({r.get('crop', '')}) — {pct_d:.1f}%  \n`{bar}`"
                        )

            # Wikipedia
            wq = build_wiki_query(primary, wiki_mode)
            if wq:
                with st.spinner("Fetching Wikipedia…"):
                    wiki = fetch_wiki(wq, wiki_lang)
                if "error" not in wiki:
                    st.markdown(f"### {wiki['title']}")
                    ext = wiki["extract"]
                    if len(ext) > wiki_chars:
                        ext = ext[:wiki_chars].rsplit(" ", 1)[0] + "…"
                    st.write(ext)
                    st.link_button("Read full article", wiki["url"])
                else:
                    st.caption(f"Wikipedia: {wiki['error']}")

        # Export
        if export_json:
            _export(merged, agg, primary, sev_label, sev_score, quality,
                    global_thresh, sizes, use_flip, preproc, orig_w, orig_h)


def _export(merged, agg, primary, sev_label, sev_score, quality,
            thresh, sizes, use_flip, preproc, orig_w, orig_h) -> None:
    def clean_row(r):
        cr = {k: v for k, v in r.items() if k not in ("bbox_xyxy", "bbox_norm")}
        cr["bbox_normalized"] = r.get("bbox_norm_orig") or r.get("bbox_norm")
        return cr

    data = {
        "predictions": [clean_row(r) for r in (merged or [])],
        "summary":     agg or [],
        "primary":     ({k: v for k, v in primary.items() if k not in ("bbox_xyxy", "bbox_norm")}
                        if primary else None),
        "severity":    {"label": sev_label, "score": sev_score},
        "action":      SEVERITY_ACTION.get(sev_label, ""),
        "image":       {"width": orig_w, "height": orig_h},
        "quality":     quality,
        "settings":    {
            "threshold":   thresh,
            "tta_sizes":   sizes,
            "tta_flip":    use_flip,
            "preproc":     preproc,
        },
    }
    st.download_button(
        "Export JSON",
        json.dumps(data, indent=2, default=str),
        "results.json",
        "application/json",
    )


if __name__ == "__main__":
    main()
