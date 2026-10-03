"""M10. Verified fixes. Each fix edits the picture; the caller re-analyses the result and reports the attention change.

Pixel fixes work on the stored frame using the element masks of its analysis, so they apply to flat images and to
editor exports alike. Fixes that need to *move or restyle* a layer (clearing the safe zone, growing text) live in the
editor, which owns the layers; the server cannot restyle pixels it has flattened.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from app.ablation.core import blur_fill, labels_from_result
from app.features.core import text_colours

FIXES = {
    "focus_subject": "Focus subject",
    "fix_text": "Fix text legibility",
    "enhance": "Enhance",
}


class FixError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _kinds(result: dict, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Union masks by role: subject (incl. editor layers marked subject), face, text (incl. text layers)."""
    labels = labels_from_result(result, shape)
    out = {k: np.zeros(shape, bool) for k in ("subject", "face", "text")}
    for i, e in enumerate(result["elements"]):
        lt = (e.get("meta") or {}).get("layer_type")
        role = e["type"] if e["type"] in out else (lt if e["type"] == "layer" and lt in out else None)
        if role:
            out[role] |= labels == i
    return out


def _close(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def focus_subject(image: np.ndarray, result: dict, cfg: dict) -> np.ndarray:
    """Blur and darken everything that is not the subject (or a face, or text), and outline the subject."""
    fc = cfg["fixes"]["focus"]
    k = _kinds(result, image.shape[:2])
    focus = k["subject"] | k["face"]
    if not focus.any():
        raise FixError("No subject or face was found to focus on")
    protect = focus | k["text"]
    soft = cv2.GaussianBlur(protect.astype(np.float32), (0, 0), fc["feather_px"])[..., None]
    dimmed = cv2.GaussianBlur(image, (0, 0), fc["blur_sigma"]).astype(np.float32) * (1 - fc["darken"])
    out = image.astype(np.float32) * soft + dimmed * (1 - soft)
    solid = _close(focus, 9)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fc["outline_px"] + 1, 2 * fc["outline_px"] + 1))
    ring = cv2.dilate(solid.astype(np.uint8), ke).astype(bool) & ~solid & ~k["text"]
    a = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 0.8)[..., None]
    out = out * (1 - a) + 255.0 * a
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


def enhance(image: np.ndarray, result: dict, cfg: dict) -> np.ndarray:
    """CLAHE on the LAB lightness channel, then a little more saturation."""
    ec = cfg["fixes"]["enhance"]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=ec["clahe_clip"], tileGridSize=(ec["clahe_tiles"], ec["clahe_tiles"]))
    lab[..., 0] = clahe.apply(lab[..., 0])
    hsv = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * ec["saturation_gain"], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _glyphs(image: np.ndarray, mask: np.ndarray, box: list[int]) -> np.ndarray:
    """The letter pixels inside a text block: those far from the colour directly behind the text (a plate colour if it
    has one), not from whatever lies outside the block."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = max(0, ys.min() - 10), ys.max() + 11, max(0, xs.min() - 10), xs.max() + 11
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    colours = text_colours(image, mask, box, 6)
    if colours is None:
        return np.zeros_like(mask)
    win = image[y0:y1, x0:x1].astype(np.float32)
    dist = np.linalg.norm(win - colours[1], axis=2)
    inside = m.astype(bool)
    thr = max(35.0, 0.4 * float(np.percentile(dist[inside], 95)))
    g = (dist > thr) & inside
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = g
    return out


def fix_text(image: np.ndarray, result: dict, cfg: dict) -> np.ndarray:
    """Give low-contrast or phone-illegible text an outline in the opposite tone and a soft shadow.
    This cannot make letters bigger (the pixels are flat); in the editor the text layer's size is raised as well."""
    tc = cfg["fixes"]["text"]
    labels = labels_from_result(result, image.shape[:2])
    out = image.astype(np.float32)
    texts, failing = 0, 0
    for i, e in enumerate(result["elements"]):
        lt = (e.get("meta") or {}).get("layer_type")
        if not (e["type"] == "text" or (e["type"] == "layer" and lt == "text")):
            continue
        texts += 1
        f = e.get("features") or {}
        ratio, legible = f.get("text_contrast_ratio"), f.get("mobile_legible")
        if not ((ratio is not None and ratio < tc["min_contrast"]) or legible is False):
            continue
        glyph = (labels == i) if e["type"] == "layer" else _glyphs(image, labels == i, e["box"])  # a layer mask is already the letters
        if glyph.sum() < 30:
            continue
        failing += 1
        text_lum = float(np.median(image[glyph], axis=0) @ np.array([0.299, 0.587, 0.114])) / 255.0
        tone = 0.0 if text_lum > 0.5 else 255.0  # the opposite tone of the letters
        sh = np.roll(np.roll(glyph, tc["shadow_offset_px"], axis=0), tc["shadow_offset_px"], axis=1).astype(np.float32)
        sh = cv2.GaussianBlur(sh, (0, 0), tc["shadow_blur_px"])[..., None] * tc["shadow_opacity"]
        out = out * (1 - sh)  # a dark shadow
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tc["stroke_px"] + 1, 2 * tc["stroke_px"] + 1))
        ring = cv2.dilate(glyph.astype(np.uint8), ke).astype(bool) & ~glyph
        a = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 0.7)[..., None]
        out = out * (1 - a) + tone * a
        out[glyph] = image[glyph]  # the letters themselves stay exactly as they were
    if texts == 0:
        raise FixError("No text was found in this thumbnail")
    if failing == 0:
        raise FixError(f"All text already passes: contrast of at least {tc['min_contrast']}:1 and readable at phone size")
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


_FIX_FUNCS = {"focus_subject": focus_subject, "fix_text": fix_text, "enhance": enhance}


def apply_fix(fix: str, image: np.ndarray, result: dict, cfg: dict) -> np.ndarray:
    if fix not in _FIX_FUNCS:
        raise FixError(f"Unknown fix {fix!r}; choose one of {sorted(_FIX_FUNCS) + ['separate_layers']}", 400)
    if not result.get("elements"):
        raise FixError("This analysis has no elements to work with")
    return _FIX_FUNCS[fix](image, result, cfg)


def _png(rgba: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 4])
    return base64.b64encode(buf.tobytes()).decode()


def make_layers(image: np.ndarray, result: dict, cfg: dict) -> list[dict]:
    """Split a flat picture into a subject cut-out (transparent background) and the picture with the subject filled in
    from its surroundings, ready to open as two layers in the editor. Text stays part of the background picture."""
    cc = cfg["fixes"]["cutout"]
    h, w = image.shape[:2]
    k = _kinds(result, (h, w))
    focus = k["subject"] | k["face"]
    if not focus.any():
        raise FixError("No subject or face was found to cut out")
    solid = _close(focus, cc["close_px"])
    alpha = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), cc["feather_px"])
    ys, xs = np.nonzero(solid)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    rgba = np.dstack([image, (alpha * 255).astype(np.uint8)])[y0:y1, x0:x1]
    grown = cv2.dilate(solid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    background = blur_fill(image, grown, cfg["ablation"]["fill_sigmas"])
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(background, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return [
        {"name": "Background", "type": "image", "x": 0, "y": 0, "width": w, "height": h, "mime": "image/jpeg", "data": base64.b64encode(jpg.tobytes()).decode()},
        {"name": "Subject", "type": "subject", "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "mime": "image/png", "data": _png(rgba)},
    ]
