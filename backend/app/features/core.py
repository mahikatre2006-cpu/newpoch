"""M5. Visual feature analysis. Measured per element and for the frame; these are the only inputs explanations may use."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

import cv2
import numpy as np

from app.elements import ElementSet, is_text_like
from app.preprocess import Context
from app.saliency.render import COLORMAPS, normalise_for_display
from .color import colour_name, lab_distance, palette, to_lab, wcag_contrast


@dataclass
class FeatureResult:
    elements: dict[str, dict]  # element id -> features
    frame: dict
    layer_views: dict


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def text_recovery(original: str, found: str) -> float:
    """Share of the original string's characters found, in order, in what OCR read at phone size (0..1).
    Counting matching characters (not whole-string distance) keeps one stray extra word from failing a match."""
    a, b = _norm(original), _norm(found)
    if not a or not b:
        return 0.0
    matched = sum(blk.size for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return matched / len(a)


def _window(box: list[int], pad: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    x0, y0, x1, y1 = box
    h, w = shape
    return slice(max(0, y0 - pad), min(h, y1 + pad)), slice(max(0, x0 - pad), min(w, x1 + pad))


def ring_of(mask: np.ndarray, box: list[int], px: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Pixels within px of the element but outside it, evaluated only in a window around the element (fast)."""
    win = _window(box, px + 2, mask.shape)
    m = mask[win].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return (cv2.dilate(m, k).astype(bool) & ~m.astype(bool)), win


def local_contrast(gray: np.ndarray, mask: np.ndarray, box: list[int], px: int) -> float | None:
    """Luminance RMS difference between the element and the ring around it, 0..1 (1 = black against white)."""
    ring, win = ring_of(mask, box, px)
    inside = mask[win]
    if ring.sum() < 20 or inside.sum() < 20:
        return None
    g = gray[win].astype(np.float32)
    ring_mean = float(g[ring].mean())
    return float(np.sqrt(np.mean((g[inside] - ring_mean) ** 2)) / 255.0)


def text_colours(rgb: np.ndarray, mask: np.ndarray, box: list[int], px: int, rim_px: int = 3, glyph_mask: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    """(text colour, colour behind the text) for a text block.
    The colour behind the text is read from the rim just INSIDE the block's edge, so text on a coloured plate (white on a
    dark pill) is measured against the plate, not against whatever lies beyond it; with no usable rim it falls back to the
    ring just outside. With glyph_mask (an editor text layer, whose mask is the letter shapes themselves) the colour behind
    the text is the ring just outside the mask. The text colour is the median of the quarter of in-mask pixels furthest from it."""
    win = _window(box, px + 2, mask.shape)
    inside = mask[win]
    if inside.sum() < 10:
        return None
    pix = rgb[win]
    m8 = inside.astype(np.uint8)
    rim = inside & ~cv2.erode(m8, np.ones((2 * rim_px + 1, 2 * rim_px + 1), np.uint8)).astype(bool)
    if glyph_mask or rim.sum() < 20:
        rim, _ = ring_of(mask, box, px)
        if rim.sum() < 20:
            return None
    bg = np.median(pix[rim], axis=0)
    px_in = pix[inside].astype(np.float32)
    dist = np.linalg.norm(px_in - bg, axis=1)
    far = px_in[dist >= np.quantile(dist, 0.75)]
    return np.median(far, axis=0), bg


def text_contrast(rgb: np.ndarray, mask: np.ndarray, box: list[int], px: int, glyph_mask: bool = False) -> float | None:
    """WCAG ratio between the text colour and the colour directly behind it (see text_colours)."""
    c = text_colours(rgb, mask, box, px, glyph_mask=glyph_mask)
    return None if c is None else wcag_contrast(*c)


def focal_points(sal: np.ndarray, cfg: dict) -> list[dict]:
    """Peaks of the attention map after non-max suppression, strongest first, in frame pixels."""
    h, w = sal.shape
    ww = cfg["work_width"]
    s = ww / w
    small = cv2.resize(sal.astype(np.float32), (ww, max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.01 * ww)
    win = int(cfg["min_distance_frac"] * ww) | 1
    peak = small == cv2.dilate(small, np.ones((win, win), np.uint8))
    top = float(small.max())
    if top <= 0:
        return []
    keep = (peak & (small >= cfg["min_strength"] * top)).astype(np.uint8)
    n, lab, _, centroids = cv2.connectedComponentsWithStats(keep, connectivity=8)  # a flat-topped peak is one region, not many
    pts = sorted(((float(small[lab == i].max()) / top, float(centroids[i][0]), float(centroids[i][1])) for i in range(1, n)), reverse=True)
    out = []
    for strength, x, y in pts:
        if all(np.hypot(x - p["x"] * s, y - p["y"] * s) >= win / 2 for p in out):
            out.append({"x": int(round(x / s)), "y": int(round(y / s)), "strength": round(strength, 3)})
        if len(out) >= cfg["max_points"]:
            break
    return out


def contrast_png(gray: np.ndarray, sigma_frac: float) -> str:
    """Local contrast map |L - blur(L)|, drawn as a transparent overlay: brighter = a sharper change in luminance."""
    g = gray.astype(np.float32)
    d = np.abs(g - cv2.GaussianBlur(g, (0, 0), sigma_frac * g.shape[1]))
    d = cv2.resize(d, (640, 360), interpolation=cv2.INTER_AREA)
    n = np.clip(d / max(float(np.percentile(d, 99)), 1e-6), 0, 1)
    bgr = cv2.applyColorMap((n * 255).astype(np.uint8), COLORMAPS["viridis"])
    alpha = (np.clip((n - 0.08) / 0.5, 0, 1) * 215).astype(np.uint8)
    ok, png = cv2.imencode(".png", np.dstack([bgr, alpha]))
    return base64.b64encode(png.tobytes()).decode()


def compute_features(ctx: Context, eset: ElementSet, sal: np.ndarray | None, cfg: dict,
                     mobile_ocr: Callable[[np.ndarray], list] | None = None) -> FeatureResult:
    """`sal` may be None (saliency failed): attention-based frame features are then left out."""
    fc = cfg["features"]
    img = ctx.image
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    frame_sat = float(sat.mean())
    frame_mean = img.reshape(-1, 3).mean(axis=0)
    thirds = np.array([(w * i / 3, h * j / 3) for i in (1, 2) for j in (1, 2)])
    diag = float(np.hypot(w, h))

    # phone-size OCR, once, only if there is any text to check
    mobile_lines = []
    needs_mobile = any(is_text_like(e) and e.text for e in eset.elements)
    if mobile_ocr is not None and needs_mobile:
        # the 168 px render enlarged 3x: OCR cannot invent detail, it only has to see what a phone viewer sees
        up = cv2.resize(ctx.mobile_small, None, fx=fc["mobile_ocr_upscale"], fy=fc["mobile_ocr_upscale"], interpolation=cv2.INTER_CUBIC)
        sx, sy = w / up.shape[1], h / up.shape[0]
        for ln in mobile_ocr(up):
            x0, y0, x1, y1 = ln.box
            ln.box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
            mobile_lines.append(ln)

    per: dict[str, dict] = {}
    for i, e in enumerate(eset.elements):
        mask = eset.mask(i)
        f: dict = {}
        ys, xs = np.nonzero(mask)
        if xs.size:
            cx, cy = float(xs.mean()), float(ys.mean())
            f["centroid"] = [round(cx / w, 3), round(cy / h, 3)]
            f["dist_to_thirds"] = round(float(np.min(np.hypot(thirds[:, 0] - cx, thirds[:, 1] - cy))) / diag, 3)
            f["dist_to_center"] = round(float(np.hypot(cx - w / 2, cy - h / 2)) / (diag / 2), 3)
            f["safe_zone_overlap_pct"] = round(100.0 * float(ctx.safe_zone[mask].mean()), 1)
        if e.type != "background" and xs.size:
            lc = local_contrast(gray, mask, e.box, fc["ring_px"])
            f["local_contrast"] = None if lc is None else round(lc, 3)
            f["saturation_pop"] = round(float(sat[mask].mean()) - frame_sat, 3)
            f["colour_distinctness"] = round(lab_distance(img[mask].mean(axis=0).round(), frame_mean.round()), 1)
            f["mean_colour"] = "#%02X%02X%02X" % tuple(int(v) for v in img[mask].mean(axis=0).round())
            f["colour_name"] = colour_name(img[mask].mean(axis=0))
        if is_text_like(e) and xs.size:
            tc = text_contrast(img, mask, e.box, fc["text_ring_px"], glyph_mask=(e.type == "layer"))
            f["text_contrast_ratio"] = None if tc is None else round(tc, 2)
            if e.text:
                bx0, by0, bx1, by1 = e.box
                found = []
                for ln in sorted(mobile_lines, key=lambda l: (l.box[1], l.box[0])):  # reading order
                    lx0, ly0, lx1, ly1 = ln.box
                    line_area = max(1, (lx1 - lx0) * (ly1 - ly0))
                    ov = max(0, min(bx1, lx1) - max(bx0, lx0)) * max(0, min(by1, ly1) - max(by0, ly0))
                    block_area = max(1, (bx1 - bx0) * (by1 - by0))
                    # a phone-size line belongs to this block if it lies mostly inside it, or covers a good part of it
                    # (OCR at phone size may merge neighbouring blocks into one wide line; extra text does not hurt recovery)
                    if ov / line_area >= 0.5 or ov / block_area >= 0.3:
                        found.append(ln.text)
                score = text_recovery(e.text, " ".join(found))
                f["mobile_match_pct"] = round(100 * score, 0)
                f["mobile_legible"] = bool(score >= fc["mobile_match_min"])
            else:
                f["mobile_legible"] = None  # script OCR cannot read: legibility is not measured, not guessed
        per[e.id] = f

    # ---- frame ----
    frame: dict = {"palette": palette(img, fc["palette_k"])}
    text_area = sum(e.area_px for e in eset.elements if is_text_like(e))
    frame["text_area_pct"] = round(100.0 * text_area / (h * w), 1)
    views: dict = {}
    if sal is not None:
        ys_i, xs_i = np.mgrid[0:h, 0:w]
        cx = float((sal * xs_i).sum() / sal.sum())
        cy = float((sal * ys_i).sum() / sal.sum())
        frame["balance_offset"] = [round((cx / w - 0.5) * 2, 3), round((cy / h - 0.5) * 2, 3)]  # -1..1: + is right / bottom
        fp = focal_points(sal, fc["focal"])
        frame["focal_points"] = len(fp)
        frame["negative_space_pct"] = round(100.0 * float((sal < fc["negative_space_factor"] / sal.size).mean()), 1)
        grid = [[round(100.0 * float(sal[j * h // 3 : (j + 1) * h // 3, i * w // 3 : (i + 1) * w // 3].sum()), 1) for i in range(3)] for j in range(3)]
        views["composition"] = {"thirds_points": [[round(float(x)), round(float(y))] for x, y in thirds], "grid_attention_pct": grid,
                                "centre_of_mass": [round(cx), round(cy)], "focal_points": fp}
    views["text"] = [
        {"element_id": e.id, "box": e.box, "text": e.text, "contrast_ratio": per[e.id].get("text_contrast_ratio"),
         "mobile_legible": per[e.id].get("mobile_legible"), "safe_zone_overlap_pct": per[e.id].get("safe_zone_overlap_pct")}
        for e in eset.elements if is_text_like(e)]
    views["colour"] = {"palette": frame["palette"], "frame_saturation": round(frame_sat, 3)}
    views["contrast_png"] = contrast_png(gray, fc["contrast_map_sigma_frac"])
    sz = cfg["preprocess"]["safe_zone"]
    views["safe_zone"] = {"boxes": [[round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)] for x0, y0, x1, y1 in sz.values()]}
    return FeatureResult(per, frame, views)
