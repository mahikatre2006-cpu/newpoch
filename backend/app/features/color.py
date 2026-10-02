"""Colour helpers: WCAG contrast, LAB distance, palette and plain-language colour names."""
from __future__ import annotations

import warnings

import cv2
import numpy as np
from sklearn.cluster import KMeans


def _lin(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def relative_luminance(rgb) -> float:
    r, g, b = _lin(np.asarray(rgb, np.float64))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def wcag_contrast(rgb1, rgb2) -> float:
    """WCAG 2.x contrast ratio: 1 (identical) to 21 (black on white)."""
    l1, l2 = relative_luminance(rgb1), relative_luminance(rgb2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """Float LAB (L 0..100, a/b about -127..127) from uint8 RGB of any shape (..., 3)."""
    flat = rgb_u8.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).reshape(rgb_u8.shape)


def lab_distance(rgb_a, rgb_b) -> float:
    """CIE76 colour difference between two RGB colours. About 2.3 is a just-noticeable difference; 50+ is a different colour."""
    a, b = to_lab(np.asarray(rgb_a, np.uint8).reshape(1, 3)), to_lab(np.asarray(rgb_b, np.uint8).reshape(1, 3))
    return float(np.linalg.norm(a - b))


def colour_name(rgb) -> str:
    r, g, b = (np.asarray(rgb, np.float32) / 255.0).tolist()
    mx, mn = max(r, g, b), min(r, g, b)
    v, s = mx, 0.0 if mx == 0 else (mx - mn) / mx
    if v < 0.18:
        return "black"
    if s < 0.12:
        return "white" if v > 0.85 else "grey"
    d = mx - mn
    if mx == r:
        h = (60 * ((g - b) / d)) % 360
    elif mx == g:
        h = 60 * ((b - r) / d) + 120
    else:
        h = 60 * ((r - g) / d) + 240
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "brown" if v < 0.55 else "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 200:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def palette(rgb: np.ndarray, k: int = 5) -> list[dict]:
    """k-means in LAB on a downsampled frame: the k dominant colours with their share of the frame, biggest first."""
    small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
    lab = to_lab(small).reshape(-1, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a flat frame has fewer than k distinct colours: harmless
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=k) / len(lab)
    out = []
    for i in np.argsort(-counts):
        centre = km.cluster_centers_[i].reshape(1, 1, 3).astype(np.float32)
        srgb = np.clip(cv2.cvtColor(centre, cv2.COLOR_LAB2RGB).reshape(3) * 255, 0, 255).round().astype(int)
        out.append({"hex": "#%02X%02X%02X" % tuple(srgb), "name": colour_name(srgb), "share": round(float(counts[i]), 3)})
    return out
