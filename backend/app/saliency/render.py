"""Turn a saliency map into the images the client draws: a transparent heatmap overlay and the frame preview."""
from __future__ import annotations

import base64

import cv2
import numpy as np

COLORMAPS = {"inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS}


def normalise_for_display(m: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """0..1 for colouring only (a probability map's values are tiny). The 99.8th percentile keeps one hot pixel
    from darkening everything else; gamma < 1 lifts the tail so secondary regions are visible."""
    hi = float(np.percentile(m, 99.8))
    return np.clip(m / hi, 0, 1) ** gamma if hi > 0 else np.zeros_like(m)


def encode_heatmap_png(m: np.ndarray, render_cfg: dict) -> str:
    """Transparent RGBA PNG: cold areas are fully transparent so the thumbnail shows through. Inferno is
    perceptually uniform and readable with common colour-vision deficiencies."""
    size = tuple(render_cfg["heatmap_size"])
    d = cv2.resize(normalise_for_display(m, render_cfg.get("display_gamma", 1.0)), size, interpolation=cv2.INTER_AREA)
    bgr = cv2.applyColorMap((d * 255).astype(np.uint8), COLORMAPS[render_cfg["colormap"]])
    alpha = np.clip((d - render_cfg["low_cut"]) / 0.5, 0, 1) * 215
    ok, png = cv2.imencode(".png", np.dstack([bgr, alpha.astype(np.uint8)]))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode()


def encode_preview_jpg(rgb: np.ndarray, render_cfg: dict, quality: int = 85) -> str:
    """The normalised frame the models saw, so the overlay lines up with exactly the same pixels."""
    small = cv2.resize(rgb, tuple(render_cfg["preview_size"]), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(jpg.tobytes()).decode()
