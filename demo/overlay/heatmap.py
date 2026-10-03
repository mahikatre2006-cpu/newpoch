"""
Heatmap generation from accumulated gaze coordinates.

Accumulates (x, y) gaze points over the tracking window, applies
2D Gaussian blur, normalises, and maps to a COLORMAP_TURBO (or JET)
colour overlay that can be alpha-blended onto the original thumbnail.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def generate_heatmap(
    coords: List[Tuple[float, float]],
    shape: Tuple[int, int, int],
    sigma: float = 40.0,
    colormap_name: str = "COLORMAP_TURBO",
    low_cut: float = 0.10,
) -> np.ndarray:
    """Build a coloured heatmap from gaze coordinates.

    Parameters
    ----------
    coords : list of (x, y) pixel positions on the thumbnail.
    shape : (H, W, C) of the target thumbnail image.
    sigma : standard deviation for the 2D Gaussian blur.
    colormap_name : OpenCV colormap constant name (e.g. "COLORMAP_TURBO").
    low_cut : normalised attention below this fraction becomes fully
              transparent in the alpha mask.

    Returns
    -------
    heatmap_bgra : np.ndarray (H, W, 4) — BGRA heatmap overlay.
    """
    h, w = shape[0], shape[1]

    # 1. Accumulate point density
    accumulator = np.zeros((h, w), dtype=np.float32)
    for (x, y) in coords:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            accumulator[iy, ix] += 1.0

    if accumulator.max() == 0:
        # No valid points — return a fully transparent overlay
        return np.zeros((h, w, 4), dtype=np.uint8)

    # 2. Gaussian blur to spread the density
    ksize = 0  # let OpenCV derive from sigma
    heatmap = cv2.GaussianBlur(accumulator, (ksize, ksize), sigma)

    # 3. Normalise to [0, 1]
    heatmap /= heatmap.max()

    # 4. Apply low-cut threshold (below this → transparent)
    mask = (heatmap > low_cut).astype(np.float32)
    # smooth the mask edges
    mask = cv2.GaussianBlur(mask, (0, 0), sigma * 0.3)

    # 5. Map to [0, 255] uint8 for the colormap
    heatmap_u8 = (heatmap * 255).astype(np.uint8)

    # 6. Apply colormap
    cmap = getattr(cv2, colormap_name, cv2.COLORMAP_TURBO)
    coloured = cv2.applyColorMap(heatmap_u8, cmap)  # (H, W, 3) BGR

    # 7. Build BGRA with alpha from the density mask
    alpha_channel = (mask * 255).astype(np.uint8)
    bgra = np.dstack([coloured, alpha_channel])

    return bgra


def blend_heatmap(
    thumbnail: np.ndarray,
    heatmap_bgra: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Alpha-blend *heatmap_bgra* onto *thumbnail*.

    Parameters
    ----------
    thumbnail : (H, W, 3) BGR
    heatmap_bgra : (H, W, 4) BGRA from :func:`generate_heatmap`
    alpha : global blend strength

    Returns
    -------
    blended : (H, W, 3) BGR
    """
    if heatmap_bgra.shape[2] != 4:
        raise ValueError("heatmap must be BGRA (4 channels)")

    h, w = thumbnail.shape[:2]
    hh, hw = heatmap_bgra.shape[:2]
    if (h, w) != (hh, hw):
        heatmap_bgra = cv2.resize(heatmap_bgra, (w, h))

    # Extract per-pixel alpha
    heat_bgr = heatmap_bgra[:, :, :3].astype(np.float32)
    heat_alpha = (heatmap_bgra[:, :, 3].astype(np.float32) / 255.0) * alpha
    heat_alpha = heat_alpha[:, :, np.newaxis]  # broadcast-ready

    blended = (
        heat_alpha * heat_bgr + (1.0 - heat_alpha) * thumbnail.astype(np.float32)
    )
    return blended.astype(np.uint8)
