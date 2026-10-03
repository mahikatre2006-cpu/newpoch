"""
Glowing "streamer-style" blob renderer.

Draws a smooth, semi-transparent radial-gradient blob at the gaze point
over the thumbnail, with:
  - Warm outer glow (large, low opacity)
  - Bright inner core
  - Sine-wave "breathing" pulse on radius / opacity
  - Optional trailing ghost positions for a comet-tail effect
"""

from __future__ import annotations

import collections
import math
import time
from typing import List, Tuple

import cv2
import numpy as np


class BlobRenderer:
    """Render a glowing gaze blob onto a frame."""

    def __init__(self, config: dict):
        self.inner_r = config.get("inner_radius", 14)
        self.outer_r = config.get("outer_radius", 50)
        self.inner_color = tuple(config.get("inner_color", [255, 255, 255]))
        self.outer_color = tuple(config.get("outer_color", [255, 200, 0]))
        self.opacity = config.get("opacity", 0.65)
        self.pulse_hz = config.get("pulse_hz", 1.5)
        self.trail_len = config.get("trail_length", 10)

        self._trail: collections.deque[Tuple[int, int]] = collections.deque(
            maxlen=max(1, self.trail_len),
        )
        self._t0 = time.time()

    # ------------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
    ) -> np.ndarray:
        """Draw the blob at *(x, y)* on *frame* (mutates and returns it)."""
        ix, iy = int(round(x)), int(round(y))
        self._trail.append((ix, iy))

        # Breathing pulse factor
        t = time.time() - self._t0
        pulse = 0.85 + 0.15 * math.sin(2 * math.pi * self.pulse_hz * t)

        # Draw trailing ghosts (oldest = faintest)
        for i, (tx, ty) in enumerate(self._trail):
            alpha = (i + 1) / len(self._trail) * 0.25 * pulse  # 0..0.25
            r = int(self.outer_r * 0.5 * pulse)
            _draw_circle_alpha(frame, tx, ty, r, self.outer_color, alpha)

        # Outer glow
        outer_r = int(self.outer_r * pulse)
        _draw_circle_alpha(frame, ix, iy, outer_r, self.outer_color,
                           self.opacity * 0.4 * pulse)

        # Mid glow
        mid_r = int((self.inner_r + self.outer_r) / 2 * pulse)
        blended = tuple(
            int(a * 0.5 + b * 0.5)
            for a, b in zip(self.inner_color, self.outer_color)
        )
        _draw_circle_alpha(frame, ix, iy, mid_r, blended,
                           self.opacity * 0.6 * pulse)

        # Inner core
        inner_r = int(self.inner_r * pulse)
        _draw_circle_alpha(frame, ix, iy, inner_r, self.inner_color,
                           self.opacity * pulse)

        return frame

    def reset(self) -> None:
        self._trail.clear()
        self._t0 = time.time()


# ------------------------------------------------------------------
# Alpha-blended filled circle helper
# ------------------------------------------------------------------

def _draw_circle_alpha(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Draw a filled circle with transparency onto *frame* in-place."""
    if radius <= 0 or alpha <= 0:
        return
    h, w = frame.shape[:2]
    # Compute bounding box (clipped to frame)
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius + 1, w)
    y2 = min(cy + radius + 1, h)
    if x1 >= x2 or y1 >= y2:
        return

    # Build a small overlay ROI
    roi = frame[y1:y2, x1:x2].copy()
    overlay = roi.copy()
    # Draw the circle relative to the ROI origin
    cv2.circle(overlay, (cx - x1, cy - y1), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
    frame[y1:y2, x1:x2] = roi
