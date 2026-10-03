"""
OpenCV window management, demo flow states, countdown overlays,
and transition effects for the live gaze tracking demo.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


class DemoState(Enum):
    SPLASH = auto()
    CALIBRATING = auto()
    TRACKING = auto()
    GENERATING = auto()
    RESULT = auto()


# ======================================================================
# Splash screen
# ======================================================================

def show_splash(
    window: str,
    screen_w: int,
    screen_h: int,
    duration_s: float = 2.0,
) -> None:
    """Show a brief title card before the demo begins."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    # Gradient background (dark blue → black)
    for y in range(screen_h):
        frac = y / screen_h
        canvas[y, :] = (int(30 * (1 - frac)), int(20 * (1 - frac)), int(10 * (1 - frac)))

    # Title
    title = "Gaze Tracking Demo"
    subtitle = "Thumbnail Attention Studio"
    tagline = "Real-time eye tracking  |  Visual saliency data collection"

    _put_centered_text(canvas, title, screen_w, int(screen_h * 0.35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 255), 3)
    _put_centered_text(canvas, subtitle, screen_w, int(screen_h * 0.48),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    _put_centered_text(canvas, tagline, screen_w, int(screen_h * 0.60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)

    # Fade in
    for alpha in _fade_steps(0.0, 1.0, 0.4):
        vis = (canvas.astype(np.float32) * alpha).astype(np.uint8)
        cv2.imshow(window, vis)
        cv2.waitKey(16)

    cv2.imshow(window, canvas)
    t0 = time.time()
    while time.time() - t0 < duration_s:
        if cv2.waitKey(30) & 0xFF in (27, ord(" ")):
            break

    # Fade out
    for alpha in _fade_steps(1.0, 0.0, 0.3):
        vis = (canvas.astype(np.float32) * alpha).astype(np.uint8)
        cv2.imshow(window, vis)
        cv2.waitKey(16)


# ======================================================================
# Countdown overlay (drawn on top of the thumbnail during tracking)
# ======================================================================

def draw_countdown(
    frame: np.ndarray,
    remaining_s: float,
    total_s: float,
    color: tuple = (0, 220, 255),
) -> np.ndarray:
    """Draw a countdown timer + progress arc in the top-right corner."""
    h, w = frame.shape[:2]

    # Arc centre
    cx, cy, r = w - 60, 60, 35
    progress = 1.0 - (remaining_s / total_s) if total_s > 0 else 1.0
    angle = int(360 * progress)

    # Background circle
    cv2.circle(frame, (cx, cy), r + 4, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r + 4, (80, 80, 80), 2, cv2.LINE_AA)
    # Progress arc
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, angle, color, 3, cv2.LINE_AA)
    # Time text
    secs = max(0, int(remaining_s) + 1)
    text = str(secs)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # "TRACKING" label
    cv2.putText(frame, "TRACKING", (w - 155, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame


# ======================================================================
# "Generating heatmap..." transition
# ======================================================================

def show_generating(
    window: str,
    thumbnail: np.ndarray,
    duration_s: float = 0.8,
) -> None:
    """Show a brief 'analysing' overlay while the heatmap is computed."""
    overlay = thumbnail.copy()
    h, w = overlay.shape[:2]

    # Darken
    dark = (overlay.astype(np.float32) * 0.4).astype(np.uint8)
    _put_centered_text(dark, "Generating heatmap...", w, h // 2,
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)

    cv2.imshow(window, dark)
    t0 = time.time()
    while time.time() - t0 < duration_s:
        cv2.waitKey(30)


# ======================================================================
# Result screen overlay
# ======================================================================

def draw_result_footer(
    frame: np.ndarray,
    color: tuple = (200, 200, 200),
) -> np.ndarray:
    """Add instructions at the bottom of the heatmap result."""
    h, w = frame.shape[:2]
    text = "Press R to restart  |  Q to quit"
    _put_centered_text(frame, text, w, h - 25,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return frame


# ======================================================================
# Fade-in animation for the heatmap overlay
# ======================================================================

def fade_in_overlay(
    window: str,
    base: np.ndarray,
    overlay: np.ndarray,
    duration_s: float = 0.5,
) -> None:
    """Smoothly blend *overlay* onto *base* over *duration_s*."""
    for alpha in _fade_steps(0.0, 1.0, duration_s):
        vis = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
        cv2.imshow(window, vis)
        cv2.waitKey(16)


# ======================================================================
# Window setup
# ======================================================================

def setup_window(name: str = "Demo", fullscreen: bool = True) -> str:
    """Create the named OpenCV window. Returns the window name."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    return name


# ======================================================================
# Helpers
# ======================================================================

def _put_centered_text(
    img: np.ndarray,
    text: str,
    frame_w: int,
    y: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.0,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame_w - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _fade_steps(start: float, end: float, duration_s: float, fps: int = 60):
    """Yield alpha values for a linear fade."""
    n_frames = max(1, int(duration_s * fps))
    for i in range(n_frames + 1):
        yield start + (end - start) * i / n_frames
