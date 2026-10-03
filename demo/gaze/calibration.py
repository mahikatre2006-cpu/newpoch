"""
Screen-space calibration for the gaze tracker.

Shows N dots on screen at known positions, records the raw gaze vectors
while the user looks at each dot, then fits an affine transform from
raw gaze output → true screen pixels.

Supports saving / loading calibration so judges don't re-calibrate
every run.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


class Calibration:
    """Affine mapping from raw gaze (x, y) to calibrated screen (x, y)."""

    def __init__(self):
        # 2×3 affine matrix (initialised to identity + zero translation)
        self._affine: np.ndarray = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    def map(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """Map a raw gaze coordinate to calibrated screen coordinates."""
        src = np.array([raw_x, raw_y, 1.0], dtype=np.float64)
        dst = self._affine @ src
        return (float(dst[0]), float(dst[1]))

    # ------------------------------------------------------------------
    def fit(
        self,
        raw_points: List[Tuple[float, float]],
        screen_points: List[Tuple[float, float]],
    ) -> None:
        """Fit an affine transform from raw → screen using least squares.

        Needs at least 3 non-collinear point pairs; 5 is recommended.
        """
        assert len(raw_points) == len(screen_points)
        n = len(raw_points)
        if n < 3:
            print("[Calibration] Not enough points, using identity mapping.")
            return

        # Build the system   A @ [a b c d e f]^T = b_vec
        # For each point: sx = a*rx + b*ry + c,  sy = d*rx + e*ry + f
        A = np.zeros((2 * n, 6), dtype=np.float64)
        b_vec = np.zeros(2 * n, dtype=np.float64)
        for i, ((rx, ry), (sx, sy)) in enumerate(zip(raw_points, screen_points)):
            A[2 * i]     = [rx, ry, 1, 0, 0, 0]
            A[2 * i + 1] = [0, 0, 0, rx, ry, 1]
            b_vec[2 * i]     = sx
            b_vec[2 * i + 1] = sy

        params, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        self._affine = np.array(
            [[params[0], params[1], params[2]],
             [params[3], params[4], params[5]]],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {"affine": self._affine.tolist()}
        with open(path, "w") as f:
            yaml.dump(data, f)
        print(f"[Calibration] Saved to {path}")

    def load(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            data = yaml.safe_load(f)
        self._affine = np.array(data["affine"], dtype=np.float64)
        print(f"[Calibration] Loaded from {path}")
        return True


# ======================================================================
# Interactive calibration routine
# ======================================================================

def _calibration_targets(screen_w: int, screen_h: int, n: int = 5) -> List[Tuple[int, int]]:
    """Return *n* calibration target positions spread across the screen."""
    cx, cy = screen_w // 2, screen_h // 2
    margin_x, margin_y = int(screen_w * 0.12), int(screen_h * 0.12)
    if n == 5:
        return [
            (cx, cy),                                          # centre
            (margin_x, margin_y),                              # top-left
            (screen_w - margin_x, margin_y),                   # top-right
            (margin_x, screen_h - margin_y),                   # bottom-left
            (screen_w - margin_x, screen_h - margin_y),        # bottom-right
        ]
    elif n == 9:
        return [
            (cx, cy),
            (margin_x, margin_y),
            (cx, margin_y),
            (screen_w - margin_x, margin_y),
            (margin_x, cy),
            (screen_w - margin_x, cy),
            (margin_x, screen_h - margin_y),
            (cx, screen_h - margin_y),
            (screen_w - margin_x, screen_h - margin_y),
        ]
    # fallback: just centre
    return [(cx, cy)]


def run_calibration(
    cam: cv2.VideoCapture,
    face_detector,
    gaze_model,
    config: dict,
    screen_w: int = 1280,
    screen_h: int = 720,
) -> Calibration:
    """Run the interactive calibration, return a fitted :class:`Calibration`.

    Parameters
    ----------
    cam : cv2.VideoCapture
    face_detector : FaceDetector instance
    gaze_model : GazeModel instance
    config : dict (the ``calibration`` section from config.yaml)
    screen_w, screen_h : display dimensions
    """
    from .face_detector import FaceDetector  # noqa: avoid circular at module level

    n_points = config.get("points", 5)
    hold_s = config.get("hold_time_s", 1.5)
    radius = config.get("marker_radius", 20)
    marker_color = tuple(config.get("marker_color", [0, 220, 255]))
    bg_color = tuple(config.get("bg_color", [30, 30, 30]))

    targets = _calibration_targets(screen_w, screen_h, n_points)
    raw_collected: List[Tuple[float, float]] = []
    screen_collected: List[Tuple[int, int]] = []

    window = "Demo"

    for idx, (tx, ty) in enumerate(targets):
        # Show the calibration dot
        canvas = np.full((screen_h, screen_w, 3), bg_color, dtype=np.uint8)
        cv2.circle(canvas, (tx, ty), radius, marker_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), radius + 6, marker_color, 2, cv2.LINE_AA)

        # Instruction text
        text = f"Look at the dot ({idx + 1}/{len(targets)})"
        cv2.putText(canvas, text, (screen_w // 2 - 180, screen_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        cv2.waitKey(500)  # brief pause before collecting

        # Collect gaze samples for hold_s seconds
        samples: List[Tuple[float, float]] = []
        t0 = time.time()
        while time.time() - t0 < hold_s:
            ret, frame = cam.read()
            if not ret:
                continue
            face = face_detector.detect(frame)
            if face is None:
                continue
            pitch, yaw = gaze_model.predict(face["face_patch"])
            raw_xy = gaze_model.gaze_to_screen(
                pitch, yaw,
                face["head_R"], face["tvec"], face["camera_matrix"],
                screen_w, screen_h,
            )
            samples.append(raw_xy)

            # Show progress ring
            elapsed = time.time() - t0
            angle = int(360 * elapsed / hold_s)
            vis = canvas.copy()
            cv2.ellipse(vis, (tx, ty), (radius + 12, radius + 12),
                        -90, 0, angle, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow(window, vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort
                break

        if samples:
            avg_x = sum(s[0] for s in samples) / len(samples)
            avg_y = sum(s[1] for s in samples) / len(samples)
            raw_collected.append((avg_x, avg_y))
            screen_collected.append((tx, ty))

    # Fit the calibration
    cal = Calibration()
    if len(raw_collected) >= 3:
        cal.fit(raw_collected, screen_collected)
    else:
        print("[Calibration] WARNING: too few valid points, using identity mapping.")

    return cal
