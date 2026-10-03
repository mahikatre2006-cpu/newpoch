"""
Unified Gaze Tracker.

Fuses:
1. MediaPipe eye blendshapes (rapid directional eye gaze)
2. Iris relative displacement within eye sockets (continuous ocular tracking)
3. Metric 3D Head Euler angles (natural vestibulo-ocular coordination)
4. Adaptive filter (ultra-steady fixation + instant saccadic response)
5. Affine calibration (works out of the box, refines with 5-point calibration)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .face_detector import FaceDetector
from .smoother import GazeSmoother


class AdaptiveSmoother:
    """Adaptive Exponential Moving Average filter.

    Low alpha (0.12) when eyes are holding fixation -> zero jitter, rock-solid.
    High alpha (0.75) when eyes jump (saccade) -> instant response without lag.
    """

    def __init__(self, min_alpha: float = 0.15, max_alpha: float = 0.75, saccade_threshold: float = 0.05):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.saccade_threshold = saccade_threshold
        self._prev: Optional[Tuple[float, float]] = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        if self._prev is None:
            self._prev = (x, y)
            return (x, y)

        dx = x - self._prev[0]
        dy = y - self._prev[1]
        dist = math.hypot(dx, dy)

        # Scale alpha based on velocity
        t = min(1.0, dist / self.saccade_threshold)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (t ** 1.5)

        new_x = alpha * x + (1.0 - alpha) * self._prev[0]
        new_y = alpha * y + (1.0 - alpha) * self._prev[1]
        self._prev = (new_x, new_y)
        return (new_x, new_y)

    def reset(self):
        self._prev = None


class GazeTracker:
    """End-to-end gaze tracking with default calibration and optional 5-point fitting."""

    def __init__(self):
        self.detector = FaceDetector()
        self.smoother = AdaptiveSmoother()

        # Calibration state
        self.is_calibrated = False
        # 2x3 affine matrix from [raw_x, raw_y, 1] -> [screen_x, screen_y]
        # Default mapping calibrated for a laptop/webcam setup
        # raw_x positive -> looking right -> screen_x increases (> 0.5)
        # raw_y positive -> looking up -> screen_y decreases (< 0.5)
        self.affine = np.array([
            [0.85, 0.0, 0.50],
            [0.0, -0.85, 0.50]
        ], dtype=np.float64)

        # Neutral center baseline
        self.center_raw_x = 0.0
        self.center_raw_y = 0.0

    def extract_raw_gaze(self, bgr_frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract fused raw gaze components from webcam frame."""
        face = self.detector.detect(bgr_frame)
        if face is None:
            return None

        bs = face["eye_blendshapes"]
        # Directional scores from MediaPipe blendshapes
        look_right = (bs.get("eyeLookInLeft", 0.0) + bs.get("eyeLookOutRight", 0.0)) / 2.0
        look_left = (bs.get("eyeLookOutLeft", 0.0) + bs.get("eyeLookInRight", 0.0)) / 2.0
        look_up = (bs.get("eyeLookUpLeft", 0.0) + bs.get("eyeLookUpRight", 0.0)) / 2.0
        look_down = (bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) / 2.0

        blend_x = float(look_right - look_left)
        blend_y = float(look_up - look_down)

        # Iris displacement
        iris_dx, iris_dy = face["iris_displacement"]
        # Head Euler (pitch, yaw, roll) in radians
        head_pitch, head_yaw, head_roll = face["head_euler"]

        # Combined raw gaze vector:
        # Eye blendshapes + scaled iris displacement + subtle head rotation contribution
        # Note: in webcam coordinates, looking right produces positive yaw and positive blend_x
        raw_x = float((0.60 * blend_x + 0.40 * (iris_dx * 3.2)) + 0.45 * head_yaw)
        raw_y = float((0.60 * blend_y - 0.40 * (iris_dy * 3.2)) + 0.45 * head_pitch)

        return {
            "raw_x": raw_x,
            "raw_y": raw_y,
            "blend_x": blend_x,
            "blend_y": blend_y,
            "iris_dx": iris_dx,
            "iris_dy": iris_dy,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "bbox": face["bbox"],
        }

    def process_frame(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """Track gaze and return calibrated, smoothed screen coordinates in [0..1]."""
        raw = self.extract_raw_gaze(bgr_frame)
        if raw is None:
            return {"status": "no_face"}

        # Apply center baseline
        rx = raw["raw_x"] - self.center_raw_x
        ry = raw["raw_y"] - self.center_raw_y

        # Map through affine matrix
        v = self.affine @ np.array([rx, ry, 1.0], dtype=np.float64)
        clamped_x = float(np.clip(v[0], 0.01, 0.99))
        clamped_y = float(np.clip(v[1], 0.01, 0.99))

        # Smooth
        sm_x, sm_y = self.smoother.filter(clamped_x, clamped_y)

        return {
            "status": "ok",
            "x": float(sm_x),
            "y": float(sm_y),
            "raw_x": float(raw["raw_x"]),
            "raw_y": float(raw["raw_y"]),
            "is_calibrated": self.is_calibrated,
            "bbox": raw["bbox"],
        }

    def recenter(self, bgr_frame: np.ndarray) -> bool:
        """Set current gaze position as neutral screen center."""
        raw = self.extract_raw_gaze(bgr_frame)
        if raw is None:
            return False
        self.center_raw_x = raw["raw_x"]
        self.center_raw_y = raw["raw_y"]
        self.smoother.reset()
        return True

    def fit_calibration(
        self,
        raw_points: List[Tuple[float, float]],
        screen_points: List[Tuple[float, float]],
    ) -> float:
        """Fit affine transform from raw (x, y) to screen targets (sx, sy).

        Returns average reprojection error in normalized units.
        """
        assert len(raw_points) == len(screen_points)
        n = len(raw_points)
        if n < 3:
            return 0.0

        A = np.zeros((2 * n, 6), dtype=np.float64)
        b_vec = np.zeros(2 * n, dtype=np.float64)
        for i, ((rx, ry), (sx, sy)) in enumerate(zip(raw_points, screen_points)):
            A[2 * i] = [rx, ry, 1, 0, 0, 0]
            A[2 * i + 1] = [0, 0, 0, rx, ry, 1]
            b_vec[2 * i] = sx
            b_vec[2 * i + 1] = sy

        params, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        self.affine = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]]
        ], dtype=np.float64)

        self.center_raw_x = 0.0
        self.center_raw_y = 0.0
        self.is_calibrated = True
        self.smoother.reset()

        # Compute error
        errors = []
        for (rx, ry), (sx, sy) in zip(raw_points, screen_points):
            v = self.affine @ np.array([rx, ry, 1.0])
            err = math.hypot(v[0] - sx, v[1] - sy)
            errors.append(err)
        return float(sum(errors) / len(errors)) if errors else 0.0

    def reset(self):
        """Reset calibration to default parameters."""
        self.is_calibrated = False
        self.center_raw_x = 0.0
        self.center_raw_y = 0.0
        self.affine = np.array([
            [0.85, 0.0, 0.50],
            [0.0, -0.85, 0.50]
        ], dtype=np.float64)
        self.smoother.reset()
