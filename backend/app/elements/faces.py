"""M4 faces: MediaPipe Face Detection. Mask = ellipse in the box grown by `expand` on every side."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from app.config import BACKEND_DIR
from .model import Detection


def face_ellipse_mask(box: tuple[float, float, float, float], expand: float, shape: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (round(cx), round(cy)), (max(1, round(w * (0.5 + expand))), max(1, round(h * (0.5 + expand)))), 0, 0, 360, 1, -1)
    return mask.astype(bool)


class FaceDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg["elements"]["face"]
        self._lock = threading.Lock()

    def load(self) -> None:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        path = BACKEND_DIR / self.cfg["model"]
        if not path.exists():
            raise FileNotFoundError(f"face model not found: {path}")
        opts = vision.FaceDetectorOptions(base_options=mpp.BaseOptions(model_asset_path=str(path)),
                                          min_detection_confidence=self.cfg["min_confidence"])
        self._det = vision.FaceDetector.create_from_options(opts)
        self._mp = mp

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        h, w = rgb.shape[:2]
        with self._lock:
            res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        faces = sorted(res.detections, key=lambda d: -d.bounding_box.width * d.bounding_box.height)
        out = []
        for i, d in enumerate(faces):
            b = d.bounding_box
            mask = face_ellipse_mask((b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height), self.cfg["expand"], (h, w))
            out.append(Detection("face", mask, "Face" if len(faces) == 1 else f"Face {i + 1}",
                                 meta={"confidence": round(float(d.categories[0].score), 3),
                                       "face_box": [int(b.origin_x), int(b.origin_y), int(b.origin_x + b.width), int(b.origin_y + b.height)]}))
        return out
