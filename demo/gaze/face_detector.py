"""
Face, Iris and Head Pose detector using MediaPipe Tasks API.

Extracts:
- 478 3D landmarks (including iris centers 468, 473 and eye corners)
- Eye blendshapes (directional gaze scores: look left, right, up, down)
- 4x4 Facial Transformation Matrix (metric 3D head rotation and translation)
- Iris normalized displacement relative to eye sockets
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional, Tuple, Dict, Any

import cv2
import mediapipe as mp
import numpy as np


def _ensure_task_file() -> str:
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
    task_path = os.path.join(weights_dir, "face_landmarker.task")
    if not os.path.exists(task_path):
        print(f"[FaceDetector] Downloading FaceLandmarker model to {task_path}...")
        os.makedirs(weights_dir, exist_ok=True)
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        urllib.request.urlretrieve(url, task_path)
    return task_path


class FaceDetector:
    """Wraps MediaPipe Face Landmarker with iris tracking, blendshapes, and 3D pose."""

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        task_path = _ensure_task_file()

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_result: Optional[dict] = None

    def detect(self, bgr_frame: np.ndarray) -> Optional[dict]:
        """Detect primary face, iris displacement, blendshapes, and head pose.

        Returns
        -------
        dict with keys:
            "landmarks"       : np.ndarray (478, 2) in image pixel coords
            "iris_displacement": (dx, dy) normalized iris offset inside eye sockets
            "eye_blendshapes" : dict of raw eye blendshape values
            "head_euler"      : (pitch, yaw, roll) in radians
            "head_R"          : (3, 3) rotation matrix
            "tvec"            : (3, 1) translation vector in cm
            "bbox"            : (x, y, w, h)
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self._landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None

        # 1. 2D Landmarks (478 points)
        lm = results.face_landmarks[0]
        pts = np.array([(l.x * w, l.y * h) for l in lm], dtype=np.float64)

        # 2. Eye blendshapes
        blend_dict = {}
        if results.face_blendshapes:
            for b in results.face_blendshapes[0]:
                if "eye" in b.category_name.lower():
                    blend_dict[b.category_name] = float(b.score)

        # 3. Iris displacement inside eye sockets
        # Subject right eye (in image left):
        # 33 = outer corner, 133 = inner corner, 159 = top lid, 145 = bottom lid, 468 = iris center
        r_outer, r_inner = pts[33], pts[133]
        r_top, r_bot = pts[159], pts[145]
        r_iris = pts[468]

        r_w = float(np.linalg.norm(r_inner - r_outer)) + 1e-6
        r_h = float(np.linalg.norm(r_bot - r_top)) + 1e-6
        r_dx = (r_iris[0] - (r_outer[0] + r_inner[0]) / 2.0) / r_w
        r_dy = (r_iris[1] - (r_top[1] + r_bot[1]) / 2.0) / r_h

        # Subject left eye (in image right):
        # 362 = inner corner, 263 = outer corner, 386 = top lid, 374 = bottom lid, 473 = iris center
        l_inner, l_outer = pts[362], pts[263]
        l_top, l_bot = pts[386], pts[374]
        l_iris = pts[473]

        l_w = float(np.linalg.norm(l_outer - l_inner)) + 1e-6
        l_h = float(np.linalg.norm(l_bot - l_top)) + 1e-6
        l_dx = (l_iris[0] - (l_inner[0] + l_outer[0]) / 2.0) / l_w
        l_dy = (l_iris[1] - (l_top[1] + l_bot[1]) / 2.0) / l_h

        # Average both eyes
        iris_dx = float((r_dx + l_dx) / 2.0)
        iris_dy = float((r_dy + l_dy) / 2.0)

        # 4. Head rotation from rigid facial transformation matrix
        pitch, yaw, roll = 0.0, 0.0, 0.0
        head_R = np.eye(3, dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        if results.facial_transformation_matrixes:
            mat = results.facial_transformation_matrixes[0]
            head_R = mat[:3, :3].astype(np.float64)
            tvec = mat[:3, 3:].astype(np.float64)

            # Decompose rotation matrix into Euler angles
            sy = np.sqrt(head_R[0, 0] ** 2 + head_R[1, 0] ** 2)
            if sy > 1e-6:
                pitch = float(np.arctan2(head_R[2, 1], head_R[2, 2]))
                yaw = float(np.arctan2(-head_R[2, 0], sy))
                roll = float(np.arctan2(head_R[1, 0], head_R[0, 0]))
            else:
                pitch = float(np.arctan2(-head_R[1, 2], head_R[1, 1]))
                yaw = float(np.arctan2(-head_R[2, 0], sy))
                roll = 0.0

        # 5. Face Bounding Box
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        res = {
            "landmarks": pts,
            "iris_displacement": (iris_dx, iris_dy),
            "eye_blendshapes": blend_dict,
            "head_euler": (pitch, yaw, roll),
            "head_R": head_R,
            "tvec": tvec,
            "bbox": bbox,
        }
        self._last_result = res
        return res

    def close(self) -> None:
        self._landmarker.close()
