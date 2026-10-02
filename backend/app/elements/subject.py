"""M4 subject: rembg (u2net) foreground, cleaned and cut down to its largest connected component(s)."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from .model import Detection


def largest_components(mask: np.ndarray, min_area: int, max_components: int, ratio: float) -> list[np.ndarray]:
    """Connected components of `mask`, biggest first: at most max_components, each at least `ratio` of the biggest
    and at least min_area pixels."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = sorted(((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)), reverse=True)
    if not comps:
        return []
    biggest = comps[0][0]
    return [lab == i for area, i in comps[:max_components] if area >= min_area and area >= ratio * biggest]


class SubjectDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg["elements"]["subject"]
        self._lock = threading.Lock()

    def load(self) -> None:
        from rembg import new_session

        self._session = new_session(self.cfg["model"], providers=["CPUExecutionProvider"])

    def foreground(self, rgb: np.ndarray) -> np.ndarray:
        from rembg import remove

        with self._lock:
            alpha = np.asarray(remove(rgb, session=self._session, only_mask=True))
        return alpha > self.cfg["threshold"]

    def detect(self, rgb: np.ndarray, exclude: np.ndarray | None = None) -> list[Detection]:
        """`exclude` (faces and text) is subtracted first, so the subject is what is left of the cutout around them."""
        h, w = rgb.shape[:2]
        fg = self.foreground(rgb)
        k = np.ones((7, 7), np.uint8)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k).astype(bool)
        if exclude is not None:
            fg &= ~exclude
        min_area = int(self.cfg["min_area_pct"] / 100 * h * w)
        comps = largest_components(fg, min_area, self.cfg["max_components"], self.cfg["component_ratio"])
        return [Detection("subject", c, "Subject" if len(comps) == 1 else f"Subject {i + 1}") for i, c in enumerate(comps)]
