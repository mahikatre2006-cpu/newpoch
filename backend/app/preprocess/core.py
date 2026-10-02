"""M2. Build the Context: model input, mobile render and YouTube safe-zone mask."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Context:
    image_hash: str
    image: np.ndarray  # 1280x720 RGB uint8, the frame every module measures
    saliency_input: np.ndarray  # same frame, downscaled with the aspect kept
    mobile_small: np.ndarray  # what a phone feed shows (168x94)
    mobile_up: np.ndarray  # mobile_small scaled back to full size, so OCR sees what a phone viewer can read
    safe_zone: np.ndarray  # H x W uint8, 1 where the duration badge or progress bar covers the thumbnail
    config: dict = field(default_factory=dict, repr=False)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def build_safe_zone(h: int, w: int, boxes: dict[str, list[float]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes.values():
        mask[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = 1
    return mask


def build_context(image: np.ndarray, image_hash: str, cfg: dict) -> Context:
    h, w = image.shape[:2]
    pre = cfg["preprocess"]
    iw = pre["saliency_input_width"]
    mw, mh = pre["mobile_size"]
    small = _resize(image, mw, mh)
    return Context(
        image_hash=image_hash,
        image=image,
        saliency_input=_resize(image, iw, round(iw * h / w)),
        mobile_small=small,
        mobile_up=cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC),
        safe_zone=build_safe_zone(h, w, pre["safe_zone"]),
        config=cfg,
    )
