"""M4 text: RapidOCR lines, merged into blocks. Recognised strings are kept for the explanation engine."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .model import Detection


@dataclass
class Line:
    text: str
    score: float
    poly: np.ndarray  # 4x2 float, frame coordinates
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    recognised: bool = True


def line_from_ocr(box, text: str, score: float, w: int, h: int) -> Line:
    poly = np.asarray(box, np.float32)
    x0, y0 = np.clip(poly.min(axis=0), 0, [w, h]).astype(int)
    x1, y1 = np.clip(poly.max(axis=0), 0, [w, h]).astype(int)
    return Line(str(text), float(score), poly, (int(x0), int(y0), int(x1), int(y1)))


def _same_block(a: Line, b: Line, gap: float) -> bool:
    """Stacked lines of one headline: similar height, horizontally overlapping, vertically close."""
    ha, hb = a.box[3] - a.box[1], b.box[3] - b.box[1]
    if min(ha, hb) <= 0 or max(ha, hb) / min(ha, hb) > 1.7:
        return False
    v_gap = max(a.box[1], b.box[1]) - min(a.box[3], b.box[3])  # negative when they overlap vertically
    if v_gap > gap * min(ha, hb):
        return False
    overlap = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
    return overlap >= 0.3 * min(a.box[2] - a.box[0], b.box[2] - b.box[0])


def merge_lines(lines: list[Line], gap: float) -> list[list[Line]]:
    """Union-find over the 'same block' relation; blocks come out in reading order (top to bottom)."""
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _same_block(lines[i], lines[j], gap):
                parent[find(j)] = find(i)
    groups: dict[int, list[Line]] = {}
    for i, ln in enumerate(lines):
        groups.setdefault(find(i), []).append(ln)
    blocks = [sorted(g, key=lambda ln: (ln.box[1], ln.box[0])) for g in groups.values()]
    return sorted(blocks, key=lambda g: (g[0].box[1], g[0].box[0]))


def block_mask(block: list[Line], shape: tuple[int, int], pad: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for ln in block:
        cv2.fillPoly(mask, [ln.poly.astype(np.int32)], 1)
    if pad > 0:
        mask = cv2.dilate(mask, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    return mask.astype(bool)


class TextDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg["elements"]["text"]
        self._lock = threading.Lock()

    def load(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        c = self.cfg
        self.engine = RapidOCR(intra_op_num_threads=c["threads"], inter_op_num_threads=1, det_limit_type="max",
                               det_limit_side_len=c["det_limit_side_len"], text_score=0.0, print_verbose=False)
        self.engine(np.full((96, 160, 3), 255, np.uint8), use_cls=False)  # warm-up

    def ocr(self, rgb: np.ndarray) -> list[Line]:
        """Every box the detector finds with its recognition score (text_score=0 so nothing is dropped inside the engine)."""
        h, w = rgb.shape[:2]
        with self._lock:
            res, _ = self.engine(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), use_cls=False)
        return [line_from_ocr(b, t, s, w, h) for b, t, s in (res or [])]

    def lines(self, rgb: np.ndarray) -> list[Line]:
        c = self.cfg
        keep = []
        for ln in self.ocr(rgb):
            height = ln.box[3] - ln.box[1]
            if height < c["min_height_px"]:
                continue
            if ln.score >= c["min_score"] and len(ln.text.strip()) >= c["min_chars"]:
                keep.append(ln)
            elif ln.score < c["min_score"] and height >= c["unrecognised_min_height_px"]:
                ln.recognised = False  # a large text-shaped box in a script OCR cannot read: still text for attention
                keep.append(ln)
        return keep

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        h, w = rgb.shape[:2]
        out = []
        for block in merge_lines(self.lines(rgb), self.cfg["merge_gap"]):
            recognised = [ln for ln in block if ln.recognised]
            text = " ".join(ln.text.strip() for ln in recognised) or None
            label = (text[:40] + "…" if text and len(text) > 40 else text) or "Text"
            out.append(Detection("text", block_mask(block, (h, w), self.cfg["pad_px"]), label, text,
                                 {"lines": len(block), "recognised": bool(recognised),
                                  "confidence": round(float(np.mean([ln.score for ln in block])), 3)}))
        return out
