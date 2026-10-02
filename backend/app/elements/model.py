"""The element set: a list of elements plus one label map, so every pixel belongs to exactly one element."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Overlaps are resolved by priority (PRD M4): text > face > subject > background.
PRIORITY = {"background": 0, "subject": 1, "face": 2, "text": 3}


@dataclass
class Detection:
    """What a detector found, before overlaps are resolved."""
    type: str  # face | text | subject
    mask: np.ndarray  # H x W bool
    label: str
    text: str | None = None
    meta: dict = field(default_factory=dict)  # detector extras (confidence, recognised, ...)


@dataclass
class Element:
    id: str
    type: str  # face | text | subject | background | layer
    label: str
    box: list[int]  # x0, y0, x1, y1 (x1 and y1 exclusive), 0 area -> [0, 0, 0, 0]
    area_px: int
    text: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class ElementSet:
    elements: list[Element]
    labels: np.ndarray  # H x W int16: index into `elements`

    def mask(self, i: int) -> np.ndarray:
        return self.labels == i

    def index(self, element_id: str) -> int:
        for i, e in enumerate(self.elements):
            if e.id == element_id:
                return i
        raise KeyError(element_id)


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def resolve(detections: list[Detection], shape: tuple[int, int]) -> ElementSet:
    """Paint detections lowest priority first so higher priority wins every overlap; whatever is left is background.
    Elements that lose all their pixels disappear; ids are numbered per type in detection order."""
    h, w = shape
    labels = np.zeros((h, w), np.int16)  # 0 = background until proven otherwise
    ordered = sorted(enumerate(detections), key=lambda t: (PRIORITY[t[1].type], t[0]))
    paint = {}  # label value -> detection
    for n, (_, d) in enumerate(ordered, start=1):
        labels[d.mask] = n
        paint[n] = d

    elements: list[Element] = [Element("background", "background", "Background", [0, 0, w, h], 0)]
    remap = {0: 0}
    counters: dict[str, int] = {}
    for n in sorted(paint, key=lambda n: (paint[n].type, ordered[n - 1][0])):
        d = paint[n]
        area = int((labels == n).sum())
        if area == 0:
            continue
        k = counters.get(d.type, 0)
        counters[d.type] = k + 1
        remap[n] = len(elements)
        elements.append(Element(f"{d.type}_{k}", d.type, d.label, bbox(labels == n), area, d.text, d.meta))
    lut = np.zeros(max(paint, default=0) + 1, np.int16)
    for old, new in remap.items():
        lut[old] = new
    labels = lut[labels]
    elements[0].area_px = int((labels == 0).sum())
    elements[0].box = bbox(labels == 0)
    return ElementSet(elements, labels)
