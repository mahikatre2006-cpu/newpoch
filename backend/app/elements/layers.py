"""M4 editor layers. When the editor sends layers they replace detection: exact masks, user-given names.

Payload: a list, bottom to top, of {"name": str, "type": "image|subject|text|shape" (optional), "mask": base64 PNG}.
The mask's alpha channel (or its grey value when it has no alpha) says where the layer is; any size, it is resized.
Where layers overlap the one on top owns the pixel (what a viewer sees); pixels no layer covers are background.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from .model import Element, ElementSet, bbox


class LayerError(ValueError):
    pass


def decode_mask(b64: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as e:  # noqa: BLE001
        raise LayerError("layer mask is not valid base64") from e
    if img is None:
        raise LayerError("layer mask is not a readable image")
    a = img[..., 3] if img.ndim == 3 and img.shape[2] == 4 else (img if img.ndim == 2 else img[..., 0])
    a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return a > 127


def elements_from_layers(layers: list, shape: tuple[int, int]) -> ElementSet:
    if not isinstance(layers, list) or not layers:
        raise LayerError("layers must be a non-empty list")
    h, w = shape
    labels = np.zeros((h, w), np.int16)
    meta: list[tuple[str, str]] = []
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or not isinstance(layer.get("mask"), str):
            raise LayerError(f"layer {i} needs a base64 'mask'")
        name = str(layer.get("name") or f"Layer {i + 1}")[:60]
        labels[decode_mask(layer["mask"], shape)] = len(meta) + 1  # later (higher) layers overwrite
        meta.append((name, str(layer.get("type") or "layer")))
    elements = [Element("background", "background", "Background", [0, 0, w, h], 0)]
    remap = np.zeros(len(meta) + 1, np.int16)
    for n, (name, ltype) in enumerate(meta, start=1):
        area = int((labels == n).sum())
        if area == 0:
            continue  # fully covered by layers above it
        remap[n] = len(elements)
        elements.append(Element(f"layer_{n - 1}", "layer", name, bbox(labels == n), area, None, {"layer_type": ltype}))
    labels = remap[labels]
    elements[0].area_px = int((labels == 0).sum())
    elements[0].box = bbox(labels == 0)
    return ElementSet(elements, labels)
