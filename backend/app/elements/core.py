"""M4. Element detection: faces, text, subject and the background remainder, or editor layer masks."""
from __future__ import annotations

import logging

import numpy as np

from app.preprocess import Context
from .faces import FaceDetector
from .layers import elements_from_layers
from .model import Detection, ElementSet, resolve
from .subject import SubjectDetector
from .text import TextDetector

log = logging.getLogger("elements")


class ElementEngine:
    """Loads the detectors once. A detector that cannot load or fails on an image is reported and skipped;
    the background remainder always exists, so there is always at least one element."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.detectors: dict[str, object] = {}
        self.load_errors: dict[str, str] = {}

    def load(self) -> None:
        for name, cls in (("faces", FaceDetector), ("text", TextDetector), ("subject", SubjectDetector)):
            try:
                d = cls(self.cfg)
                d.load()
                self.detectors[name] = d
            except Exception as e:  # noqa: BLE001
                self.load_errors[name] = f"{type(e).__name__}: {e}"
                log.warning("element detector %s unavailable: %s", name, self.load_errors[name])

    @property
    def status(self) -> dict:
        return {"loaded": list(self.detectors), "unavailable": self.load_errors}

    @property
    def mobile_ocr(self):
        """OCR for the phone-size legibility test (M5), or None if the text detector is not loaded."""
        t = self.detectors.get("text")
        return t.ocr if t is not None else None

    def detect(self, ctx: Context, layers: list | None = None) -> tuple[ElementSet, list[str]]:
        """Returns the element set and the errors of any detector that failed (its part is simply missing)."""
        h, w = ctx.image.shape[:2]
        if layers:
            return elements_from_layers(layers, (h, w)), []  # layers replace detection entirely
        errors = [f"{n}: {e}" for n, e in self.load_errors.items()]
        dets: list[Detection] = []
        for name in ("faces", "text"):
            d = self.detectors.get(name)
            if d is None:
                continue
            try:
                dets += d.detect(ctx.image)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {e}")
        subject = self.detectors.get("subject")
        if subject is not None:
            try:
                taken = np.zeros((h, w), bool)
                for d in dets:
                    taken |= d.mask
                dets += subject.detect(ctx.image, exclude=taken)
            except Exception as e:  # noqa: BLE001
                errors.append(f"subject: {type(e).__name__}: {e}")
        return resolve(dets, (h, w)), errors
