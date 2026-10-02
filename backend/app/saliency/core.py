"""M3. Saliency with an automatic fallback chain, so the pipeline never breaks."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from app.preprocess import Context
from .backends import REGISTRY

log = logging.getLogger("saliency")


@dataclass
class SaliencyResult:
    map: np.ndarray  # H x W float32 summing to 1
    model: str  # which backend actually produced it
    errors: list[str] = field(default_factory=list)  # backends that were tried first and failed


class SaliencyEngine:
    """Loads every configured backend once at startup, then serves predictions.
    A backend that fails to load is skipped; one that fails at run time hands over to the next."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backends = []
        self.load_errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        for name in self.cfg["saliency"]["backends"]:
            try:
                b = REGISTRY[name](self.cfg)
                b.load()
                if not getattr(b, "lazy", False):  # lazy backends import their framework on first use, so no warm-up
                    b.predict(np.full((144, 256, 3), 128, np.uint8), (256, 144))  # the first pass is slow (allocations)
                self.backends.append(b)
            except Exception as e:  # noqa: BLE001: any failure must degrade to the next backend
                self.load_errors[name] = f"{type(e).__name__}: {e}"
                log.warning("saliency backend %s unavailable: %s", name, self.load_errors[name])
        if not self.backends:
            raise RuntimeError(f"No saliency backend could load: {self.load_errors}")

    @property
    def status(self) -> dict:
        return {"loaded": [b.name for b in self.backends], "unavailable": self.load_errors,
                "fallback_active": bool(self.backends) and self.backends[0].name != self.cfg["saliency"]["backends"][0]}

    def predict(self, ctx: Context) -> SaliencyResult:
        h, w = ctx.image.shape[:2]
        optional = set(self.cfg["saliency"].get("optional_backends", []))
        errors = [f"{n}: {e}" for n, e in self.load_errors.items() if n not in optional]
        for b in self.backends:
            try:
                with self._lock:  # one forward pass at a time: the model is the CPU bottleneck anyway
                    m = b.predict(ctx.saliency_input, (w, h))
                return SaliencyResult(map=m, model=b.name, errors=errors)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{b.name}: {type(e).__name__}: {e}")
                log.warning("saliency backend %s failed at run time: %s", b.name, e)
        raise RuntimeError("; ".join(errors) or "no saliency backend")
