"""M3b. Predicted gaze sequence with DeepGaze III: each next fixation is predicted from the image and the fixations so far,
so the order (and the inhibition of return) comes from a model of human scanpaths rather than from ranking peaks of a static map.

The chain is deterministic: free viewing starts at the centre of the frame, and at every step the most likely next
fixation is taken, after ruling out the area around the start and around every fixation already made (people do not
re-fixate the same spot straight away). Without that exclusion the model's best guess is simply "stay where you are"."""
from __future__ import annotations

import logging
import threading

import numpy as np

from app.config import BACKEND_DIR
from app.preprocess import Context

log = logging.getLogger("scanpath")


def _memoise_features(model):
    """DeepGaze III recomputes its DenseNet features for every fixation although the image never changes.
    Wrap the feature extractor so the (expensive) backbone runs once per image and the light heads run per fixation."""
    import torch

    class Memo(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.key = None
            self.out = None

        def clear(self):
            self.key = self.out = None

        def forward(self, x):
            if self.key is not None and self.key.shape == x.shape and torch.equal(self.key, x):
                return self.out
            self.out = self.inner(x)
            self.key = x.clone()
            return self.out

    model.features = Memo(model.features)
    return model.features


class ScanpathEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sc = cfg["scanpath"]
        self.model = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Optional: if DeepGaze III cannot load, the pipeline falls back to the winner-take-all path on the saliency map."""
        try:
            import torch

            import deepgaze_pytorch

            torch.set_num_threads(int(self.cfg["saliency"].get("torch_threads", 4)))
            path = self.cfg["saliency"]["deepgaze_iie"]["centerbias"]
            path = BACKEND_DIR / path
            self._centerbias = np.load(path)
            self._torch = torch
            self.model = deepgaze_pytorch.DeepGazeIII(pretrained=True).eval()
            self._memo = _memoise_features(self.model)
        except Exception as e:  # noqa: BLE001
            self.model = None
            self.error = f"{type(e).__name__}: {e}"
            log.warning("scanpath model unavailable: %s", self.error)

    @property
    def status(self) -> dict:
        return {"loaded": ["deepgaze_iii"] if self.model is not None else [], "unavailable": {"deepgaze_iii": self.error} if self.error else {}}

    def predict(self, ctx: Context) -> list[tuple[int, int]]:
        """Fixations as (x, y) in frame (1280x720) pixels, in viewing order."""
        from scipy.ndimage import zoom
        from scipy.special import logsumexp

        torch = self._torch
        img = ctx.saliency_input
        h, w = img.shape[:2]
        cb = zoom(self._centerbias, (h / self._centerbias.shape[0], w / self._centerbias.shape[1]), order=0, mode="nearest")
        cb = cb - logsumexp(cb)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)[None])).float()
        c = torch.from_numpy(cb[None]).float()
        radius = self.sc["exclusion_frac"] * w
        yy, xx = np.mgrid[0:h, 0:w]
        hx, hy = [w / 2.0] * 4, [h / 2.0] * 4  # free viewing starts at the centre
        out: list[tuple[int, int]] = []
        with self._lock:
            self._memo.clear()
            for _ in range(self.sc["fixations"]):
                with torch.inference_mode():
                    ld = self.model(x, c, torch.tensor([hx[-4:]]).float(), torch.tensor([hy[-4:]]).float())[0, 0].numpy().copy()
                for px, py in zip(hx[3:], hy[3:]):  # the start and every fixation so far
                    ld[(xx - px) ** 2 + (yy - py) ** 2 < radius**2] = -np.inf
                if not np.isfinite(ld).any():
                    break
                iy, ix = np.unravel_index(int(np.argmax(ld)), ld.shape)
                hx.append(float(ix))
                hy.append(float(iy))
                out.append((min(int(round((ix + 0.5) * ctx.image.shape[1] / w)), ctx.image.shape[1] - 1),
                            min(int(round((iy + 0.5) * ctx.image.shape[0] / h)), ctx.image.shape[0] - 1)))
        return out
