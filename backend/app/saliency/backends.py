"""Saliency backends. Each turns an RGB frame into a probability map (H x W, float32, sums to 1)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import BACKEND_DIR


def to_distribution(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (w, h), clamp at zero, normalise to sum 1 so element shares are well defined."""
    m = cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    m = np.maximum(m, 0.0)
    total = float(m.sum())
    if not np.isfinite(total) or total <= 1e-12:
        # a flat frame has nothing to attend to: say so with a uniform distribution instead of failing
        return np.full(m.shape, 1.0 / m.size, np.float32)
    return (m / total).astype(np.float32)


class OpenCVFineGrained:
    """Classical bottom-up saliency. No weights, so it can always run; this is the safety net."""

    name = "opencv_fine_grained"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load(self) -> None:
        if not hasattr(cv2, "saliency"):
            raise RuntimeError("opencv-contrib-python-headless is required for cv2.saliency")
        self._engine = cv2.saliency.StaticSaliencyFineGrained_create()

    def predict(self, rgb: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        ok, sal = self._engine.computeSaliency(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("OpenCV saliency failed")
        sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=max(rgb.shape[1] / 60, 1.0))
        return to_distribution(sal, out_size)


class DeepGazeIIE:
    """DeepGaze IIE (deepgaze_pytorch): the primary model. A fixation-density model, so its output is
    already a probability distribution over where people look."""

    name = "deepgaze_iie"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load(self) -> None:
        import torch
        from scipy.special import logsumexp  # noqa: F401  (checked here so a missing dep fails at startup, not per request)

        import deepgaze_pytorch

        torch.set_num_threads(int(self.cfg["saliency"].get("torch_threads", 4)))
        path = Path(self.cfg["saliency"]["deepgaze_iie"]["centerbias"])
        path = path if path.is_absolute() else BACKEND_DIR / path
        self._centerbias = np.load(path)  # 1024x1024 log density from MIT1003
        self._torch = torch
        self._model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()

    def _centerbias_for(self, h: int, w: int) -> np.ndarray:
        from scipy.ndimage import zoom
        from scipy.special import logsumexp

        cb = zoom(self._centerbias, (h / self._centerbias.shape[0], w / self._centerbias.shape[1]), order=0, mode="nearest")
        return cb - logsumexp(cb)

    def predict(self, rgb: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        torch = self._torch
        h, w = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])).float()
        cb = torch.from_numpy(self._centerbias_for(h, w)[None]).float()
        with torch.inference_mode():
            log_density = self._model(x, cb)[0, 0].numpy()
        log_density -= log_density.max()
        return to_distribution(np.exp(log_density), out_size)


class MSINetTF:
    """MSI-Net (Kroner et al. 2020) as the original TensorFlow SavedModel: the optional second backend.
    TensorFlow is imported on first use, so the app starts (and DeepGaze runs) without it; if TensorFlow or the
    weights are missing the backend reports itself unavailable and the chain moves on."""

    name = "msinet_tf"
    lazy = True
    SIZE = (320, 240)  # W, H the model was trained at; 16:9 frames are letterboxed into it

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._sig = None

    def load(self) -> None:
        import importlib.util

        if importlib.util.find_spec("tensorflow") is None:
            raise RuntimeError("tensorflow is not installed")
        path = Path(self.cfg["saliency"]["msinet_tf"]["saved_model"])
        self._path = path if path.is_absolute() else BACKEND_DIR / path
        if not (self._path / "saved_model.pb").exists():
            raise FileNotFoundError(f"MSI-Net SavedModel not found in {self._path}")

    def _signature(self):
        if self._sig is None:
            import tensorflow as tf  # first use; may fail where TensorFlow cannot load (it is blocked on some Windows setups)

            model = tf.saved_model.load(str(self._path))
            self._sig = model.signatures["serving_default"]
            self._in_name = list(self._sig.structured_input_signature[1].keys())[0]
            outs = self._sig.structured_outputs
            self._out_name = next((k for k, v in outs.items() if len(v.shape) == 4), next(iter(outs)))
            self._tf = tf
        return self._sig

    def _infer(self, x: np.ndarray) -> np.ndarray:
        """x: 1 x 240 x 320 x 3 float32, raw 0..255 RGB -> 1 x 240 x 320 x 1 map."""
        sig = self._signature()
        return sig(**{self._in_name: self._tf.constant(x)})[self._out_name].numpy()

    def predict(self, rgb: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        tw, th = self.SIZE
        h, w = rgb.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        top, left = (th - nh) // 2, (tw - nw) // 2
        canvas = np.zeros((th, tw, 3), np.float32)
        canvas[top : top + nh, left : left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        m = self._infer(canvas[None])[0, :, :, 0]
        return to_distribution(m[top : top + nh, left : left + nw], out_size)  # drop the padding before resizing


REGISTRY = {OpenCVFineGrained.name: OpenCVFineGrained, DeepGazeIIE.name: DeepGazeIIE, MSINetTF.name: MSINetTF}
