"""M8. Element ablation: remove one element from the picture and see where its attention goes.

Small elements (text, graphics) are removed with Telea inpainting; large ones (subject, face) are replaced by a heavily
blurred version of the surrounding picture, because inpainting a big hole smears. The saliency model then runs on the
edited picture. The removed pixels now look like background, so they are counted as background: that makes the changes
of the remaining elements add up to exactly the share that was removed.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.elements import decode_rle
from app.preprocess import build_context
from app.saliency import encode_heatmap_png, encode_preview_jpg

LARGE_TYPES = {"subject", "face"}


class AblationError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def blur_fill(image: np.ndarray, hole: np.ndarray, sigmas: list[int]) -> np.ndarray:
    """Fill `hole` with a blurred copy of what surrounds it (normalised convolution: blur the known pixels and divide by
    the blurred 'known' weight). Finer blurs fill the edges, coarser ones reach into the middle of big holes."""
    img = image.astype(np.float32)
    known = (~hole).astype(np.float32)
    out = img.copy()
    filled = ~hole
    for sigma in sigmas:
        num = cv2.GaussianBlur(img * known[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(known, (0, 0), sigma)
        ok = (den > 1e-3) & ~filled
        out[ok] = num[ok] / den[ok][:, None]
        filled |= ok
        if filled.all():
            break
    if not filled.all():  # a frame that is entirely hole: fall back to the mean colour of whatever is known
        out[~filled] = img[~hole].mean(axis=0) if (~hole).any() else 127
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_element(image: np.ndarray, mask: np.ndarray, large: bool, cfg: dict) -> np.ndarray:
    """The picture without the masked element."""
    ac = cfg["ablation"]
    k = ac["dilate_px"] * (2 if large else 1)
    hole = cv2.dilate(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    if large:
        filled = blur_fill(image, hole, ac["fill_sigmas"])
        soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2.0)[..., None]  # feather the seam
        return np.clip(filled * soft + image * (1 - soft), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, hole.astype(np.uint8) * 255, ac["inpaint_radius"], cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def labels_from_result(result: dict, shape: tuple[int, int]) -> np.ndarray:
    """Rebuild the element label map (index into result['elements']) from the stored run-length masks."""
    labels = np.zeros(shape, np.int16)
    for i, e in enumerate(result["elements"]):
        labels[decode_rle(e["mask_rle"], shape)] = i
    return labels


def ablate(saliency, cfg: dict, image: np.ndarray, result: dict, element_id: str) -> dict:
    """`saliency` is the SaliencyEngine; `image` the stored 1280x720 frame; `result` the stored analysis."""
    els = result.get("elements")
    if not els:
        raise AblationError("This analysis has no elements to remove")
    idx = next((i for i, e in enumerate(els) if e["id"] == element_id), None)
    if idx is None:
        raise AblationError(f"Unknown element {element_id!r}", 404)
    target = els[idx]
    if target["type"] == "background":
        raise AblationError("The background is what is left when everything else is removed; pick an element")
    h, w = image.shape[:2]
    labels = labels_from_result(result, (h, w))
    mask = labels == idx
    large = target["type"] in LARGE_TYPES or target["area_pct"] >= cfg["ablation"]["inpaint_max_area_pct"]
    edited = remove_element(image, mask, large, cfg)

    ctx = build_context(edited, "ablation", cfg)
    sal = saliency.predict(ctx)
    bg = next(i for i, e in enumerate(els) if e["type"] == "background")
    new_labels = labels.copy()
    new_labels[mask] = bg  # the removed pixels now show background
    shares = np.bincount(new_labels.ravel(), weights=sal.map.ravel().astype(np.float64), minlength=len(els))
    shares = shares / shares.sum() * 100

    removed_pct = target["attention_pct"]
    flow = [{"element_id": e["id"], "label": e["label"], "before_pct": e["attention_pct"], "after_pct": round(float(shares[i]), 2),
             "delta_pct": round(float(shares[i]) - e["attention_pct"], 2)}
            for i, e in enumerate(els) if i != idx and not (e["type"] == "background" and e["area_pct"] == 0 and shares[i] < 0.005)]
    flow.sort(key=lambda f: -f["delta_pct"])
    gains = [f for f in flow if f["delta_pct"] >= 0.1][:4]
    if gains:
        parts = ", ".join(f"{f['label']} {f['delta_pct']:+.1f}%" for f in gains)
        message = f"Without {target['label']}, its {removed_pct:.1f}% splits: {parts}."
    else:
        message = f"Without {target['label']}, its {removed_pct:.1f}% of attention spreads thinly across the frame."
    render = cfg["saliency"]["render"]
    return {
        "removed": {"element_id": element_id, "label": target["label"], "attention_pct": removed_pct, "method": "blur_fill" if large else "inpaint"},
        "flow": flow,
        "ablated_heatmap_png": encode_heatmap_png(sal.map, render),
        "ablated_image_jpg": encode_preview_jpg(edited, render),
        "message": message,
        "saliency_model": sal.model,
    }
