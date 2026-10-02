"""M6. Attention distribution and hierarchy: share, density, viewing order, scanpath and the intent check.

The saliency map is a probability distribution (sums to 1) and the element masks partition the frame, so the
element shares sum to 100% by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.elements import ElementSet, encode_rle


@dataclass
class AttentionResult:
    elements: list[dict]
    hierarchy: list[str]
    scanpath: list[dict]
    intent_check: dict | None
    warnings: list[str]


def predict_scanpath(attention: np.ndarray, labels: np.ndarray, element_ids: list[str], cfg: dict) -> list[dict]:
    """Winner-take-all with inhibition of return: take the highest remaining peak, suppress a Gaussian around it so
    the next winner is elsewhere. Fixations weaker than min_ratio x the first are dropped, but at least
    min_fixations are kept (for a map that is not flat). Each fixation is labelled with the element it lands on."""
    h, w = attention.shape
    scale = cfg["work_width"] / w
    small = cv2.resize(attention.astype(np.float32), (cfg["work_width"], max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    if float(small.max()) <= 0 or float(small.max()) - float(small.min()) <= 1e-12 * float(small.max()):
        return []  # a flat map has no winner
    sh, sw = small.shape
    sigma = cfg["inhibition_sigma_frac"] * sw
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    work = small.copy()
    first = float(work.max())
    path: list[dict] = []
    for order in range(1, cfg["max_fixations"] + 1):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if order > cfg["min_fixations"] and float(work[y, x]) < cfg["min_ratio"] * first:
            break
        px, py = min(int(round((x + 0.5) / scale)), w - 1), min(int(round((y + 0.5) / scale)), h - 1)
        path.append({"x": px, "y": py, "order": order, "element_id": element_ids[int(labels[py, px])]})
        work *= 1.0 - np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        work[(xx - x) ** 2 + (yy - y) ** 2 < (cfg["exclusion_frac"] * sw) ** 2] = 0.0  # hard minimum distance between fixations
    return path


def label_fixations(points: list[tuple[int, int]], labels: np.ndarray, element_ids: list[str]) -> list[dict]:
    return [{"x": int(x), "y": int(y), "order": i, "element_id": element_ids[int(labels[y, x])]} for i, (x, y) in enumerate(points, start=1)]


def intent_check(intent: list[str], hierarchy: list[str]) -> tuple[dict, dict[str, int], list[str]]:
    """Compare the order the creator wants with the predicted viewing order, over the elements they ranked.
    Returns (check, intended_rank per element, warnings)."""
    seen: list[str] = []
    for i in intent:
        if i in hierarchy and i not in seen:
            seen.append(i)
    unknown = [i for i in dict.fromkeys(intent) if i not in hierarchy]
    intended = {eid: n for n, eid in enumerate(seen, start=1)}
    predicted_order = [e for e in hierarchy if e in intended]
    mismatches = []
    for rank, eid in enumerate(predicted_order, start=1):
        if rank != intended[eid]:
            mismatches.append({"element_id": eid, "intended": intended[eid], "predicted": rank, "gap": rank - intended[eid]})
    mismatches.sort(key=lambda m: intended[m["element_id"]])
    check = {"matches": not mismatches, "mismatches": mismatches}
    return check, intended, [f"intent: unknown element id {u!r}" for u in unknown]


def analyze_attention(sal: np.ndarray, eset: ElementSet, scan_cfg: dict, intent: list[str] | None = None,
                      fixations: list[tuple[int, int]] | None = None) -> AttentionResult:
    """`fixations`: a predicted gaze sequence from a scanpath model. When given it sets the viewing order (elements in the
    order they are first looked at, then the rest by peak attention); otherwise the order is by peak attention and the
    scanpath comes from winner-take-all on the map."""
    h, w = sal.shape
    n = len(eset.elements)
    flat_labels = eset.labels.ravel()
    share = np.bincount(flat_labels, weights=sal.ravel().astype(np.float64), minlength=n)
    share = share / share.sum()  # guards float drift; the map already sums to 1
    peaks = np.zeros(n)
    for i in range(n):
        m = eset.labels == i
        peaks[i] = float(sal[m].max()) if m.any() else 0.0

    area = np.array([e.area_px for e in eset.elements], dtype=np.float64) / (h * w)
    order = sorted(range(n), key=lambda i: (-peaks[i], -share[i], i))  # peak first, share breaks ties, index keeps it deterministic
    ids = [e.id for e in eset.elements]
    path = label_fixations(fixations, eset.labels, ids) if fixations else predict_scanpath(sal, eset.labels, ids, scan_cfg)
    if fixations:  # what viewers look at first is what the sequence visits first
        visited = list(dict.fromkeys(ids.index(p["element_id"]) for p in path))
        order = visited + [i for i in order if i not in visited]
    hierarchy = [eset.elements[i].id for i in order]
    rank = {eset.elements[i].id: r for r, i in enumerate(order, start=1)}

    check, intended, warnings = (None, {}, [])
    if intent:
        check, intended, warnings = intent_check(intent, hierarchy)

    elements = []
    for i, e in enumerate(eset.elements):
        d = {
            "id": e.id, "type": e.type, "label": e.label, "box": e.box,
            "mask_rle": encode_rle(eset.labels == i),
            "area_pct": round(float(area[i]) * 100, 2),
            "attention_pct": round(float(share[i]) * 100, 2),
            "density": round(float(share[i] / area[i]), 2) if area[i] > 0 else None,
            "predicted_rank": rank[e.id],
            "intent_rank": intended.get(e.id),
            "features": None, "top_driver": None, "explanations": None,  # M5 and M7
        }
        if e.text is not None:
            d["text"] = e.text
        if e.meta:
            d["meta"] = e.meta
        elements.append(d)

    return AttentionResult(elements, hierarchy, path, check, warnings)
