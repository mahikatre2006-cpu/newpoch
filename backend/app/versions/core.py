"""M11. Versions and comparison: what changed between two analyses of the same thumbnail."""
from __future__ import annotations

import uuid

import numpy as np

from app.elements import decode_rle

IOU_MIN = 0.25  # two detected elements of the same type are "the same element" if their masks overlap this much


def new_version(session_id: str, analysis_id: str, label: str, kind: str, parent_id: str | None = None) -> dict:
    return {"version_id": str(uuid.uuid4()), "session_id": session_id, "analysis_id": analysis_id,
            "label": label[:80], "kind": kind, "parent_id": parent_id}


def _masks(result: dict) -> list[np.ndarray]:
    # a quarter of the resolution is plenty for overlap and 16x cheaper
    return [decode_rle(e["mask_rle"], (720, 1280))[::4, ::4] for e in result["elements"]]


def match_elements(a: dict, b: dict) -> list[tuple[int | None, int | None]]:
    """Pairs (index in a, index in b); None marks an element that exists on one side only.
    Background pairs with background, editor layers pair by name, detected elements by mask overlap and type."""
    ea, eb = a["elements"], b["elements"]
    pairs: list[tuple[int | None, int | None]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    def take(i, j):
        pairs.append((i, j))
        used_a.add(i)
        used_b.add(j)

    for i, x in enumerate(ea):
        if x["type"] == "background":
            j = next((j for j, y in enumerate(eb) if y["type"] == "background"), None)
            if j is not None:
                take(i, j)
    for i, x in enumerate(ea):  # editor layers: the creator's own names
        if x["type"] == "layer" and i not in used_a:
            j = next((j for j, y in enumerate(eb) if y["type"] == "layer" and y["label"] == x["label"] and j not in used_b), None)
            if j is not None:
                take(i, j)
    rest_a = [i for i in range(len(ea)) if i not in used_a and ea[i]["type"] != "layer"]
    rest_b = [j for j in range(len(eb)) if j not in used_b and eb[j]["type"] != "layer"]
    if rest_a and rest_b:
        ma, mb = _masks(a), _masks(b)
        cands = []
        for i in rest_a:
            for j in rest_b:
                if ea[i]["type"] != eb[j]["type"]:
                    continue
                inter = float((ma[i] & mb[j]).sum())
                union = float((ma[i] | mb[j]).sum())
                if union and inter / union >= IOU_MIN:
                    cands.append((inter / union, i, j))
        for _, i, j in sorted(cands, reverse=True):
            if i not in used_a and j not in used_b:
                take(i, j)
    pairs += [(i, None) for i in range(len(ea)) if i not in used_a]
    pairs += [(None, j) for j in range(len(eb)) if j not in used_b]
    return pairs


def compare_results(a: dict, b: dict) -> dict:
    """Per-element attention change from version a to version b, plus both viewing orders."""
    rows = []
    for i, j in match_elements(a, b):
        x = a["elements"][i] if i is not None else None
        y = b["elements"][j] if j is not None else None
        pa, pb = (x["attention_pct"] if x else None), (y["attention_pct"] if y else None)
        rows.append({
            "label": (y or x)["label"], "type": (y or x)["type"], "a_id": x["id"] if x else None, "b_id": y["id"] if y else None,
            "a_pct": pa, "b_pct": pb, "delta_pct": round(pb - pa, 2) if x and y else None,
            "status": "matched" if x and y else ("removed" if x else "new"),
        })
    rows = [r for r in rows if not (r["type"] == "background" and (r["a_pct"] or 0) < 0.05 and (r["b_pct"] or 0) < 0.05)]
    rows.sort(key=lambda r: -max(r["a_pct"] or 0, r["b_pct"] or 0))

    def order(res: dict) -> list[str]:
        lab = {e["id"]: e["label"] for e in res["elements"]}
        return [lab[h] for h in res["hierarchy"] or [] if h in lab]

    return {"rows": rows, "hierarchy_a": order(a), "hierarchy_b": order(b)}


def describe_change(rows: list[dict], fix_label: str) -> str:
    """One sentence for a fix: the biggest movers, with before and after."""
    moved = [r for r in rows if r["status"] == "matched" and r["delta_pct"] is not None and abs(r["delta_pct"]) >= 0.5]
    moved.sort(key=lambda r: -abs(r["delta_pct"]))
    if not moved:
        return f"{fix_label}: no element's attention share moved by 0.5 points or more."
    parts = ", ".join(f"{r['label']} {r['a_pct']:.1f}% → {r['b_pct']:.1f}%" for r in moved[:3])
    return f"{fix_label}: {parts}."
