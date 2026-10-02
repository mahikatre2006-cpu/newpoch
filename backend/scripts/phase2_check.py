"""Phase 2 exit check: on every thumbnail, element masks must cover every pixel exactly once and attention shares must sum to 100%.
Writes a contact sheet (element colours + attention % + scanpath) to data/phase2/.

    .venv\\Scripts\\python scripts\\phase2_check.py [--limit 24]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import data_dir, load_config  # noqa: E402
from app.elements import decode_rle  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.storage import SqliteStore  # noqa: E402
import base64  # noqa: E402

COLORS = {"face": (0, 200, 255), "text": (255, 220, 0), "subject": (170, 90, 255), "background": (90, 90, 90), "layer": (0, 255, 120)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()
    cfg = {**load_config(), "cache": {"enabled": False}}
    t = time.perf_counter()
    pipe = Pipeline(cfg, store=SqliteStore(data_dir() / "check"))
    print(f"loaded in {time.perf_counter() - t:.1f}s: {pipe.status}")

    tiles, totals, bad = [], [], 0
    for p in sorted((data_dir() / "thumbnails").glob("*.jpg"))[: args.limit]:
        r = pipe.analyze(p.read_bytes())
        totals.append(r["timing_ms"]["total"])
        els = r["elements"]
        cover = np.zeros((720, 1280), np.int16)
        overlay = cv2.imdecode(np.frombuffer(base64.b64decode(r["image_jpg"]), np.uint8), cv2.IMREAD_COLOR)
        overlay = cv2.resize(overlay, (1280, 720))
        tint = np.zeros_like(overlay)
        for e in els:
            m = decode_rle(e["mask_rle"], (720, 1280))
            cover += m
            tint[m] = COLORS[e["type"]][::-1]
        ok_partition = bool((cover == 1).all())
        total_share = sum(e["attention_pct"] for e in els)
        ok_sum = abs(total_share - 100) <= 0.5
        bad += not (ok_partition and ok_sum)
        out = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0)
        for s in r["scanpath"]:
            cv2.circle(out, (s["x"], s["y"]), 26, (255, 255, 255), 4)
            cv2.putText(out, str(s["order"]), (s["x"] - 10, s["y"] + 11), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        for e in els:
            if e["type"] != "background" and e["attention_pct"] >= 1:
                x0, y0, x1, y1 = e["box"]
                cv2.putText(out, f"{e['type']} {e['attention_pct']:.0f}%", (x0 + 6, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4, cv2.LINE_AA)
        tile = cv2.resize(out, (480, 270))
        tiles.append(tile)
        summary = " ".join(f"{e['id']}={e['attention_pct']:.0f}%/{e['area_pct']:.0f}%" for e in els)
        print(f"{p.stem[:26]:26s} {r['timing_ms']['total']:5d}ms part={'ok' if ok_partition else 'BAD'} sum={total_share:6.2f} {'ok' if ok_sum else 'BAD'} | {summary}")
        if r["errors"]:
            print("   errors:", r["errors"])
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    out_dir = data_dir() / "phase2"
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "contact_elements.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"median {np.median(totals):.0f} ms, max {max(totals)} ms; failures: {bad}; sheet -> {out_dir / 'contact_elements.jpg'}")


if __name__ == "__main__":
    main()
