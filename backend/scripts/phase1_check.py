"""Phase 1 exit check: run /analyze's pipeline on every thumbnail in data/thumbnails, log timing, save overlay contact sheets.

    .venv\\Scripts\\python scripts\\phase1_check.py [--limit 10] [--force-fallback]
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import data_dir, load_config  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.storage import SqliteStore  # noqa: E402


def overlay(result: dict) -> np.ndarray:
    base = cv2.imdecode(np.frombuffer(base64.b64decode(result["image_jpg"]), np.uint8), cv2.IMREAD_COLOR)
    hm = cv2.imdecode(np.frombuffer(base64.b64decode(result["heatmap_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
    hm = cv2.resize(hm, (base.shape[1], base.shape[0]))
    a = hm[..., 3:4].astype(np.float32) / 255
    return (base * (1 - a) + hm[..., :3] * a).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--force-fallback", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    cfg = {**cfg, "cache": {"enabled": False}}
    if args.force_fallback:
        cfg["saliency"] = {**cfg["saliency"], "backends": ["opencv_fine_grained"]}
    t = time.perf_counter()
    pipe = Pipeline(cfg, store=SqliteStore(data_dir() / "check"))
    print(f"models loaded in {time.perf_counter() - t:.1f}s: {pipe.status}")

    files = sorted(p for p in (data_dir() / "thumbnails").glob("*.jpg"))[: args.limit]
    out_dir = data_dir() / "phase1"
    out_dir.mkdir(exist_ok=True)
    tiles, totals = [], []
    for p in files:
        r = pipe.analyze(p.read_bytes())
        totals.append(r["timing_ms"]["total"])
        tile = cv2.resize(overlay(r), (480, 270))
        cv2.putText(tile, p.stem[:28], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        print(f"{p.name:34s} {r['saliency_model']:20s} total {r['timing_ms']['total']:5d} ms  (saliency {r['timing_ms']['saliency']} ms)  errors={len(r['errors'])}")
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    name = "contact_fallback.jpg" if args.force_fallback else "contact_deepgaze.jpg"
    cv2.imwrite(str(out_dir / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"median {np.median(totals):.0f} ms, max {max(totals)} ms -> {out_dir / name}")


if __name__ == "__main__":
    main()
