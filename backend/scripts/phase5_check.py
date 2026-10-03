"""Phase 5 exit check on real thumbnails: ablation deltas add up to the removed share; each fix shows a measured change.
Writes a contact sheet (original | ablated | focus | text | enhance) to data/phase5/.

    python scripts\\phase5_check.py [--limit 6]
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import data_dir, load_config  # noqa: E402
from app.fixes import FixError  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.storage import SqliteStore  # noqa: E402


def jpg(b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = cv2.resize(img, (426, 240))
    cv2.rectangle(out, (0, 0), (426, 22), (0, 0, 0), -1)
    cv2.putText(out, text[:50], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--every", type=int, default=4, help="take every Nth thumbnail so the sample is varied")
    args = ap.parse_args()
    cfg = load_config()
    pipe = Pipeline(cfg, store=SqliteStore(data_dir() / "check5"))
    files = sorted((data_dir() / "thumbnails").glob("*.jpg"))[:: args.every][: args.limit]
    rows, worst = [], 0.0
    for p in files:
        r = pipe.analyze(p.read_bytes())
        aid = r["analysis_id"]
        v0 = pipe.record_version(r["session_id"], aid, p.stem, "upload")
        print(f"\n=== {p.stem}  elements: {[(e['id'], e['attention_pct']) for e in r['elements'] if e['attention_pct'] >= 1]}")
        tiles = [label(jpg(r["image_jpg"]), "original")]

        # ---- ablation of the element with the most attention that is not background ----
        top = max((e for e in r["elements"] if e["type"] != "background"), key=lambda e: e["attention_pct"], default=None)
        if top:
            a = pipe.ablate(aid, top["id"])
            total = sum(f["delta_pct"] for f in a["flow"])
            err = abs(total - a["removed"]["attention_pct"])
            worst = max(worst, err)
            print(f"  ablate {top['id']} ({a['removed']['method']}): removed {a['removed']['attention_pct']}%, deltas sum {total:.2f} (error {err:.2f}) | {a['message']}")
            tiles.append(label(jpg(a["ablated_image_jpg"]), f"without {top['label'][:24]}"))
        # ---- fixes ----
        for fix in ("focus_subject", "fix_text", "enhance"):
            try:
                f = pipe.fix(aid, fix, r["session_id"])
                print(f"  {fix}: {f['message']}")
                tiles.append(label(jpg(f["analysis"]["image_jpg"]), f["label"]))
            except FixError as e:
                print(f"  {fix}: {e}")
                tiles.append(label(np.zeros((240, 426, 3), np.uint8), f"{fix}: n/a"))
        try:
            lay = pipe.fix(aid, "separate_layers", r["session_id"])
            print("  separate_layers:", [(l["name"], l["width"], l["height"], len(l["data"]) // 1024, "KB") for l in lay["layers"]])
        except FixError as e:
            print("  separate_layers:", e)
        vs = pipe.versions(r["session_id"])
        if len(vs) >= 2:
            c = pipe.compare(vs[0]["version_id"], vs[-1]["version_id"])
            print(f"  versions: {[v['label'] for v in vs]}; compare first->last rows: {[(x['label'], x['a_pct'], x['b_pct']) for x in c['rows'][:3]]}")
        while len(tiles) < 5:
            tiles.append(np.zeros((240, 426, 3), np.uint8))
        rows.append(np.hstack(tiles[:5]))
    out = data_dir() / "phase5"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "contact_fixes.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\nworst ablation sum error: {worst:.2f} points (limit 1.0) -> {out / 'contact_fixes.jpg'}")


if __name__ == "__main__":
    main()
