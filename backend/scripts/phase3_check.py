"""Phase 3 exit check: every element of every thumbnail has at least one explanation citing a measured number.
Prints the measurements and sentences so they can be judged by eye.

    python scripts\\phase3_check.py [--limit 24] [--show 6]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import data_dir, load_config  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.storage import SqliteStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()
    cfg = {**load_config(), "cache": {"enabled": False}}
    pipe = Pipeline(cfg, store=SqliteStore(data_dir() / "check"))
    bad, n_el, totals = 0, 0, []
    for k, p in enumerate(sorted((data_dir() / "thumbnails").glob("*.jpg"))[: args.limit]):
        r = pipe.analyze(p.read_bytes())
        totals.append(r["timing_ms"]["total"])
        for e in r["elements"]:
            n_el += 1
            ex = e["explanations"] or []
            if not ex or not all(re.search(r"\d", m) for m in ex):
                bad += 1
                print("MISSING/NO NUMBER:", p.stem, e["id"], ex)
        if r["errors"]:
            print(p.stem, "errors:", r["errors"])
        if k < args.show:
            f = r["frame"]
            print(f"\n=== {p.stem}  ({r['timing_ms']['total']} ms, features {r['timing_ms'].get('features')} ms)")
            print("palette:", ", ".join(f"{c['name']} {c['share']:.0%}" for c in f["palette"]), "| balance", f["balance_offset"], "| focal", f["focal_points"], "| empty", f["negative_space_pct"], "%")
            for m in f["explanations"]:
                print("  [frame]", m)
            for e in r["elements"]:
                ft = e["features"]
                extra = ""
                if e["type"] == "text":
                    extra = f" contrast={ft.get('text_contrast_ratio')} mobile={ft.get('mobile_legible')}({ft.get('mobile_match_pct')}%)"
                print(f"  {e['id']:12s} {e['label'][:22]:22s} att={e['attention_pct']:5.1f}% area={e['area_pct']:5.1f}% lc={ft.get('local_contrast')} sat={ft.get('saturation_pop')} dist={ft.get('colour_distinctness')} safe={ft.get('safe_zone_overlap_pct')}{extra}")
                for m in e["explanations"]:
                    print("      -", m)
    print(f"\nelements checked: {n_el}, without a numbered explanation: {bad}; median {sorted(totals)[len(totals) // 2]} ms")


if __name__ == "__main__":
    main()
