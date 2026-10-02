"""Time each module alone and all together on one thumbnail (scratch benchmark)."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import data_dir, load_config  # noqa: E402
from app.ingest import ingest_upload  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.preprocess import build_context  # noqa: E402
from app.storage import SqliteStore  # noqa: E402

cfg = {**load_config(), "cache": {"enabled": False}}
p = Pipeline(cfg, store=SqliteStore(data_dir() / "check"))
f = sorted((data_dir() / "thumbnails").glob("*.jpg"))[3]
ing = ingest_upload(f.read_bytes(), cfg["ingest"])
ctx = build_context(ing.image, ing.sha256, cfg)
fns = {"saliency": lambda: p.saliency.predict(ctx), "scanpath": lambda: p.scanpath.predict(ctx), "elements": lambda: p.elements.detect(ctx)}
for n, fn in fns.items():
    fn()  # warm
    t = time.perf_counter()
    for _ in range(3):
        fn()
    print(f"{n:9s} alone: {(time.perf_counter() - t) / 3 * 1000:6.0f} ms")
for label, names in (("saliency+elements", ["saliency", "elements"]), ("all three", list(fns))):
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda n: fns[n](), names))
        ts.append((time.perf_counter() - t) * 1000)
    print(f"{label:18s} parallel: {sum(ts) / 3:6.0f} ms")
