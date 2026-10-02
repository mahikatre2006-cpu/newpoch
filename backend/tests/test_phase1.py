"""Phase 1: M1 ingest, M2 preprocess, M3 saliency (+ fallback), /analyze, cache."""
from __future__ import annotations

import io
from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import load_config
from app.ingest import IngestError, ingest_upload, strip_bars
from app.pipeline import Pipeline
from app.preprocess import build_context
from app.saliency import SaliencyEngine
from app.saliency.backends import REGISTRY
from app.storage import SqliteStore
from tests.conftest import NullElements, NullScanpath

CFG = load_config()


def png_bytes(w=1280, h=720, fmt="PNG", color=(200, 30, 30)) -> bytes:
    img = np.full((h, w, 3), 40, np.uint8)
    img[h // 4 : h // 2, w // 8 : w // 3] = color  # a bright block to attend to
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, fmt)
    return buf.getvalue()


def fallback_cfg() -> dict:
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["opencv_fine_grained"]
    return cfg


@pytest.fixture()
def pipe(tmp_path):
    return Pipeline(fallback_cfg(), store=SqliteStore(tmp_path), elements=NullElements(), scanpath=NullScanpath())


# ---------- M1 ----------
def test_ingest_normalises_to_1280x720_and_hashes():
    a = ingest_upload(png_bytes(1920, 1080), CFG["ingest"])
    assert a.image.shape == (720, 1280, 3) and a.image.dtype == np.uint8
    assert a.sha256 == ingest_upload(png_bytes(1920, 1080), CFG["ingest"]).sha256
    assert a.sha256 != ingest_upload(png_bytes(1920, 1080, color=(30, 200, 30)), CFG["ingest"]).sha256


def test_layer_payload_changes_the_cache_key():
    base = ingest_upload(png_bytes(), CFG["ingest"]).sha256
    with_layers = ingest_upload(png_bytes(), CFG["ingest"], layers=[{"name": "Title"}]).sha256
    assert base != with_layers
    assert with_layers == ingest_upload(png_bytes(), CFG["ingest"], layers=[{"name": "Title"}]).sha256


def test_ingest_far_from_16x9_is_letterboxed_not_cropped():
    a = ingest_upload(png_bytes(600, 600), CFG["ingest"])
    assert a.info["fit"] == "letterbox" and a.image.shape == (720, 1280, 3)


@pytest.mark.parametrize("data,msg", [(b"", "Empty"), (b"not an image", "readable"), (png_bytes(100, 50), "too small")])
def test_ingest_rejects_bad_input(data, msg):
    with pytest.raises(IngestError, match=msg):
        ingest_upload(data, CFG["ingest"])


def test_ingest_rejects_other_formats():
    buf = io.BytesIO()
    Image.new("RGB", (640, 360)).save(buf, "GIF")
    with pytest.raises(IngestError, match="Unsupported"):
        ingest_upload(buf.getvalue(), CFG["ingest"])


def test_strip_bars_removes_letterbox():
    img = np.zeros((360, 480, 3), np.uint8)
    img[45:315] = 180
    assert strip_bars(img).shape[0] == 270


def test_video_id_is_validated_before_any_request():
    from app.ingest import ingest_video_id

    with pytest.raises(IngestError, match="valid"):
        ingest_video_id("../../etc/passwd", CFG["ingest"])


# ---------- M2 ----------
def test_context_has_all_views_and_safe_zone():
    ing = ingest_upload(png_bytes(), CFG["ingest"])
    ctx = build_context(ing.image, ing.sha256, CFG)
    assert ctx.saliency_input.shape == (288, 512, 3)
    assert ctx.mobile_small.shape == (94, 168, 3) and ctx.mobile_up.shape == (720, 1280, 3)
    sz = ctx.safe_zone
    assert sz[700, 1250] == 1 and sz[10, 10] == 0 and sz[715, 10] == 1  # badge, free area, progress bar
    badge_w = (sz[690] == 1).sum() / 1280
    assert 0.13 < badge_w < 0.15


# ---------- M3 ----------
def test_fallback_map_is_a_distribution(pipe):
    ctx = build_context(*(lambda i: (i.image, i.sha256))(ingest_upload(png_bytes(), CFG["ingest"])), CFG)
    res = pipe.saliency.predict(ctx)
    assert res.model == "opencv_fine_grained"
    assert res.map.shape == (720, 1280) and abs(float(res.map.sum()) - 1.0) < 1e-4 and res.map.min() >= 0


def test_failing_primary_falls_back_automatically(monkeypatch):
    class Boom:
        name = "deepgaze_iie"

        def __init__(self, cfg): ...
        def load(self): ...
        def predict(self, rgb, size):
            if rgb.shape[1] <= 256:  # the start-up warm-up passes; it breaks later, at run time
                return np.full((size[1], size[0]), 1.0 / (size[0] * size[1]), np.float32)
            raise RuntimeError("weights exploded")

    monkeypatch.setitem(REGISTRY, "deepgaze_iie", Boom)
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["deepgaze_iie", "opencv_fine_grained"]
    eng = SaliencyEngine(cfg)
    eng.load()
    ing = ingest_upload(png_bytes(), cfg["ingest"])
    res = eng.predict(build_context(ing.image, ing.sha256, cfg))
    assert res.model == "opencv_fine_grained" and any("weights exploded" in e for e in res.errors)


def test_missing_primary_at_load_time_is_reported_not_fatal(monkeypatch):
    class Missing:
        name = "deepgaze_iie"

        def __init__(self, cfg): ...
        def load(self):
            raise FileNotFoundError("centerbias missing")

    monkeypatch.setitem(REGISTRY, "deepgaze_iie", Missing)
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["deepgaze_iie", "opencv_fine_grained"]
    eng = SaliencyEngine(cfg)
    eng.load()
    assert eng.status["fallback_active"] and "deepgaze_iie" in eng.status["unavailable"]


def test_no_backend_at_all_raises():
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = []
    with pytest.raises(RuntimeError):
        SaliencyEngine(cfg).load()


# ---------- pipeline + API ----------
def test_analyze_returns_contract_fields_and_caches(pipe):
    r1 = pipe.analyze(png_bytes())
    assert r1["saliency_model"] == "opencv_fine_grained" and not r1["cached"]
    assert {"ingest", "preprocess", "saliency", "render", "total"} <= set(r1["timing_ms"])
    r2 = pipe.analyze(png_bytes())
    assert r2["cached"] and r2["analysis_id"] == r1["analysis_id"] and r2["heatmap_png"] == r1["heatmap_png"]


def test_cache_is_not_shared_across_model_changes(tmp_path):
    store = SqliteStore(tmp_path)
    Pipeline(fallback_cfg(), store=store, elements=NullElements(), scanpath=NullScanpath()).analyze(png_bytes())
    other = deepcopy(CFG)
    other["saliency"]["backends"] = ["opencv_fine_grained"]
    other["saliency"]["render"]["display_gamma"] = 0.9  # any config change -> new signature
    assert not Pipeline(other, store=store, elements=NullElements(), scanpath=NullScanpath()).analyze(png_bytes())["cached"]


@pytest.fixture()
def client(pipe):
    from app.main import app

    app.state.pipeline = pipe  # the lifespan keeps an injected pipeline instead of loading DeepGaze
    with TestClient(app) as c:
        yield c
    app.state.pipeline = None


def test_api_analyze_and_health(client):
    r = client.post("/analyze", files={"image": ("t.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saliency_model"] and body["heatmap_png"] and body["image_jpg"] and body["errors"] == []
    assert [e["id"] for e in body["elements"]] == ["background"]
    assert client.get("/health").json()["status"] == "ok"


def test_api_rejects_garbage_and_missing_input(client):
    assert client.post("/analyze", files={"image": ("t.png", b"nope", "image/png")}).status_code == 422
    assert client.post("/analyze", data={}).status_code == 422
    assert client.post("/analyze", data={"video_id": "bad id"}).status_code == 422
    r = client.post("/analyze", files={"image": ("t.png", png_bytes(), "image/png")}, data={"intent": "{not json"})
    assert r.status_code == 422


def test_registry_lists_all_backends():
    assert set(REGISTRY) == {"deepgaze_iie", "msinet_tf", "opencv_fine_grained"}


# ---------- optional TensorFlow backend ----------
def test_msinet_tf_letterboxes_and_unpads(monkeypatch):
    from app.saliency.backends import MSINetTF

    b = MSINetTF(CFG)
    seen = {}

    def fake(x):
        seen["shape"], seen["pad_rows"] = x.shape, (x[0].max(axis=(1, 2)) == 0).sum()
        out = np.zeros((1, 240, 320, 1), np.float32)
        out[0, 60:180] = 1.0  # saliency only inside the real picture (16:9 -> 180 rows, 30 px bars)
        out[0, 100:140, 150:200] = 5.0
        return out

    monkeypatch.setattr(b, "_infer", fake)
    m = b.predict(np.full((288, 512, 3), 200, np.uint8), (1280, 720))
    assert seen["shape"] == (1, 240, 320, 3) and seen["pad_rows"] == 60  # 30 black rows top and bottom
    assert m.shape == (720, 1280) and abs(float(m.sum()) - 1) < 1e-4
    # hot patch: canvas rows 100:140 -> 70:110 after dropping 30 padded rows -> x4 = 280:440; cols 150:200 -> 600:800
    assert m[280:440, 600:800].sum() > 0.12  # ~15% of the mass in 3.5% of the area, so it landed in the right place


def test_msinet_tf_unavailable_without_tensorflow_is_silent(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "tensorflow" else object())
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["msinet_tf", "opencv_fine_grained"]
    eng = SaliencyEngine(cfg)
    eng.load()
    assert "msinet_tf" in eng.status["unavailable"] and eng.status["loaded"] == ["opencv_fine_grained"]
    ing = ingest_upload(png_bytes(), cfg["ingest"])
    assert eng.predict(build_context(ing.image, ing.sha256, cfg)).errors == []  # optional: not an error for the user


def test_flat_frame_gives_a_uniform_map_not_an_error(pipe):
    flat = np.full((720, 1280, 3), 128, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(flat).save(buf, "PNG")
    r = pipe.analyze(buf.getvalue())
    assert r["saliency_model"] == "opencv_fine_grained" and r["errors"] == []
