"""Phase 4: what the editor sends (flattened image + one named mask per layer) and what the backend does with it."""
from __future__ import annotations

import base64
import io
from copy import deepcopy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import load_config
from app.elements import elements_from_layers, is_text_like
from app.explain import Explainer
from app.features import compute_features
from app.ingest import ingest_upload
from app.pipeline import Pipeline
from app.preprocess import build_context
from app.storage import SqliteStore
from tests.conftest import NullScanpath

CFG = load_config()
H, W = 720, 1280


def mask_png(x0, y0, x1, y1) -> str:
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 255
    ok, png = cv2.imencode(".png", m)
    return base64.b64encode(png.tobytes()).decode()


def scene(title_x: int) -> tuple[bytes, list[dict]]:
    """Dark background, an orange block, and a white title bar at x = title_x: what the editor would flatten and send."""
    img = np.full((H, W, 3), 25, np.uint8)
    img[160:560, 760:1160] = (245, 150, 20)
    img[330:450, title_x : title_x + 480] = 250
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    layers = [
        {"name": "Backdrop", "type": "shape", "mask": mask_png(0, 0, W, H)},
        {"name": "Product", "type": "shape", "mask": mask_png(760, 160, 1160, 560)},
        {"name": "Title", "type": "text", "text": "BIG TITLE", "mask": mask_png(title_x, 330, title_x + 480, 450)},
    ]
    return buf.tobytes(), layers


def fast_pipe(tmp_path):
    class NoDetect:  # layers bypass detection, so the real engine is not needed
        status = {"loaded": ["none"], "unavailable": {}}

        def detect(self, ctx, layers=None):
            return elements_from_layers(layers, ctx.image.shape[:2]), []

    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["opencv_fine_grained"]
    return Pipeline(cfg, store=SqliteStore(tmp_path), elements=NoDetect(), scanpath=NullScanpath())


def share(result, label):
    return next(e["attention_pct"] for e in result["elements"] if e["label"] == label)


def test_text_layers_carry_their_string_and_are_text_like():
    _, layers = scene(100)
    es = elements_from_layers(layers, (H, W))
    title = next(e for e in es.elements if e.label == "Title")
    assert title.text == "BIG TITLE" and is_text_like(title) and title.meta["layer_type"] == "text"
    assert next(e for e in es.elements if e.label == "Product").text is None
    layers[2]["text"] = "   "
    assert next(e for e in elements_from_layers(layers, (H, W)).elements if e.label == "Title").text is None
    layers[2]["text"] = "x" * 500
    assert len(next(e for e in elements_from_layers(layers, (H, W)).elements if e.label == "Title").text) == 200


def test_text_layer_gets_text_measurements_and_text_rules():
    png, layers = scene(100)
    ing = ingest_upload(png, CFG["ingest"])
    ctx = build_context(ing.image, ing.sha256, CFG)
    es = elements_from_layers(layers, (H, W))
    sal = np.full((H, W), 1.0 / (H * W), np.float32)

    class Line:
        def __init__(self):
            self.text, self.box, self.score = "other words", (0, 0, 10, 10), 0.9

    fr = compute_features(ctx, es, sal, CFG, mobile_ocr=lambda im: [Line()])  # phone OCR finds nothing at the title
    title = fr.elements["layer_2"]
    assert title["text_contrast_ratio"] > 10 and title["mobile_legible"] is False and title["mobile_match_pct"] == 0
    assert "text_contrast_ratio" not in fr.elements["layer_1"]  # a shape is not text
    assert fr.layer_views["text"][0]["element_id"] == "layer_2"
    from app.attention import analyze_attention

    att = analyze_attention(sal, es, CFG["attention"]["scanpath"])
    Explainer().explain(att.elements, fr.elements, fr.frame)
    title_el = next(e for e in att.elements if e["id"] == "layer_2")
    assert any("not readable at phone size" in m for m in title_el["explanations"])  # the text rule applied to a layer


def test_layers_flow_through_the_pipeline_with_names_as_labels(tmp_path):
    png, layers = scene(100)
    r = fast_pipe(tmp_path).analyze(png, layers=layers)
    assert [e["label"] for e in r["elements"]][:1] == ["Background"]
    assert {"Backdrop", "Product", "Title"} <= {e["label"] for e in r["elements"]}
    assert abs(sum(e["attention_pct"] for e in r["elements"]) - 100) < 0.05
    assert r["errors"] == []


def test_moving_a_text_layer_changes_its_share_of_attention(tmp_path):
    """The phase 4 exit check, on the backend side: same layers, title moved, share of the title moves."""
    p = fast_pipe(tmp_path)
    a = p.analyze(*scene(100)[:1], layers=scene(100)[1])
    b = p.analyze(*scene(380)[:1], layers=scene(380)[1])  # title slid right, now touching the orange block's neighbourhood
    assert abs(share(a, "Title") - share(b, "Title")) > 0.3
    assert a["analysis_id"] != b["analysis_id"]  # different pixels and masks: separate cache entries
    c = p.analyze(*scene(100)[:1], layers=scene(100)[1])
    assert c["cached"] and share(c, "Title") == share(a, "Title")


def test_a_fully_covered_layer_disappears_from_the_budget(tmp_path):
    png, layers = scene(100)
    layers.append({"name": "Cover", "type": "shape", "mask": mask_png(0, 0, W, H)})  # hides everything below
    r = fast_pipe(tmp_path).analyze(png, layers=layers)
    labels = {e["label"] for e in r["elements"]}
    assert "Cover" in labels and not labels & {"Backdrop", "Product", "Title"}


def test_layers_over_http_including_bad_payloads(tmp_path):
    import json

    from app.main import app

    app.state.pipeline = fast_pipe(tmp_path)
    png, layers = scene(100)
    with TestClient(app) as c:
        ok = c.post("/analyze", files={"image": ("e.png", png, "image/png")}, data={"layers": json.dumps(layers)})
        assert ok.status_code == 200 and "Title" in {e["label"] for e in ok.json()["elements"]}
        bad = c.post("/analyze", files={"image": ("e.png", png, "image/png")}, data={"layers": json.dumps([{"name": "x", "mask": "###"}])})
        assert bad.status_code == 422
    app.state.pipeline = None


def test_an_empty_layer_list_means_ordinary_detection(tmp_path):
    from app.main import app
    from tests.conftest import NullElements

    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["opencv_fine_grained"]
    app.state.pipeline = Pipeline(cfg, store=SqliteStore(tmp_path), elements=NullElements(), scanpath=NullScanpath())
    with TestClient(app) as c:
        r = c.post("/analyze", files={"image": ("e.png", scene(100)[0], "image/png")}, data={"layers": "[]"})
        assert r.status_code == 200 and [e["id"] for e in r.json()["elements"]] == ["background"]
    app.state.pipeline = None


def test_an_element_that_fills_the_frame_is_explained_without_crashing(tmp_path):
    """A photo layer over the background covers 100% of the frame: no surroundings, so local contrast is None.
    (This crashed the explanation step and left the frame without explanations.)"""
    png, _ = scene(100)
    layers = [{"name": "Background", "type": "shape", "mask": mask_png(0, 0, W, H)}, {"name": "Photo", "type": "image", "mask": mask_png(0, 0, W, H)}]
    r = fast_pipe(tmp_path).analyze(png, layers=layers)
    assert r["errors"] == []
    assert r["frame"]["explanations"] is not None
    photo = next(e for e in r["elements"] if e["label"] == "Photo")
    assert photo["features"]["local_contrast"] is None and photo["explanations"] and all(any(c.isdigit() for c in m) for m in photo["explanations"])


def test_top_driver_phrase_survives_missing_features():
    from app.explain.engine import top_driver

    e = {"type": "layer", "area_pct": 100.0}
    key, phrase = top_driver(e, {"local_contrast": None, "saturation_pop": 0.4, "colour_distinctness": None, "dist_to_center": 0.0})
    assert key == "saturation" and "None" not in phrase


def test_an_explanation_failure_still_returns_the_frame_with_empty_explanations(tmp_path, monkeypatch):
    p = fast_pipe(tmp_path)
    monkeypatch.setattr(p.explainer, "explain", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rules exploded")))
    png, layers = scene(100)
    r = p.analyze(png, layers=layers)
    assert r["frame"]["explanations"] == [] and r["frame"]["palette"] and any("explanations: RuntimeError" in e for e in r["errors"])


def test_masks_are_read_from_the_alpha_channel_when_there_is_one():
    """What the editor sends: RGBA with the layer's coverage in alpha and the colour zeroed."""
    from app.elements.layers import decode_mask

    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[100:300, 200:600, 3] = 255
    b64 = base64.b64encode(cv2.imencode(".png", rgba)[1].tobytes()).decode()
    m = decode_mask(b64, (H, W))
    assert m.sum() == 200 * 400 and m[150, 300] and not m[50, 50]
    opaque_grey = base64.b64encode(cv2.imencode(".png", np.where(np.arange(W) < 640, 255, 0).astype(np.uint8)[None, :].repeat(H, 0))[1].tobytes()).decode()
    assert decode_mask(opaque_grey, (H, W)).sum() == 640 * H  # a plain grey mask still works
    full = np.full((H, W, 4), 255, np.uint8)  # an opaque colour PNG would mean "the whole frame": that is what alpha means
    assert decode_mask(base64.b64encode(cv2.imencode(".png", full)[1].tobytes()).decode(), (H, W)).all()
