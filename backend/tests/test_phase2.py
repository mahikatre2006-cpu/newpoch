"""Phase 2: M4 element detection (partition, priority, text blocks, layers, RLE) and M6 attention distribution."""
from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.attention import analyze_attention, intent_check, predict_scanpath
from app.config import data_dir, load_config
from app.elements import (Detection, ElementEngine, LayerError, decode_rle, elements_from_layers, encode_rle, resolve)
from app.elements.faces import face_ellipse_mask
from app.elements.subject import largest_components
from app.elements.text import Line, block_mask, merge_lines
from app.pipeline import Pipeline, apply_intent
from app.preprocess import build_context
from app.saliency import SaliencyEngine
from app.storage import SqliteStore
from tests.conftest import FixedScanpath, NullScanpath
from app.ingest import ingest_upload

CFG = load_config()
H, W = 90, 160  # small frames keep the unit tests instant


def rect(x0, y0, x1, y1, shape=(H, W)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def png_b64(mask: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", (mask * 255).astype(np.uint8))
    return base64.b64encode(png.tobytes()).decode()


# ---------- RLE ----------
@pytest.mark.parametrize("seed", range(5))
def test_rle_roundtrip(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((H, W)) > 0.6
    assert (decode_rle(encode_rle(m), m.shape) == m).all()


def test_rle_edges_and_validation():
    full, empty = np.ones((4, 5), bool), np.zeros((4, 5), bool)
    assert encode_rle(full).startswith("0,") and (decode_rle(encode_rle(full), (4, 5)) == full).all()
    assert (decode_rle(encode_rle(empty), (4, 5)) == empty).all()
    with pytest.raises(ValueError):
        decode_rle("1,2,3", (4, 5))


# ---------- partition and priority ----------
def test_every_pixel_belongs_to_exactly_one_element_and_priority_wins():
    dets = [
        Detection("subject", rect(20, 10, 120, 80), "Subject"),
        Detection("face", rect(50, 15, 90, 55), "Face"),
        Detection("text", rect(60, 40, 140, 60), "Hi", "Hi"),
    ]
    es = resolve(dets, (H, W))
    assert [e.id for e in es.elements] == ["background", "face_0", "subject_0", "text_0"]
    assert sum(int(es.mask(i).sum()) for i in range(len(es.elements))) == H * W  # partition
    assert sum(e.area_px for e in es.elements) == H * W
    ids = {e.id: i for i, e in enumerate(es.elements)}
    assert es.labels[50, 70] == ids["text_0"]  # text over face over subject
    assert es.labels[20, 60] == ids["face_0"]
    assert es.labels[70, 30] == ids["subject_0"]
    assert es.labels[2, 2] == 0
    for i, e in enumerate(es.elements):  # box encloses the mask, area matches
        if e.area_px:
            ys, xs = np.nonzero(es.mask(i))
            assert e.box == [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def test_fully_covered_detection_disappears():
    es = resolve([Detection("subject", rect(10, 10, 20, 20), "Subject"), Detection("text", rect(0, 0, 40, 40), "T", "T")], (H, W))
    assert [e.id for e in es.elements] == ["background", "text_0"]


def test_no_detections_is_just_background():
    es = resolve([], (H, W))
    assert [e.id for e in es.elements] == ["background"] and es.elements[0].area_px == H * W


# ---------- faces, text, subject helpers ----------
def test_face_mask_is_an_expanded_ellipse():
    m = face_ellipse_mask((60, 20, 100, 60), 0.15, (H, W))
    assert m[40, 80] and not m[40, 5]
    assert m[40, 103] and not m[40, 108]  # grown ~15% (6 px) beyond the 40 px box on the right
    assert m.sum() > np.pi * 20 * 20  # larger than the un-expanded ellipse


def line(x0, y0, x1, y1, text="x"):
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    return Line(text, 0.9, poly, (x0, y0, x1, y1))


def test_stacked_lines_merge_into_one_block_and_far_lines_do_not():
    a, b = line(10, 10, 100, 40, "PYTHON"), line(12, 44, 98, 74, "COURSE")  # headline, 4 px gap
    c = line(10, 300, 100, 330, "footer")  # far below
    d = line(300, 12, 400, 42, "side")  # same row, no horizontal overlap
    blocks = merge_lines([c, b, a, d], gap=0.6)
    texts = [[ln.text for ln in blk] for blk in blocks]
    assert ["PYTHON", "COURSE"] in texts and ["footer"] in texts and ["side"] in texts and len(texts) == 3


def test_lines_of_very_different_height_do_not_merge():
    assert len(merge_lines([line(10, 10, 100, 60), line(10, 64, 100, 80)], gap=0.6)) == 2


def test_block_mask_covers_its_lines():
    m = block_mask([line(10, 10, 50, 20)], (H, W), pad=2)
    assert m[15, 30] and m[8, 30] and not m[40, 30]


def test_largest_components_keeps_big_ones_only():
    m = rect(5, 5, 60, 60) | rect(100, 10, 150, 50) | rect(80, 80, 83, 83)
    comps = largest_components(m, min_area=50, max_components=2, ratio=0.25)
    assert len(comps) == 2 and comps[0].sum() >= comps[1].sum()
    assert len(largest_components(m, 50, 1, 0.25)) == 1
    assert largest_components(np.zeros((H, W), bool), 10, 2, 0.25) == []


# ---------- editor layers ----------
def test_layers_replace_detection_with_exact_masks_and_names():
    layers = [{"name": "Backdrop", "mask": png_b64(rect(0, 0, 160, 90))},
              {"name": "Title", "type": "text", "mask": png_b64(rect(10, 10, 80, 30))},
              {"name": "Hidden", "mask": png_b64(rect(10, 10, 20, 20))}]  # top of Title: Title loses those pixels
    es = elements_from_layers(layers, (H, W))
    assert [e.label for e in es.elements] == ["Background", "Backdrop", "Title", "Hidden"]
    assert [e.id for e in es.elements][1:] == ["layer_0", "layer_1", "layer_2"]
    assert sum(e.area_px for e in es.elements) == H * W
    assert es.labels[15, 15] == 3 and es.labels[15, 50] == 2 and es.labels[60, 100] == 1
    assert es.elements[2].meta["layer_type"] == "text"


def test_layers_hidden_completely_disappear_and_uncovered_is_background():
    layers = [{"name": "A", "mask": png_b64(rect(0, 0, 20, 20))}, {"name": "B", "mask": png_b64(rect(0, 0, 40, 40))}]
    es = elements_from_layers(layers, (H, W))
    assert [e.label for e in es.elements] == ["Background", "B"] and es.labels[80, 150] == 0


@pytest.mark.parametrize("bad", [[], "x", [{"name": "a"}], [{"mask": "!!!"}], [{"mask": base64.b64encode(b"nope").decode()}]])
def test_bad_layers_raise(bad):
    with pytest.raises(LayerError):
        elements_from_layers(bad, (H, W))


# ---------- M6 attention ----------
def two_element_set():
    return resolve([Detection("face", rect(0, 0, 40, 45), "Face")], (H, W))


def test_shares_sum_to_100_and_density_is_share_over_area():
    sal = np.full((H, W), 1.0 / (H * W), np.float32)  # uniform: density 1 everywhere
    sal[:45, :40] *= 5
    sal /= sal.sum()
    es = two_element_set()
    r = analyze_attention(sal, es, CFG["attention"]["scanpath"])
    byid = {e["id"]: e for e in r.elements}
    assert abs(sum(e["attention_pct"] for e in r.elements) - 100) < 0.05
    assert byid["face_0"]["attention_pct"] > 40 and byid["face_0"]["density"] > 2 > byid["background"]["density"]
    assert r.hierarchy[0] == "face_0" and byid["face_0"]["predicted_rank"] == 1 and byid["background"]["predicted_rank"] == 2
    assert byid["face_0"]["area_pct"] == pytest.approx(100 * 40 * 45 / (H * W), abs=0.01)
    assert decode_rle(byid["face_0"]["mask_rle"], (H, W)).sum() == 40 * 45


def test_hierarchy_ranks_by_peak_then_share_and_is_deterministic():
    rng = np.random.default_rng(1)
    sal = rng.random((H, W)).astype(np.float32)
    sal /= sal.sum()
    es = resolve([Detection("face", rect(0, 0, 30, 30), "F"), Detection("text", rect(100, 50, 150, 80), "T", "T")], (H, W))
    runs = [analyze_attention(sal, es, CFG["attention"]["scanpath"]) for _ in range(3)]
    assert runs[0].hierarchy == runs[1].hierarchy == runs[2].hierarchy
    peaks = {e.id: float(sal[es.mask(i)].max()) for i, e in enumerate(es.elements)}
    assert runs[0].hierarchy == sorted(peaks, key=lambda k: -peaks[k])


def test_scanpath_visits_separate_peaks_and_labels_them():
    sal = np.full((720, 1280), 1e-9, np.float32)
    for (x, y, a) in [(200, 150, 1.0), (1000, 500, 0.7), (640, 600, 0.5)]:
        yy, xx = np.mgrid[0:720, 0:1280]
        sal += a * np.exp(-(((xx - x) / 40) ** 2 + ((yy - y) / 40) ** 2) / 2).astype(np.float32)
    sal /= sal.sum()
    es = resolve([Detection("face", rect(150, 100, 260, 200, (720, 1280)), "F", None)], (720, 1280))
    path = predict_scanpath(sal, es.labels, [e.id for e in es.elements], CFG["attention"]["scanpath"])
    assert [p["order"] for p in path] == list(range(1, len(path) + 1)) and 3 <= len(path) <= 5
    assert abs(path[0]["x"] - 200) < 12 and abs(path[0]["y"] - 150) < 12 and path[0]["element_id"] == "face_0"
    assert abs(path[1]["x"] - 1000) < 12 and abs(path[2]["x"] - 640) < 12
    assert len({(p["x"], p["y"]) for p in path}) == len(path)  # inhibition of return: never the same spot twice


def test_flat_map_has_no_scanpath():
    flat = np.full((H, W), 1.0 / (H * W), np.float32)
    assert predict_scanpath(flat, np.zeros((H, W), np.int16), ["background"], CFG["attention"]["scanpath"]) == []


def test_intent_check_flags_gaps_over_the_ranked_elements():
    hierarchy = ["face_0", "subject_0", "text_0", "background"]
    ok, intended, _ = intent_check(["face_0", "subject_0", "text_0"], hierarchy)
    assert ok["matches"] and intended == {"face_0": 1, "subject_0": 2, "text_0": 3}
    bad, intended, _ = intent_check(["face_0", "text_0", "subject_0"], hierarchy)  # creator wants the title second
    assert not bad["matches"]
    assert {"element_id": "text_0", "intended": 2, "predicted": 3, "gap": 1} in bad["mismatches"]
    assert {"element_id": "subject_0", "intended": 3, "predicted": 2, "gap": -1} in bad["mismatches"]


def test_intent_with_unknown_and_duplicate_ids():
    check, intended, warnings = intent_check(["ghost", "face_0", "face_0", "background"], ["face_0", "background"])
    assert intended == {"face_0": 1, "background": 2} and check["matches"] and "ghost" in warnings[0]


def test_apply_intent_does_not_mutate_the_cached_result():
    base = {"elements": [{"id": "a", "intent_rank": None}, {"id": "b", "intent_rank": None}], "hierarchy": ["a", "b"],
            "intent_check": None, "errors": []}
    out = apply_intent(base, ["b", "a"])
    assert out["intent_check"]["matches"] is False and out["elements"][1]["intent_rank"] == 1
    assert base["intent_check"] is None and base["elements"][1]["intent_rank"] is None
    assert apply_intent(out, None)["intent_check"] is None


# ---------- pipeline wiring (fast saliency + fake element engine) ----------
class FakeElements:
    status = {"loaded": ["fake"], "unavailable": {}}

    def __init__(self, fail=False):
        self.fail = fail

    def detect(self, ctx, layers=None):
        if layers:
            return elements_from_layers(layers, ctx.image.shape[:2]), []
        h, w = ctx.image.shape[:2]
        errs = ["text: RuntimeError: ocr crashed"] if self.fail else []
        return resolve([Detection("face", rect(100, 100, 400, 400, (h, w)), "Face")], (h, w)), errs


def fast_pipe(tmp_path, elements=None, scanpath=None):
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["opencv_fine_grained"]
    return Pipeline(cfg, store=SqliteStore(tmp_path), elements=elements or FakeElements(), scanpath=scanpath or NullScanpath())


def sample_png() -> bytes:
    img = np.full((720, 1280, 3), 60, np.uint8)
    img[100:400, 100:400] = (230, 40, 40)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_pipeline_returns_elements_hierarchy_scanpath(tmp_path):
    r = fast_pipe(tmp_path).analyze(sample_png())
    assert [e["id"] for e in r["elements"]] == ["background", "face_0"]
    assert abs(sum(e["attention_pct"] for e in r["elements"]) - 100) < 0.05
    assert r["hierarchy"] and r["scanpath"] and {"saliency", "elements", "attention", "total"} <= set(r["timing_ms"])
    assert r["intent_check"] is None and r["errors"] == []


def test_partial_failure_is_reported_not_fatal(tmp_path):
    r = fast_pipe(tmp_path, FakeElements(fail=True)).analyze(sample_png())
    assert r["elements"] and any("ocr crashed" in e for e in r["errors"])


def test_intent_applies_on_fresh_and_cached_results(tmp_path):
    p = fast_pipe(tmp_path)
    fresh = p.analyze(sample_png(), intent=["background", "face_0"])
    assert fresh["intent_check"] is not None and not fresh["cached"]
    cached = p.analyze(sample_png(), intent=["face_0", "background"])  # a different intent on the same pixels
    assert cached["cached"] and cached["intent_check"] is not None
    assert p.analyze(sample_png())["intent_check"] is None


def test_layers_flow_through_the_pipeline_and_change_the_cache_key(tmp_path):
    p = fast_pipe(tmp_path)
    layers = [{"name": "Hero", "mask": png_b64(rect(100, 100, 400, 400, (720, 1280)))}]
    r = p.analyze(sample_png(), layers=layers)
    assert [e["label"] for e in r["elements"]] == ["Background", "Hero"]
    assert r["analysis_id"] != p.analyze(sample_png())["analysis_id"]


def test_invalid_layers_are_a_client_error_over_http(tmp_path):
    from app.main import app

    app.state.pipeline = fast_pipe(tmp_path)
    with TestClient(app) as c:
        r = c.post("/analyze", files={"image": ("t.png", sample_png(), "image/png")}, data={"layers": '[{"name": "x"}]'})
        assert r.status_code == 422 and "mask" in r.json()["detail"]
        ok = c.post("/analyze", files={"image": ("t.png", sample_png(), "image/png")}, data={"intent": '["face_0","background"]'})
        assert ok.status_code == 200 and ok.json()["intent_check"] is not None
        assert c.get("/health").json()["status"] == "ok"
    app.state.pipeline = None


# ---------- real detectors on a real thumbnail ----------
THUMB = data_dir() / "thumbnails" / "education_KorR0O7kiEk.jpg"


@pytest.mark.skipif(not THUMB.exists(), reason="sample thumbnail not downloaded")
def test_real_detectors_on_a_real_thumbnail():
    eng = ElementEngine(CFG)
    eng.load()
    assert set(eng.status["loaded"]) == {"faces", "text", "subject"}, eng.status
    ing = ingest_upload(THUMB.read_bytes(), CFG["ingest"])
    es, errors = eng.detect(build_context(ing.image, ing.sha256, CFG))
    assert errors == []
    assert (np.stack([es.mask(i) for i in range(len(es.elements))]).sum(axis=0) == 1).all()
    types = {e.type for e in es.elements}
    assert {"face", "text", "subject", "background"} <= types
    face = next(e for e in es.elements if e.type == "face")
    assert face.box[0] > 640  # the presenter stands on the right of this thumbnail


# ---------- gaze sequence (DeepGaze III) and the winner-take-all fallback ----------
def test_wta_fixations_keep_a_minimum_distance():
    rng = np.random.default_rng(3)
    sal = np.full((720, 1280), 1e-9, np.float32)
    yy, xx = np.mgrid[0:720, 0:1280]
    for x, y in [(600, 300), (640, 330), (680, 300), (900, 500)]:  # three peaks crowded on one face, one far away
        sal += np.exp(-(((xx - x) / 30) ** 2 + ((yy - y) / 30) ** 2) / 2).astype(np.float32)
    sal /= sal.sum()
    es = resolve([], (720, 1280))
    path = predict_scanpath(sal, es.labels, ["background"], CFG["attention"]["scanpath"])
    pts = [(p["x"], p["y"]) for p in path]
    min_d = CFG["attention"]["scanpath"]["exclusion_frac"] * 1280 * 0.95
    assert all(np.hypot(a[0] - b[0], a[1] - b[1]) >= min_d for i, a in enumerate(pts) for b in pts[:i])


def test_model_sequence_sets_the_viewing_order(tmp_path):
    # background-heavy attention map would rank the face last; the sequence visits the face first and the viewer sees it first
    sal = np.full((H, W), 1.0 / (H * W), np.float32)
    es = resolve([Detection("face", rect(0, 0, 30, 30), "F"), Detection("text", rect(100, 50, 150, 80), "T", "T")], (H, W))
    r = analyze_attention(sal, es, CFG["attention"]["scanpath"], fixations=[(120, 60), (10, 10), (80, 80)])
    assert r.hierarchy[:3] == ["text_0", "face_0", "background"]
    assert [p["element_id"] for p in r.scanpath] == ["text_0", "face_0", "background"]
    assert [p["order"] for p in r.scanpath] == [1, 2, 3]


def test_pipeline_uses_the_model_sequence_when_present(tmp_path):
    p = fast_pipe(tmp_path, scanpath=FixedScanpath([(200, 200), (900, 600), (300, 300)]))
    r = p.analyze(sample_png())
    assert [(s["x"], s["y"]) for s in r["scanpath"]] == [(200, 200), (900, 600), (300, 300)]
    assert r["hierarchy"][0] == "face_0" and r["scanpath"][0]["element_id"] == "face_0"
    assert "scanpath" in r["timing_ms"]


def test_a_crashing_scanpath_model_falls_back_to_the_map(tmp_path):
    class Crash(FixedScanpath):
        def predict(self, ctx):
            raise RuntimeError("dg3 exploded")

    r = fast_pipe(tmp_path, scanpath=Crash([])).analyze(sample_png())
    assert r["scanpath"] and any("dg3 exploded" in e for e in r["errors"])
