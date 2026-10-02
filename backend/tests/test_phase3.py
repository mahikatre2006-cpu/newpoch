"""Phase 3: M5 visual features and M7 explanations."""
from __future__ import annotations

import io
import textwrap
from copy import deepcopy

import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import load_config
from app.elements import Detection, resolve
from app.explain import Explainer, RuleError, load_rules, select
from app.explain.engine import NUMERIC, element_facts, frame_facts, render, top_driver
from app.features import colour_name, compute_features, focal_points, lab_distance, palette, text_recovery, wcag_contrast
from app.features.core import local_contrast, text_contrast
from app.ingest import ingest_upload
from app.pipeline import Pipeline, apply_intent
from app.preprocess import build_context
from app.storage import SqliteStore
from tests.conftest import FixedScanpath, NullScanpath

CFG = load_config()
H, W = 720, 1280


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def ctx_for(img: np.ndarray):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    ing = ingest_upload(buf.getvalue(), CFG["ingest"])
    return build_context(ing.image, ing.sha256, CFG)


def uniform_sal():
    return np.full((H, W), 1.0 / (H * W), np.float32)


# ---------- colour maths ----------
def test_wcag_known_values():
    assert wcag_contrast((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0, abs=0.01)
    assert wcag_contrast((120, 120, 120), (120, 120, 120)) == pytest.approx(1.0)
    assert wcag_contrast((119, 119, 119), (255, 255, 255)) == pytest.approx(4.48, abs=0.05)  # the classic #777 on white


def test_lab_distance_and_names():
    assert lab_distance((10, 20, 30), (10, 20, 30)) == 0
    assert lab_distance((255, 0, 0), (0, 0, 255)) > 100 > lab_distance((250, 0, 0), (255, 10, 10)) > 0
    for rgb, name in [((230, 30, 30), "red"), ((250, 120, 20), "orange"), ((250, 230, 20), "yellow"), ((20, 200, 60), "green"),
                      ((20, 60, 220), "blue"), ((150, 40, 200), "purple"), ((5, 5, 5), "black"), ((250, 250, 250), "white"),
                      ((128, 128, 128), "grey"), ((110, 60, 20), "brown"), ((250, 100, 180), "pink")]:
        assert colour_name(rgb) == name, rgb


def test_palette_finds_the_dominant_colours():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :800] = (220, 30, 30)
    img[:, 800:] = (30, 30, 220)
    p = palette(img, 3)
    assert p[0]["name"] == "red" and p[0]["share"] == pytest.approx(0.625, abs=0.03) and {c["name"] for c in p} >= {"red", "blue"}
    assert sum(c["share"] for c in p) == pytest.approx(1.0, abs=0.01)
    assert palette(np.full((720, 1280, 3), 90, np.uint8), 5)  # a flat frame must not crash


# ---------- text recovery ----------
@pytest.mark.parametrize("orig,found,lo,hi", [("PYTHON 2026", "python2026", 1.0, 1.0), ("PYTHON FULL COURSE", "PYTHON", 0.3, 0.4),
                                              ("don't buy stocks", "", 0.0, 0.0), ("HELLO", "HELLO WORLD EXTRA", 1.0, 1.0),
                                              ("HELLO", "HELL0", 0.7, 0.9), ("abc", "xyz", 0.0, 0.0)])
def test_text_recovery(orig, found, lo, hi):
    assert lo <= text_recovery(orig, found) <= hi


# ---------- measurements on synthetic frames ----------
def test_local_contrast_high_for_bright_on_dark_and_zero_for_blend():
    img = np.full((H, W, 3), 30, np.uint8)
    img[200:400, 300:700] = 240
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = rect(300, 200, 700, 400)
    assert local_contrast(gray, m, [300, 200, 700, 400], 24) > 0.7
    flat = cv2.cvtColor(np.full((H, W, 3), 90, np.uint8), cv2.COLOR_RGB2GRAY)
    assert local_contrast(flat, m, [300, 200, 700, 400], 24) < 0.01


def test_text_contrast_measures_text_against_its_surroundings():
    box = [300, 300, 700, 380]
    for text_col, bg_col, lo, hi in [((255, 255, 255), (20, 20, 20), 15, 21.5), ((120, 120, 120), (135, 135, 135), 1.0, 1.4)]:
        img = np.full((H, W, 3), bg_col, np.uint8)
        for x in range(310, 690, 30):  # letter-like strokes inside the block
            img[310:370, x : x + 14] = text_col
        r = text_contrast(img, rect(300, 300, 700, 380), box, 6)
        assert lo <= r <= hi, (text_col, bg_col, r)


def test_focal_points_counts_separate_peaks_once():
    yy, xx = np.mgrid[0:H, 0:W]
    sal = sum(np.exp(-(((xx - x) / 40) ** 2 + ((yy - y) / 40) ** 2) / 2) for x, y in [(300, 200), (900, 500)]).astype(np.float32)
    sal /= sal.sum()
    pts = focal_points(sal, CFG["features"]["focal"])
    assert len(pts) == 2 and pts[0]["strength"] == 1.0
    assert abs(pts[0]["x"] - 300) < 20 or abs(pts[0]["x"] - 900) < 20
    plateau = np.zeros((H, W), np.float32)
    plateau[300:420, 500:700] = 1
    assert len(focal_points(plateau / plateau.sum(), CFG["features"]["focal"])) == 1  # a flat top is one focal point
    assert focal_points(np.zeros((H, W), np.float32), CFG["features"]["focal"]) == []


def build_scene():
    img = np.full((H, W, 3), 40, np.uint8)
    img[100:400, 100:400] = (230, 40, 40)  # a saturated block
    img[300:380, 800:1200] = 20  # text block region
    for x in range(810, 1190, 30):
        img[310:370, x : x + 14] = 250
    img[640:720, 1100:1280] = (250, 250, 250)  # under the duration badge
    es = resolve([Detection("subject", rect(100, 100, 400, 400), "Hero"), Detection("text", rect(800, 300, 1200, 380), "HELLO", "HELLO"),
                  Detection("subject", rect(1100, 640, 1280, 720), "Badge")], (H, W))
    return img, es


def test_compute_features_on_a_scene():
    img, es = build_scene()
    ctx = ctx_for(img)
    sal = uniform_sal()
    sal[100:400, 100:400] *= 6
    sal /= sal.sum()

    class Line:
        def __init__(self, text, box):
            self.text, self.box, self.score = text, box, 0.9

    fr = compute_features(ctx, es, sal, CFG, mobile_ocr=lambda im: [Line("hello", (int(800 * im.shape[1] / W), int(300 * im.shape[0] / H), int(1200 * im.shape[1] / W), int(380 * im.shape[0] / H)))])
    ids = {e.id: e for e in es.elements}
    hero, text, badge = fr.elements["subject_0"], fr.elements["text_0"], fr.elements["subject_1"]
    assert hero["saturation_pop"] > 0.05 and hero["local_contrast"] > 0.1 and hero["colour_distinctness"] > 20
    assert hero["safe_zone_overlap_pct"] == 0.0 and badge["safe_zone_overlap_pct"] > 90  # the badge sits under the timestamp
    assert text["text_contrast_ratio"] > 10 and text["mobile_legible"] is True and text["mobile_match_pct"] == 100
    assert "local_contrast" not in fr.elements["background"] and 0 <= hero["dist_to_thirds"] < 0.5
    f = fr.frame
    assert f["balance_offset"][0] < 0 and f["balance_offset"][1] < 0  # attention sits upper-left
    assert 0 <= f["negative_space_pct"] <= 100 and f["focal_points"] >= 1 and f["text_area_pct"] == pytest.approx(100 * 400 * 80 / (H * W), abs=0.2)
    assert len(f["palette"]) == 5
    v = fr.layer_views
    assert {"composition", "text", "colour", "contrast_png", "safe_zone"} <= set(v)
    assert sum(sum(row) for row in v["composition"]["grid_attention_pct"]) == pytest.approx(100, abs=0.6)
    assert len(v["composition"]["thirds_points"]) == 4 and v["text"][0]["mobile_legible"] is True and v["safe_zone"]["boxes"]


def test_mobile_legibility_false_when_phone_ocr_finds_nothing_and_unmeasured_when_unreadable_script():
    img, es = build_scene()
    fr = compute_features(ctx_for(img), es, uniform_sal(), CFG, mobile_ocr=lambda im: [])
    assert fr.elements["text_0"]["mobile_legible"] is False and fr.elements["text_0"]["mobile_match_pct"] == 0
    es2 = resolve([Detection("text", rect(800, 300, 1200, 380), "Text", None)], (H, W))  # script OCR cannot read
    fr2 = compute_features(ctx_for(img), es2, uniform_sal(), CFG, mobile_ocr=lambda im: [])
    assert fr2.elements["text_0"]["mobile_legible"] is None  # not measured, not guessed


def test_features_without_saliency_still_work():
    img, es = build_scene()
    fr = compute_features(ctx_for(img), es, None, CFG)
    assert "balance_offset" not in fr.frame and fr.elements["subject_0"]["local_contrast"] is not None


# ---------- rules file and evaluator ----------
def write_rules(tmp_path, body: str):
    p = tmp_path / "r.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_shipped_rules_are_valid_and_every_message_cites_a_number():
    rules = load_rules()
    assert len(rules) >= 20 and len({r["id"] for r in rules}) == len(rules)
    for r in rules:
        assert r["_fields"] & NUMERIC, r["id"]


@pytest.mark.parametrize("body,msg", [
    ("- {id: a, applies_to: any, when: 'true', priority: 1, message: 'no numbers here {label}'}", "no measured number"),
    ("- {id: a, applies_to: any, when: 'true', priority: 1, message: '{label} {nonsense}'}", "unknown placeholder"),
    ("- {id: a, applies_to: any, when: 'secret > 1', priority: 1, message: '{area_pct}'}", "unknown fact"),
    ("- {id: a, applies_to: any, when: '__import__(\"os\")', priority: 1, message: '{area_pct}'}", "abs/min/max|not allowed"),
    ("- {id: a, applies_to: any, when: '[x for x in area_pct]', priority: 1, message: '{area_pct}'}", "not allowed"),
    ("- {id: a, applies_to: weird, when: 'true', priority: 1, message: '{area_pct}'}", "applies_to"),
    ("- {id: a, applies_to: any, priority: 1, message: '{area_pct}'}", "missing"),
    ("- {id: a, applies_to: any, when: 'area_pct >', priority: 1, message: '{area_pct}'}", "bad expression"),
    ("- {id: a, applies_to: any, when: 'true', priority: 1, message: '{area_pct}'}\n- {id: a, applies_to: any, when: 'true', priority: 1, message: '{area_pct}'}", "duplicate"),
])
def test_bad_rules_are_rejected_at_load(tmp_path, msg, body):
    with pytest.raises(RuleError, match=msg):
        load_rules(write_rules(tmp_path, body))


def test_every_rule_renders_with_all_facts_present_and_never_prints_none():
    facts = {"label": "Title", "text": "HELLO", "type": "text", "attention_pct": 12.34, "area_pct": 5.6, "density": 2.2, "predicted_rank": 3,
             "intent_rank": 2, "local_contrast": 0.123, "saturation_pop": 0.25, "saturation_pop_pct": 25.0, "colour_distinctness": 31.2,
             "dist_to_thirds": 0.04, "dist_to_center": 0.3, "safe_zone_overlap_pct": 30.0, "text_contrast_ratio": 2.4, "mobile_match_pct": 40,
             "mobile_legible": False, "top_driver": "face", "top_driver_phrase": "being a face (5.0% of the frame)", "colour_name": "red",
             "balance_x": 0.4, "balance_y": -0.3, "balance_x_pct": 40.0, "balance_y_pct": 30.0, "balance_side_x": "right", "balance_side_y": "top",
             "focal_points": 2, "negative_space_pct": 55.5, "palette_top_name": "red", "palette_top_pct": 50.0, "text_area_pct": 35.0, "face_count": 1}
    for r in load_rules():
        out = render(r, facts)
        assert out and "None" not in out and any(ch.isdigit() for ch in out), r["id"]
        assert render(r, {**facts, **{k: None for k in r["_fields"] & NUMERIC}}) is None  # a missing number suppresses the message


def test_evaluator_treats_missing_numbers_as_not_applicable(tmp_path):
    rules = load_rules(write_rules(tmp_path, """
        - {id: low, applies_to: any, when: "text_contrast_ratio < 3.0", priority: 5, message: "contrast {text_contrast_ratio}"}
        - {id: rank, applies_to: any, when: "intent_rank is not None and predicted_rank != intent_rank", priority: 9, message: "#{intent_rank} vs #{predicted_rank}"}
        - {id: neg, applies_to: any, when: "not (area_pct > 10) and abs(local_contrast) >= 0.1", priority: 1, message: "{area_pct}% {local_contrast}"}
    """))
    assert select(rules, "text", {"text_contrast_ratio": None, "intent_rank": None, "area_pct": 5, "local_contrast": None}, 4) == []
    out = select(rules, "text", {"text_contrast_ratio": 2.0, "intent_rank": 1, "predicted_rank": 3, "area_pct": 5, "local_contrast": 0.2}, 4)
    assert out == ["#1 vs #3", "contrast 2.0", "5.0% 0.20"]  # priority order, sensible rounding
    assert select(rules, "text", {"text_contrast_ratio": 2.0, "intent_rank": 1, "predicted_rank": 3, "area_pct": 5, "local_contrast": 0.2}, 1) == ["#1 vs #3"]


def test_select_limit_applies_to_filters_and_fallback(tmp_path):
    rules = load_rules(write_rules(tmp_path, """
        - {id: a, applies_to: face, when: "true", priority: 9, message: "A {area_pct}"}
        - {id: b, applies_to: text, when: "true", priority: 8, message: "B {area_pct}"}
        - {id: c, applies_to: any, when: "true", priority: 7, message: "C {area_pct}"}
        - {id: d, applies_to: any, when: "true", priority: 6, message: "D {area_pct}"}
        - {id: base, applies_to: any, when: "true", fallback: true, priority: 1, message: "BASE {area_pct}"}
    """))
    f = {"area_pct": 1.0}
    assert select(rules, "face", f, 2) == ["A 1.0", "C 1.0"]  # limit; text-only rule filtered out; fallback not needed
    only_base = load_rules(write_rules(tmp_path, """
        - {id: base, applies_to: any, when: "true", fallback: true, priority: 1, message: "BASE {area_pct}"}
    """))
    assert select(only_base, "face", f, 4, minimum=2) == ["BASE 1.0"]  # fallback fills in when nothing else matched
    assert select(only_base, "face", f, 4, minimum=0) == []


def test_top_driver_picks_the_largest_normalised_feature():
    face = {"type": "face", "area_pct": 7.0}
    assert top_driver(face, {"local_contrast": 0.1})[0] == "face"
    el = {"type": "subject", "area_pct": 10.0}
    assert top_driver(el, {"local_contrast": 0.44, "saturation_pop": 0.05, "colour_distinctness": 10, "dist_to_center": 0.9})[0] == "local_contrast"
    assert top_driver(el, {"local_contrast": 0.02, "saturation_pop": 0.3, "colour_distinctness": 10, "dist_to_center": 0.9})[0] == "saturation"
    assert top_driver(el, {"local_contrast": 0.02, "saturation_pop": 0.0, "colour_distinctness": 3, "dist_to_center": 0.95}) == (None, "a combination of position and size")
    assert "0.44" in top_driver(el, {"local_contrast": 0.44})[1]


def test_explainer_fills_elements_and_frame_and_intent_message_goes_first():
    img, es = build_scene()
    ctx = ctx_for(img)
    sal = uniform_sal()
    sal[100:400, 100:400] *= 6
    sal /= sal.sum()
    from app.attention import analyze_attention

    att = analyze_attention(sal, es, CFG["attention"]["scanpath"])
    fr = compute_features(ctx, es, sal, CFG)
    ex = Explainer()
    ex.explain(att.elements, fr.elements, fr.frame)
    for e in att.elements:
        assert 1 <= len(e["explanations"]) <= 4 and all(any(c.isdigit() for c in m) for m in e["explanations"]), e["id"]
        assert e["features"] is fr.elements[e["id"]]
    assert 1 <= len(fr.frame["explanations"]) <= 3
    result = {"elements": att.elements, "hierarchy": att.hierarchy, "intent_check": None, "errors": [], "frame": fr.frame}
    wrong = list(reversed([e["id"] for e in att.elements]))
    out = apply_intent(result, wrong, ex)
    mism = {m["element_id"] for m in out["intent_check"]["mismatches"]}
    assert mism
    for e in out["elements"]:
        if e["id"] in mism:
            assert e["explanations"][0].startswith("You want ") and len(e["explanations"]) <= 4
    assert all(not e["explanations"][0].startswith("You want ") for e in result["elements"])  # the cached base is untouched


# ---------- pipeline ----------
class FakeElements:
    status = {"loaded": ["fake"], "unavailable": {}}

    def detect(self, ctx, layers=None):
        h, w = ctx.image.shape[:2]
        return resolve([Detection("subject", rect(100, 100, 400, 400), "Hero")], (h, w)), []


def scene_png():
    img, _ = build_scene()
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def test_pipeline_returns_features_frame_layer_views_and_explanations(tmp_path):
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["opencv_fine_grained"]
    p = Pipeline(cfg, store=SqliteStore(tmp_path), elements=FakeElements(), scanpath=NullScanpath())
    r = p.analyze(scene_png())
    assert r["errors"] == [] and r["frame"]["palette"] and r["frame"]["explanations"] is not None and "contrast_png" in r["layer_views"]
    for e in r["elements"]:
        assert e["explanations"] and e["features"] is not None
    assert "features" in r["timing_ms"]
    again = p.analyze(scene_png(), intent=["background", "subject_0"])  # cached result + a new intent
    assert again["cached"] and again["frame"] == r["frame"]
    assert again["intent_check"] is not None and again["elements"][0]["intent_rank"] == 1


def test_feature_failure_is_reported_and_attention_still_returned(tmp_path, monkeypatch):
    import app.pipeline as pl

    monkeypatch.setattr(pl, "compute_features", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["opencv_fine_grained"]
    r = Pipeline(cfg, store=SqliteStore(tmp_path), elements=FakeElements(), scanpath=NullScanpath()).analyze(scene_png())
    assert r["elements"] and r["frame"] is None and any("features: RuntimeError: boom" in e for e in r["errors"])
    assert all(e["explanations"] is None for e in r["elements"])
