"""Phase 5: M8 ablation, M10 fixes, M11 versions and compare, and the endpoints that expose them."""
from __future__ import annotations

import base64
from copy import deepcopy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ablation import AblationError, ablate, blur_fill, remove_element
from app.attention import analyze_attention
from app.config import load_config
from app.elements import Detection, decode_rle, resolve
from app.fixes import FixError, apply_fix, make_layers
from app.pipeline import Pipeline
from app.saliency.core import SaliencyResult
from app.storage import SqliteStore
from app.versions import compare_results, match_elements
from app.versions.core import describe_change
from tests.conftest import NullScanpath

CFG = load_config()
H, W = 720, 1280


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def ellipse(cx, cy, rx, ry):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return m.astype(bool)


def scene_image() -> np.ndarray:
    img = np.full((H, W, 3), 60, np.uint8)
    img[150:600, 100:420] = (200, 120, 60)  # the subject
    cv2.ellipse(img, (260, 260), (70, 85), 0, 0, 360, (230, 190, 160), -1)  # a face on it
    img[300:400, 700:1200] = 120  # a text block, background of the block
    for x in range(710, 1190, 34):  # letter-like strokes in a near-identical grey: low contrast
        img[310:390, x : x + 16] = 150
    return img


def scene_elements():
    return resolve([Detection("subject", rect(100, 150, 420, 600), "Hero"), Detection("face", ellipse(260, 260, 70, 85), "Face"),
                    Detection("text", rect(700, 300, 1200, 400), "TITLE", "TITLE")], (H, W))


class SceneSaliency:
    """A saliency 'model' with a controllable map: attention proportional to weights per region of the image it is given."""

    def __init__(self, weights=None):
        self.weights = weights

    def predict(self, ctx):
        img = ctx.image.astype(np.float32)
        m = cv2.GaussianBlur(np.abs(img - np.median(img.reshape(-1, 3), axis=0)).sum(axis=2) + 1.0, (0, 0), 25)
        m = np.maximum(m, 1e-6)
        return SaliencyResult(map=(m / m.sum()).astype(np.float32), model="test")


def analysed_scene():
    img, es = scene_image(), scene_elements()
    sal = SceneSaliency().predict(type("C", (), {"image": img})())
    att = analyze_attention(sal.map, es, CFG["attention"]["scanpath"])
    from app.explain import Explainer
    from app.features import compute_features
    from app.ingest import ingest_upload
    from app.preprocess import build_context
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    ing = ingest_upload(buf.getvalue(), CFG["ingest"])
    ctx = build_context(ing.image, ing.sha256, CFG)
    fr = compute_features(ctx, es, sal.map, CFG)
    Explainer().explain(att.elements, fr.elements, fr.frame)
    return ing.image, {"elements": att.elements, "hierarchy": att.hierarchy, "frame": fr.frame}


# ---------- M8 ----------
def test_blur_fill_fills_every_pixel_of_a_big_hole_from_the_surroundings():
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :640] = (200, 0, 0)
    img[:, 640:] = (0, 0, 200)
    hole = rect(300, 100, 980, 620)  # a huge hole spanning both colours
    out = blur_fill(img, hole, CFG["ablation"]["fill_sigmas"])
    assert (out[~hole] == img[~hole]).all()  # outside the hole nothing changes
    assert out[360, 400, 0] > out[360, 400, 2] and out[360, 880, 2] > out[360, 880, 0]  # each side takes its neighbour's colour
    assert out[hole].std() > 0 and np.isfinite(out).all()
    assert blur_fill(img, np.ones((H, W), bool), CFG["ablation"]["fill_sigmas"]).shape == img.shape  # all hole: no crash


@pytest.mark.parametrize("large", [False, True])
def test_remove_element_makes_the_region_look_like_its_surroundings(large):
    img = np.full((H, W, 3), 80, np.uint8)
    img[300:420, 500:700] = (250, 40, 40)
    out = remove_element(img, rect(500, 300, 700, 420), large, CFG)
    assert np.abs(out[310:410, 510:690].astype(int) - 80).max() < 25  # the red block is gone
    assert (out[:200] == img[:200]).all()  # far away pixels untouched


def test_ablation_deltas_add_up_to_the_removed_share_and_background_absorbs_the_hole():
    img, result = analysed_scene()
    out = ablate(SceneSaliency(), CFG, img, result, "text_0")
    removed = out["removed"]["attention_pct"]
    assert sum(f["delta_pct"] for f in out["flow"]) == pytest.approx(removed, abs=0.05)
    assert {f["element_id"] for f in out["flow"]} == {"background", "face_0", "subject_0"}
    assert out["removed"]["method"] == "inpaint" and out["ablated_heatmap_png"] and out["ablated_image_jpg"]
    assert "Without TITLE" in out["message"]
    for f in out["flow"]:  # the bookkeeping is consistent: after = before + delta, and the shares still add to 100
        assert f["after_pct"] == pytest.approx(f["before_pct"] + f["delta_pct"], abs=0.02)
    assert sum(f["after_pct"] for f in out["flow"]) == pytest.approx(100.0, abs=0.1)  # the removed element no longer exists


def test_large_elements_use_the_blurred_fill():
    img, result = analysed_scene()
    assert ablate(SceneSaliency(), CFG, img, result, "subject_0")["removed"]["method"] == "blur_fill"
    assert ablate(SceneSaliency(), CFG, img, result, "face_0")["removed"]["method"] == "blur_fill"


def test_ablation_rejects_unknown_background_and_empty():
    img, result = analysed_scene()
    with pytest.raises(AblationError) as e:
        ablate(SceneSaliency(), CFG, img, result, "background")
    assert e.value.status == 422
    with pytest.raises(AblationError) as e:
        ablate(SceneSaliency(), CFG, img, result, "ghost")
    assert e.value.status == 404
    with pytest.raises(AblationError):
        ablate(SceneSaliency(), CFG, img, {"elements": None}, "x")


# ---------- M10 ----------
def test_focus_subject_blurs_and_darkens_the_rest_keeps_the_subject_and_outlines_it():
    img, result = analysed_scene()
    out = apply_fix("focus_subject", img, result, CFG)
    assert (out[400:500, 150:350] == img[400:500, 150:350]).all()  # subject interior untouched
    assert (out[310:390, 710:1190] == img[310:390, 710:1190]).all()  # text stays sharp too (its edge is feathered, the inside is not)
    bg_before, bg_after = img[20:120, 600:1100].astype(float), out[20:120, 600:1100].astype(float)
    assert bg_after.mean() == pytest.approx(bg_before.mean() * 0.75, rel=0.03)  # darkened by 25%
    ring = out[145:150, 200:300]
    assert (ring > 240).mean() > 0.5  # the white outline just outside the subject's top edge


def test_focus_needs_a_subject():
    img, result = analysed_scene()
    result["elements"] = [e for e in result["elements"] if e["type"] in ("background", "text")]
    with pytest.raises(FixError, match="No subject"):
        apply_fix("focus_subject", img, result, CFG)


def test_enhance_raises_contrast_and_saturation():
    img, result = analysed_scene()
    out = apply_fix("enhance", img, result, CFG)
    assert out.shape == img.shape and not (out == img).all()
    sat = lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1].astype(float).mean()
    assert sat(out) > sat(img)
    grey = lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)[300:400, 700:1200].astype(float)
    assert grey(out).std() > grey(img).std()  # CLAHE lifts contrast locally: here, in the low-contrast text block


def test_fix_text_outlines_low_contrast_text_and_keeps_the_letters_exactly():
    img, result = analysed_scene()
    title = next(e for e in result["elements"] if e["id"] == "text_0")
    assert title["features"]["text_contrast_ratio"] < 4.5  # the scene's text really is low contrast
    out = apply_fix("fix_text", img, result, CFG)
    stroke_zone = out[310:390, 710:726]  # letter pixels
    assert (stroke_zone == img[310:390, 710:726]).all()
    gap = out[330:370, 727:730]  # just beside a letter: now has the outline
    assert np.abs(gap.astype(int) - img[330:370, 727:730].astype(int)).max() > 20
    assert (out[:200, :600] == img[:200, :600]).all()  # nothing else changed


def test_fix_text_refuses_when_there_is_nothing_to_fix():
    img, result = analysed_scene()
    result["elements"] = [e for e in result["elements"] if e["type"] != "text"]
    with pytest.raises(FixError, match="No text"):
        apply_fix("fix_text", img, result, CFG)
    img2, r2 = analysed_scene()
    for e in r2["elements"]:
        if e["type"] == "text":
            e["features"]["text_contrast_ratio"], e["features"]["mobile_legible"] = 12.0, True
    with pytest.raises(FixError, match="already passes"):
        apply_fix("fix_text", img2, r2, CFG)


def test_unknown_fix_is_a_400_and_separate_layers_gives_a_cutout():
    img, result = analysed_scene()
    with pytest.raises(FixError) as e:
        apply_fix("make_it_viral", img, result, CFG)
    assert e.value.status == 400
    layers = make_layers(img, result, CFG)
    assert [l["name"] for l in layers] == ["Background", "Subject"] and layers[1]["type"] == "subject"
    sub = layers[1]
    arr = cv2.imdecode(np.frombuffer(base64.b64decode(sub["data"]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert arr.shape == (sub["height"], sub["width"], 4) and arr[sub["height"] // 2, sub["width"] // 2, 3] == 255  # solid inside
    assert arr[0, 0, 3] < 255 and arr[0, sub["width"] // 2, 3] < 255  # the edge is feathered, not a hard cut
    assert sub["x"] <= 100 and sub["y"] <= 150 and sub["x"] + sub["width"] >= 420
    bg = cv2.imdecode(np.frombuffer(base64.b64decode(layers[0]["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert bg.shape == (H, W, 3) and abs(int(bg[400, 250, 2]) - 200) > 60  # the person is filled in from the surroundings, not still there


# ---------- M11 ----------
def el(i, typ, label, pct, mask=None):
    from app.elements import encode_rle

    m = rect(0, 0, 10, 10) if mask is None else mask
    return {"id": i, "type": typ, "label": label, "attention_pct": pct, "mask_rle": encode_rle(m), "area_pct": 1.0}


def res(elements, hierarchy):
    return {"elements": elements, "hierarchy": hierarchy}


def test_matching_by_background_layer_name_and_mask_overlap():
    a = res([el("background", "background", "Background", 20, ~rect(100, 100, 300, 300)), el("face_0", "face", "Face", 50, rect(100, 100, 300, 300)),
             el("text_0", "text", "A", 30, rect(500, 500, 700, 600))], ["face_0", "text_0", "background"])
    b = res([el("background", "background", "Background", 15, ~rect(110, 100, 310, 300)), el("face_3", "face", "Face", 60, rect(110, 100, 310, 300))],
            ["face_3", "background"])
    pairs = match_elements(a, b)
    assert (0, 0) in pairs and (1, 1) in pairs and (2, None) in pairs  # the moved face still matches; the text vanished
    d = compare_results(a, b)
    face = next(r for r in d["rows"] if r["label"] == "Face")
    assert face["delta_pct"] == 10 and face["status"] == "matched"
    assert next(r for r in d["rows"] if r["label"] == "A")["status"] == "removed"
    assert d["hierarchy_a"] == ["Face", "A", "Background"] and d["hierarchy_b"] == ["Face", "Background"]


def test_editor_layers_match_by_name_and_new_ones_are_flagged():
    a = res([el("layer_0", "layer", "Title", 40), el("layer_1", "layer", "Product", 60)], ["layer_1", "layer_0"])
    b = res([el("layer_0", "layer", "Product", 55), el("layer_1", "layer", "Title", 30), el("layer_2", "layer", "Arrow", 15)], ["layer_0", "layer_1", "layer_2"])
    rows = {r["label"]: r for r in compare_results(a, b)["rows"]}
    assert rows["Title"]["delta_pct"] == -10 and rows["Product"]["delta_pct"] == -5 and rows["Arrow"]["status"] == "new" and rows["Arrow"]["a_pct"] is None


def test_describe_change_names_the_biggest_movers():
    rows = [{"label": "Subject", "status": "matched", "delta_pct": 23.0, "a_pct": 21.0, "b_pct": 44.0},
            {"label": "Background", "status": "matched", "delta_pct": -22.0, "a_pct": 52.0, "b_pct": 30.0},
            {"label": "Tiny", "status": "matched", "delta_pct": 0.2, "a_pct": 1.0, "b_pct": 1.2}]
    s = describe_change(rows, "Focus subject")
    assert s == "Focus subject: Subject 21.0% → 44.0%, Background 52.0% → 30.0%." and "Tiny" not in s
    assert "no element" in describe_change([], "Enhance")


def test_storage_images_and_versions_roundtrip(tmp_path):
    s = SqliteStore(tmp_path)
    s.save_image("a1", b"\x89PNGdata", [{"name": "x"}])
    s.save_image("a1", b"other", None)  # the first copy wins: an analysis id is immutable
    assert s.get_image("a1") == (b"\x89PNGdata", [{"name": "x"}]) and s.get_image("zz") is None
    for i, label in enumerate(["one", "two", "three"]):
        s.add_version({"version_id": f"v{i}", "session_id": "S", "analysis_id": f"a{i}", "parent_id": f"v{i-1}" if i else None, "label": label, "kind": "fix"})
    s.add_version({"version_id": "other", "session_id": "T", "analysis_id": "a9", "label": "x", "kind": "upload"})
    assert [v["label"] for v in s.list_versions("S")] == ["one", "two", "three"] and s.get_version("v2")["parent_id"] == "v1"


# ---------- pipeline + API ----------
class SceneEngine:
    status = {"loaded": ["scene"], "unavailable": {}}

    def detect(self, ctx, layers=None):
        return scene_elements(), []


def scene_pipe(tmp_path):
    cfg = deepcopy(CFG)
    cfg["saliency"]["backends"] = ["opencv_fine_grained"]
    return Pipeline(cfg, store=SqliteStore(tmp_path), elements=SceneEngine(), scanpath=NullScanpath())


def scene_png() -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(scene_image(), cv2.COLOR_RGB2BGR))
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path):
    from app.main import app

    app.state.pipeline = scene_pipe(tmp_path)
    with TestClient(app) as c:
        yield c
    app.state.pipeline = None


def upload(client, **data):
    r = client.post("/analyze", files={"image": ("s.png", scene_png(), "image/png")}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_can_be_saved_as_a_version_without_duplicates(client):
    a = upload(client, save_version="true", label="My thumbnail")
    assert a["version_id"]
    b = upload(client, save_version="true", session_id=a["session_id"])  # same pixels again: no new version
    assert b["version_id"] == a["version_id"]
    vs = client.get(f"/versions/{a['session_id']}").json()["versions"]
    assert [v["label"] for v in vs] == ["My thumbnail"] and vs[0]["kind"] == "upload"
    assert upload(client)["version_id"] is None  # not asked to save: no version


def test_ablate_endpoint(client):
    a = upload(client)
    r = client.post("/ablate", json={"analysis_id": a["analysis_id"], "element_id": "text_0"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert sum(f["delta_pct"] for f in out["flow"]) == pytest.approx(out["removed"]["attention_pct"], abs=1.0)
    assert client.post("/ablate", json={"analysis_id": a["analysis_id"], "element_id": "background"}).status_code == 422
    assert client.post("/ablate", json={"analysis_id": a["analysis_id"], "element_id": "nope"}).status_code == 404
    assert client.post("/ablate", json={"analysis_id": "f" * 64, "element_id": "text_0"}).status_code == 404
    assert client.post("/ablate", json={"analysis_id": a["analysis_id"]}).status_code == 422  # missing field


def test_fix_creates_a_version_with_measured_changes_and_compare_shows_them(client):
    a = upload(client, save_version="true", label="Original")
    r = client.post("/fix", json={"analysis_id": a["analysis_id"], "fix": "focus_subject", "session_id": a["session_id"]})
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["label"] == "Focus subject" and f["analysis"]["analysis_id"] != a["analysis_id"] and f["analysis"]["version_id"] == f["version"]["version_id"]
    assert f["version"]["parent_id"] == a["version_id"] and f["message"].startswith("Focus subject:")
    assert {"Background", "Hero"} <= {c["label"] for c in f["changes"]}
    vs = client.get(f"/versions/{a['session_id']}").json()["versions"]
    assert [v["label"] for v in vs] == ["Original", "Focus subject"]
    cmp = client.get("/compare", params={"a": vs[0]["version_id"], "b": vs[1]["version_id"]}).json()
    assert cmp["a"]["heatmap_png"] and cmp["b"]["image_jpg"] and len(cmp["rows"]) >= 3
    assert abs(sum(r["a_pct"] or 0 for r in cmp["rows"]) - 100) < 0.2 and abs(sum(r["b_pct"] or 0 for r in cmp["rows"]) - 100) < 0.2


def test_fix_errors_and_separate_layers(client):
    a = upload(client)
    assert client.post("/fix", json={"analysis_id": a["analysis_id"], "fix": "nonsense"}).status_code == 400
    assert client.post("/fix", json={"analysis_id": "0" * 64, "fix": "enhance"}).status_code == 404
    r = client.post("/fix", json={"analysis_id": a["analysis_id"], "fix": "separate_layers"})
    assert r.status_code == 200 and r.json()["analysis"] is None and [l["name"] for l in r.json()["layers"]] == ["Background", "Subject"]
    assert client.get("/compare", params={"a": "x", "b": "y"}).status_code == 404
    assert client.get("/versions/unknown-session").json()["versions"] == []


def test_each_fix_that_applies_gives_a_before_after(client):
    a = upload(client, save_version="true")
    for fix in ("focus_subject", "fix_text", "enhance"):
        r = client.post("/fix", json={"analysis_id": a["analysis_id"], "fix": fix, "session_id": a["session_id"]})
        assert r.status_code == 200, (fix, r.text)
        assert r.json()["changes"], fix
    assert [v["label"] for v in client.get(f"/versions/{a['session_id']}").json()["versions"]] == ["Upload", "Focus subject", "Fix text legibility", "Enhance"]


def test_analysis_can_be_fetched_by_id(client):
    a = upload(client)
    assert client.get(f"/analysis/{a['analysis_id']}").json()["analysis_id"] == a["analysis_id"]
    assert client.get("/analysis/nope").status_code == 404


def test_text_on_a_plate_is_measured_and_fixed_against_the_plate_not_the_outside():
    """White letters on a dark pill over a bright frame: the colour behind the text is the pill, and the letters (not the pill) get the outline."""
    from app.features.core import text_colours, text_contrast
    from app.fixes.core import _glyphs

    img = np.full((H, W, 3), 235, np.uint8)  # bright frame
    img[300:400, 500:1000] = (40, 40, 60)  # dark pill
    for x in range(520, 980, 40):
        img[320:380, x : x + 18] = (245, 245, 245)  # white letters
    mask = rect(500, 300, 1000, 400)
    box = [500, 300, 1000, 400]
    text, behind = text_colours(img, mask, box, 6)
    assert behind.mean() < 70 and text.mean() > 230  # behind the text = the pill
    assert text_contrast(img, mask, box, 6) > 12  # white on near-black, not white on bright
    glyph = _glyphs(img, mask, box)
    assert glyph[320:380, 520:538].mean() > 0.95 and glyph[305:315, 600:900].mean() < 0.05  # letters yes, pill no
    # an editor text layer's mask is the letters themselves: the colour behind them is the ring just outside the mask
    letters = np.zeros((H, W), bool)
    for x in range(520, 980, 40):
        letters[320:380, x : x + 18] = True
    ratio = text_contrast(img, letters, [520, 320, 960, 380], 6, glyph_mask=True)
    assert ratio > 12


def test_stored_analysis_with_intent_and_stored_image(client):
    a = upload(client)
    ids = [e["id"] for e in a["elements"] if e["type"] != "background"]
    import json as _json

    r = client.get(f"/analysis/{a['analysis_id']}", params={"intent": _json.dumps(list(reversed(ids)))})
    assert r.status_code == 200 and r.json()["intent_check"] is not None and r.json()["analysis_id"] == a["analysis_id"]
    assert client.get(f"/analysis/{a['analysis_id']}").json()["intent_check"] is None
    assert client.get(f"/analysis/{a['analysis_id']}", params={"intent": "{bad"}).status_code == 422
    img = client.get(f"/image/{a['analysis_id']}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    arr = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
    assert arr.shape == (720, 1280, 3)
    assert client.get("/image/nope").status_code == 404
