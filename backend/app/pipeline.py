"""One analysis, end to end (PRD section 6). Modules plug in here; a module that fails leaves its part null
and adds an entry to `errors`, so the request never fails as a whole."""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app.ablation import ablate as run_ablation
from app.attention import analyze_attention, intent_check
from app.config import load_config
from app.elements import ElementEngine
from app.explain import Explainer
from app.features import compute_features
from app.fixes import FIXES, apply_fix, make_layers
from app.ingest import IngestError, ingest_upload, ingest_video_id
from app.preprocess import build_context
from app.saliency import SaliencyEngine, ScanpathEngine, encode_heatmap_png, encode_preview_jpg
from app.storage import Store, make_store
from app.versions import compare_results, new_version
from app.versions.core import describe_change

RESULT_FORMAT = 5  # bump when the shape or meaning of a stored result changes


class LookupFailed(ValueError):
    status = 404


def _png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


def apply_intent(result: dict, intent: list[str] | None, explainer: Explainer | None = None) -> dict:
    """The creator's intended order only changes `intent_rank`, `intent_check` and the mismatch sentence, never the
    analysis itself, so it is applied to a copy after the (cacheable) result exists."""
    out = copy.deepcopy(result)
    for e in out.get("elements") or []:
        e["intent_rank"] = None
    out["intent_check"] = None
    if intent and out.get("hierarchy"):
        check, intended, warnings = intent_check(intent, out["hierarchy"])
        out["intent_check"] = check
        for e in out["elements"]:
            e["intent_rank"] = intended.get(e["id"])
        if explainer is not None:
            for m in check["mismatches"]:
                e = next(x for x in out["elements"] if x["id"] == m["element_id"])
                msg = explainer.intent_message(e["label"], m["intended"], m["predicted"])
                if msg and e.get("explanations") is not None:
                    e["explanations"] = ([msg] + e["explanations"])[:4]
        out["errors"] = list(out["errors"]) + warnings
    return out


class Pipeline:
    def __init__(self, cfg: dict | None = None, store: Store | None = None,
                 engine: SaliencyEngine | None = None, elements: ElementEngine | None = None,
                 scanpath: ScanpathEngine | None = None, explainer: Explainer | None = None):
        self.cfg = cfg or load_config()
        self.store = store or make_store()
        self.saliency = engine or SaliencyEngine(self.cfg)
        if engine is None:
            self.saliency.load()  # models load once, at startup
        self.elements = elements or ElementEngine(self.cfg)
        if elements is None:
            self.elements.load()
        self.scanpath = scanpath or ScanpathEngine(self.cfg)
        if scanpath is None:
            self.scanpath.load()
        self.explainer = explainer or Explainer()
        self.pool = ThreadPoolExecutor(max_workers=self.cfg["elements"]["workers"], thread_name_prefix="m3m4")
        self.signature = hashlib.sha256(json.dumps(
            [RESULT_FORMAT, self.cfg, self.saliency.status["loaded"], self.elements.status["loaded"], self.scanpath.status["loaded"]],
            sort_keys=True, default=str).encode()).hexdigest()[:16]

    @property
    def status(self) -> dict:
        return {**self.saliency.status, "elements": self.elements.status, "scanpath": self.scanpath.status}

    def close(self) -> None:
        self.pool.shutdown(wait=False)
        self.store.close()

    def analyze(self, data: bytes | None = None, *, video_id: str | None = None, layers: list | None = None,
                intent: list[str] | None = None, session_id: str | None = None) -> dict:
        t_all = time.perf_counter()
        timing: dict[str, int] = {}

        t = time.perf_counter()
        if video_id:
            ing = ingest_video_id(video_id, self.cfg["ingest"])
        elif data is not None:
            ing = ingest_upload(data, self.cfg["ingest"], layers)
        else:
            raise IngestError("Send an image file or a YouTube video ID")
        timing["ingest"] = _ms(t)

        if self.cfg["cache"]["enabled"]:
            hit = self.store.get_analysis(ing.sha256, self.signature)
            if hit is not None:
                hit = apply_intent(hit, intent, self.explainer)
                hit.update(session_id=session_id or hit.get("session_id"), cached=True, timing_ms={"cache": _ms(t_all), "total": _ms(t_all)})
                return hit

        t = time.perf_counter()
        ctx = build_context(ing.image, ing.sha256, self.cfg)
        timing["preprocess"] = _ms(t)
        errors: list[str] = []

        # M3 and M4 are independent: run them side by side
        def timed(fn, *a):
            t0 = time.perf_counter()
            return fn(*a), _ms(t0)

        f_sal = self.pool.submit(timed, self.saliency.predict, ctx)
        f_el = self.pool.submit(timed, self.elements.detect, ctx, layers)
        f_sp = self.pool.submit(timed, self.scanpath.predict, ctx) if self.scanpath.model is not None else None

        sal, eset = None, None
        try:
            sal, timing["saliency"] = f_sal.result()
            errors.extend(sal.errors)
        except Exception as e:  # noqa: BLE001: even the fallback failed: no heatmap, but the rest still reports
            errors.append(f"saliency: {type(e).__name__}: {e}")
        # invalid editor layers are the caller's mistake (LayerError), so that one propagates as a 422
        (eset, el_errors), timing["elements"] = f_el.result()
        errors.extend(el_errors)

        fixations = None
        if f_sp is not None:
            try:
                fixations, timing["scanpath"] = f_sp.result()
            except Exception as e:  # noqa: BLE001: the winner-take-all path on the map takes over
                errors.append(f"scanpath: {type(e).__name__}: {e}")

        t = time.perf_counter()
        att = None
        if sal is not None:
            att = analyze_attention(sal.map, eset, self.cfg["attention"]["scanpath"], fixations=fixations)
        timing["attention"] = _ms(t)

        frame, layer_views = None, None
        if att is not None:
            t = time.perf_counter()
            try:
                feat = compute_features(ctx, eset, sal.map, self.cfg, getattr(self.elements, "mobile_ocr", None))
                frame, layer_views = feat.frame, feat.layer_views
            except Exception as e:  # noqa: BLE001: no features / explanations, but the attention result still stands
                errors.append(f"features: {type(e).__name__}: {e}")
            if frame is not None:
                try:
                    self.explainer.explain(att.elements, feat.elements, frame)
                except Exception as e:  # noqa: BLE001: measurements are still useful without sentences
                    errors.append(f"explanations: {type(e).__name__}: {e}")
                    frame["explanations"] = []
            timing["features"] = _ms(t)

        t = time.perf_counter()
        render = self.cfg["saliency"]["render"]
        timing["render"] = 0
        heatmap = encode_heatmap_png(sal.map, render) if sal is not None else None
        preview = encode_preview_jpg(ing.image, render)
        timing["render"] = _ms(t)
        timing["total"] = _ms(t_all)

        result = {
            "analysis_id": ing.sha256,
            "session_id": session_id or str(uuid.uuid4()),
            "saliency_model": sal.model if sal else "none",
            "heatmap_png": heatmap,
            "image_jpg": preview,
            "image_info": ing.info,
            "elements": att.elements if att else None,
            "hierarchy": att.hierarchy if att else None,
            "scanpath": att.scanpath if att else None,
            "frame": frame, "intent_check": None, "layer_views": layer_views,
            "timing_ms": timing, "errors": errors, "cached": False,
        }
        if self.cfg["cache"]["enabled"] and sal is not None:  # never cache a degraded result with no heatmap
            self.store.save_analysis(result, self.signature)
            self.store.save_image(ing.sha256, _png(ing.image), layers)  # ablation and fixes work on this exact frame
        return apply_intent(result, intent, self.explainer)


    # ---------------- M8, M10, M11: work on a stored analysis ----------------
    def get_analysis(self, analysis_id: str) -> dict | None:
        return self.store.get_analysis(analysis_id, self.signature)

    def analysis_with_intent(self, analysis_id: str, intent: list[str] | None) -> dict | None:
        result = self.get_analysis(analysis_id)
        return None if result is None else apply_intent(result, intent, self.explainer)

    def image_png(self, analysis_id: str) -> bytes | None:
        stored = self.store.get_image(analysis_id)
        return None if stored is None else stored[0]

    def _load(self, analysis_id: str) -> tuple[dict, np.ndarray, list | None]:
        result = self.get_analysis(analysis_id)
        stored = self.store.get_image(analysis_id)
        if result is None or stored is None:
            raise LookupFailed("Unknown analysis: analyse the image again first")
        png, layers = stored
        bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        return result, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), layers

    def record_version(self, session_id: str, analysis_id: str, label: str, kind: str, parent_id: str | None = None) -> dict:
        """Save a version. Saving the same analysis twice in a row returns the existing version instead of a duplicate."""
        existing = self.store.list_versions(session_id)
        if existing and existing[-1]["analysis_id"] == analysis_id:
            return existing[-1]
        v = new_version(session_id, analysis_id, label, kind, parent_id or (existing[-1]["version_id"] if existing else None))
        self.store.add_version(v)
        return self.store.get_version(v["version_id"]) or v

    def versions(self, session_id: str) -> list[dict]:
        return self.store.list_versions(session_id)

    def ablate(self, analysis_id: str, element_id: str) -> dict:
        result, image, _ = self._load(analysis_id)
        t = time.perf_counter()
        out = run_ablation(self.saliency, self.cfg, image, result, element_id)
        out["timing_ms"] = {"total": _ms(t)}
        return out

    def fix(self, analysis_id: str, fix: str, session_id: str | None = None) -> dict:
        """Apply a fix to a stored analysis. Pixel fixes are re-analysed and saved as a new version with the measured change;
        'separate_layers' returns the two layers for the editor instead."""
        result, image, layers = self._load(analysis_id)
        label = FIXES.get(fix, "Separate layers" if fix == "separate_layers" else fix)
        if fix == "separate_layers":
            return {"fix": fix, "label": label, "layers": make_layers(image, result, self.cfg), "analysis": None}
        fixed = apply_fix(fix, image, result, self.cfg)
        sid = session_id or result.get("session_id") or str(uuid.uuid4())
        before_version = next((v for v in reversed(self.store.list_versions(sid)) if v["analysis_id"] == analysis_id), None)
        new = self.analyze(_png(fixed), layers=layers, session_id=sid)
        diff = compare_results(result, new)
        version = self.record_version(sid, new["analysis_id"], label, "fix", before_version["version_id"] if before_version else None)
        new["version_id"] = version["version_id"]
        return {"fix": fix, "label": label, "version": version, "analysis": new, "changes": diff["rows"],
                "hierarchy_before": diff["hierarchy_a"], "hierarchy_after": diff["hierarchy_b"],
                "message": describe_change(diff["rows"], label)}

    def compare(self, version_a: str, version_b: str) -> dict:
        va, vb = self.store.get_version(version_a), self.store.get_version(version_b)
        if va is None or vb is None:
            raise LookupFailed("Unknown version")
        ra, rb = self.get_analysis(va["analysis_id"]), self.get_analysis(vb["analysis_id"])
        if ra is None or rb is None:
            raise LookupFailed("A version's analysis is no longer available: analyse it again")
        diff = compare_results(ra, rb)

        def side(v, r):
            return {"version": v, "image_jpg": r["image_jpg"], "heatmap_png": r["heatmap_png"], "saliency_model": r["saliency_model"]}

        return {"a": side(va, ra), "b": side(vb, rb), **diff}
