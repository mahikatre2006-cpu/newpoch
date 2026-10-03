"""HTTP routes. Thin: parse the request, call the pipeline, return the contract."""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile

from app.ablation import AblationError
from app.api.schemas import AblateRequest, AnalysisResult, FixRequest, Health
from app.elements import LayerError
from app.fixes import FixError
from app.ingest import IngestError
from app.pipeline import LookupFailed

router = APIRouter()


@router.get("/health", response_model=Health)
def health(request: Request):
    status = request.app.state.pipeline.status
    degraded = status["fallback_active"] or bool(status["elements"]["unavailable"]) or bool(status["scanpath"]["unavailable"])
    return Health(status="degraded" if degraded else "ok", saliency=status)


@router.post("/analyze", response_model=AnalysisResult)
def analyze(
    request: Request,
    image: UploadFile | None = File(None),
    video_id: str | None = Form(None),
    layers: str | None = Form(None),
    intent: str | None = Form(None),
    session_id: str | None = Form(None),
    save_version: bool = Form(False),
    label: str | None = Form(None),
):
    """An image file (or a YouTube video ID), with optional editor `layers` (JSON) and `intent` (JSON list of element ids)."""
    try:
        layers_obj = json.loads(layers) if layers else None
        intent_list = json.loads(intent) if intent else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="layers and intent must be valid JSON")
    if intent_list is not None and not (isinstance(intent_list, list) and all(isinstance(i, str) for i in intent_list)):
        raise HTTPException(status_code=422, detail="intent must be a JSON list of element ids")
    data = image.file.read() if image is not None else None
    pipe = request.app.state.pipeline
    try:
        result = pipe.analyze(data, video_id=video_id, layers=layers_obj, intent=intent_list, session_id=session_id)
    except (IngestError, LayerError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    if save_version:  # an explicit "keep this" (an upload, or the editor's Save version): not every debounced edit
        kind = "edit" if layers_obj else "upload"
        name = (label or ("Editor version" if layers_obj else "Upload")).strip()
        result["version_id"] = pipe.record_version(result["session_id"], result["analysis_id"], name, kind)["version_id"]
    return result


def _guard(fn, *args):
    try:
        return fn(*args)
    except (AblationError, FixError, LookupFailed) as e:
        raise HTTPException(status_code=e.status, detail=str(e))


@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str, request: Request, intent: str | None = None):
    """A stored analysis, optionally with the creator's intended order (JSON list of element ids) applied to it."""
    try:
        intent_list = json.loads(intent) if intent else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="intent must be valid JSON")
    if intent_list is not None and not (isinstance(intent_list, list) and all(isinstance(i, str) for i in intent_list)):
        raise HTTPException(status_code=422, detail="intent must be a JSON list of element ids")
    result = request.app.state.pipeline.analysis_with_intent(analysis_id, intent_list)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown analysis id")
    return result


@router.get("/image/{analysis_id}")
def get_image(analysis_id: str, request: Request):
    """The stored 1280x720 frame of an analysis (what the editor opens)."""
    png = request.app.state.pipeline.image_png(analysis_id)
    if png is None:
        raise HTTPException(status_code=404, detail="Unknown analysis id")
    return Response(content=png, media_type="image/png")


@router.post("/ablate")
def ablate(body: AblateRequest, request: Request):
    """Remove one element and see where its attention goes."""
    return _guard(request.app.state.pipeline.ablate, body.analysis_id, body.element_id)


@router.post("/fix")
def fix(body: FixRequest, request: Request):
    """Apply a fix, re-analyse, save a new version and report what changed."""
    return _guard(request.app.state.pipeline.fix, body.analysis_id, body.fix, body.session_id)


@router.get("/versions/{session_id}")
def versions(session_id: str, request: Request):
    return {"session_id": session_id, "versions": request.app.state.pipeline.versions(session_id)}


@router.get("/compare")
def compare(a: str, b: str, request: Request):
    """Two versions side by side: both heatmaps and the per-element attention change."""
    return _guard(request.app.state.pipeline.compare, a, b)
