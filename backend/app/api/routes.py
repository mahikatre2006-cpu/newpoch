"""HTTP routes. Thin: parse the request, call the pipeline, return the contract."""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.schemas import AnalysisResult, Health
from app.elements import LayerError
from app.ingest import IngestError

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
    try:
        return request.app.state.pipeline.analyze(data, video_id=video_id, layers=layers_obj, intent=intent_list, session_id=session_id)
    except (IngestError, LayerError) as e:
        raise HTTPException(status_code=422, detail=str(e))
