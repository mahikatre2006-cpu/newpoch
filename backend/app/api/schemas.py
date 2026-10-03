"""The response shapes (PRD section 9). Parts owned by modules that are not built yet stay null."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    session_id: str
    saliency_model: str
    heatmap_png: str | None  # null if saliency failed entirely (see errors)
    image_jpg: str  # the normalised 16:9 frame the models saw, so the overlay lines up exactly
    image_info: dict[str, Any]
    elements: list[dict[str, Any]] | None = None
    hierarchy: list[str] | None = None
    scanpath: list[dict[str, Any]] | None = None
    frame: dict[str, Any] | None = None
    intent_check: dict[str, Any] | None = None
    layer_views: dict[str, Any] | None = None
    timing_ms: dict[str, int]
    errors: list[str]
    cached: bool = False
    version_id: str | None = None  # set when this analysis was saved as a version


class Health(BaseModel):
    status: str  # "ok" | "degraded" (running on a fallback)
    saliency: dict[str, Any]


class AblateRequest(BaseModel):
    analysis_id: str
    element_id: str


class FixRequest(BaseModel):
    analysis_id: str
    fix: str  # focus_subject | fix_text | enhance | separate_layers
    session_id: str | None = None
