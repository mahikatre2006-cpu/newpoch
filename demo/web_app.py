"""
Web backend for the Standalone Gaze Tracking & Lab Validation Demo.

Endpoints:
  GET  /api/status       - Returns tracker readiness and calibration status
  GET  /api/thumbnails   - Lists bundled sample YouTube thumbnails
  POST /api/track        - Returns smoothed, normalized screen coordinates (x, y)
  POST /api/recenter     - Sets current user gaze as the neutral center (0.5, 0.5)
  POST /api/calibrate    - Fits affine transform from 5-point calibration
  POST /api/reset        - Resets calibration to defaults
  POST /api/heatmap      - Generates thermal heatmap PNG (Inferno or Turbo)
"""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gaze.tracker import GazeTracker
from overlay.heatmap import generate_heatmap

# ── Global Gaze Tracker ───────────────────────────────────────────────
tracker: Optional[GazeTracker] = None


def get_tracker() -> GazeTracker:
    global tracker
    if tracker is None:
        print("[Web] Initializing GazeTracker...")
        tracker = GazeTracker()
        print("[Web] GazeTracker ready.")
    return tracker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-initialize tracker on startup."""
    try:
        get_tracker()
    except Exception as e:
        print(f"[Web] Tracker initialization warning: {e}")
    yield


app = FastAPI(title="Thumbnail Attention Studio - Gaze Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_frame(raw_bytes: bytes) -> Optional[np.ndarray]:
    nparr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ── API Endpoints ─────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    t = get_tracker()
    return {
        "ready": True,
        "is_calibrated": t.is_calibrated,
    }


@app.get("/api/thumbnails")
async def list_thumbnails():
    """Return pre-loaded sample thumbnails for the demo."""
    samples = [
        {
            "id": "tech",
            "title": "Next-Gen Tech Gadget",
            "tag": "Tech Review",
            "file": "tech_review.jpg",
            "url": "/thumbnails/tech_review.jpg",
            "saliency_hints": ["Glowing handheld gadget", "Creator face & reaction", "Top title text"],
        },
        {
            "id": "gaming",
            "title": "Ancient Temple Found!",
            "tag": "Gaming & Survival",
            "file": "gaming_adventure.jpg",
            "url": "/thumbnails/gaming_adventure.jpg",
            "saliency_hints": ["Golden temple center", "Gamer with neon visor", "Yellow title header"],
        },
        {
            "id": "sample",
            "title": "Production Studio Test",
            "tag": "Creator Studio",
            "file": "sample.jpg",
            "url": "/thumbnails/sample.jpg",
            "saliency_hints": ["High contrast central elements", "Text hierarchy"],
        },
    ]
    return samples


@app.post("/api/track")
async def track(frame: UploadFile = File(...)):
    """Extract real-time gaze coordinates from webcam frame."""
    t = get_tracker()
    contents = await frame.read()
    img = _decode_frame(contents)
    if img is None:
        return {"status": "error", "message": "Failed to decode frame"}

    result = t.process_frame(img)
    return result


@app.post("/api/recenter")
async def recenter(frame: UploadFile = File(...)):
    """Set current user gaze as the center baseline."""
    t = get_tracker()
    contents = await frame.read()
    img = _decode_frame(contents)
    if img is None:
        return {"status": "error", "message": "Failed to decode frame"}

    ok = t.recenter(img)
    return {"status": "ok" if ok else "no_face"}


@app.post("/api/calibrate")
async def calibrate(data: str = Form(...)):
    """Fit affine transformation from 5-point calibration."""
    t = get_tracker()
    parsed = json.loads(data)
    raw_pts = [tuple(p) for p in parsed["raw_points"]]
    scr_pts = [tuple(p) for p in parsed["screen_points"]]

    avg_err = t.fit_calibration(raw_pts, scr_pts)
    print(f"[Web] Calibration fitted on {len(raw_pts)} points. Avg error: {avg_err:.3f}")
    return {"status": "ok", "avg_error": avg_err, "n_points": len(raw_pts)}


@app.post("/api/reset")
async def reset():
    """Reset calibration to defaults."""
    t = get_tracker()
    t.reset()
    return {"status": "ok"}


@app.post("/api/heatmap")
async def generate_heatmap_endpoint(
    points_x: str = Form(...),
    points_y: str = Form(...),
    width: int = Form(1280),
    height: int = Form(720),
    colormap: str = Form("inferno"),
    sigma: float = Form(45.0),
):
    """Generate a thermal heatmap overlay from accumulated gaze coordinates."""
    # points_x and points_y are comma-separated normalized floats [0..1]
    px = [float(x) for x in points_x.split(",") if x.strip()]
    py = [float(y) for y in points_y.split(",") if y.strip()]

    # Map normalized coordinates to pixel space
    coords = [(x * width, y * height) for x, y in zip(px, py)]

    # Colormap selection (Inferno matches main app, Turbo is bright thermal)
    cv_cmap = "COLORMAP_INFERNO" if colormap.lower() == "inferno" else "COLORMAP_TURBO"

    heatmap_bgra = generate_heatmap(
        coords,
        (height, width, 3),
        sigma=sigma,
        colormap_name=cv_cmap,
        low_cut=0.08,
    )
    _, buffer = cv2.imencode(".png", heatmap_bgra)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "status": "ok",
        "heatmap_base64": f"data:image/png;base64,{img_b64}",
        "n_points": len(coords),
    }


# ── Static directories ────────────────────────────────────────────────
os.makedirs("thumbnails", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting Web Demo at http://localhost:8000")
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
