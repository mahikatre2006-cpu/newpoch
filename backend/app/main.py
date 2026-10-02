"""FastAPI app. Run from backend/:  python -m uvicorn app.main:app --port 8000"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "pipeline", None) is None:  # tests inject their own
        app.state.pipeline = Pipeline()  # DeepGaze and the other models load once, here
    yield
    app.state.pipeline.close()


app = FastAPI(title="Thumbnail Attention Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error(_, exc: RequestValidationError):
    """FastAPI's default 422 echoes the offending input back, which breaks the response when that input is
    NaN or Infinity (not valid JSON). Return only where and why."""
    return JSONResponse(status_code=422, content={"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]})


app.include_router(router)
