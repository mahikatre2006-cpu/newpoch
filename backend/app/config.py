"""Paths, config.yaml and environment. The only module that touches the filesystem layout."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")
WEIGHTS_DIR = BACKEND_DIR / "weights"

# keep every downloaded model (DeepGaze backbones, rembg) inside the project so the demo machine runs offline
os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR / "torch"))
os.environ.setdefault("U2NET_HOME", str(WEIGHTS_DIR / "u2net"))


def data_dir() -> Path:
    d = Path(os.environ.get("THUMB_DATA_DIR", BACKEND_DIR.parent / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(BACKEND_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
