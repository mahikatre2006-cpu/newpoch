# Thumbnail Attention Studio

Predicts where viewers look on a YouTube thumbnail, explains why, and shows what changes when you edit it.
The spec is [docs/PRD.md](docs/PRD.md) (v2). v1 code is parked in `_v1_archive/` for reference only.

## Run

```
cd backend
.venv\Scripts\python -m uvicorn app.main:app --port 8000     # models load at start-up (about 10 s)
.venv\Scripts\python -m pytest                                # tests

cd frontend
npm install
npm run dev                                                   # http://localhost:5174, /api is proxied to :8000
```

Phase checks (timings, overlays and contact sheets land in `data/phase1/` and `data/phase2/`):
`python scripts\phase1_check.py` (heatmaps) and `python scripts\phase2_check.py` (elements: every pixel in exactly one element, shares sum to 100%), run from `backend/` with the venv.

## Setup that is not in git

- `backend/weights/` (git-ignored): `centerbias_mit1003.npy` (from the DeepGaze release page) and `torch/` (DeepGaze IIE and its backbones, about 900 MB, downloaded on first load).
- `backend/.env`: `DATABASE_URL` (Neon, optional; empty = local SQLite), later `YOUTUBE_API_KEY`.
- Windows Application Control blocks some newer wheels on the dev laptop; `requirements.txt` pins versions that load.

## Layout

`backend/app/{ingest,preprocess,saliency,...}` one package per PRD module · `frontend/src/{pages,analysis,editor,api}` · `ml/` evaluation · `data/` samples and cache
