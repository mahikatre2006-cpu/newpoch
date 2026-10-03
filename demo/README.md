# Gaze Tracking Demo — Thumbnail Attention Studio

> **Standalone interactive demo** for hackathon judges.
> Tracks where you look on a YouTube thumbnail using a webcam, then generates a real-time thermal heatmap.

This module is **completely isolated** from the core predictive ML backend. It demonstrates **how ground-truth saliency data is collected in the real world**, setting up the pitch:

> *"This is how empirical data is gathered — it's expensive and slow. That's why we built our core predictive model to do this instantly."*

---

## Quick Start

```bash
cd demo

# 1. Create a venv (separate from the backend)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Web Demo (Recommended for Judges & Presentations)
python web_app.py
# Open http://localhost:8000 in your browser!

# 4. Or run the Desktop OpenCV version
python run_demo.py
```

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to the config file |
| `--thumbnail` | from config | Override the thumbnail image path |
| `--duration` | `4` (seconds) | How long to track gaze |
| `--skip-calibration` | off | Reuse saved `calibration.yaml` instead of re-calibrating |
| `--no-fullscreen` | off | Run in a resizable window |

**Recommended for the live pitch:**
```bash
# Pre-calibrate before judges arrive
python run_demo.py

# Then for the actual demo (skip calibration for speed)
python run_demo.py --skip-calibration
```

---

## Demo Flow

1. **Splash screen** (2 s) — title card
2. **Calibration** (~15 s) — 5 dots appear; look at each one
3. **Thumbnail appears** — fade-in animation
4. **Live tracking** (4 s) — glowing blob follows your gaze; countdown timer
5. **Heatmap generation** — brief "analysing" transition
6. **Result** — thermal heatmap overlay; press **R** to restart, **Q** to quit

---

## Project Structure

```
demo/
├── run_demo.py            # Entry point
├── config.yaml            # All tunables
├── requirements.txt       # Isolated dependencies
├── gaze/
│   ├── face_detector.py   # MediaPipe Face Mesh + head pose
│   ├── gaze_model.py      # ETH-XGaze ResNet-18
│   ├── calibration.py     # 5-point screen calibration
│   └── smoother.py        # EMA coordinate smoothing
├── overlay/
│   ├── blob_renderer.py   # Glowing blob with trails
│   ├── heatmap.py         # Gaussian density → COLORMAP_TURBO
│   └── ui.py              # Window, countdown, transitions
├── thumbnails/
│   └── sample.jpg         # Place your thumbnail here
└── weights/
    └── eth-xgaze_resnet18.pth  # Download separately
```

---

## Model Weights

The ETH-XGaze ResNet-18 checkpoint (~45 MB) is **not** included in this repo.

**Option A (recommended):** Clone [gaze_track_webcam](https://github.com/ChiShengChen/gaze_track_webcam) and copy the model file.

**Option B:** The demo will still run without weights (using random ResNet-18 parameters), but gaze tracking will be inaccurate. Useful for testing the UI flow.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Cannot open webcam" | Check Windows Settings → Privacy → Camera → Allow apps to access your camera |
| `opencv-python` conflicts | Uninstall all opencv packages first: `pip uninstall opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless`, then `pip install opencv-contrib-python==4.11.0.86` |
| Gaze is inaccurate | Re-calibrate (don't use `--skip-calibration`); ensure good lighting; sit ~60 cm from screen |
| "DLL load failed" | Windows Application Control is blocking a wheel — use the pinned versions in requirements.txt |
