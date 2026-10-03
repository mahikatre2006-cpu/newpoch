#!/usr/bin/env python3
"""
Standalone Gaze Tracking Demo — Thumbnail Attention Studio
==========================================================

Single entry-point that orchestrates the full demo flow:

  1. Splash screen
  2. Webcam calibration (or load saved calibration)
  3. Display thumbnail + live glowing blob following gaze
  4. After N seconds, generate thermal heatmap overlay
  5. Show result until user quits or restarts

Usage
-----
    python run_demo.py                        # full flow
    python run_demo.py --skip-calibration     # reuse saved calibration
    python run_demo.py --thumbnail path.jpg   # custom thumbnail
    python run_demo.py --duration 4           # track for 4 seconds
    python run_demo.py --no-fullscreen        # windowed mode
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaze.tracker import GazeTracker
from overlay.blob_renderer import BlobRenderer
from overlay.heatmap import generate_heatmap, blend_heatmap
from overlay.ui import (
    setup_window,
    show_splash,
    draw_countdown,
    show_generating,
    draw_result_footer,
    fade_in_overlay,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_demo(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    if args.thumbnail:
        config["thumbnail"]["path"] = args.thumbnail
    if args.duration:
        config["tracking"]["duration_s"] = args.duration

    screen_w, screen_h = config["thumbnail"]["display_size"]
    fullscreen = (not args.no_fullscreen) and config["ui"].get("fullscreen", True)

    print("[Demo] Initialising camera...")
    cam_cfg = config["camera"]
    cam = cv2.VideoCapture(cam_cfg.get("index", 0))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
    if not cam.isOpened():
        print("[Demo] ERROR: Cannot open webcam. Check camera permissions.")
        sys.exit(1)

    print("[Demo] Initialising GazeTracker (MediaPipe Iris + 3D Pose)...")
    tracker = GazeTracker()

    blob = BlobRenderer(config["blob"])
    window = setup_window("Demo", fullscreen=fullscreen)

    while True:
        _run_single_demo(cam, tracker, blob, config, window, screen_w, screen_h, args)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[Demo] Exiting.")
                cam.release()
                cv2.destroyAllWindows()
                return
            if key in (ord("r"), ord("R")):
                tracker.reset()
                blob.reset()
                break


def _run_single_demo(
    cam: cv2.VideoCapture,
    tracker: GazeTracker,
    blob: BlobRenderer,
    config: dict,
    window: str,
    screen_w: int,
    screen_h: int,
    args: argparse.Namespace,
) -> None:
    # 1. Splash
    splash_dur = config["ui"].get("splash_duration_s", 1.5)
    show_splash(window, screen_w, screen_h, duration_s=splash_dur)

    # 2. Calibration (optional)
    if not args.skip_calibration:
        print("[Demo] Quick center alignment: look at the screen center...")
        t0 = time.time()
        while time.time() - t0 < 1.0:
            ret, frame = cam.read()
            if ret:
                tracker.recenter(frame)
                break

    # 3. Load thumbnail
    thumb_path = config["thumbnail"]["path"]
    if not os.path.exists(thumb_path):
        thumb_path = "thumbnails/tech_review.jpg"

    thumbnail = cv2.imread(thumb_path)
    if thumbnail is None:
        thumbnail = _placeholder_thumbnail(screen_w, screen_h)
    else:
        thumbnail = cv2.resize(thumbnail, (screen_w, screen_h))

    # Fade in thumbnail
    for alpha in _linspace(0.0, 1.0, 10):
        vis = (thumbnail.astype(np.float32) * alpha).astype(np.uint8)
        cv2.imshow(window, vis)
        cv2.waitKey(16)

    # 4. Live tracking
    duration_s = config["tracking"].get("duration_s", 4.0)
    countdown_color = tuple(config["ui"].get("countdown_color", [0, 220, 255]))
    coords: list[tuple[float, float]] = []

    print(f"[Demo] Tracking for {duration_s}s — look at the thumbnail!")
    t_start = time.time()

    while True:
        elapsed = time.time() - t_start
        remaining = duration_s - elapsed
        if remaining <= 0:
            break

        ret, frame = cam.read()
        if not ret:
            continue

        res = tracker.process_frame(frame)
        display = thumbnail.copy()

        if res["status"] == "ok":
            sx = float(res["x"] * screen_w)
            sy = float(res["y"] * screen_h)
            coords.append((sx, sy))
            display = blob.draw(display, sx, sy)
        else:
            cv2.putText(display, "Position face in front of webcam", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        display = draw_countdown(display, remaining, duration_s, countdown_color)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord("c"), ord("C")):
            tracker.recenter(frame)

    # 5. Generate heatmap
    show_generating(window, thumbnail, duration_s=0.5)

    heat_cfg = config["heatmap"]
    heatmap_bgra = generate_heatmap(
        coords,
        thumbnail.shape,
        sigma=heat_cfg.get("sigma", 42),
        colormap_name=heat_cfg.get("colormap", "COLORMAP_INFERNO"),
        low_cut=heat_cfg.get("low_cut", 0.08),
    )
    result = blend_heatmap(thumbnail, heatmap_bgra, alpha=heat_cfg.get("alpha", 0.65))

    fade_dur = config["ui"].get("fade_duration_s", 0.4)
    fade_in_overlay(window, thumbnail, result, duration_s=fade_dur)

    result = draw_result_footer(result)
    cv2.imshow(window, result)
    print(f"[Demo] Heatmap generated from {len(coords)} gaze fixations.")
    print("[Demo] Press R to restart, Q to quit.")


def _placeholder_thumbnail(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        frac = y / h
        img[y, :] = (int(40 + 30 * frac), int(30 + 20 * frac), int(50 + 40 * frac))
    cv2.putText(img, "Thumbnail Attention Studio Demo",
                (w // 2 - 250, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
    return img


def _linspace(start: float, end: float, n: int):
    for i in range(n):
        yield start + (end - start) * i / max(1, n - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Gaze Tracking Demo — Thumbnail Attention Studio",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--thumbnail", default=None, help="Override thumbnail image path")
    parser.add_argument("--duration", type=float, default=4.0, help="Tracking duration in seconds")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration")
    parser.add_argument("--no-fullscreen", action="store_true", help="Run in a window")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
