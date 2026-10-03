"""
ETH-XGaze gaze estimation model (ResNet-18 backbone).

Loads a pre-trained checkpoint and predicts a 2D gaze direction
(pitch, yaw) from a normalised 224 × 224 face patch + head pose.

Weights are auto-downloaded from HuggingFace (hysts/ptgaze-eth-xgaze-resnet18)
on first run and cached to demo/weights/.  No manual download needed.

The gaze vector is then projected onto a virtual screen plane to produce
a raw (x, y) screen coordinate that the calibration module refines.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# HuggingFace auto-download
_HF_REPO = "hysts/ptgaze-eth-xgaze-resnet18"
_HF_FILENAME = "model.safetensors"


def _ensure_weights(weights_path: str) -> str:
    """Return *weights_path* if it exists; otherwise auto-download from HuggingFace.

    Downloads safetensors weights from hysts/ptgaze-eth-xgaze-resnet18,
    converts to a PyTorch state dict, and saves as a .pth file.
    """
    if os.path.isfile(weights_path):
        return weights_path

    print(f"[GazeModel] Weights not found at '{weights_path}'.")
    print(f"[GazeModel] Auto-downloading from HuggingFace ({_HF_REPO})...")

    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        sf_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_HF_FILENAME,
            local_dir=os.path.dirname(weights_path) or ".",
        )
        # Load safetensors → ordinary state dict → save as .pth
        state = load_file(sf_path)
        torch.save(state, weights_path)
        # Remove the intermediate safetensors file to keep things tidy
        if sf_path != weights_path and os.path.isfile(sf_path):
            os.remove(sf_path)
        print(f"[GazeModel] Weights saved to {weights_path}")
    except Exception as e:
        print(f"[GazeModel] WARNING: auto-download failed: {e}")
        print("[GazeModel]          Running with random weights (gaze inaccurate).")

    return weights_path


# ======================================================================
# Lightweight ResNet-18 gaze head (matches the ETH-XGaze checkpoint)
# ======================================================================

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class GazeResNet18(nn.Module):
    """ResNet-18 with a 2-unit gaze output (pitch, yaw)."""

    def __init__(self, num_out: int = 2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_out)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ======================================================================
# High-level wrapper
# ======================================================================

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GazeModel:
    """Load ETH-XGaze weights and predict gaze direction from a face patch."""

    def __init__(self, weights_path: str, device: str = "cpu", input_size: int = 224):
        self.device = torch.device(device)
        self.input_size = input_size

        self._model = GazeResNet18(num_out=2)

        weights_path = _ensure_weights(weights_path)
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            # Handle both raw state_dict and {'model': ...} wrappers
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            # Strip 'module.' prefix if saved with DataParallel
            state = {k.replace("module.", ""): v for k, v in state.items()}
            self._model.load_state_dict(state, strict=False)
            print(f"[GazeModel] Loaded weights from {weights_path}")
        else:
            print(f"[GazeModel] WARNING: weights not found at {weights_path}")
            print("[GazeModel]          Running with random weights (gaze will be inaccurate).")

        self._model.to(self.device).eval()

    # ------------------------------------------------------------------
    def predict(self, face_patch_bgr: np.ndarray) -> Tuple[float, float]:
        """Return (pitch, yaw) in radians from a 224×224 BGR face patch."""
        rgb = cv2.cvtColor(face_patch_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self._model(tensor)

        pitch = float(out[0, 0])
        yaw = float(out[0, 1])
        return (pitch, yaw)

    # ------------------------------------------------------------------
    @staticmethod
    def gaze_to_screen(
        pitch: float,
        yaw: float,
        head_R: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        screen_w: int = 1280,
        screen_h: int = 720,
    ) -> Tuple[float, float]:
        """Project gaze pitch/yaw through head rotation onto screen coords.

        This is a simplified geometric projection.  The calibration module
        further refines the mapping with an affine transform.
        """
        # Gaze direction in head coordinate system
        gaze_dir = np.array([
            -math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
            -math.cos(pitch) * math.cos(yaw),
        ], dtype=np.float64)

        # Rotate gaze into camera coordinate system
        gaze_cam = head_R @ gaze_dir

        # Intersect with the screen plane (z = focal_length * scale)
        # We use a simplified projection: the gaze ray from the face centre
        # direction is (gaze_cam), and we project it onto the image plane.
        if abs(gaze_cam[2]) < 1e-6:
            # Looking almost parallel to the camera — clamp to centre
            return (screen_w / 2.0, screen_h / 2.0)

        # Scale factor to project onto screen
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Project the gaze onto the image plane
        px = fx * (gaze_cam[0] / gaze_cam[2]) + cx
        py = fy * (gaze_cam[1] / gaze_cam[2]) + cy

        # Map from camera image coords to screen coords
        # Camera is typically 640×480, screen display is 1280×720
        cam_w = cx * 2
        cam_h = cy * 2
        sx = px / cam_w * screen_w
        sy = py / cam_h * screen_h

        return (float(sx), float(sy))
