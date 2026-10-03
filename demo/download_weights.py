#!/usr/bin/env python3
"""
One-time weight download script for the ETH-XGaze demo module.
Downloads from HuggingFace (hysts/ptgaze-eth-xgaze-resnet18) and
saves to demo/weights/eth-xgaze_resnet18.pth

Usage:
    cd demo
    python download_weights.py
"""

import os
import sys

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "eth-xgaze_resnet18.pth")
HF_REPO = "hysts/ptgaze-eth-xgaze-resnet18"
HF_FILENAME = "model.safetensors"


def main():
    if os.path.isfile(WEIGHTS_PATH):
        print(f"[OK] Weights already present at {WEIGHTS_PATH}")
        return

    print(f"Downloading ETH-XGaze weights from HuggingFace...")
    print(f"  Repo: {HF_REPO}")
    print(f"  Saving to: {WEIGHTS_PATH}")
    print()

    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import torch
    except ImportError as e:
        print(f"ERROR: Missing package: {e}")
        print("Make sure you have activated the demo venv:")
        print("  .venv\\Scripts\\activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    print("[1/2] Downloading model.safetensors ...")
    sf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        local_dir=os.path.dirname(WEIGHTS_PATH),
    )

    print("[2/2] Converting safetensors -> .pth ...")
    state = load_file(sf_path)
    torch.save(state, WEIGHTS_PATH)

    # Clean up the intermediate safetensors file
    if sf_path != WEIGHTS_PATH and os.path.isfile(sf_path):
        os.remove(sf_path)

    size_mb = os.path.getsize(WEIGHTS_PATH) / 1e6
    print(f"\nDone!  Saved {size_mb:.1f} MB -> {WEIGHTS_PATH}")
    print("\nYou can now run the demo:")
    print("  python run_demo.py --skip-calibration")


if __name__ == "__main__":
    main()
