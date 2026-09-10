"""Download the deployable PyTorch weights from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "MilanMagik/AstroStarDetect"
ROOT = Path(__file__).resolve().parent

MODEL_FILES = {
    "denoise": ["StarDenoise/models/deep_denoise_cnn_AI3_6.pth"],
    "starnet": [
        "StarNetPyTorch/models/weights_G_RGB.pth",
        "StarNetPyTorch/models/weights_G_Greyscale.pth",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AstroColorProcess model weights from Hugging Face."
    )
    parser.add_argument(
        "--only",
        choices=("all", "denoise", "starnet"),
        default="all",
        help="Select which model weights to download (default: all).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch, tag, or commit (default: main).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when a cached file is available.",
    )
    args = parser.parse_args()

    groups = MODEL_FILES if args.only == "all" else {args.only: MODEL_FILES[args.only]}
    for filenames in groups.values():
        for filename in filenames:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                revision=args.revision,
                local_dir=ROOT,
                force_download=args.force,
            )
            print(f"Downloaded: {local_path}")


if __name__ == "__main__":
    main()
