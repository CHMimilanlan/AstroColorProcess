"""Simple full-reference metrics for the bundled short/long M51 pair."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference, candidate = load_rgb(args.reference), load_rgb(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    print(f"PSNR: {psnr(reference, candidate):.3f} dB")
    print(f"MAE:  {np.mean(np.abs(reference - candidate)):.6f}")


if __name__ == "__main__":
    main()

