"""Reproducible 1080p benchmark using three CC BY 4.0 ESO images.

The ESO publication images are already processed and comparatively clean, so this
script creates deterministic Poisson + Gaussian sensor-noise inputs.  The original
downloads remain untouched and act as full-reference targets.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from denoise import denoise_array
from model import load_model


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "inputs_1080p" / "references"
NOISY_DIR = ROOT / "inputs_1080p" / "noisy"
OUTPUT_DIR = ROOT / "outputs_1080p"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"
WEIGHTS = ROOT / "models" / "deep_denoise_cnn_AI3_6.pth"

IMAGES = {
    "vbg_011_nebulae": {
        "filename": "vbg_011_nebulae.jpg",
        "title": "Eagle / Omega / Sh2-54 nebulae",
        "source": "https://www.eso.org/public/products/virtualbackgrounds/vbg_011/",
        "credit": "ESO",
    },
    "vbg_012_messier17": {
        "filename": "vbg_012_messier17.jpg",
        "title": "Messier 17",
        "source": "https://www.eso.org/public/products/virtualbackgrounds/vbg_012/",
        "credit": "ESO",
    },
    "vbg_013_orion": {
        "filename": "vbg_013_orion.jpg",
        "title": "Orion Nebula",
        "source": "https://www.eso.org/public/products/virtualbackgrounds/vbg_013/",
        "credit": "ESO/Igor Chekalin",
    },
}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(array, "RGB").save(path)


def add_sensor_noise(clean: np.ndarray, seed: int) -> np.ndarray:
    """Add signal-dependent shot noise plus channel-wise read noise.

    ``peak_electrons`` controls the Poisson component.  The values are deliberately
    moderate: visible at 100% but not so strong that the test becomes artificial.
    """
    rng = np.random.default_rng(seed)
    peak_electrons = 350.0
    shot = rng.poisson(np.clip(clean, 0, 1) * peak_electrons).astype(np.float32) / peak_electrons
    read = rng.normal(0.0, 0.010, clean.shape).astype(np.float32)
    # A small row-correlated component approximates residual banding/read pattern.
    row_pattern = rng.normal(0.0, 0.003, (clean.shape[0], 1, 3)).astype(np.float32)
    return np.clip(shot + read + row_pattern, 0.0, 1.0)


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Standard local SSIM averaged over RGB channels."""
    c1, c2 = 0.01**2, 0.03**2
    scores = []
    for channel in range(3):
        x, y = reference[..., channel], candidate[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean(np.abs(reference - candidate)))


def make_comparison(path: Path, noisy: np.ndarray, denoised: np.ndarray, reference: np.ndarray, title: str) -> None:
    panel_size = (640, 360)
    labels = ("NOISY INPUT", "DENOISED", "ESO REFERENCE")
    images = (noisy, denoised, reference)
    canvas = Image.new("RGB", (1920, 410), "#111111")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="white")
    for index, (label, array) in enumerate(zip(labels, images)):
        panel = Image.fromarray(np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
        panel = panel.resize(panel_size, Image.Resampling.LANCZOS)
        x = index * panel_size[0]
        canvas.paste(panel, (x, 50))
        draw.text((x + 12, 30), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


def main() -> None:
    NOISY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(WEIGHTS, device)
    results = []

    for seed, (key, metadata) in enumerate(IMAGES.items(), start=20260901):
        reference_path = REFERENCE_DIR / metadata["filename"]
        reference = load_rgb(reference_path)
        if reference.shape[:2] != (1080, 1920):
            raise ValueError(f"Expected 1920x1080: {reference_path} -> {reference.shape}")
        noisy = add_sensor_noise(reference, seed)
        noisy_path = NOISY_DIR / f"{key}_noisy.png"
        save_rgb(noisy_path, noisy)
        # Benchmark the exact 8-bit file that a user can pass to denoise.py.
        noisy = load_rgb(noisy_path)

        started = time.perf_counter()
        denoised = denoise_array(
            noisy,
            model,
            device,
            strength=0.85,
            color_strength=0.75,
            mode="full",
            tile_size=256,
            overlap=64,
            batch_size=1,
            half_precision=device.type == "cuda",
            linear_stretch="never",
        )
        elapsed = time.perf_counter() - started
        output_path = OUTPUT_DIR / f"{key}_denoised.png"
        save_rgb(output_path, denoised)
        denoised = load_rgb(output_path)
        make_comparison(
            COMPARISON_DIR / f"{key}_comparison.jpg",
            noisy,
            denoised,
            reference,
            metadata["title"],
        )

        result = {
            "id": key,
            **metadata,
            "resolution": "1920x1080",
            "device": str(device),
            "elapsed_seconds": round(elapsed, 3),
            "noisy_psnr_db": round(psnr(reference, noisy), 3),
            "denoised_psnr_db": round(psnr(reference, denoised), 3),
            "noisy_ssim": round(ssim(reference, noisy), 5),
            "denoised_ssim": round(ssim(reference, denoised), 5),
            "noisy_mae": round(mae(reference, noisy), 6),
            "denoised_mae": round(mae(reference, denoised), 6),
            "input": str(noisy_path.relative_to(ROOT)),
            "output": str(output_path.relative_to(ROOT)),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    (OUTPUT_DIR / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
