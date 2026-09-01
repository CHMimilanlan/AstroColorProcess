#!/usr/bin/env python3
"""Deconvolve an image that has already been stretched (JPEG/TIFF/PNG)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from compare_fits import render_pair
from star_deconvolution import (
    DeconvolutionConfig,
    _robust_noise,
    _to_luminance,
    load_image,
    save_image,
    star_deconvolution,
)


def run(input_path: Path, output_dir: Path, prefix: str, config: DeconvolutionConfig,
        overwrite: bool = False) -> dict:
    image, _ = load_image(input_path)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Input must be an already-stretched RGB image")
    source = image[..., :3]
    restored, processing_report, diagnostics = star_deconvolution(source, config)

    before_lum = _to_luminance(source)
    after_lum = _to_luminance(restored)
    before_detail = before_lum - cv2.GaussianBlur(before_lum, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101)
    after_detail = after_lum - cv2.GaussianBlur(after_lum, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101)
    dark = before_lum <= np.percentile(before_lum, 35.0)
    noise_before = _robust_noise(before_detail[dark])
    noise_after = _robust_noise(after_detail[dark])
    report = {
        "pipeline": "already-stretched RGB image -> automatic PSF deconvolution",
        "input": str(input_path.resolve()),
        "quality_control": {
            "dark_region_detail_sigma_before": noise_before,
            "dark_region_detail_sigma_after": noise_after,
            "dark_region_detail_change_percent": 100.0 * (noise_after / noise_before - 1.0),
            "mean_absolute_change": float(np.mean(np.abs(restored - source))),
        },
        "deconvolution": processing_report,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(output_dir / f"{prefix}_restored.tif", restored, overwrite=overwrite)
    save_image(output_dir / f"{prefix}_restored.png", restored, overwrite=overwrite)
    for name in ("psf", "target_psf", "star_mask"):
        save_image(output_dir / f"{prefix}_{name}.tif", diagnostics[name], overwrite=overwrite)
    iio.imwrite(
        output_dir / f"{prefix}_comparison_hd.png",
        render_pair(source, restored, ("STRETCHED JPEG INPUT", "AUTO PSF DECONVOLUTION"), 86),
    )
    height, width = source.shape[:2]
    size = min(1000, height, width)
    x0, y0 = (width - size) // 2, (height - size) // 2
    crop = render_pair(
        source[y0:y0 + size, x0:x0 + size],
        restored[y0:y0 + size, x0:x0 + size],
        ("INPUT / CENTER CROP", "RESTORED / CENTER CROP"),
        64,
    )
    iio.imwrite(output_dir / f"{prefix}_comparison_crop.png", crop)
    (output_dir / f"{prefix}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--target-fwhm", type=float, required=True)
    parser.add_argument("--stellar-strength", type=float, default=0.55)
    parser.add_argument("--nonstellar-strength", type=float, default=0.18)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--correction-limit", type=float, default=1.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = DeconvolutionConfig(
        solver="quality",
        iterations=args.iterations,
        target_fwhm_px=args.target_fwhm,
        stellar_strength=args.stellar_strength,
        nonstellar_strength=args.nonstellar_strength,
        patch_size=25,
        max_stars=300,
        detection_sigma=5.0,
        damping=args.damping,
        correction_limit=args.correction_limit,
    )
    print(json.dumps(run(args.input, args.output_dir, args.prefix, config, args.overwrite),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
