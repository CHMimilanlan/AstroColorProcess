#!/usr/bin/env python3
"""Run the project's reference FITS stretch before automatic star deconvolution."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_STRETCH_ROOT = Path(r"D:\Workman\MyProject\ImageStack\RASPAstroStacker")

from compare_fits import render_pair
from star_deconvolution import (
    DeconvolutionConfig,
    _robust_noise,
    _to_luminance,
    load_image,
    save_image,
    star_deconvolution,
)


def _load_reference_stretcher(root: Path):
    sys.path.insert(0, str(root))
    from ImageStretch.stretch import ImageDebayerAndStretch
    return ImageDebayerAndStretch


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    prefix: str,
    config: DeconvolutionConfig,
    shadow_co: float = -2.0,
    stretch_root: Path = DEFAULT_STRETCH_ROOT,
    overwrite: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stretch = _load_reference_stretcher(stretch_root)
    start = time.perf_counter()
    _, stretched_bgr, _, rgb_flag = stretch(
        input_path, shadow_co=shadow_co, color_calib=False, do_debug=False
    )
    if stretched_bgr is None or not rgb_flag:
        raise RuntimeError("Reference ImageStretch did not return a valid RGB image")
    stretched = cv2.cvtColor(stretched_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    stretch_seconds = time.perf_counter() - start

    _, metadata = load_image(input_path)
    restored, processing_report, diagnostics = star_deconvolution(stretched, config)

    before_lum = _to_luminance(stretched)
    after_lum = _to_luminance(restored)
    before_detail = before_lum - cv2.GaussianBlur(before_lum, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101)
    after_detail = after_lum - cv2.GaussianBlur(after_lum, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101)
    dark = before_lum <= np.percentile(before_lum, 35.0)
    noise_before = _robust_noise(before_detail[dark])
    noise_after = _robust_noise(after_detail[dark])

    report = {
        "pipeline": "FITS -> RASPAstroStacker/ImageStretch MTF -> automatic PSF deconvolution",
        "input": str(input_path.resolve()),
        "reference_stretch": {
            "module": str((stretch_root / "ImageStretch").resolve()),
            "shadow_co": shadow_co,
            "color_calib": False,
            "native_output_dtype": "uint8",
            "elapsed_seconds": stretch_seconds,
        },
        "quality_control": {
            "dark_region_detail_sigma_before": noise_before,
            "dark_region_detail_sigma_after": noise_after,
            "dark_region_detail_change_percent": 100.0 * (noise_after / noise_before - 1.0),
            "mean_absolute_change": float(np.mean(np.abs(restored - stretched))),
        },
        "deconvolution": processing_report,
    }

    save_image(output_dir / f"{prefix}_stretched_input.tif", stretched, overwrite=overwrite)
    save_image(output_dir / f"{prefix}_stretched_restored.tif", restored, overwrite=overwrite)
    save_image(output_dir / f"{prefix}_stretched_restored.fits", restored, metadata, overwrite=overwrite)
    for name in ("psf", "target_psf", "star_mask"):
        save_image(output_dir / f"{prefix}_{name}.tif", diagnostics[name], overwrite=overwrite)

    hd = render_pair(stretched, restored, ("REFERENCE MTF STRETCH", "STRETCH + AUTO PSF DECONV"), 86)
    iio.imwrite(output_dir / f"{prefix}_comparison_hd.png", hd)
    height, width = stretched.shape[:2]
    size = min(1200, height, width)
    x0, y0 = (width - size) // 2, (height - size) // 2
    crop = render_pair(
        stretched[y0:y0 + size, x0:x0 + size],
        restored[y0:y0 + size, x0:x0 + size],
        ("STRETCHED / CENTER CROP", "DECONVOLVED / CENTER CROP"),
        64,
    )
    iio.imwrite(output_dir / f"{prefix}_comparison_crop.png", crop)
    report_path = output_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--stretch-root", type=Path, default=DEFAULT_STRETCH_ROOT)
    parser.add_argument("--shadow-co", type=float, default=-2.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--target-fwhm", type=float, required=True)
    parser.add_argument("--stellar-strength", type=float, default=0.55)
    parser.add_argument("--nonstellar-strength", type=float, default=0.20)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--correction-limit", type=float, default=1.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = DeconvolutionConfig(
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
    report = run_pipeline(
        args.input, args.output_dir, args.prefix, cfg, args.shadow_co,
        args.stretch_root, args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
