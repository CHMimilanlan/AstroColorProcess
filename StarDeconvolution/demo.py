#!/usr/bin/env python3
"""Build a reproducible degraded M51 case and benchmark the restoration."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from star_deconvolution import (
    DeconvolutionConfig,
    convolve,
    gaussian_psf,
    load_image,
    save_image,
    star_deconvolution,
)


def _render_panels(panels: list[np.ndarray], labels: list[str], label_height: int) -> np.ndarray:
    """Add labels to native-resolution RGB panels and concatenate them."""
    rendered = []
    font_scale = max(0.62, panels[0].shape[1] / 700.0)
    thickness = max(1, int(round(font_scale * 1.4)))
    for panel, label in zip(panels, labels):
        height, width = panel.shape[:2]
        bgr = cv2.cvtColor(np.round(np.clip(panel, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (width, label_height), (0, 0, 0), -1)
        baseline = int(label_height * 0.72)
        cv2.putText(bgr, label, (16, baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        rendered.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.concatenate(rendered, axis=1).astype(np.float32) / 255.0


def _preview(reference: np.ndarray, degraded: np.ndarray, automatic: np.ndarray, calibrated: np.ndarray) -> np.ndarray:
    # A detail-rich crop makes stellar shape changes visible at normal zoom.
    h, w = reference.shape[:2]
    crop_size = min(420, h, w)
    x0, y0 = max(0, (w - crop_size) // 2), max(0, (h - crop_size) // 2)
    panels = [item[y0 : y0 + crop_size, x0 : x0 + crop_size] for item in (reference, degraded, automatic, calibrated)]
    return _render_panels(panels, ["REFERENCE", "SYNTHETIC BLUR", "AUTO PSF", "KNOWN PSF"], 34)


def _preview_hd(reference: np.ndarray, degraded: np.ndarray, automatic: np.ndarray, calibrated: np.ndarray) -> np.ndarray:
    """Native-resolution 4000x1000 comparison for lossless zoom inspection."""
    return _render_panels(
        [reference, degraded, automatic, calibrated],
        ["REFERENCE", "SYNTHETIC BLUR", "AUTO PSF", "KNOWN PSF"],
        64,
    )


def run_demo(sample: Path, output_dir: Path, overwrite: bool) -> dict:
    reference, metadata = load_image(sample)
    if reference.ndim != 3 or reference.shape[2] < 3:
        raise ValueError("The demo sample must be RGB")
    reference = reference[..., :3]
    degradation_psf = gaussian_psf(25, fwhm=4.2, axis_ratio=0.62, angle_deg=28.0)
    degraded = convolve(reference, degradation_psf)
    rng = np.random.default_rng(20260830)
    degraded = np.clip(degraded + rng.normal(0.0, 0.0018, degraded.shape).astype(np.float32), 0.0, 1.0)

    automatic_config = DeconvolutionConfig(
        solver="quality",
        iterations=50,
        target_fwhm_px=1.25,
        stellar_strength=0.70,
        nonstellar_strength=0.70,
        patch_size=25,
        max_stars=250,
        detection_sigma=4.5,
        damping=0.0,
        correction_limit=3.0,
    )
    calibrated_config = DeconvolutionConfig(
        solver="quality",
        iterations=80,
        target_fwhm_px=0.0,
        stellar_strength=1.0,
        nonstellar_strength=1.0,
        patch_size=25,
        max_stars=250,
        detection_sigma=4.5,
        damping=0.50,
        correction_limit=3.0,
        noise_sigma=0.0018,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    start = time.perf_counter()
    automatic, automatic_report, diagnostics = star_deconvolution(degraded, automatic_config)
    automatic_wall_time = time.perf_counter() - start
    calibrated_start = time.perf_counter()
    calibrated, calibrated_report, _ = star_deconvolution(degraded, calibrated_config, degradation_psf)
    calibrated_wall_time = time.perf_counter() - calibrated_start
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    data_range = 1.0
    degraded_psnr = float(peak_signal_noise_ratio(reference, degraded, data_range=data_range))
    automatic_psnr = float(peak_signal_noise_ratio(reference, automatic, data_range=data_range))
    calibrated_psnr = float(peak_signal_noise_ratio(reference, calibrated, data_range=data_range))
    degraded_ssim = float(structural_similarity(reference, degraded, data_range=data_range, channel_axis=-1))
    automatic_ssim = float(structural_similarity(reference, automatic, data_range=data_range, channel_axis=-1))
    calibrated_ssim = float(structural_similarity(reference, calibrated, data_range=data_range, channel_axis=-1))
    report = {
        "reference": str(sample),
        "synthetic_degradation": "elliptical Gaussian FWHM=4.2 px, axis_ratio=0.62, angle=28 deg, Gaussian noise sigma=0.0018",
        "degraded_psnr_db": degraded_psnr,
        "degraded_ssim": degraded_ssim,
        "automatic_psf": {
            "psnr_db": automatic_psnr,
            "psnr_gain_db": automatic_psnr - degraded_psnr,
            "ssim": automatic_ssim,
            "ssim_gain": automatic_ssim - degraded_ssim,
            "measured_wall_seconds": automatic_wall_time,
            "processing_report": automatic_report,
        },
        "known_psf_solver_ceiling": {
            "psnr_db": calibrated_psnr,
            "psnr_gain_db": calibrated_psnr - degraded_psnr,
            "ssim": calibrated_ssim,
            "ssim_gain": calibrated_ssim - degraded_ssim,
            "measured_wall_seconds": calibrated_wall_time,
            "processing_report": calibrated_report,
            "note": "Uses the exact synthetic degradation kernel; this is a transparent solver validation, not an auto-PSF claim.",
        },
        "peak_traced_allocation_mb": traced_peak / (1024.0 * 1024.0),
    }

    save_image(output_dir / "01_reference.tif", reference, metadata, overwrite)
    save_image(output_dir / "02_degraded_input.tif", degraded, metadata, overwrite)
    save_image(output_dir / "03_restored_auto_psf.tif", automatic, metadata, overwrite)
    save_image(output_dir / "04_restored_known_psf.tif", calibrated, metadata, overwrite)
    save_image(output_dir / "05_star_mask.tif", diagnostics["star_mask"], overwrite=overwrite)
    save_image(output_dir / "06_estimated_psf.tif", diagnostics["psf"], overwrite=overwrite)
    save_image(output_dir / "07_target_psf.tif", diagnostics["target_psf"], overwrite=overwrite)
    save_image(output_dir / "comparison.png", _preview(reference, degraded, automatic, calibrated), overwrite=overwrite)
    save_image(output_dir / "comparison_hd.png", _preview_hd(reference, degraded, automatic, calibrated), overwrite=overwrite)
    report_path = output_dir / "metrics.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=here / "samples" / "m45_dss2_cds.jpg")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run_demo(args.sample, args.output_dir, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
