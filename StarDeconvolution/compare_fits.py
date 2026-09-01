#!/usr/bin/env python3
"""Create linked-stretch original/restored previews for linear FITS data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from star_deconvolution import _robust_noise, _to_luminance, load_image


def linked_asinh(original: np.ndarray, restored: np.ndarray, strength: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical per-channel display parameters derived from original."""
    source = np.asarray(original[..., :3], np.float32)
    result = np.asarray(restored[..., :3], np.float32)
    black = np.percentile(source, 0.5, axis=(0, 1)).astype(np.float32)
    white = np.percentile(source, 99.85, axis=(0, 1)).astype(np.float32)
    scale = np.maximum(white - black, 1e-6)

    def stretch(image: np.ndarray) -> np.ndarray:
        normalized = np.maximum((image - black) / scale, 0.0)
        return np.clip(np.arcsinh(strength * normalized) / np.arcsinh(strength), 0.0, 1.0)

    return stretch(source), stretch(result)


def render_pair(left: np.ndarray, right: np.ndarray, labels: tuple[str, str], label_height: int) -> np.ndarray:
    rendered: list[np.ndarray] = []
    for image, label in zip((left, right), labels):
        panel = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], label_height), (0, 0, 0), -1)
        font_scale = max(0.8, min(2.0, bgr.shape[1] / 1500.0))
        thickness = max(1, int(round(font_scale * 1.5)))
        cv2.putText(bgr, label, (20, int(label_height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        rendered.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.concatenate(rendered, axis=1)


def make_comparison(original_path: Path, restored_path: Path, output_prefix: Path) -> dict[str, float | list[int]]:
    original, _ = load_image(original_path)
    restored, _ = load_image(restored_path)
    if original.shape != restored.shape or original.ndim != 3:
        raise ValueError(f"Expected matching RGB data, got {original.shape} and {restored.shape}")
    shown_original, shown_restored = linked_asinh(original, restored)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    hd = render_pair(shown_original, shown_restored, ("ORIGINAL FITS", "RESTORED / AUTO PSF"), 86)
    iio.imwrite(output_prefix.with_name(output_prefix.name + "_comparison_hd.png"), hd)

    height, width = original.shape[:2]
    crop_size = min(1200, height, width)
    x0, y0 = (width - crop_size) // 2, (height - crop_size) // 2
    crop = render_pair(
        shown_original[y0:y0 + crop_size, x0:x0 + crop_size],
        shown_restored[y0:y0 + crop_size, x0:x0 + crop_size],
        ("ORIGINAL / CENTER CROP", "RESTORED / CENTER CROP"),
        64,
    )
    iio.imwrite(output_prefix.with_name(output_prefix.name + "_comparison_crop.png"), crop)

    before_lum = _to_luminance(original)
    after_lum = _to_luminance(restored)
    before_detail = before_lum - cv2.GaussianBlur(before_lum, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101)
    after_detail = after_lum - cv2.GaussianBlur(after_lum, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101)
    dark = before_lum <= np.percentile(before_lum, 35.0)
    before_noise = _robust_noise(before_detail[dark])
    after_noise = _robust_noise(after_detail[dark])
    return {
        "image_shape": list(original.shape),
        "dark_region_detail_sigma_before": before_noise,
        "dark_region_detail_sigma_after": after_noise,
        "dark_region_detail_change_percent": 100.0 * (after_noise / before_noise - 1.0),
        "mean_absolute_linear_change": float(np.mean(np.abs(restored - original))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = make_comparison(args.original, args.restored, args.output_prefix)
    report_path = args.output_prefix.with_name(args.output_prefix.name + "_preview_metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
