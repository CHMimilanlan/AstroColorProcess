"""Command-line inference demo for Cosmic Clarity AI 3.6."""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from image_io import load_image, save_image
from model import load_model, model_statistics


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "deep_denoise_cnn_AI3_6.pth"


def rgb_to_ycbcr(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
        dtype=np.float32,
    )
    converted = image @ matrix.T
    return converted[..., 0], converted[..., 1] + 0.5, converted[..., 2] + 0.5


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    converted = np.stack((y, cb - 0.5, cr - 0.5), axis=-1)
    matrix = np.array(
        [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]],
        dtype=np.float32,
    )
    return np.clip(converted @ matrix.T, 0.0, 1.0)


def _mtf_stretch(image: np.ndarray, target_median: float = 0.25) -> tuple[np.ndarray, float, list[float]]:
    original_min = float(image.min())
    stretched = image - original_min
    medians = [float(np.median(stretched[..., channel])) for channel in range(3)]
    for channel, median in enumerate(medians):
        if median == 0:
            continue
        x = stretched[..., channel]
        denominator = median * (target_median + x - 1.0) - target_median * x
        stretched[..., channel] = ((median - 1.0) * target_median * x) / np.where(
            np.abs(denominator) < 1e-12, 1e-12, denominator
        )
    return np.clip(stretched, 0.0, 1.0), original_min, medians


def _mtf_unstretch(image: np.ndarray, original_min: float, original_medians: list[float]) -> np.ndarray:
    result = image.copy()
    current_medians = [float(np.median(result[..., channel])) for channel in range(3)]
    for channel, current in enumerate(current_medians):
        target = original_medians[channel]
        if current == 0 or target == 0:
            continue
        x = result[..., channel]
        denominator = current * (target + x - 1.0) - target * x
        result[..., channel] = ((current - 1.0) * target * x) / np.where(
            np.abs(denominator) < 1e-12, 1e-12, denominator
        )
    return np.clip(result + original_min, 0.0, 1.0)


def _guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    kernel = (2 * radius + 1, 2 * radius + 1)
    kwargs = {"ddepth": -1, "ksize": kernel, "borderType": cv2.BORDER_REFLECT}
    mean_i = cv2.boxFilter(guide, **kwargs)
    mean_p = cv2.boxFilter(source, **kwargs)
    corr_i = cv2.boxFilter(guide * guide, **kwargs)
    corr_ip = cv2.boxFilter(guide * source, **kwargs)
    variance_i = corr_i - mean_i * mean_i
    covariance_ip = corr_ip - mean_i * mean_p
    a = covariance_ip / (variance_i + epsilon)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, **kwargs) * guide + cv2.boxFilter(b, **kwargs)


def denoise_chroma(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
    effective = float(np.clip(strength * 2.0, 0.0, 1.0))
    if effective == 0:
        return cb, cr
    radius = 2 + int(round(10 * effective))
    epsilon = (0.001 + 0.05 * effective) ** 2
    cb_filtered = _guided_filter(y, cb, radius, epsilon)
    cr_filtered = _guided_filter(y, cr, radius, epsilon)
    return (
        (1.0 - effective) * cb + effective * cb_filtered,
        (1.0 - effective) * cr + effective * cr_filtered,
    )


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


def _batches(items: list[tuple[int, int, np.ndarray]], size: int) -> Iterator[list[tuple[int, int, np.ndarray]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def infer_channel(
    channel: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int,
    overlap: int,
    batch_size: int,
    half_precision: bool,
) -> np.ndarray:
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile-size")
    height, width = channel.shape
    pad_y, pad_x = max(0, tile_size - height), max(0, tile_size - width)
    padded = np.pad(channel, ((0, pad_y), (0, pad_x)), mode="reflect")
    ph, pw = padded.shape
    tiles = [
        (y, x, padded[y : y + tile_size, x : x + tile_size])
        for y in _positions(ph, tile_size, overlap)
        for x in _positions(pw, tile_size, overlap)
    ]
    # A Hann window suppresses tile seams; the floor guarantees edge coverage.
    window_1d = np.hanning(tile_size).astype(np.float32)
    window = np.maximum(np.outer(window_1d, window_1d), 1e-3)
    accumulated = np.zeros_like(padded, dtype=np.float32)
    weights = np.zeros_like(padded, dtype=np.float32)

    use_amp = half_precision and device.type == "cuda"
    for batch in _batches(tiles, batch_size):
        array = np.stack([item[2] for item in batch])
        tensor = torch.from_numpy(array[:, None]).repeat(1, 3, 1, 1).to(device)
        amp = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            prediction = model(tensor)[:, 0].float().cpu().numpy()
        for (y, x, _), tile_prediction in zip(batch, prediction):
            accumulated[y : y + tile_size, x : x + tile_size] += tile_prediction * window
            weights[y : y + tile_size, x : x + tile_size] += window
    return (accumulated / np.maximum(weights, 1e-8))[:height, :width]


def denoise_array(
    pixels: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    strength: float = 0.9,
    color_strength: float | None = None,
    mode: str = "full",
    tile_size: int = 256,
    overlap: int = 64,
    batch_size: int = 1,
    half_precision: bool = True,
    linear_stretch: str = "auto",
) -> np.ndarray:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if strength == 0.0:
        return pixels.copy()
    mono = pixels.shape[-1] == 1
    rgb = np.repeat(pixels, 3, axis=-1) if mono else pixels[..., :3].copy()

    should_stretch = linear_stretch == "always" or (
        linear_stretch == "auto" and float(np.median(rgb - rgb.min())) < 0.05
    )
    stretch_state = None
    if should_stretch:
        rgb, minimum, medians = _mtf_stretch(rgb)
        stretch_state = (minimum, medians)

    def run(channel: np.ndarray) -> np.ndarray:
        return infer_channel(channel, model, device, tile_size, overlap, batch_size, half_precision)

    if mono:
        denoised = ((1.0 - strength) * rgb[..., 0] + strength * run(rgb[..., 0]))[..., None]
        result = np.repeat(denoised, 3, axis=-1)
    elif mode == "separate":
        result = np.stack(
            [(1.0 - strength) * rgb[..., c] + strength * run(rgb[..., c]) for c in range(3)],
            axis=-1,
        )
    elif mode in ("luminance", "full"):
        y, cb, cr = rgb_to_ycbcr(rgb)
        y = (1.0 - strength) * y + strength * run(y)
        if mode == "full":
            cb, cr = denoise_chroma(y, cb, cr, strength if color_strength is None else color_strength)
        result = ycbcr_to_rgb(y, cb, cr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if stretch_state is not None:
        result = _mtf_unstretch(result, *stretch_state)
    return result[..., :1] if mono else result


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astronomy-specific Cosmic Clarity AI 3.6 denoising demo")
    parser.add_argument("input", type=Path, help="Input JPG/PNG/TIFF/FITS")
    parser.add_argument("output", type=Path, help="Output path; extension selects format")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--color-strength", type=float, default=None)
    parser.add_argument("--mode", choices=("luminance", "full", "separate"), default="full")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--linear-stretch", choices=("auto", "always", "never"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    model = load_model(args.weights, device)
    record = load_image(args.input)
    started = time.perf_counter()
    output = denoise_array(
        record.pixels,
        model,
        device,
        strength=args.strength,
        color_strength=args.color_strength,
        mode=args.mode,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        half_precision=args.precision == "fp16",
        linear_stretch=args.linear_stretch,
    )
    save_image(args.output, record, output)
    elapsed = time.perf_counter() - started
    statistics = model_statistics(model)
    print(f"Model: Cosmic Clarity AI 3.6 ({statistics['parameters']:,} parameters)")
    print(f"Device: {device}; input: {record.pixels.shape}; elapsed: {elapsed:.3f}s")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

