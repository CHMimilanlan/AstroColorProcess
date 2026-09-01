#!/usr/bin/env python3
"""Effect-first, star-guided astronomical image deconvolution demo.

The implementation estimates a global empirical PSF from isolated stars,
performs regularized deconvolution to a finite circular target PSF, and can use
an externally calibrated PSF when one is available.  It is a classical,
interpretable implementation inspired by public BXT concepts, not a BXT model
clone or a substitute for BXT's proprietary neural network.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.float32]


@dataclass
class ImageMetadata:
    kind: str
    original_dtype: str
    fits_header: Any = None
    channel_first: bool = False


@dataclass
class DeconvolutionConfig:
    solver: str = "quality"
    iterations: int = 40
    target_ratio: float = 0.40
    target_fwhm_px: float | None = 1.35
    stellar_strength: float = 0.82
    nonstellar_strength: float = 0.65
    patch_size: int = 25
    max_stars: int = 250
    detection_sigma: float = 5.0
    saturation: float = 0.985
    max_eccentricity: float = 0.82
    damping: float = 0.50
    correction_limit: float = 3.0
    wiener_regularization: float = 1e-3
    noise_sigma: float | None = None

    def validate(self) -> None:
        if self.solver not in {"quality", "balanced", "fast"}:
            raise ValueError("solver must be one of: quality, balanced, fast")
        if not 1 <= self.iterations <= 150:
            raise ValueError("iterations must be in [1, 150]")
        if not 0.25 <= self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be in [0.25, 1.0]")
        if not 0.0 <= self.stellar_strength <= 1.0:
            raise ValueError("stellar_strength must be in [0, 1]")
        if not 0.0 <= self.nonstellar_strength <= 1.0:
            raise ValueError("nonstellar_strength must be in [0, 1]")
        if self.target_fwhm_px is not None and not 0.0 <= self.target_fwhm_px <= 20.0:
            raise ValueError("target_fwhm_px must be in [0, 20], or None")
        if self.patch_size < 11:
            raise ValueError("patch_size must be at least 11")
        if self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        if self.max_stars < 5:
            raise ValueError("max_stars must be at least 5")
        if self.detection_sigma <= 0 or self.damping < 0:
            raise ValueError("detection_sigma must be positive and damping nonnegative")
        if self.correction_limit <= 1.0:
            raise ValueError("correction_limit must be greater than 1")
        if self.wiener_regularization <= 0:
            raise ValueError("wiener_regularization must be positive")
        if self.noise_sigma is not None and self.noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive, or None for automatic estimation")


@dataclass
class StarMeasurement:
    x: float
    y: float
    peak: float
    snr: float
    fwhm: float
    eccentricity: float


def _to_luminance(image: np.ndarray) -> FloatImage:
    if image.ndim == 2:
        return np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")
    if image.shape[2] == 1:
        return np.asarray(image[..., 0], dtype=np.float32)
    rgb = np.asarray(image[..., :3], dtype=np.float32)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def _robust_noise(detail: np.ndarray) -> float:
    values = np.asarray(detail, dtype=np.float32).ravel()
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, 1e-6)


def _measure_patch(patch: np.ndarray, noise: float) -> tuple[float, float, float, float, float, FloatImage] | None:
    size = patch.shape[0]
    edge = np.concatenate((patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]))
    background = float(np.median(edge))
    signal = np.maximum(np.asarray(patch, dtype=np.float32) - background, 0.0)
    peak = float(signal.max())
    # Positive-only background noise has a large second moment and would make
    # the measured PSF much broader than the visible stellar core.
    weights = np.maximum(signal - 1.5 * noise, 0.0)
    flux = float(weights.sum())
    if flux <= 1e-8 or peak <= 0:
        return None
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cx = float((weights * xx).sum() / flux)
    cy = float((weights * yy).sum() / flux)
    dx, dy = xx - cx, yy - cy
    cxx = float((weights * dx * dx).sum() / flux)
    cyy = float((weights * dy * dy).sum() / flux)
    cxy = float((weights * dx * dy).sum() / flux)
    eigenvalues = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64))
    minor_var, major_var = np.maximum(eigenvalues, 1e-6)
    fwhm = 2.35482 * float((minor_var * major_var) ** 0.25)
    eccentricity = math.sqrt(max(0.0, 1.0 - minor_var / major_var))
    snr = peak / noise
    return cx, cy, fwhm, eccentricity, snr, weights


def detect_stars(
    luminance: np.ndarray,
    config: DeconvolutionConfig,
) -> tuple[list[StarMeasurement], list[FloatImage], float, FloatImage]:
    """Detect isolated, unsaturated point sources and return normalized patches."""
    lum = np.asarray(luminance, dtype=np.float32)
    background = cv2.GaussianBlur(lum, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101)
    detail = lum - background
    noise = _robust_noise(detail)
    threshold = config.detection_sigma * noise
    local_max = cv2.dilate(detail, np.ones((5, 5), np.uint8))
    candidates = np.argwhere((detail >= local_max - 1e-7) & (detail > threshold) & (lum < config.saturation))
    if candidates.size == 0:
        return [], [], noise, detail

    scores = detail[candidates[:, 0], candidates[:, 1]]
    candidates = candidates[np.argsort(scores)[::-1]]
    radius = config.patch_size // 2
    height, width = lum.shape
    measurements: list[StarMeasurement] = []
    patches: list[FloatImage] = []
    occupied: list[tuple[int, int]] = []

    for y, x in candidates:
        y, x = int(y), int(x)
        if x < radius or y < radius or x >= width - radius or y >= height - radius:
            continue
        # Prevent a bright star's diffraction/halo peaks from being counted repeatedly.
        if any((x - ox) ** 2 + (y - oy) ** 2 < (radius * 0.75) ** 2 for oy, ox in occupied):
            continue
        patch = lum[y - radius : y + radius + 1, x - radius : x + radius + 1]
        if float(patch.max()) >= config.saturation:
            continue
        measured = _measure_patch(patch, noise)
        if measured is None:
            continue
        cx, cy, fwhm, eccentricity, snr, signal = measured
        # Keep moderately broad stars; the later robust width filter removes
        # extended objects while allowing oversampled/blurred survey data.
        if not (1.1 <= fwhm <= radius * 0.92):
            continue
        if eccentricity > config.max_eccentricity or snr < config.detection_sigma:
            continue
        if abs(cx - radius) > radius * 0.30 or abs(cy - radius) > radius * 0.30:
            continue

        # Subpixel register each normalized star before taking a robust median.
        transform = np.float32([[1.0, 0.0, radius - cx], [0.0, 1.0, radius - cy]])
        aligned = cv2.warpAffine(signal, transform, (config.patch_size, config.patch_size), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned = np.maximum(aligned, 0.0)
        total = float(aligned.sum())
        if total <= 1e-8:
            continue
        aligned /= total
        measurements.append(StarMeasurement(x + cx - radius, y + cy - radius, float(patch.max()), snr, fwhm, eccentricity))
        patches.append(aligned.astype(np.float32))
        occupied.append((y, x))
        if len(measurements) >= config.max_stars:
            break

    if len(measurements) >= 8:
        widths = np.array([m.fwhm for m in measurements])
        median = float(np.median(widths))
        mad = max(float(np.median(np.abs(widths - median))), 0.15)
        keep = np.abs(widths - median) <= 3.0 * mad
        measurements = [m for m, flag in zip(measurements, keep) if flag]
        patches = [p for p, flag in zip(patches, keep) if flag]
    return measurements, patches, noise, detail


def estimate_empirical_psf(patches: list[np.ndarray]) -> FloatImage:
    if len(patches) < 5:
        raise RuntimeError(
            f"Only {len(patches)} valid stars found; need at least 5. "
            "Try a lower --detection-sigma, a larger --patch-size, or a star-richer image."
        )
    stack = np.stack(patches, axis=0).astype(np.float32)
    psf = np.median(stack, axis=0)
    psf = np.maximum(psf, 0.0)
    psf /= max(float(psf.sum()), 1e-12)
    return psf.astype(np.float32)


def gaussian_psf(size: int, fwhm: float, axis_ratio: float = 1.0, angle_deg: float = 0.0) -> FloatImage:
    """Create a normalized elliptical Gaussian PSF (axis_ratio = minor/major)."""
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    theta = math.radians(angle_deg)
    xr = math.cos(theta) * xx + math.sin(theta) * yy
    yr = -math.sin(theta) * xx + math.cos(theta) * yy
    sigma_major = max(fwhm / 2.35482, 0.45)
    sigma_minor = max(sigma_major * axis_ratio, 0.45)
    kernel = np.exp(-0.5 * ((xr / sigma_major) ** 2 + (yr / sigma_minor) ** 2))
    kernel /= float(kernel.sum())
    return kernel.astype(np.float32)


def convolve(image: np.ndarray, kernel: np.ndarray) -> FloatImage:
    return cv2.filter2D(np.asarray(image, dtype=np.float32), -1, np.asarray(kernel, dtype=np.float32),
                        borderType=cv2.BORDER_REFLECT101)


def damped_richardson_lucy(
    observed: np.ndarray,
    psf: np.ndarray,
    iterations: int,
    noise: float,
    damping: float,
    correction_limit: float,
) -> FloatImage:
    """Positivity-preserving RL with residual damping and bounded updates."""
    obs = np.maximum(np.asarray(observed, dtype=np.float32), 0.0)
    estimate = np.maximum(obs.copy(), 1e-7)
    flipped = np.flip(np.asarray(psf, dtype=np.float32), axis=(0, 1)).copy()
    epsilon = max(float(obs.max()) * 1e-7, 1e-8)
    damping_scale = max(damping * noise, 0.0)
    lower, upper = 1.0 / correction_limit, correction_limit

    for _ in range(iterations):
        blurred = np.maximum(convolve(estimate, psf), epsilon)
        residual = obs - blurred
        if damping_scale > 0:
            # Small residuals are likely noise; large residuals approach normal RL.
            residual *= np.abs(residual) / (np.abs(residual) + damping_scale)
        ratio = np.clip(1.0 + residual / blurred, lower, upper)
        estimate *= np.clip(convolve(ratio, flipped), lower, upper)
        estimate = np.clip(estimate, 0.0, 1.5)
    return estimate.astype(np.float32)


def _psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert a centered spatial PSF to an optical transfer function."""
    kernel = np.asarray(psf, dtype=np.float32)
    if kernel.shape[0] > shape[0] or kernel.shape[1] > shape[1]:
        raise ValueError("PSF cannot be larger than the requested OTF shape")
    padded = np.zeros(shape, dtype=np.float32)
    padded[: kernel.shape[0], : kernel.shape[1]] = kernel
    padded = np.roll(padded, -(kernel.shape[0] // 2), axis=0)
    padded = np.roll(padded, -(kernel.shape[1] // 2), axis=1)
    return np.fft.rfft2(padded)


def wiener_deconvolution_to_target(
    observed: np.ndarray,
    source_psf: np.ndarray,
    target_psf: np.ndarray,
    regularization: float = 3e-4,
) -> FloatImage:
    """Map source PSF to target PSF with regularized frequency inversion.

    Unlike unsharp masking this filter explicitly applies H_target/H_source.
    Reflection padding suppresses FFT wraparound at image boundaries.
    """
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    image = np.asarray(observed, dtype=np.float32)
    radius = max(source_psf.shape + target_psf.shape) // 2
    if image.ndim == 2:
        padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    elif image.ndim == 3:
        padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    else:
        raise ValueError(f"Expected a 2-D or 3-D image, got {image.shape}")

    shape = padded.shape[:2]
    source_otf = _psf_to_otf(source_psf, shape)
    target_otf = _psf_to_otf(target_psf, shape)
    transfer = np.conj(source_otf) * target_otf / (
        np.abs(source_otf) ** 2 + float(regularization)
    )
    spectrum = np.fft.rfft2(padded, axes=(0, 1))
    if padded.ndim == 3:
        spectrum *= transfer[..., None]
    else:
        spectrum *= transfer
    restored = np.fft.irfft2(spectrum, s=shape, axes=(0, 1)).real
    restored = restored[radius:-radius, radius:-radius]
    return np.asarray(restored, dtype=np.float32)


def _make_masks(
    luminance: np.ndarray,
    stars: list[StarMeasurement],
    noise: float,
    source_fwhm: float,
) -> tuple[FloatImage, FloatImage]:
    height, width = luminance.shape
    star_mask = np.zeros((height, width), dtype=np.float32)
    core_radius = max(2, int(math.ceil(2.2 * source_fwhm)))
    for star in stars:
        cv2.circle(star_mask, (int(round(star.x)), int(round(star.y))), core_radius, 1.0, -1, lineType=cv2.LINE_AA)
    star_mask = cv2.GaussianBlur(star_mask, (0, 0), max(1.0, source_fwhm * 0.65))
    star_mask = np.clip(star_mask, 0.0, 1.0)

    smooth = cv2.GaussianBlur(np.asarray(luminance, np.float32), (0, 0), 1.0)
    structure = cv2.GaussianBlur(np.abs(cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)), (0, 0), 1.0)
    low = max(2.5 * noise, 1e-6)
    high = max(float(np.percentile(structure, 99.0)), low * 2.0)
    structure_mask = np.clip((structure - low) / (high - low), 0.0, 1.0)
    structure_mask = cv2.GaussianBlur(structure_mask, (0, 0), 0.8)
    structure_mask *= 1.0 - star_mask
    return star_mask.astype(np.float32), structure_mask.astype(np.float32)


def _measure_at_stars(luminance: np.ndarray, stars: list[StarMeasurement], patch_size: int, noise: float) -> dict[str, float]:
    radius = patch_size // 2
    values: list[tuple[float, float, float]] = []
    for star in stars:
        x, y = int(round(star.x)), int(round(star.y))
        if x < radius or y < radius or x >= luminance.shape[1] - radius or y >= luminance.shape[0] - radius:
            continue
        patch = luminance[y - radius : y + radius + 1, x - radius : x + radius + 1]
        measured = _measure_patch(patch, noise)
        if measured is not None:
            edge = np.concatenate((patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]))
            signal = np.maximum(np.asarray(patch, np.float32) - float(np.median(edge)), 0.0)
            # Measure the connected half-maximum stellar core at 4x sampling.
            # This is closer to literal FWHM than the detection moment, which
            # deliberately includes faint wings and can grow after restoration.
            upsample = 4
            fine = cv2.resize(signal, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC)
            fine = np.maximum(fine, 0.0)
            peak_y, peak_x = np.unravel_index(int(np.argmax(fine)), fine.shape)
            half_max = (fine >= 0.5 * float(fine[peak_y, peak_x])).astype(np.uint8)
            _, labels = cv2.connectedComponents(half_max, connectivity=8)
            label = int(labels[peak_y, peak_x])
            area = float(np.count_nonzero(labels == label)) / (upsample * upsample)
            core_fwhm = 2.0 * math.sqrt(max(area, 1e-6) / math.pi)
            values.append((core_fwhm, measured[2], measured[3]))
    if not values:
        return {
            "median_fwhm": float("nan"),
            "median_moment_width": float("nan"),
            "median_eccentricity": float("nan"),
        }
    return {
        "median_fwhm": float(np.median([v[0] for v in values])),
        "median_moment_width": float(np.median([v[1] for v in values])),
        "median_eccentricity": float(np.median([v[2] for v in values])),
    }


def star_deconvolution(
    image: np.ndarray,
    config: DeconvolutionConfig | None = None,
    source_psf: np.ndarray | None = None,
) -> tuple[FloatImage, dict[str, Any], dict[str, FloatImage]]:
    """Restore an HxW or HxWxC image normalized to [0, 1].

    ``source_psf=None`` selects an empirical PSF from image stars.  Passing a
    normalized PSF is useful for calibration frames or controlled benchmarks.
    In the latter case ``target_fwhm_px=0`` means recovering the latent image
    before that supplied incremental blur; auto PSF mode should normally use a
    finite target FWHM to avoid an ill-conditioned zero-width target.
    """
    cfg = config or DeconvolutionConfig()
    cfg.validate()
    source = np.asarray(image, dtype=np.float32)
    if source.ndim not in (2, 3) or source.size == 0:
        raise ValueError(f"Unsupported image shape: {source.shape}")
    if source.ndim == 3 and source.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got {source.shape[2]}")
    if not np.all(np.isfinite(source)):
        raise ValueError("Input contains NaN or Inf")
    if float(source.min()) < -1e-6 or float(source.max()) > 1.0 + 1e-6:
        raise ValueError(f"Input must be normalized to [0, 1], got [{source.min()}, {source.max()}]")
    source = np.clip(source, 0.0, 1.0)

    start = time.perf_counter()
    luminance = _to_luminance(source)
    stars, patches, noise, _ = detect_stars(luminance, cfg)
    psf_origin = "empirical stars"
    if source_psf is None:
        psf = estimate_empirical_psf(patches)
    else:
        psf = np.asarray(source_psf, dtype=np.float32)
        if (psf.ndim != 2 or psf.shape[0] != psf.shape[1] or psf.shape[0] % 2 == 0
                or min(psf.shape) < 3 or not np.all(np.isfinite(psf))):
            raise ValueError("source_psf must be a finite odd-sized square 2-D kernel")
        if np.any(psf < 0) or float(psf.sum()) <= 0:
            raise ValueError("source_psf must be nonnegative with positive sum")
        psf = psf / float(psf.sum())
        psf_origin = "supplied/calibrated"
    psf_measurement = _measure_patch(psf, 1e-8)
    if psf_measurement is None:
        raise RuntimeError("Estimated PSF has no measurable core")
    source_fwhm = float(psf_measurement[2])
    sampled_star_fwhm = float(np.median([s.fwhm for s in stars])) if stars else float("nan")
    target_fwhm = (
        float(cfg.target_fwhm_px)
        if cfg.target_fwhm_px is not None
        else max(1.25, source_fwhm * cfg.target_ratio)
    )
    target_psf = (
        gaussian_psf(cfg.patch_size, target_fwhm)
        if target_fwhm > 0
        else np.pad(
            np.ones((1, 1), np.float32),
            ((cfg.patch_size // 2, cfg.patch_size // 2),) * 2,
        )
    )

    process = source[..., :3] if source.ndim == 3 else source
    solve_signal = luminance if cfg.solver == "balanced" and process.ndim == 3 else process
    if cfg.solver == "fast":
        restored_signal = wiener_deconvolution_to_target(
            solve_signal, psf, target_psf, cfg.wiener_regularization
        )
    else:
        channels = ([solve_signal] if solve_signal.ndim == 2
                    else [solve_signal[..., c] for c in range(solve_signal.shape[2])])
        restored_channels = []
        for channel in channels:
            channel_noise = (
                float(cfg.noise_sigma) if cfg.noise_sigma is not None
                else _robust_noise(channel - cv2.GaussianBlur(
                    channel, (0, 0), 12.0, borderType=cv2.BORDER_REFLECT101
                ))
            )
            latent = damped_richardson_lucy(
                channel, psf, cfg.iterations, channel_noise, cfg.damping,
                cfg.correction_limit,
            )
            restored_channels.append(convolve(latent, target_psf) if target_fwhm > 0 else latent)
        restored_signal = (restored_channels[0] if solve_signal.ndim == 2
                           else np.stack(restored_channels, axis=-1))

    if cfg.solver == "balanced" and process.ndim == 3:
        ratio = np.divide(restored_signal, np.maximum(luminance, 2e-4),
                          out=np.ones_like(luminance))
        restored_signal = process * np.clip(ratio, 0.35, 2.5)[..., None]

    restored_luminance = _to_luminance(restored_signal)
    star_mask, structure_mask = _make_masks(luminance, stars, noise, source_fwhm)
    # Nonstellar strength is a real whole-field reconstruction control.  The
    # previous implementation multiplied it by a sparse edge mask, which made
    # the visible output remain almost identical to the blurred input.
    blend_mask = np.clip(
        cfg.nonstellar_strength * (1.0 - star_mask) + cfg.stellar_strength * star_mask,
        0.0, 1.0,
    )
    if process.ndim == 2:
        output = process + blend_mask * (restored_signal - process)
    else:
        output_rgb = process + blend_mask[..., None] * (restored_signal - process)
        if source.shape[2] == 4:
            output = source.copy()
            output[..., :3] = output_rgb
        else:
            output = output_rgb
    output = np.clip(output, 0.0, 1.0).astype(np.float32)

    after_luminance = _to_luminance(output)
    before_metrics = _measure_at_stars(luminance, stars, cfg.patch_size, noise)
    after_metrics = _measure_at_stars(after_luminance, stars, cfg.patch_size, noise)
    report: dict[str, Any] = {
        "algorithm": (
            "regularized PSF-to-target Wiener" if cfg.solver == "fast"
            else "RGB damped Richardson-Lucy to finite target PSF"
        ),
        "psf_origin": psf_origin,
        "image_shape": list(source.shape),
        "valid_stars": len(stars),
        "estimated_noise_sigma": noise,
        "solver_noise_sigma": cfg.noise_sigma if cfg.noise_sigma is not None else "per-channel automatic",
        "source_fwhm_px": source_fwhm,
        "sampled_star_median_fwhm_px": sampled_star_fwhm,
        "target_fwhm_px": target_fwhm,
        "before": before_metrics,
        "after": after_metrics,
        "fwhm_change_percent": 100.0 * (after_metrics["median_fwhm"] / before_metrics["median_fwhm"] - 1.0),
        "elapsed_seconds": time.perf_counter() - start,
        "config": asdict(cfg),
        "notes": [
            "Effect-first quality mode solves RGB channels independently; balanced/fast reduce cost.",
            "Global PSF demo; spatially varying aberrations require tiled local PSFs or a learned model.",
            "Use linear 32-bit data for scientific processing; stretched samples are visual demos only.",
        ],
    }
    diagnostics = {
        "psf": psf,
        "target_psf": target_psf,
        "star_mask": star_mask,
        "structure_mask": structure_mask,
        "blend_mask": blend_mask,
        "restored_luminance": restored_luminance,
    }
    return output, report, diagnostics


def load_image(path: str | Path) -> tuple[FloatImage, ImageMetadata]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".fit", ".fits", ".fts"}:
        from astropy.io import fits

        with fits.open(input_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data)
            header = hdul[0].header.copy()
        channel_first = data.ndim == 3 and data.shape[0] in (1, 3, 4)
        if channel_first:
            data = np.moveaxis(data, 0, -1)
        metadata = ImageMetadata("fits", str(data.dtype), header, channel_first)
    elif suffix == ".npy":
        data = np.load(input_path, allow_pickle=False)
        metadata = ImageMetadata("npy", str(data.dtype))
    elif suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        import imageio.v3 as iio

        data = np.asarray(iio.imread(input_path))
        metadata = ImageMetadata("raster", str(data.dtype))
    else:
        raise ValueError(f"Unsupported input format: {suffix}")
    if data.size == 0:
        raise ValueError(f"Empty image: {input_path}")
    if np.issubdtype(data.dtype, np.integer):
        if np.iinfo(data.dtype).min < 0:
            raise ValueError("Signed integer image input is not supported")
        image = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        image = data.astype(np.float32)
    return image, metadata


def save_image(
    path: str | Path,
    image: np.ndarray,
    metadata: ImageMetadata | None = None,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    data = np.asarray(image, dtype=np.float32)
    if suffix in {".fit", ".fits", ".fts"}:
        from astropy.io import fits

        if metadata and metadata.channel_first and data.ndim == 3:
            data = np.moveaxis(data, -1, 0)
        header = metadata.fits_header.copy() if metadata and metadata.fits_header is not None else None
        if header is not None:
            header.add_history("StarDeconvolution: empirical PSF + damped Richardson-Lucy")
        fits.PrimaryHDU(data=data, header=header).writeto(output_path, overwrite=overwrite)
    elif suffix == ".npy":
        np.save(output_path, data, allow_pickle=False)
    elif suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        import imageio.v3 as iio

        clipped = np.clip(data, 0.0, 1.0)
        if suffix in {".tif", ".tiff"}:
            encoded = np.round(clipped * 65535.0).astype(np.uint16)
        else:
            encoded = np.round(clipped * 255.0).astype(np.uint8)
        iio.imwrite(output_path, encoded)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver", choices=("quality", "balanced", "fast"), default="quality")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--target-ratio", type=float, default=0.40)
    parser.add_argument("--target-fwhm", type=float, default=1.35,
                        help="finite target PSF FWHM; use 0 only with a calibrated incremental PSF")
    parser.add_argument("--stellar-strength", type=float, default=0.82)
    parser.add_argument("--nonstellar-strength", type=float, default=0.65)
    parser.add_argument("--patch-size", type=int, default=25)
    parser.add_argument("--max-stars", type=int, default=250)
    parser.add_argument("--detection-sigma", type=float, default=5.0)
    parser.add_argument("--saturation", type=float, default=0.985)
    parser.add_argument("--damping", type=float, default=0.50)
    parser.add_argument("--correction-limit", type=float, default=3.0)
    parser.add_argument("--wiener-regularization", type=float, default=1e-3)
    parser.add_argument("--noise-sigma", type=float,
                        help="optional calibrated normalized noise sigma")
    parser.add_argument("--psf-fwhm", type=float,
                        help="optional calibrated source PSF FWHM in pixels")
    parser.add_argument("--psf-axis-ratio", type=float, default=1.0)
    parser.add_argument("--psf-angle", type=float, default=0.0)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    image, metadata = load_image(args.input)
    config = DeconvolutionConfig(
        solver=args.solver,
        iterations=args.iterations,
        target_ratio=args.target_ratio,
        target_fwhm_px=args.target_fwhm,
        stellar_strength=args.stellar_strength,
        nonstellar_strength=args.nonstellar_strength,
        patch_size=args.patch_size,
        max_stars=args.max_stars,
        detection_sigma=args.detection_sigma,
        saturation=args.saturation,
        damping=args.damping,
        correction_limit=args.correction_limit,
        wiener_regularization=args.wiener_regularization,
        noise_sigma=args.noise_sigma,
    )
    supplied_psf = None
    if args.psf_fwhm is not None:
        supplied_psf = gaussian_psf(
            args.patch_size, args.psf_fwhm, args.psf_axis_ratio, args.psf_angle
        )
    output, report, diagnostics = star_deconvolution(image, config, supplied_psf)
    save_image(args.output, output, metadata, args.overwrite)
    if args.diagnostics_dir:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        for name, diagnostic in diagnostics.items():
            save_image(args.diagnostics_dir / f"{name}.tif", diagnostic, overwrite=args.overwrite)
    report_path = args.report or args.output.with_suffix(".json")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
