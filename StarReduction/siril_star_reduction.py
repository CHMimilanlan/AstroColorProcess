#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Siril-compatible star reduction implemented with NumPy.

The algorithm reproduces the PixelMath expression used by Siril's
DSA-Star_Reduction.py workflow, while exposing a callback interface for an
external/custom star-removal algorithm.

Expected input:
    - stretched / non-linear image
    - floating-point values normalized to [0, 1]
    - original and starless arrays with exactly the same shape

Core Siril PixelMath expression (v = reduction value):
    ~((~mtf(~v, original) / ~mtf(~v, starless)) * ~starless)

In Siril PixelMath, ~x means 1-x. The default implementation below uses an
algebraically equivalent but numerically more stable closed form.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatImage: TypeAlias = NDArray[np.floating]


class StarRemover(Protocol):
    """Callable interface expected from a custom star-removal algorithm."""

    def __call__(self, image: FloatImage, **kwargs: Any) -> FloatImage:
        """Return a starless image with the same shape and [0, 1] range."""


@dataclass
class ImageMetadata:
    """Minimal metadata used to preserve the input file representation."""

    kind: str
    original_dtype: np.dtype
    fits_header: Any = None


def _as_float_image(image: np.ndarray, name: str) -> FloatImage:
    """Convert an image to float64 and check for invalid values."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        count = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise ValueError(f"{name} contains {count} NaN/Inf values.")
    return array


def _check_unit_range(
    image: FloatImage,
    name: str,
    *,
    tolerance: float = 1e-6,
    clip: bool = False,
) -> FloatImage:
    """Validate the [0, 1] range required by Siril PixelMath."""
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if min_value < -tolerance or max_value > 1.0 + tolerance:
        message = (
            f"{name} must be normalized to [0, 1], but its range is "
            f"[{min_value:.8g}, {max_value:.8g}]."
        )
        if not clip:
            raise ValueError(message + " Normalize the image first or set clip_input=True.")
        warnings.warn(message + " Values will be clipped.", RuntimeWarning, stacklevel=2)
    return np.clip(image, 0.0, 1.0) if clip else image


def midtones_transfer_function(m: float, x: np.ndarray) -> FloatImage:
    r"""Apply Siril's/PixInsight-style Midtones Transfer Function.

    .. math::
        MTF(m, x) = ((m - 1)x) / (((2m - 1)x) - m)

    ``m=0.5`` is the identity transform. Inputs are expected in [0, 1].
    """
    if not 0.0 < m < 1.0:
        raise ValueError(f"m must be strictly between 0 and 1, got {m}.")

    x_array = np.asarray(x, dtype=np.float64)
    denominator = (2.0 * m - 1.0) * x_array - m

    # For 0 < m < 1 and x in [0, 1], the denominator does not cross zero.
    # The explicit endpoints avoid tiny floating-point deviations.
    result = ((m - 1.0) * x_array) / denominator
    result = np.where(x_array <= 0.0, 0.0, result)
    result = np.where(x_array >= 1.0, 1.0, result)
    return result


def _siril_literal_formula(
    original: FloatImage,
    starless: FloatImage,
    value: float,
    *,
    epsilon: float,
) -> FloatImage:
    """Literal translation of the Siril PixelMath expression.

    This is retained mainly for validation/debugging. The stable closed form in
    ``siril_star_reduction`` is preferred because the literal expression may
    encounter 0/0 near saturated pixels.
    """
    m = 1.0 - value
    numerator = 1.0 - midtones_transfer_function(m, original)
    denominator = 1.0 - midtones_transfer_function(m, starless)

    ratio = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=np.abs(denominator) > epsilon,
    )
    return 1.0 - ratio * (1.0 - starless)


def siril_star_reduction(
    original: np.ndarray,
    *,
    value: float = 0.2,
    starless: Optional[np.ndarray] = None,
    star_remover: Optional[StarRemover] = None,
    star_remover_kwargs: Optional[Mapping[str, Any]] = None,
    clip_input: bool = False,
    clip_output: bool = True,
    warn_on_starless_overshoot: bool = True,
    implementation: str = "stable",
    epsilon: float = 1e-12,
    output_dtype: np.dtype | type = np.float32,
) -> FloatImage:
    """Reduce stars using the Siril DSA PixelMath method.

    Exactly one of ``starless`` and ``star_remover`` must normally be supplied.
    ``star_remover`` is the extension interface for the caller's own algorithm.

    Parameters
    ----------
    original:
        Stretched/non-linear image normalized to [0, 1]. Any channel layout is
        accepted because the operation is elementwise.
    value:
        Siril reduction value. ``0.5`` is identity; values below ``0.5`` reduce
        stars; values above ``0.5`` enlarge them. The useful range is (0, 1).
    starless:
        Precomputed starless image with the same shape as ``original``.
    star_remover:
        Callable that receives ``original`` and returns the starless image.
    star_remover_kwargs:
        Optional keyword arguments forwarded to ``star_remover``.
    clip_input:
        Clip input values to [0, 1]. Disabled by default because silent clipping
        can hide a normalization error.
    clip_output:
        Clip the result to [0, 1]. Recommended for image export.
    warn_on_starless_overshoot:
        Warn if the starless result is brighter than the original in many pixels.
        Such overshoot can be valid locally, but a large amount often indicates
        a mismatch or a star-removal artifact.
    implementation:
        ``"stable"`` uses an algebraically simplified formula. ``"literal"``
        follows Siril PixelMath step by step and is intended for comparison.
    epsilon:
        Safe-division threshold used by the literal implementation.
    output_dtype:
        NumPy dtype of the returned image.

    Returns
    -------
    np.ndarray
        Star-reduced image with the same shape as ``original``.
    """
    if not 0.0 < value < 1.0:
        raise ValueError(f"value must be strictly between 0 and 1, got {value}.")
    if starless is not None and star_remover is not None:
        raise ValueError("Provide either starless or star_remover, not both.")
    if starless is None and star_remover is None:
        raise ValueError("A starless image or star_remover callback is required.")

    original_float = _check_unit_range(
        _as_float_image(original, "original"),
        "original",
        clip=clip_input,
    )

    if starless is None:
        kwargs = dict(star_remover_kwargs or {})
        generated = star_remover(original_float.copy(), **kwargs)  # type: ignore[misc]
        starless_float = _as_float_image(generated, "star_remover output")
    else:
        starless_float = _as_float_image(starless, "starless")

    if starless_float.shape != original_float.shape:
        raise ValueError(
            "original and starless must have exactly the same shape: "
            f"{original_float.shape} != {starless_float.shape}."
        )

    starless_float = _check_unit_range(
        starless_float,
        "starless",
        clip=clip_input,
    )

    if warn_on_starless_overshoot:
        overshoot = starless_float > original_float + 1e-5
        fraction = float(np.mean(overshoot))
        if fraction > 1e-3:
            warnings.warn(
                f"starless is brighter than original in {fraction:.3%} of pixels. "
                "The Siril formula will still run, but inspect the star-removal output.",
                RuntimeWarning,
                stacklevel=2,
            )

    implementation = implementation.lower()
    if implementation == "literal":
        reduced = _siril_literal_formula(
            original_float,
            starless_float,
            value,
            epsilon=epsilon,
        )
    elif implementation == "stable":
        # Siril PixelMath:
        #   ~((~mtf(~v, O) / ~mtf(~v, S)) * ~S)
        # where ~x = 1-x, O=original and S=starless.
        #
        # Let m=1-v. Using the analytical MTF definition, this simplifies to:
        #   R = 1 - (1-O) * [m-(2m-1)S] / [m-(2m-1)O]
        #
        # This form is mathematically equivalent but avoids 0/0 cancellation at
        # saturated pixels where O or S is close to 1.
        m = 1.0 - value
        coefficient = 2.0 * m - 1.0
        numerator = m - coefficient * starless_float
        denominator = m - coefficient * original_float

        # In the valid domain, denominator is positive. Keep an explicit guard
        # for malformed/out-of-range data and extreme floating-point cases.
        safe_denominator = np.where(
            np.abs(denominator) > epsilon,
            denominator,
            np.copysign(epsilon, denominator + (denominator == 0.0)),
        )
        reduced = 1.0 - (1.0 - original_float) * numerator / safe_denominator
    else:
        raise ValueError("implementation must be 'stable' or 'literal'.")

    if clip_output:
        reduced = np.clip(reduced, 0.0, 1.0)

    return reduced.astype(output_dtype, copy=False)


def load_image(path: str | Path) -> tuple[FloatImage, ImageMetadata]:
    """Load FITS, NPY, TIFF, PNG or JPEG as a normalized floating image."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix in {".fit", ".fits", ".fts"}:
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError("FITS support requires: pip install astropy") from exc

        with fits.open(input_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data)
            header = hdul[0].header.copy()
        if data.size == 0:
            raise ValueError(f"FITS file contains no image data: {input_path}")
        original_dtype = data.dtype
        float_data = data.astype(np.float64)
        # Integer FITS is normalized using its dtype range. Floating FITS must
        # already follow Siril's [0, 1] convention.
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            if info.min < 0:
                raise ValueError("Signed integer FITS input is not auto-normalized safely.")
            float_data /= float(info.max)
        return float_data, ImageMetadata("fits", original_dtype, header)

    if suffix == ".npy":
        data = np.load(input_path, allow_pickle=False)
        original_dtype = data.dtype
        float_data = data.astype(np.float64)
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            if info.min < 0:
                raise ValueError("Signed integer NPY input is not auto-normalized safely.")
            float_data /= float(info.max)
        return float_data, ImageMetadata("npy", original_dtype)

    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("Raster image support requires: pip install imageio pillow tifffile") from exc

        data = np.asarray(iio.imread(input_path))
        original_dtype = data.dtype
        float_data = data.astype(np.float64)
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            if info.min < 0:
                raise ValueError("Signed integer raster input is not supported.")
            float_data /= float(info.max)
        return float_data, ImageMetadata("raster", original_dtype)

    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def save_image(
    path: str | Path,
    image: np.ndarray,
    *,
    metadata: Optional[ImageMetadata] = None,
    overwrite: bool = False,
) -> None:
    """Save an image. FITS/NPY keep float32; raster files use 8/16-bit output."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    suffix = output_path.suffix.lower()
    data = np.asarray(image, dtype=np.float32)

    if suffix in {".fit", ".fits", ".fts"}:
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError("FITS support requires: pip install astropy") from exc

        header = metadata.fits_header.copy() if metadata and metadata.fits_header is not None else None
        if header is not None:
            header["HISTORY"] = "Star reduction: Siril DSA PixelMath-compatible NumPy implementation"
        fits.PrimaryHDU(data=data, header=header).writeto(output_path, overwrite=overwrite)
        return

    if suffix == ".npy":
        np.save(output_path, data, allow_pickle=False)
        return

    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("Raster image support requires: pip install imageio pillow tifffile") from exc

        clipped = np.clip(data, 0.0, 1.0)
        if suffix in {".tif", ".tiff"}:
            encoded = np.round(clipped * 65535.0).astype(np.uint16)
        else:
            # Pillow/imageio support for 16-bit RGB PNG is backend-dependent;
            # use portable uint8 for PNG/JPEG and prefer TIFF/FITS for precision.
            encoded = np.round(clipped * 255.0).astype(np.uint8)
        iio.imwrite(output_path, encoded)
        return

    raise ValueError(f"Unsupported output format: {output_path.suffix}")


def load_star_remover(spec: str) -> StarRemover:
    """Load a callback from ``module:function`` or ``/path/file.py:function``."""
    if ":" not in spec:
        raise ValueError("Star remover must use 'module:function' or '/path/file.py:function'.")
    module_name, function_name = spec.rsplit(":", 1)

    module_path = Path(module_name)
    if module_path.suffix == ".py" or module_path.exists():
        import importlib.util

        resolved = module_path.expanduser().resolve()
        module_spec = importlib.util.spec_from_file_location(resolved.stem, resolved)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot import Python file: {resolved}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    callback = getattr(module, function_name, None)
    if callback is None or not callable(callback):
        raise AttributeError(f"Callable '{function_name}' not found in '{module_name}'.")
    return callback


def _parse_json_mapping(text: Optional[str]) -> dict[str, Any]:
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("--star-remover-kwargs must decode to a JSON object.")
    return parsed


def _self_test() -> None:
    rng = np.random.default_rng(20260723)
    starless = rng.uniform(0.0, 0.65, size=(64, 48, 3))
    star_signal = rng.uniform(0.0, 1.0 - starless)
    original = starless + star_signal

    identity = siril_star_reduction(original, starless=starless, value=0.5)
    np.testing.assert_allclose(identity, original, rtol=0.0, atol=2e-7)

    no_stars = siril_star_reduction(starless, starless=starless, value=0.2)
    np.testing.assert_allclose(no_stars, starless, rtol=0.0, atol=2e-7)

    stable = siril_star_reduction(
        original,
        starless=starless,
        value=0.2,
        implementation="stable",
        output_dtype=np.float64,
    )
    literal = siril_star_reduction(
        original,
        starless=starless,
        value=0.2,
        implementation="literal",
        output_dtype=np.float64,
    )
    np.testing.assert_allclose(stable, literal, rtol=1e-10, atol=1e-10)

    if not np.all((stable >= 0.0) & (stable <= 1.0)):
        raise AssertionError("Output is outside [0, 1].")
    if not np.all(stable <= original + 1e-10):
        raise AssertionError("value < 0.5 should not brighten valid star pixels.")
    if not np.all(stable >= starless - 1e-10):
        raise AssertionError("value < 0.5 should stay above the starless baseline.")

    print("Self-test passed.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reduce stars with the Siril DSA PixelMath-compatible algorithm."
    )
    parser.add_argument("--input", type=Path, help="Original stretched image.")
    parser.add_argument("--output", type=Path, help="Output star-reduced image.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--starless", type=Path, help="Precomputed starless image.")
    source.add_argument(
        "--star-remover",
        type=str,
        help="Custom callback: module:function or /path/file.py:function.",
    )
    parser.add_argument(
        "--star-remover-kwargs",
        type=str,
        default=None,
        help='JSON object forwarded to the callback, e.g. \'{"tile_size": 512}\'.',
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.2,
        help="Reduction value: <0.5 reduce, 0.5 identity, >0.5 enlarge (default: 0.2).",
    )
    parser.add_argument(
        "--clip-input",
        action="store_true",
        help="Clip out-of-range input values instead of failing.",
    )
    parser.add_argument(
        "--implementation",
        choices=["stable", "literal"],
        default="stable",
        help="Use the stable equivalent formula or literal PixelMath translation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file.")
    parser.add_argument("--self-test", action="store_true", help="Run numerical tests and exit.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used.")
    if args.starless is None and args.star_remover is None:
        parser.error("Provide --starless or --star-remover.")

    original, metadata = load_image(args.input)

    if args.starless is not None:
        starless, _ = load_image(args.starless)
        result = siril_star_reduction(
            original,
            starless=starless,
            value=args.value,
            clip_input=args.clip_input,
            implementation=args.implementation,
        )
    else:
        callback = load_star_remover(args.star_remover)
        callback_kwargs = _parse_json_mapping(args.star_remover_kwargs)
        result = siril_star_reduction(
            original,
            star_remover=callback,
            star_remover_kwargs=callback_kwargs,
            value=args.value,
            clip_input=args.clip_input,
            implementation=args.implementation,
        )

    save_image(args.output, result, metadata=metadata, overwrite=args.overwrite)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
