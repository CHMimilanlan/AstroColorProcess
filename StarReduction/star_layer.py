#!/usr/bin/env python3
"""Extract reusable star layers and add them back to starless images.

The layer is stored as an additive RGB image:

    stars = source - starless
    composed = starless + stars

``source`` can be either the original image or the output of the Siril-compatible
star-reduction algorithm.  A black-background TIFF can be used with an
"Addition" / "Linear Dodge (Add)" blend mode.  FITS and NPY are preferable when
negative residuals must be preserved exactly.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from siril_star_reduction import (
    ImageMetadata,
    load_image,
    save_image,
    siril_star_reduction,
)

FloatImage = NDArray[np.floating]
RASTER_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _as_finite_float_image(image: np.ndarray, name: str) -> FloatImage:
    """Return a finite float64 image with at least two dimensions."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        invalid_count = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise ValueError(f"{name} contains {invalid_count} NaN/Inf values.")
    return array


def _validate_pair(
    first: np.ndarray,
    first_name: str,
    second: np.ndarray,
    second_name: str,
) -> tuple[FloatImage, FloatImage]:
    """Convert two images and require identical dimensions/channel layout."""
    first_float = _as_finite_float_image(first, first_name)
    second_float = _as_finite_float_image(second, second_name)
    if first_float.shape != second_float.shape:
        raise ValueError(
            f"{first_name} and {second_name} must have exactly the same shape: "
            f"{first_float.shape} != {second_float.shape}."
        )
    return first_float, second_float


def _check_unit_range(
    image: FloatImage,
    name: str,
    *,
    clip: bool,
    tolerance: float = 1e-6,
) -> FloatImage:
    """Validate the normalized range expected by the existing project."""
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        message = (
            f"{name} must be normalized to [0, 1], but its range is "
            f"[{minimum:.8g}, {maximum:.8g}]."
        )
        if not clip:
            raise ValueError(message + " Normalize it first or enable clip_input.")
        warnings.warn(message + " Values will be clipped.", RuntimeWarning, stacklevel=2)
    return np.clip(image, 0.0, 1.0) if clip else image


def extract_star_layer(
    original: np.ndarray,
    starless: np.ndarray,
    *,
    mode: str = "original",
    value: float = 0.2,
    implementation: str = "stable",
    clip_input: bool = False,
    clip_negative: bool = False,
    output_dtype: np.dtype | type = np.float32,
) -> FloatImage:
    """Extract an additive star layer from an original/starless image pair.

    Parameters
    ----------
    original, starless:
        Registered images with exactly the same shape and values in [0, 1].
    mode:
        ``"original"`` extracts the full-size stars as ``original-starless``.
        ``"reduced"`` first applies the Siril-compatible reduction and extracts
        the already-reduced stars as ``reduced-starless``.
    value:
        Siril reduction value used only by ``mode="reduced"``.
    implementation:
        ``"stable"`` or ``"literal"``; forwarded to the reduction function.
    clip_input:
        Clip input images to [0, 1] instead of rejecting invalid ranges.
    clip_negative:
        Replace negative residuals with zero. This produces a conventional
        black-background layer, but is not exactly reversible where the
        starless image is brighter than the source.
    output_dtype:
        NumPy dtype of the returned layer.
    """
    original_float, starless_float = _validate_pair(
        original, "original", starless, "starless"
    )
    normalized_original = _check_unit_range(
        original_float, "original", clip=clip_input
    )
    normalized_starless = _check_unit_range(
        starless_float, "starless", clip=clip_input
    )

    normalized_mode = mode.lower()
    if normalized_mode == "original":
        source = normalized_original
    elif normalized_mode == "reduced":
        source = siril_star_reduction(
            normalized_original,
            starless=normalized_starless,
            value=value,
            implementation=implementation,
            clip_input=False,
            output_dtype=np.float64,
        )
    else:
        raise ValueError("mode must be 'original' or 'reduced'.")

    layer = source - normalized_starless
    # Algebraically equal background pixels can differ by a few float64 ULPs
    # after the reduction formula. They are zero residuals, not real negative
    # image data, and should not make an otherwise valid TIFF export fail.
    numerical_negative = (layer < 0.0) & (layer >= -1e-12)
    layer = np.where(numerical_negative, 0.0, layer)
    negative_mask = layer < 0.0
    if np.any(negative_mask):
        negative_fraction = float(np.mean(negative_mask))
        minimum = float(np.min(layer))
        action = "Negative values will be clipped to zero." if clip_negative else (
            "Keep them in FITS/NPY for an exact round trip, or enable "
            "clip_negative for a black-background raster layer."
        )
        warnings.warn(
            f"The star layer contains {negative_fraction:.3%} negative channel "
            f"values (minimum {minimum:.8g}). {action}",
            RuntimeWarning,
            stacklevel=2,
        )
        if clip_negative:
            layer = np.maximum(layer, 0.0)

    return layer.astype(output_dtype, copy=False)


def compose_star_layer(
    starless: np.ndarray,
    stars: np.ndarray,
    *,
    clip_output: bool = True,
    output_dtype: np.dtype | type = np.float32,
) -> FloatImage:
    """Add a star layer to a registered starless image."""
    starless_float, stars_float = _validate_pair(
        starless, "starless", stars, "stars"
    )
    composed = starless_float + stars_float

    out_of_range = (composed < 0.0) | (composed > 1.0)
    if clip_output and np.any(out_of_range):
        clipped_fraction = float(np.mean(out_of_range))
        warnings.warn(
            f"Composed image has {clipped_fraction:.3%} channel values outside "
            "[0, 1]; they will be clipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        composed = np.clip(composed, 0.0, 1.0)

    return composed.astype(output_dtype, copy=False)


def _save_star_layer(
    output: Path,
    layer: np.ndarray,
    *,
    metadata: ImageMetadata,
    overwrite: bool,
) -> None:
    """Protect signed residuals from silent clipping by raster encoders."""
    suffix = output.suffix.lower()
    if suffix in RASTER_SUFFIXES and float(np.min(layer)) < 0.0:
        raise ValueError(
            f"{output.suffix} cannot preserve this layer's negative residuals. "
            "Use a .fits/.npy output or rerun extract with --clip-negative."
        )
    if suffix in {".png", ".jpg", ".jpeg"}:
        warnings.warn(
            "PNG/JPEG output is only 8-bit in this project; 16-bit TIFF or "
            "32-bit FITS/NPY is recommended for a star layer.",
            RuntimeWarning,
            stacklevel=2,
        )
    save_image(output, layer, metadata=metadata, overwrite=overwrite)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract additive star layers or add them to starless images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract a full-size or reduced additive star layer."
    )
    extract_parser.add_argument(
        "--input", type=Path, required=True, help="Original stretched image."
    )
    extract_parser.add_argument(
        "--starless", type=Path, required=True, help="Matching starless image."
    )
    extract_parser.add_argument(
        "--output", type=Path, required=True, help="Output star-layer image."
    )
    extract_parser.add_argument(
        "--mode",
        choices=["original", "reduced"],
        default="original",
        help="Extract original stars or stars after reduction (default: original).",
    )
    extract_parser.add_argument(
        "--value",
        type=float,
        default=0.2,
        help="Reduction value for --mode reduced (default: 0.2).",
    )
    extract_parser.add_argument(
        "--implementation",
        choices=["stable", "literal"],
        default="stable",
        help="Reduction formula used by --mode reduced (default: stable).",
    )
    extract_parser.add_argument(
        "--clip-input",
        action="store_true",
        help="Clip out-of-range inputs instead of failing.",
    )
    extract_parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="Clip negative residuals for a black-background raster layer.",
    )
    extract_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file."
    )

    compose_parser = subparsers.add_parser(
        "compose", help="Add a saved star layer to a starless image."
    )
    compose_parser.add_argument(
        "--starless", type=Path, required=True, help="Target starless image."
    )
    compose_parser.add_argument(
        "--stars", type=Path, required=True, help="Additive star-layer image."
    )
    compose_parser.add_argument(
        "--output", type=Path, required=True, help="Output composed image."
    )
    compose_parser.add_argument(
        "--no-clip-output",
        action="store_false",
        dest="clip_output",
        help="Keep values outside [0, 1] (effective for FITS/NPY output).",
    )
    compose_parser.set_defaults(clip_output=True)
    compose_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file."
    )
    return parser


def _run_extract(args: argparse.Namespace) -> int:
    original, metadata = load_image(args.input)
    starless, _ = load_image(args.starless)
    layer = extract_star_layer(
        original,
        starless,
        mode=args.mode,
        value=args.value,
        implementation=args.implementation,
        clip_input=args.clip_input,
        clip_negative=args.clip_negative,
    )
    _save_star_layer(
        args.output,
        layer,
        metadata=metadata,
        overwrite=args.overwrite,
    )
    print(
        f"Saved {args.mode} star layer: {args.output} "
        f"(range {float(np.min(layer)):.8g} .. {float(np.max(layer)):.8g})"
    )
    return 0


def _run_compose(args: argparse.Namespace) -> int:
    starless, metadata = load_image(args.starless)
    stars, _ = load_image(args.stars)
    composed = compose_star_layer(
        starless,
        stars,
        clip_output=args.clip_output,
    )
    save_image(
        args.output,
        composed,
        metadata=metadata,
        overwrite=args.overwrite,
    )
    print(
        f"Saved composed image: {args.output} "
        f"(range {float(np.min(composed)):.8g} .. {float(np.max(composed)):.8g})"
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.command == "extract":
        return _run_extract(args)
    if args.command == "compose":
        return _run_compose(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
