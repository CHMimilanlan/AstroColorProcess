"""Image I/O with FITS metadata and dynamic-range preservation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from astropy.io import fits
from PIL import Image


FITS_EXTENSIONS = {".fit", ".fits", ".fts"}
TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class ImageRecord:
    pixels: np.ndarray  # HWC float32, normalized to [0, 1]
    offset: float
    scale: float
    original_dtype: np.dtype
    original_layout: str
    header: Any = None


def _to_hwc(array: np.ndarray) -> tuple[np.ndarray, str]:
    if array.ndim == 2:
        return array[..., None], "hw"
    if array.ndim != 3:
        raise ValueError(f"Only mono/RGB 2-D images are supported, got shape {array.shape}")
    if array.shape[-1] in (1, 3, 4):
        return array[..., :3], "hwc"
    if array.shape[0] in (1, 3, 4):
        return np.moveaxis(array[:3], 0, -1), "chw"
    raise ValueError(f"Cannot determine channel axis for shape {array.shape}")


def _normalize(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Image contains no finite pixels")
    clean = np.nan_to_num(array, copy=True)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        offset = float(max(0, info.min))
        scale = float(info.max - offset)
    else:
        minimum = float(np.min(clean[finite]))
        maximum = float(np.max(clean[finite]))
        if minimum >= 0.0 and maximum <= 1.0:
            offset, scale = 0.0, 1.0
        else:
            offset, scale = minimum, maximum - minimum
    if scale <= 0:
        scale = 1.0
    normalized = np.clip((clean.astype(np.float32) - offset) / scale, 0.0, 1.0)
    return normalized, offset, scale


def load_image(path: str | Path) -> ImageRecord:
    path = Path(path)
    extension = path.suffix.lower()
    header = None
    if extension in FITS_EXTENSIONS:
        with fits.open(path, memmap=False) as hdul:
            index = next((i for i, hdu in enumerate(hdul) if hdu.data is not None), None)
            if index is None:
                raise ValueError(f"No image HDU in {path}")
            array = np.asarray(hdul[index].data)
            header = hdul[index].header.copy()
    elif extension in TIFF_EXTENSIONS:
        array = np.asarray(tifffile.imread(path))
    else:
        with Image.open(path) as image:
            if image.mode not in ("L", "I;16", "I", "F"):
                image = image.convert("RGB")
            array = np.asarray(image)

    original_dtype = array.dtype
    hwc, layout = _to_hwc(array)
    pixels, offset, scale = _normalize(hwc)
    return ImageRecord(pixels, offset, scale, original_dtype, layout, header)


def _restore_layout(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "hw":
        return array[..., 0]
    if layout == "chw":
        return np.moveaxis(array, -1, 0)
    return array


def _restore_range(record: ImageRecord, pixels: np.ndarray) -> np.ndarray:
    restored = np.clip(pixels, 0.0, 1.0) * record.scale + record.offset
    if np.issubdtype(record.original_dtype, np.integer):
        info = np.iinfo(record.original_dtype)
        restored = np.clip(np.rint(restored), info.min, info.max).astype(record.original_dtype)
    else:
        restored = restored.astype(record.original_dtype)
    return _restore_layout(restored, record.original_layout)


def save_image(path: str | Path, record: ImageRecord, pixels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    restored = _restore_range(record, pixels)
    if extension in FITS_EXTENSIONS:
        fits.PrimaryHDU(restored, header=record.header).writeto(path, overwrite=True)
    elif extension in TIFF_EXTENSIONS:
        tifffile.imwrite(path, restored)
    else:
        # JPEG/PNG demo output is display-oriented. Preserve mono when applicable.
        display = np.clip(pixels * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if record.original_layout == "hw":
            display = display[..., 0]
        Image.fromarray(display).save(path, quality=95)

