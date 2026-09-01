"""Small deterministic checks; runnable without pytest."""

from __future__ import annotations

import numpy as np

from star_deconvolution import (
    DeconvolutionConfig, convolve, gaussian_psf, star_deconvolution,
    wiener_deconvolution_to_target,
)


def synthetic_field(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(1234)
    image = np.full((size, size), 0.015, dtype=np.float32)
    for _ in range(55):
        x, y = rng.integers(15, size - 15, size=2)
        image[y, x] += rng.uniform(0.15, 0.75)
    image = convolve(image, gaussian_psf(21, 4.0, 0.7, 25.0))
    image += rng.normal(0.0, 0.0008, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def run_tests() -> None:
    kernel = gaussian_psf(21, 3.5, 0.7, 20.0)
    assert kernel.shape == (21, 21)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-6)
    image = synthetic_field()
    output, report, diagnostics = star_deconvolution(
        image,
        DeconvolutionConfig(
            solver="quality", iterations=8, target_fwhm_px=1.5, patch_size=21, max_stars=80,
            detection_sigma=4.0, stellar_strength=1.0, nonstellar_strength=0.0,
        ),
    )
    assert output.shape == image.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all() and 0 <= output.min() <= output.max() <= 1
    assert report["valid_stars"] >= 5
    assert diagnostics["psf"].shape == (21, 21)
    assert report["after"]["median_fwhm"] < report["before"]["median_fwhm"]
    fast = wiener_deconvolution_to_target(image, kernel, gaussian_psf(21, 1.8), 1e-3)
    assert fast.shape == image.shape and np.isfinite(fast).all()

    rgb = np.repeat(image[..., None], 3, axis=-1)
    calibrated, calibrated_report, _ = star_deconvolution(
        rgb,
        DeconvolutionConfig(
            solver="quality", iterations=6, target_fwhm_px=0.0,
            patch_size=21, max_stars=80, detection_sigma=4.0,
            stellar_strength=1.0, nonstellar_strength=1.0,
        ),
        kernel,
    )
    assert calibrated.shape == rgb.shape
    assert calibrated_report["psf_origin"] == "supplied/calibrated"
    print("All StarDeconvolution tests passed.")


if __name__ == "__main__":
    run_tests()
