from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from denoise import denoise_array, rgb_to_ycbcr, ycbcr_to_rgb  # noqa: E402
from image_io import load_image, save_image  # noqa: E402
from model import CosmicClarityDenoiseCNN, model_statistics  # noqa: E402


class CoreTests(unittest.TestCase):
    def test_model_shape_and_parameter_count(self) -> None:
        model = CosmicClarityDenoiseCNN()
        output = model(torch.rand(1, 3, 16, 16))
        self.assertEqual(tuple(output.shape), (1, 3, 16, 16))
        self.assertEqual(model_statistics(model)["parameters"], 2_945_283)

    def test_ycbcr_roundtrip(self) -> None:
        image = np.random.default_rng(3).random((20, 21, 3), dtype=np.float32)
        reconstructed = ycbcr_to_rgb(*rgb_to_ycbcr(image))
        np.testing.assert_allclose(image, reconstructed, atol=3e-5)

    def test_zero_strength_is_identity(self) -> None:
        image = np.random.default_rng(4).random((9, 10, 3), dtype=np.float32)
        result = denoise_array(image, torch.nn.Identity(), torch.device("cpu"), strength=0.0)
        np.testing.assert_array_equal(image, result)

    def test_fits_roundtrip_preserves_channel_layout(self) -> None:
        from astropy.io import fits

        original = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.fits"
            output = Path(directory) / "output.fits"
            fits.PrimaryHDU(original).writeto(source)
            record = load_image(source)
            save_image(output, record, record.pixels)
            with fits.open(output) as hdul:
                restored = hdul[0].data
        self.assertEqual(restored.shape, original.shape)
        np.testing.assert_array_equal(restored, original)


if __name__ == "__main__":
    unittest.main()
