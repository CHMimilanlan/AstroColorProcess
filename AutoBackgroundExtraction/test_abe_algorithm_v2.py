import unittest

import numpy as np

import abe_algorithm_v2 as abe


class ABEAlgorithmTests(unittest.TestCase):
    def test_fast_median_metrics_match_scipy_reference(self):
        if abe.cv2 is None:
            self.skipTest("OpenCV acceleration is not installed")

        rng = np.random.default_rng(1234)
        image = rng.random((67, 91), dtype=np.float32)
        accelerated = abe._build_sampling_metrics(image)

        saved_cv2 = abe.cv2
        try:
            abe.cv2 = None
            reference = abe._build_sampling_metrics(image)
        finally:
            abe.cv2 = saved_cv2

        for name in ("smooth", "abs_residual", "local_residual"):
            np.testing.assert_array_equal(accelerated[name], reference[name])
        self.assertEqual(accelerated["residual_med"], reference["residual_med"])
        self.assertEqual(accelerated["residual_sigma"], reference["residual_sigma"])

    def test_regular_grid_polynomial_evaluation_is_bit_exact(self):
        shape = (43, 71)
        terms = abe._polynomial_terms(4)
        coeffs = np.linspace(-0.03, 0.04, len(terms), dtype=np.float32)
        x_grid, y_grid = np.meshgrid(
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[0], dtype=np.float32),
        )

        reference = abe.evaluate_polynomial_2d(
            x_grid,
            y_grid,
            coeffs,
            terms=terms,
            image_shape=shape,
        )
        accelerated = abe._evaluate_polynomial_grid(shape, coeffs, terms)
        np.testing.assert_array_equal(accelerated, reference)

    def test_batched_grid_sampling_matches_scalar_statistics(self):
        rng = np.random.default_rng(99)
        image = rng.random((53, 61), dtype=np.float32)
        residual = rng.random(image.shape, dtype=np.float32) * 0.01
        structure_mask = rng.random(image.shape) > 0.92
        x_coords = np.array([5, 17, 29, 41, 55])
        y_coords = np.array([5, 16, 27, 38, 47])
        patch_size = 7
        threshold = 0.009

        accelerated = abe._sample_background_from_grid(
            image,
            structure_mask,
            x_coords,
            y_coords,
            residual,
            threshold,
            patch_size,
            max_structure_ratio=0.2,
            residual_percentile=90.0,
        )

        reference = []
        half_patch = patch_size // 2
        for y in y_coords:
            for x in x_coords:
                y0, y1, x0, x1 = abe._patch_bounds(
                    x, y, half_patch, image.shape[1], image.shape[0]
                )
                residual_score = float(
                    np.percentile(residual[y0:y1, x0:x1], 90.0)
                )
                structure_ratio = float(np.mean(structure_mask[y0:y1, x0:x1]))
                if residual_score <= threshold and structure_ratio <= 0.2:
                    reference.append(
                        {
                            "x": float(x),
                            "y": float(y),
                            "z": float(np.median(image[y0:y1, x0:x1])),
                            "residual_score": residual_score,
                            "struct_ratio": structure_ratio,
                        }
                    )

        self.assertEqual(accelerated, reference)

    def test_color_layouts_share_samples_and_results(self):
        rng = np.random.default_rng(7)
        y, x = np.mgrid[:72, :96].astype(np.float32)
        gradient = 0.05 + 0.0002 * x + 0.0001 * y
        image = np.stack(
            [gradient, gradient * 1.1, gradient * 0.9], axis=2
        ).astype(np.float32)
        image += rng.normal(0.0, 1e-4, image.shape).astype(np.float32)
        mask = np.zeros(image.shape[:2], dtype=bool)

        kwargs = dict(
            structure_mask=mask,
            grid_size=12,
            threshold_factor=2.0,
            degree=2,
            correction_mode="neutralize",
            return_background=True,
            return_samples=True,
        )
        hwc = abe.automatic_background_extraction(image, **kwargs)
        chw = abe.automatic_background_extraction(
            np.transpose(image, (2, 0, 1)), **kwargs
        )

        np.testing.assert_array_equal(hwc[0], np.transpose(chw[0], (1, 2, 0)))
        np.testing.assert_array_equal(hwc[1], np.transpose(chw[1], (1, 2, 0)))
        np.testing.assert_array_equal(hwc[2][0], chw[2][0])
        np.testing.assert_array_equal(hwc[2][1], chw[2][1])

    def test_empty_sample_fallback_does_not_raise(self):
        image = np.ones((32, 40), dtype=np.float32)
        mask = np.ones_like(image, dtype=bool)
        corrected, background, samples = abe.automatic_background_extraction(
            image,
            structure_mask=mask,
            grid_size=8,
            max_structure_ratio=0.0,
            return_background=True,
            return_samples=True,
        )
        np.testing.assert_array_equal(corrected, image)
        np.testing.assert_array_equal(background, np.zeros_like(image))
        self.assertEqual(samples[0].size, 0)
        self.assertEqual(samples[1].size, 0)


if __name__ == "__main__":
    unittest.main()
