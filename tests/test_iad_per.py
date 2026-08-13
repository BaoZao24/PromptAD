import unittest

import numpy as np
import torch
import torch.nn.functional as F

from utils.iad_per import iad_per_score


def official_upstream_formula(x: torch.Tensor, reconstruction: torch.Tensor) -> np.ndarray:
    """The PER lines in QXSLAB/vae_ism_ano upstream/main:util.py."""
    diff = x - reconstruction
    mask = F.max_pool2d(x, kernel_size=3, stride=1, padding=1) < 0.05
    pool = -F.max_pool2d(-diff.abs(), kernel_size=3, stride=1, padding=1)
    background = np.percentile((pool * mask).flatten(start_dim=1).numpy(), 90, axis=1)
    signal = np.percentile((pool * ~mask).flatten(start_dim=1).numpy(), 99, axis=1)
    return 2 * background + signal


class IADPERTests(unittest.TestCase):
    def test_batched_shape_components_and_backends(self) -> None:
        x_np = np.linspace(0.0, 0.2, 2 * 1 * 5 * 7, dtype=np.float32).reshape(2, 1, 5, 7)
        reconstruction_np = x_np * np.float32(0.8)

        score_np, components_np = iad_per_score(
            x_np, reconstruction_np, return_components=True
        )
        score_torch, components_torch = iad_per_score(
            torch.from_numpy(x_np),
            torch.from_numpy(reconstruction_np),
            return_components=True,
        )

        self.assertEqual(score_np.shape, (2,))
        self.assertEqual(tuple(score_torch.shape), (2,))
        self.assertEqual(components_np["background_mask"].shape, x_np.shape)
        self.assertEqual(tuple(components_torch["signal_mask"].shape), x_np.shape)
        np.testing.assert_allclose(score_np, score_torch.numpy(), rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(
            components_np["background_mask"], components_torch["background_mask"].numpy()
        )

    def test_hand_computed_masked_percentiles(self) -> None:
        x = np.array([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=np.float32)
        error = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        reconstruction = x - error

        score, components = iad_per_score(
            x,
            reconstruction,
            alpha=0.5,
            gamma=1,
            xi=50,
            eta=50,
            return_components=True,
        )

        # Official masking retains zeros outside the selected region:
        # background [1, 0, 3, 0] -> median 0.5;
        # signal     [0, 2, 0, 4] -> median 1.0.
        np.testing.assert_allclose(components["background"], [0.5])
        np.testing.assert_allclose(components["signal"], [1.0])
        np.testing.assert_allclose(score, [2.0])

    def test_no_background_and_no_signal_have_zero_missing_term(self) -> None:
        signal_only = np.ones((1, 1, 2, 2), dtype=np.float32)
        background_only = np.zeros((1, 1, 2, 2), dtype=np.float32)

        signal_score, signal_parts = iad_per_score(
            signal_only,
            np.zeros_like(signal_only),
            gamma=1,
            return_components=True,
        )
        background_score, background_parts = iad_per_score(
            background_only,
            np.ones_like(background_only),
            gamma=1,
            return_components=True,
        )

        np.testing.assert_allclose(signal_parts["background"], [0.0])
        np.testing.assert_allclose(signal_parts["signal"], [1.0])
        np.testing.assert_allclose(signal_score, [1.0])
        np.testing.assert_allclose(background_parts["background"], [1.0])
        np.testing.assert_allclose(background_parts["signal"], [0.0])
        np.testing.assert_allclose(background_score, [2.0])

    def test_min_pool_boundary_ignores_outside_image(self) -> None:
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        reconstruction = -np.array([[[[4.0, 3.0], [2.0, 1.0]]]], dtype=np.float32)

        score, components = iad_per_score(x, reconstruction, gamma=3, return_components=True)

        # Every clipped 3x3 neighbourhood contains the full 2x2 image. If the
        # min-pool boundary were zero-padded, this would incorrectly be zero.
        np.testing.assert_allclose(components["pooled_error"], np.ones_like(x))
        np.testing.assert_allclose(score, [2.0])

    def test_cpu_torch_output_is_finite(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(7)
        x = torch.rand((4, 2, 8, 9), generator=generator, device="cpu")
        reconstruction = torch.rand((4, 2, 8, 9), generator=generator, device="cpu")

        score = iad_per_score(x, reconstruction)

        self.assertEqual(score.device.type, "cpu")
        self.assertTrue(torch.isfinite(score).all().item())

    def test_regression_matches_official_upstream_formula(self) -> None:
        x = torch.tensor(
            [
                [[[0.00, 0.01, 0.02, 0.20],
                  [0.01, 0.02, 0.03, 0.10],
                  [0.00, 0.01, 0.08, 0.09],
                  [0.00, 0.00, 0.01, 0.02]]],
                [[[0.10, 0.10, 0.10, 0.10],
                  [0.10, 0.01, 0.01, 0.10],
                  [0.10, 0.01, 0.01, 0.10],
                  [0.10, 0.10, 0.10, 0.10]]],
            ],
            dtype=torch.float32,
        )
        reconstruction = x + torch.tensor(
            [
                [[[0.04, 0.03, 0.02, 0.01],
                  [0.05, 0.08, 0.07, 0.02],
                  [0.06, 0.09, 0.10, 0.03],
                  [0.04, 0.05, 0.06, 0.07]]],
                [[[0.09, 0.08, 0.07, 0.06],
                  [0.05, 0.04, 0.03, 0.02],
                  [0.01, 0.02, 0.03, 0.04],
                  [0.05, 0.06, 0.07, 0.08]]],
            ],
            dtype=torch.float32,
        )

        expected = official_upstream_formula(x, reconstruction)
        actual_torch = iad_per_score(x, reconstruction)
        actual_numpy = iad_per_score(x.numpy(), reconstruction.numpy())

        np.testing.assert_allclose(expected, [0.11, 0.0385], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual_torch.numpy(), expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual_numpy, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
