import unittest

import torch

from utils.normal_subspace import (
    fit_normal_subspace,
    fit_support_score_calibrator,
    fuse_support_calibrated_scores,
    leave_one_out_cosine_distance,
    leave_one_out_topk_cosine_distance,
)


class NormalSubspaceTests(unittest.TestCase):
    def test_low_rank_normal_features_have_small_residual(self):
        generator = torch.Generator().manual_seed(7)
        basis = torch.randn(2, 8, generator=generator)
        coeff = torch.randn(64, 2, generator=generator)
        normal = coeff @ basis
        model = fit_normal_subspace(
            normal,
            variance_threshold=0.99,
            max_components=4,
            seed=7,
        )
        self.assertEqual(model.feature_dim, 8)
        self.assertLessEqual(model.rank, 4)
        self.assertLess(float(model.score(normal).mean()), 1e-4)

    def test_anomalous_direction_has_larger_residual(self):
        normal = torch.zeros(32, 6)
        normal[:, 0] = torch.arange(32, dtype=torch.float32)
        model = fit_normal_subspace(
            normal,
            variance_threshold=0.99,
            max_components=2,
            seed=3,
        )
        query = normal[:2].clone()
        query[0, 1] = 5.0
        self.assertGreater(float(model.score(query)[0]), float(model.score(query)[1]))

    def test_dimension_mismatch_is_rejected(self):
        model = fit_normal_subspace(torch.randn(8, 4), max_components=2)
        with self.assertRaises(ValueError):
            model.score(torch.randn(2, 3))

    def test_leave_one_out_excludes_self_match(self):
        features = torch.eye(4)
        distances = leave_one_out_cosine_distance(features, chunk_size=2)
        self.assertTrue(torch.allclose(distances, torch.full((4,), 0.5)))

    def test_leave_one_out_topk_keeps_patch_shape(self):
        features = torch.eye(4).reshape(2, 2, 4)
        distances = leave_one_out_topk_cosine_distance(features, topk=2, chunk_size=2)
        self.assertEqual(distances.shape, (2, 2))
        self.assertTrue(torch.allclose(distances, torch.full((2, 2), 0.5)))

    def test_hybrid_agreement_is_support_scaled(self):
        calibrator = (
            fit_support_score_calibrator(torch.tensor([0.1, 0.2, 0.3, 0.4])),
            fit_support_score_calibrator(torch.tensor([1.0, 2.0, 3.0, 4.0])),
        )
        fused = fuse_support_calibrated_scores(
            torch.tensor([0.8]),
            torch.tensor([8.0]),
            calibrator,
        )
        self.assertTrue(torch.isfinite(fused).all())
        self.assertEqual(fused.shape, (1,))


if __name__ == "__main__":
    unittest.main()
