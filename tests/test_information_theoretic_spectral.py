import unittest

import numpy as np

from utils.information_theoretic_spectral import ICAFrozen, KLDRef


class InformationTheoreticSpectralTests(unittest.TestCase):
    def test_kld_matches_hand_calculation_with_kt_pseudocount(self):
        reference_image = np.array([[0.0, 0.0], [1.0, 1.0]])
        query_image = np.zeros((2, 2), dtype=np.float64)
        model = KLDRef.fit(
            [reference_image], n_bins=2, value_range=(0.0, 2.0)
        )

        expected_query = np.array([4.5, 0.5]) / 5.0
        expected_reference = np.array([2.5, 2.5]) / 5.0
        expected = np.sum(
            expected_query * np.log2(expected_query / expected_reference)
        )

        self.assertEqual(model.reference.pseudocount, 0.5)
        self.assertAlmostEqual(model.score(query_image), expected)

    def test_kld_empty_reference_bins_do_not_produce_infinity(self):
        model = KLDRef.fit(
            [np.zeros((4, 4))], n_bins=2, value_range=(0.0, 2.0)
        )
        score = model.score(np.full((4, 4), 1.5))

        self.assertTrue(np.all(model.reference.probabilities > 0.0))
        self.assertTrue(np.isfinite(score))

    def test_rgb_and_grayscale_use_the_same_fixed_histogram(self):
        grayscale = np.array([[0.0, 1.0], [1.0, 0.0]])
        rgb = np.repeat(grayscale[..., None], 3, axis=-1)
        model = KLDRef.fit([rgb], n_bins=2, value_range=(0.0, 2.0))

        self.assertAlmostEqual(model.score(rgb), model.score(grayscale))

    def test_query_does_not_change_reference_or_bin_range(self):
        model = KLDRef.fit(
            [np.array([[0.0, 0.25], [0.5, 1.0]])],
            n_bins=4,
            value_range=(0.0, 1.0),
        )
        counts_before = model.reference.counts.copy()
        probabilities_before = model.reference.probabilities.copy()
        edges_before = model.reference.bin_edges.copy()

        first = model.score(np.full((3, 3), 1000.0))
        second = model.score(np.full((3, 3), 1000.0))

        self.assertAlmostEqual(first, second)
        np.testing.assert_array_equal(model.reference.counts, counts_before)
        np.testing.assert_array_equal(
            model.reference.probabilities, probabilities_before
        )
        np.testing.assert_array_equal(model.reference.bin_edges, edges_before)
        self.assertEqual(model.reference.value_range, (0.0, 1.0))
        self.assertFalse(model.reference.counts.flags.writeable)
        self.assertFalse(model.reference.probabilities.flags.writeable)
        self.assertFalse(model.reference.bin_edges.flags.writeable)

    def test_ica_information_content_matches_hand_calculation(self):
        reference = np.array([[0.0, 0.0], [0.0, 1.0]])
        model = ICAFrozen.fit(
            [reference],
            n_bins=2,
            value_range=(0.0, 2.0),
            information_threshold=0.0,
        )
        query = np.array([[0.0, 1.0]])

        expected_probabilities = np.array([3.5, 1.5]) / 5.0
        expected_information = -np.log2(expected_probabilities)

        np.testing.assert_allclose(
            model.reference.probabilities, expected_probabilities
        )
        np.testing.assert_allclose(
            model.information_map(query), expected_information[None, :]
        )

    def test_ica_cluster_follows_the_requested_time_axis(self):
        normal = np.zeros((4, 4), dtype=np.float64)
        model = ICAFrozen.fit(
            [normal],
            n_bins=2,
            value_range=(0.0, 2.0),
            information_threshold=2.0,
        )
        # Three rare events are contiguous only along axis 1.
        query = np.array(
            [
                [1.5, 1.5, 1.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        along_columns = model.score(query, time_axis=1, cluster_length=3)
        along_rows = model.score(query, time_axis=0, cluster_length=3)

        self.assertGreater(along_columns, 0.0)
        self.assertLess(along_rows, 0.0)

    def test_ica_cluster_score_is_continuous_and_reductions_are_configurable(self):
        normal = np.concatenate(
            [np.zeros(31, dtype=np.float64), np.ones(1, dtype=np.float64)]
        ).reshape(4, 8)
        model = ICAFrozen.fit(
            [normal],
            n_bins=2,
            value_range=(0.0, 2.0),
            information_threshold=1.0,
        )
        query = np.array([[1.0, 1.0, 1.0, 0.0]])

        score_map = model.cluster_score_map(query, time_axis=1)
        maximum = model.score(query, time_axis=1, image_reduction="max")
        mean = model.score(query, time_axis=1, image_reduction="mean")
        topk = model.score(
            query,
            time_axis=1,
            image_reduction="topk_mean",
            topk_fraction=0.5,
        )
        quantile = model.score(
            query,
            time_axis=1,
            image_reduction="quantile",
            quantile=0.75,
        )

        self.assertEqual(score_map.shape, (1, 2))
        self.assertAlmostEqual(maximum, float(score_map.max()))
        self.assertAlmostEqual(mean, float(score_map.mean()))
        self.assertAlmostEqual(topk, float(score_map.max()))
        self.assertAlmostEqual(quantile, float(np.quantile(score_map, 0.75)))
        self.assertNotEqual(maximum, float(maximum > 0.0))

    def test_ica_scores_remain_finite_with_unseen_events_and_default_n_equals_two(self):
        model = ICAFrozen.fit(
            [np.zeros((5, 5))], n_bins=20, value_range=(0.0, 20.0)
        )
        query = np.full((3, 6), 19.5)

        score_map = model.cluster_score_map(query, time_axis=1)
        score = model.score(query, time_axis=1)

        self.assertEqual(
            score_map.shape, (3, 4)
        )  # six events, default length N+1 = 3
        self.assertTrue(np.all(np.isfinite(model.information_map(query))))
        self.assertTrue(np.all(np.isfinite(score_map)))
        self.assertTrue(np.isfinite(score))


if __name__ == "__main__":
    unittest.main()
