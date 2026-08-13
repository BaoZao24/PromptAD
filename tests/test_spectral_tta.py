import unittest

import numpy as np

from utils.spectral_tta import SPECTRAL_TTA_BUNDLES, augment_spectrogram


class SpectralTTATests(unittest.TestCase):
    def test_response_bundle_keeps_time_alignment_views(self):
        self.assertEqual(
            SPECTRAL_TTA_BUNDLES["rf_time_alignment_v1"],
            ("identity", "rf_time_shift_up", "rf_time_shift_down"),
        )
        self.assertEqual(
            SPECTRAL_TTA_BUNDLES["rf_frequency_response_only_v1"],
            ("identity", "rf_frequency_response_jitter"),
        )
        self.assertEqual(
            SPECTRAL_TTA_BUNDLES["rf_spectral_response_v1"],
            (
                "identity",
                "rf_time_shift_up",
                "rf_time_shift_down",
                "rf_frequency_response_jitter",
            ),
        )
        self.assertEqual(
            SPECTRAL_TTA_BUNDLES["ofdma_spectral_response_v1"],
            (
                "identity",
                "ofdma_time_shift_left",
                "ofdma_time_shift_right",
                "ofdma_frequency_response_jitter",
            ),
        )

    def test_rf_response_is_constant_over_time_for_each_frequency_bin(self):
        image = np.full((12, 18), 128, dtype=np.uint8)
        augmented = augment_spectrogram(
            image,
            "rf_frequency_response_jitter",
            frequency_response_strength=3.0,
        )

        delta = augmented.astype(np.int16) - image.astype(np.int16)
        self.assertEqual(augmented.dtype, image.dtype)
        self.assertEqual(augmented.shape, image.shape)
        np.testing.assert_array_equal(delta, np.repeat(delta[:1], 12, axis=0))
        self.assertGreater(np.std(delta), 0.0)
        self.assertAlmostEqual(float(np.mean(delta)), 0.0, delta=0.6)

    def test_ofdma_response_is_constant_over_time_for_each_frequency_bin(self):
        image = np.full((18, 12), 128, dtype=np.uint8)
        augmented = augment_spectrogram(
            image,
            "ofdma_frequency_response_jitter",
            frequency_response_strength=3.0,
        )

        delta = augmented.astype(np.int16) - image.astype(np.int16)
        np.testing.assert_array_equal(delta, np.repeat(delta[:, :1], 12, axis=1))
        self.assertGreater(np.std(delta), 0.0)
        self.assertAlmostEqual(float(np.mean(delta)), 0.0, delta=0.6)


if __name__ == "__main__":
    unittest.main()
