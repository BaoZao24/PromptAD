import unittest
from types import SimpleNamespace

import torch

from tools.eval_fedjam_fewshot_dual import vit_query_scores, vit_tta_modes


class FedJamTTAMemoryLayoutTests(unittest.TestCase):
    def test_spectral_tta_bundle_selection(self):
        self.assertEqual(
            vit_tta_modes(SimpleNamespace(vit_tta="none")),
            ("identity",),
        )
        self.assertEqual(
            vit_tta_modes(SimpleNamespace(vit_tta="stft_shift_blur")),
            ("identity", "blur", "time_shift_up", "time_shift_down"),
        )
        self.assertEqual(
            vit_tta_modes(SimpleNamespace(vit_tta="rf_spectral_response_v1")),
            (
                "identity",
                "rf_time_shift_up",
                "rf_time_shift_down",
                "rf_frequency_response_jitter",
            ),
        )

    def test_merged_layout_queries_identity_features_once(self):
        args = SimpleNamespace(
            vit_tta_memory_layout="merged",
            distance_chunk_size=16,
            nn_topk=1,
        )
        identity = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        scores = vit_query_scores({"identity": identity}, gallery, args)

        self.assertEqual(scores.shape, (1,))
        self.assertAlmostEqual(float(scores[0]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
