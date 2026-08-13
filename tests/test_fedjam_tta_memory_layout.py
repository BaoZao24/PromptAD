import unittest
from types import SimpleNamespace

import torch

from tools.eval_fedjam_fewshot_dual import vit_query_scores


class FedJamTTAMemoryLayoutTests(unittest.TestCase):
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
