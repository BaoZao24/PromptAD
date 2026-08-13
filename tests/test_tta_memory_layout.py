import unittest
from types import SimpleNamespace

import torch

from tools.eval_cls_vit_patchcore_gallery import finalize_vit_nn_gallery


class TTAMemoryLayoutTests(unittest.TestCase):
    def test_merged_gallery_keeps_all_augmented_normal_features_without_coreset(self):
        args = SimpleNamespace(
            coreset_ratio=1.0,
            rowwise_coreset=False,
            coreset_method="random",
            seed=7,
        )
        views = [torch.randn(6, 4) for _ in range(4)]
        gallery = torch.cat(views, dim=0)
        rows = torch.arange(gallery.shape[0]) % 3
        cols = torch.arange(gallery.shape[0]) % 2

        merged, merged_rows, merged_cols = finalize_vit_nn_gallery(gallery, rows, cols, args)

        self.assertEqual(merged.shape, (24, 4))
        self.assertTrue(torch.allclose(merged.norm(dim=-1), torch.ones(24), atol=1e-6))
        self.assertTrue(torch.equal(merged_rows, rows))
        self.assertTrue(torch.equal(merged_cols, cols))


if __name__ == "__main__":
    unittest.main()
