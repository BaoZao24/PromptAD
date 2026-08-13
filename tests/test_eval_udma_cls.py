import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tools.eval_udma_cls import (
    FrozenSpectralStatistics,
    build_jobs,
    fit_udma,
    predict_job,
    safe_metrics,
)


def _args(output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=str(output_root),
        seed=7,
        image_height=32,
        image_width=32,
        batch_size=2,
        num_workers=0,
        reference_extractor="spectral_stats",
        teacher_epochs=1,
        student_epochs=1,
        teacher_lr=1e-3,
        student_lr=1e-3,
        weight_decay=0.0,
        feature_channels=4,
        teacher_hidden_1=4,
        teacher_hidden_2=8,
        encoder_channels_1=4,
        encoder_channels_2=4,
        encoder_channels_3=4,
        latent_channels=4,
        memory_size=3,
        shrink_threshold=None,
        update_threshold=None,
        memory_update_rate=0.1,
        separateness_margin=1.0,
        weight_teacher_ae=0.5,
        weight_teacher_memae=0.5,
        weight_ae_memae=0.5,
        compactness_weight=0.1,
        separateness_weight=0.1,
        log_every=0,
        normal_sampling="1shot",
    )


class EvalUDMATest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _sample(self, name: str, label: int, offset: int) -> dict:
        grid = np.indices((32, 32)).sum(axis=0).astype(np.uint8)
        image = np.clip(grid + offset, 0, 255).astype(np.uint8)
        if label:
            image[:, 12:20] = 255
        path = self.root / f"{name}.png"
        self.assertTrue(cv2.imwrite(str(path), image))
        return {"path": str(path), "label": label, "name": name}

    def test_frozen_spectral_reference_is_deterministic(self):
        extractor = FrozenSpectralStatistics().eval()
        data = torch.linspace(0, 1, 2 * 32 * 32).reshape(2, 1, 32, 32)
        first = extractor(data)
        second = extractor(data)
        self.assertEqual(tuple(first.shape), (2, 16))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(sum(parameter.numel() for parameter in extractor.parameters()), 0)

    def test_fit_predict_freezes_state_and_saves_complete_payload(self):
        train = [
            self._sample("train-0", 0, 4),
            self._sample("train-1", 0, 8),
        ]
        test = [
            self._sample("normal-0", 0, 5),
            self._sample("normal-1", 0, 9),
            self._sample("abnormal-0", 1, 20),
            self._sample("abnormal-1", 1, 30),
        ]
        args = _args(self.root / "output")
        device = torch.device("cpu")
        model = fit_udma(train, args, device)

        self.assertTrue(bool(model.teacher_statistics_fitted.item()))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(len(model.training_history["teacher_loss"]), 1)
        self.assertEqual(len(model.training_history["student_loss"]), 1)

        job = {
            "dataset": "synthetic",
            "category": "signal",
            "scene": "scene-1",
            "jsr": "jsr-1",
            "train_samples": train,
            "test_samples": test,
        }
        row = predict_job(model, job, args, device)
        self.assertEqual(row["method"], "udma_reimplementation")
        self.assertEqual(row["num_test_normal"], 2)
        self.assertEqual(row["num_test_abnormal"], 2)
        self.assertTrue(np.isfinite(row["image_auroc"]))
        self.assertTrue(np.isfinite(row["image_auprc"]))
        self.assertTrue(np.isfinite(row["fpr95"]))

        payloads = list((Path(args.output_root) / "scores").glob("*.npz"))
        self.assertEqual(len(payloads), 1)
        with np.load(payloads[0]) as payload:
            self.assertEqual(
                set(payload.files),
                {"labels", "scores", "names", "image_paths", "method"},
            )
            self.assertEqual(payload["labels"].tolist(), [0, 0, 1, 1])
            self.assertEqual(payload["names"].tolist(), [sample["name"] for sample in test])
            self.assertEqual(
                payload["image_paths"].tolist(), [sample["path"] for sample in test]
            )
            self.assertTrue(np.isfinite(payload["scores"]).all())

    def test_support_only_guard_and_metric_guard(self):
        args = _args(self.root / "output")
        abnormal = self._sample("bad-support", 1, 20)
        with self.assertRaisesRegex(ValueError, "support-only"):
            fit_udma([abnormal], args, torch.device("cpu"))
        metrics = safe_metrics([0, 0], [0.1, 0.2])
        self.assertTrue(np.isnan(metrics["image_auroc"]))

    def test_public_rf_requires_explicit_manifest(self):
        args = SimpleNamespace(protocol="public_rf", support_manifest="")
        with self.assertRaisesRegex(ValueError, "explicit --support-manifest"):
            build_jobs(args)


if __name__ == "__main__":
    unittest.main()
