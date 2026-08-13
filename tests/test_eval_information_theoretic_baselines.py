import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools import eval_information_theoretic_baselines as evaluator


class InformationTheoreticEvaluatorTest(unittest.TestCase):
    def test_both_methods_fit_and_score_with_paper_defaults(self):
        support = [np.zeros((8, 9), dtype=np.uint8), np.ones((8, 9), dtype=np.uint8)]
        models = evaluator.fit_detectors(support, evaluator.METHODS)
        scores = evaluator.score_image(
            models, np.full((8, 9), 255, dtype=np.uint8), time_axis=0
        )
        self.assertEqual(set(models), set(evaluator.METHODS))
        self.assertEqual(set(scores), set(evaluator.METHODS))
        self.assertTrue(all(np.isfinite(value) for value in scores.values()))
        config = evaluator.method_protocol(time_axis=0)
        self.assertEqual(config["kld_reference"]["bins"], 32)
        self.assertEqual(config["ica_frozen"]["bins"], 20)
        self.assertEqual(config["ica_frozen"]["cluster_length"], 3)

    def test_binary_smoke_limit_is_balanced_and_order_preserving(self):
        samples = [
            {"name": f"n{index}", "label": label}
            for index, label in enumerate([0, 0, 1, 0, 1, 1])
        ]
        limited = evaluator.limit_binary_samples(samples, 2)
        self.assertEqual([item["name"] for item in limited], ["n0", "n1", "n2", "n4"])
        self.assertEqual([item["label"] for item in limited], [0, 0, 1, 1])

    def test_ofdma_uses_max_over_exactly_21_sus(self):
        count = 42
        payload = {
            "observation_ids": np.asarray(["obs0"] * 21 + ["obs1"] * 21),
            "labels": np.asarray([0] * 21 + [1] * 21, dtype=np.int32),
            "jammer_types": np.asarray(["no jammer"] * 21 + ["barrage"] * 21),
            "kld_reference": np.arange(count, dtype=np.float32),
            "ica_frozen": np.arange(count, dtype=np.float32)[::-1],
        }
        output = evaluator.aggregate_ofdma(payload, evaluator.METHODS)
        np.testing.assert_array_equal(output["labels"], [0, 1])
        np.testing.assert_allclose(output["kld_reference"], [20, 41])
        np.testing.assert_allclose(output["ica_frozen"], [41, 20])
        broken = dict(payload)
        broken["observation_ids"] = np.asarray(["obs0"] * 20 + ["obs1"] * 22)
        with self.assertRaisesRegex(ValueError, "expected 21"):
            evaluator.aggregate_ofdma(broken, evaluator.METHODS)

    def test_metrics_and_macro_are_percentage_scaled(self):
        perfect = evaluator.safe_metrics([0, 0, 1, 1], [0.0, 0.1, 0.8, 0.9])
        self.assertEqual(perfect["auroc"], 100.0)
        self.assertEqual(perfect["auprc"], 100.0)
        self.assertEqual(perfect["fpr95"], 0.0)
        rows = []
        for scene, scores in (("a", [0.0, 1.0]), ("b", [1.0, 0.0])):
            rows.append(
                evaluator.metric_row(
                    row_type="scene",
                    dataset="ofdma",
                    category="ofdma",
                    scene=scene,
                    jsr="1shot",
                    shot=1,
                    scope="overall",
                    method="kld_reference",
                    num_support=21,
                    labels=np.asarray([0, 1]),
                    scores=np.asarray(scores),
                )
            )
        macro = evaluator.append_macro_rows(rows)
        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["auroc"], 50.0)

    def test_fedjam_macro_averages_attack_cells(self):
        rows = []
        for attack, auroc in (("pulse", 60.0), ("single_tone", 70.0), ("wideband", 80.0)):
            rows.append(
                {
                    "shot": 1,
                    "method": "ica_frozen",
                    "num_test_normal": 2,
                    "num_test_abnormal": 1,
                    "auroc": auroc,
                    "auprc": auroc - 10.0,
                    "fpr95": 100.0 - auroc,
                    "scope": attack,
                }
            )
        macro = evaluator.fedjam_attack_macro_rows(rows)
        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["scope"], "macro_attack")
        self.assertEqual(macro[0]["auroc"], 70.0)
        self.assertEqual(macro[0]["num_test_abnormal"], 3)

    def test_rf_builder_delegates_and_keeps_explicit_manifest(self):
        fake_job = {
            "dataset": "public_rf",
            "category": "x",
            "scene": "s",
            "jsr": "j",
            "train_samples": [],
            "test_samples": [{"label": 0}, {"label": 1}],
        }

        def fake_public(builder_args):
            self.assertEqual(builder_args.support_manifest, "/tmp/frozen.json")
            builder_args.support_manifest_sha256 = "manifest-hash"
            builder_args.test_normal_paths_sha256 = "test-hash"
            builder_args.support_policy = "k2_per_frequency"
            builder_args.per_frequency_k = 2
            return [fake_job]

        args = SimpleNamespace(
            protocol="public_rf",
            output_root="/tmp/out",
            normal_sampling="per_frequency",
            seed=111,
            support_seed=111,
            support_manifest="/tmp/frozen.json",
            public_rf_signals=("x",),
            max_test_per_class=0,
        )
        with mock.patch(
            "tools.eval_cls_aux_cnn_gallery.public_rf_jobs", side_effect=fake_public
        ) as called:
            jobs, metadata = evaluator.build_rf_jobs(args)
        called.assert_called_once()
        self.assertIs(jobs[0], fake_job)
        self.assertEqual(metadata["support_manifest_sha256"], "manifest-hash")
        self.assertEqual(metadata["per_frequency_k"], 2)

    def test_ofdma_builder_delegates_to_formal_build_jobs(self):
        normalization = {"min_db": -120.0, "max_db": 0.0}
        source = {
            "source_protocol": {
                "protocol_name": "target-scene-test",
                "normalization": normalization,
            }
        }
        args = SimpleNamespace(
            dataset_root="/tmp/ofdma",
            split="test",
            scene_ids=[],
            shots=[1],
            max_normal_observations=1,
            max_anomaly_observations_per_type=1,
        )
        fake_preprocessor = object()
        with mock.patch(
            "tools.eval_ofdma_target_scene_baselines.source_protocol",
            return_value=source,
        ), mock.patch(
            "datasets.ofdma_target_scene.load_target_scene_manifest",
            return_value="manifest",
        ), mock.patch(
            "tools.eval_ofdma_target_scene_baselines.build_jobs",
            return_value=([{"dataset": "ofdma"}], ["scene-a"]),
        ) as build, mock.patch(
            "datasets.ofdma_spectrum.OFDMASpectrogramPreprocessor",
            return_value=fake_preprocessor,
        ):
            jobs, metadata, preprocessor = evaluator.build_ofdma_protocol(args)
        build.assert_called_once_with(args, "manifest", normalization)
        self.assertEqual(jobs, [{"dataset": "ofdma"}])
        self.assertEqual(metadata["su_reduction"], "maximum score over 21 sensing units")
        self.assertIs(preprocessor, fake_preprocessor)

    def test_path_job_scores_before_copying_labels(self):
        samples = [
            {"path": "/fake/support.png", "name": "support", "label": 0},
            {"path": "/fake/test.png", "name": "test", "label": 1},
        ]
        job = {
            "train_samples": samples[:1],
            "test_samples": samples[1:],
        }
        args = SimpleNamespace(
            protocol="rf_target",
            methods=("kld_reference",),
            time_axis=0,
            progress_every=0,
        )
        with mock.patch.object(
            evaluator, "load_path_image", return_value=np.zeros((4, 4), dtype=np.uint8)
        ), mock.patch.object(
            evaluator, "fit_detectors", return_value={"kld_reference": object()}
        ) as fit, mock.patch.object(
            evaluator, "score_image", return_value={"kld_reference": 3.5}
        ):
            payload, _ = evaluator.evaluate_path_job(job, args)
        fit.assert_called_once()
        np.testing.assert_array_equal(payload["labels"], [1])
        np.testing.assert_allclose(payload["kld_reference"], [3.5])

    def test_cli_defaults_to_both_methods_and_cpu_protocol_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            rf = evaluator.parse_args(
                ["--protocol", "rf_target", "--output-root", directory]
            )
            ofdma = evaluator.parse_args(
                ["--protocol", "ofdma", "--output-root", directory]
            )
        self.assertEqual(rf.methods, evaluator.METHODS)
        self.assertEqual(rf.time_axis, 0)
        self.assertEqual(ofdma.time_axis, 1)


if __name__ == "__main__":
    unittest.main()
