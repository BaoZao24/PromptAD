from types import SimpleNamespace

import numpy as np
import pytest

from tools.eval_resad_formal import (
    Sample,
    assert_sample_isolation,
    build_support_fit_records,
)


def make_sample(index: int) -> Sample:
    image = np.full((12, 16, 3), 32 + index, dtype=np.uint8)
    return Sample(None, 0, "target/test", f"train/normal-{index}", image_bgr=image)


def test_support_fit_uses_two_disjoint_folds_without_external_samples():
    support = [make_sample(index) for index in range(4)]
    args = SimpleNamespace(seed=111, singleton_noise_std=1.0)

    normal, references, metadata = build_support_fit_records(
        support, args, "fedjam/4shot"
    )

    assert len(normal) == 4
    assert sorted(len(values) for values in references.values()) == [2, 2]
    assert {sample.name for sample in normal} == {sample.name for sample in support}
    assert {
        sample.name for values in references.values() for sample in values
    } == {sample.name for sample in support}
    for query in normal:
        assert query.name not in {sample.name for sample in references[query.group]}
    assert metadata["dataset_scope"] == "fedjam"
    assert metadata["support_count"] == 4
    assert metadata["anomaly_count"] == 0


def test_singleton_support_uses_only_a_deterministic_weak_view():
    support = [make_sample(0)]
    args = SimpleNamespace(seed=111, singleton_noise_std=1.0)

    normal_a, references_a, metadata = build_support_fit_records(
        support, args, "fedjam/1shot"
    )
    normal_b, references_b, _ = build_support_fit_records(
        support, args, "fedjam/1shot"
    )

    assert metadata["fit_strategy"] == "singleton weak-view residual fitting"
    assert references_a[normal_a[0].group][0].name == support[0].name
    assert references_b[normal_b[0].group][0].name == support[0].name
    np.testing.assert_array_equal(normal_a[0].image_bgr, normal_b[0].image_bgr)
    assert not np.array_equal(normal_a[0].image_bgr, support[0].image_bgr)


def test_support_test_identity_overlap_is_rejected_for_pathless_samples():
    support = [make_sample(0)]
    duplicated_test = [
        Sample(None, 1, "target/test", support[0].name, image_bgr=support[0].image_bgr)
    ]

    with pytest.raises(RuntimeError, match="support/test leakage"):
        assert_sample_isolation(support, duplicated_test, "fedjam/1shot")
