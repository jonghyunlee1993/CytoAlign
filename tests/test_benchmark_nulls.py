import numpy as np
import pytest

from src.benchmark.artifacts import LockedFeatureScales, TrainingReference
from src.evaluation.distribution_metrics import (
    make_sliced_wasserstein_projections,
    sliced_wasserstein_distance,
)
from src.evaluation.individualization import (
    derangements_digest,
    evaluate_matched_patient_gain,
    patient_derangement,
)
from src.models.nulls import TargetPriorSampler


DIGEST = "a" * 64


def _reference(values, patients, specimens, markers=("Y1",)):
    return TrainingReference(
        values=np.asarray(values),
        patient_ids=tuple(patients),
        specimen_ids=tuple(specimens),
        marker_names=tuple(markers),
        split_role="outer_training",
        bank_role="null_prior_bank",
        fold=0,
        reference_bank_id="fold0-null-prior",
        manifest_digest=DIGEST,
    )


def _scales(markers=("Y1",), values=None):
    if values is None:
        values = np.ones(len(markers))
    return LockedFeatureScales(
        values=np.asarray(values),
        marker_names=tuple(markers),
        fit_role="outer_training_target",
        fold=0,
        patient_digest="b" * 64,
        manifest_digest=DIGEST,
        scale_id="fold0-iqr",
    )


def _derangements(patients, count=20):
    mappings = [
        patient_derangement(patients, random_state=11 + index)
        for index in range(count)
    ]
    return mappings, derangements_digest(mappings)


def test_target_prior_sampler_is_reproducible_and_preserves_joint_rows():
    target = np.asarray([[0.0, 10.0], [1.0, 11.0], [5.0, 15.0]])
    reference = _reference(
        target,
        ["P1", "P1", "P2"],
        ["S1", "S1", "S2"],
        markers=("Y1", "Y2"),
    )
    sampler = TargetPriorSampler(random_state=7).fit(reference)

    first = sampler.predict(100, token="specimen-A")
    second = sampler.predict(100, token="specimen-A")

    np.testing.assert_array_equal(first, second)
    assert all(tuple(row) in set(map(tuple, target)) for row in first)
    assert np.allclose(first[:, 1] - first[:, 0], 10.0)
    assert sampler.reference_bank_id_ == "fold0-null-prior"


def test_target_prior_sampler_balances_patients_before_cells():
    target = np.concatenate([np.zeros((1000, 1)), np.ones((1, 1))])
    reference = _reference(
        target,
        ["large"] * 1000 + ["small"],
        ["large-S1"] * 1000 + ["small-S1"],
    )
    draw = TargetPriorSampler(random_state=9).fit(reference).predict(
        10_000, token="balance"
    )
    assert 0.45 < float(draw.mean()) < 0.55


def test_target_prior_sampler_balances_longitudinal_specimens_before_cells():
    target = np.concatenate([np.zeros((1000, 1)), np.ones((1, 1))])
    reference = _reference(
        target,
        ["P1"] * 1001,
        ["large-visit"] * 1000 + ["small-visit"],
    )
    draw = TargetPriorSampler(random_state=9).fit(reference).predict(
        10_000, token="visits"
    )
    assert 0.45 < float(draw.mean()) < 0.55


def test_training_reference_rejects_nontraining_or_missing_provenance():
    with pytest.raises(ValueError, match="outer_training"):
        TrainingReference(
            values=np.asarray([[1.0]]),
            patient_ids=("P1",),
            specimen_ids=("S1",),
            marker_names=("Y1",),
            split_role="test",
            bank_role="null_prior_bank",
            fold=0,
            reference_bank_id="bad",
            manifest_digest=DIGEST,
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        _reference([[1.0]], ["P1"], [""])


def test_patient_derangement_has_no_self_match_and_rejects_duplicates():
    patients = ["P1", "P2", "P3", "P4"]
    first = patient_derangement(patients, random_state=3)
    second = patient_derangement(patients, random_state=3)
    assert first == second
    assert set(first) == set(patients)
    assert set(first.values()) == set(patients)
    assert all(patient != wrong for patient, wrong in first.items())
    with pytest.raises(ValueError, match="unique"):
        patient_derangement(["P1", "P1", "P2"], random_state=3)


def test_matched_patient_gain_is_positive_and_uses_frozen_derangements():
    values = {
        "S1": np.asarray([[0.0], [0.1], [0.2]]),
        "S2": np.asarray([[5.0], [5.1], [5.2]]),
        "S3": np.asarray([[10.0], [10.1], [10.2]]),
    }
    patients = {"S1": "P1", "S2": "P2", "S3": "P3"}
    mappings, digest = _derangements(("P1", "P2", "P3"))
    result = evaluate_matched_patient_gain(
        values,
        values,
        patients,
        _scales(),
        marker_names=("Y1",),
        derangements=mappings,
        derangement_manifest_digest=digest,
    )
    assert result["matched_error"] == 0.0
    assert result["wrong_patient_error"] > 1.0
    assert result["individualization_gain"] > 1.0
    assert len(result["per_derangement"]) == 20


def test_source_blind_shared_prior_has_zero_aggregate_individualization_gain():
    shared_prior = np.asarray([[0.0], [5.0], [10.0]])
    predictions = {
        "S1": shared_prior,
        "S2": shared_prior,
        "S3": shared_prior,
    }
    targets = {
        "S1": np.zeros((10, 1)),
        "S2": np.full((10, 1), 5.0),
        "S3": np.full((10, 1), 10.0),
    }
    patient_by_specimen = {"S1": "P1", "S2": "P2", "S3": "P3"}
    mappings, digest = _derangements(("P1", "P2", "P3"))
    result = evaluate_matched_patient_gain(
        predictions,
        targets,
        patient_by_specimen,
        _scales(),
        marker_names=("Y1",),
        derangements=mappings,
        derangement_manifest_digest=digest,
    )
    assert result["individualization_gain"] == pytest.approx(0.0, abs=1e-12)


def test_matched_patient_gain_equal_weights_longitudinal_specimens():
    predictions = {
        "P1-large": np.zeros((500, 1)),
        "P1-small": np.asarray([[10.0]]),
        "P2-only": np.full((20, 1), 20.0),
    }
    targets = {
        "P1-large": np.zeros((1000, 1)),
        "P1-small": np.asarray([[10.0]]),
        "P2-only": np.full((5, 1), 20.0),
    }
    patient_by_specimen = {
        "P1-large": "P1",
        "P1-small": "P1",
        "P2-only": "P2",
    }
    mappings, digest = _derangements(("P1", "P2"), count=1)
    result = evaluate_matched_patient_gain(
        predictions,
        targets,
        patient_by_specimen,
        _scales(),
        marker_names=("Y1",),
        derangements=mappings,
        derangement_manifest_digest=digest,
    )
    assert result["per_patient"]["P1"]["matched_error"] == pytest.approx(0.0)
    assert result["per_patient"]["P1"]["target_specimen_count"] == 2


def test_matched_patient_gain_rejects_key_nonfinite_and_digest_errors():
    mappings, digest = _derangements(("P1", "P2"), count=1)
    kwargs = {
        "patient_by_specimen": {"S1": "P1", "S2": "P2"},
        "feature_scales": _scales(),
        "marker_names": ("Y1",),
        "derangements": mappings,
        "derangement_manifest_digest": digest,
    }
    with pytest.raises(ValueError, match="specimen keys differ"):
        evaluate_matched_patient_gain(
            {"S1": np.asarray([[0.0]]), "S2": np.asarray([[1.0]])},
            {"S1": np.asarray([[0.0]]), "S3": np.asarray([[1.0]])},
            **kwargs,
        )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_matched_patient_gain(
            {"S1": np.asarray([[np.nan]]), "S2": np.asarray([[1.0]])},
            {"S1": np.asarray([[0.0]]), "S2": np.asarray([[1.0]])},
            **kwargs,
        )
    with pytest.raises(ValueError, match="digest"):
        evaluate_matched_patient_gain(
            {"S1": np.asarray([[0.0]]), "S2": np.asarray([[1.0]])},
            {"S1": np.asarray([[0.0]]), "S2": np.asarray([[1.0]])},
            **{**kwargs, "derangement_manifest_digest": "0" * 64},
        )


def test_sliced_wasserstein_is_zero_for_equal_and_positive_for_shifted_data():
    rng = np.random.RandomState(13)
    values = rng.normal(size=(200, 3))
    scales = _scales(("M1", "M2", "M3"))
    projections = make_sliced_wasserstein_projections(
        3, n_projections=32, random_state=5
    )
    exact = sliced_wasserstein_distance(
        values,
        values,
        scales,
        marker_names=("M1", "M2", "M3"),
        projections=projections,
    )
    shifted = sliced_wasserstein_distance(
        values,
        values + 2.0,
        scales,
        marker_names=("M1", "M2", "M3"),
        projections=projections,
    )
    assert exact == pytest.approx(0.0, abs=1e-12)
    assert shifted > 0.5


def test_sliced_wasserstein_identical_input_remains_zero_above_row_cap():
    values = np.random.RandomState(19).normal(size=(10_000, 3))
    distance = sliced_wasserstein_distance(
        values,
        values,
        _scales(("M1", "M2", "M3")),
        marker_names=("M1", "M2", "M3"),
        n_projections=32,
        max_rows=1000,
        random_state=5,
    )
    assert distance == pytest.approx(0.0, abs=1e-12)
