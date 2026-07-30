import json

import numpy as np
import pytest

from src.training.cell_type_probe import (
    FINE_CELL_TYPES,
    paired_patient_specimens,
    probe_metrics,
    shuffle_rows_within_groups,
    summarize_cell_type_probe,
)


def test_probe_metrics_reports_rare_class_auprc():
    labels = np.asarray(FINE_CELL_TYPES)
    predictions = labels.copy()
    probabilities = np.full((len(labels), len(labels)), 0.01)
    np.fill_diagonal(probabilities, 0.93)
    result = probe_metrics(labels, predictions, probabilities, FINE_CELL_TYPES)
    assert result["balanced_accuracy"] == 1.0
    assert result["rare_t_macro_auprc"] == 1.0
    assert result["per_class"]["T cell DP"]["recall"] == 1.0


def test_shuffle_rows_within_groups_preserves_group_marginals():
    values = np.arange(12).reshape(6, 2)
    groups = np.asarray(["a", "a", "a", "b", "b", "b"])
    shuffled = shuffle_rows_within_groups(values, groups, seed=7)
    assert sorted(map(tuple, shuffled[:3])) == sorted(map(tuple, values[:3]))
    assert sorted(map(tuple, shuffled[3:])) == sorted(map(tuple, values[3:]))
    assert not np.array_equal(shuffled, values)


def test_paired_patient_subsets_are_nested_and_keep_longitudinal_specimens():
    patients = ["P1", "P2", "P3", "P4"]
    mapping = {
        "P1": ["P1_a", "P1_b"],
        "P2": ["P2_a"],
        "P3": ["P3_a"],
        "P4": ["P4_a"],
    }
    _, one_patients, one_specimens = paired_patient_specimens(
        patients, mapping, 1, seed=7
    )
    _, two_patients, two_specimens = paired_patient_specimens(
        patients, mapping, 2, seed=7
    )
    key, all_patients, all_specimens = paired_patient_specimens(
        patients, mapping, "all", seed=7
    )

    assert set(one_patients) < set(two_patients) < set(all_patients)
    assert set(one_specimens) <= set(two_specimens) <= set(all_specimens)
    assert key == "all"
    assert len(all_patients) == 4
    if "P1" in one_patients:
        assert {"P1_a", "P1_b"} <= set(one_specimens)
    if "P1" in two_patients:
        assert {"P1_a", "P1_b"} <= set(two_specimens)


@pytest.mark.parametrize("count", [0, 5, True, "bad"])
def test_paired_patient_subset_rejects_invalid_counts(count):
    with pytest.raises(ValueError):
        paired_patient_specimens(
            ["P1", "P2"],
            {"P1": ["P1_a"], "P2": ["P2_a"]},
            count,
            seed=7,
        )


def test_group_shuffle_is_stable_when_nested_groups_are_appended():
    small_values = np.arange(12).reshape(6, 2)
    small_groups = np.asarray(["b", "b", "a", "a", "c", "c"])
    large_values = np.vstack([small_values, [[12, 13], [14, 15]]])
    large_groups = np.concatenate([small_groups, ["d", "d"]])

    small = shuffle_rows_within_groups(small_values, small_groups, seed=9)
    large = shuffle_rows_within_groups(large_values, large_groups, seed=9)

    assert np.array_equal(small, large[: len(small)])


def _patient_metric(value, classes):
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "macro_f1": value,
        "macro_auprc": value,
        "t_subtype_macro_auprc": value,
        "rare_t_macro_auprc": value,
        "per_class": {
            label: {
                "support": 1,
                "recall": value,
                "precision": value,
                "auprc": value,
            }
            for label in classes
        },
    }


def _curve_resolutions(patient, baseline, residual):
    classes = ["T cell DN", "T cell DP"]
    methods = {}
    for method, value in (
        ("translated_y_h_only", baseline),
        ("translated_y_h_residual_ungated", residual),
    ):
        methods[method] = {
            "overall": {
                **_patient_metric(value, classes),
                "confusion_matrix": [[1, 0], [0, 1]],
            },
            "patients": {patient: _patient_metric(value, classes)},
        }
    return {
        resolution: {"classes": classes, "methods": methods}
        for resolution in ("fine", "coarse")
    }


def test_seed_filtered_summary_aggregates_each_paired_count(tmp_path):
    root = tmp_path / "curve"
    for fold in range(2):
        patient = f"P{fold}"
        points = {}
        for key, paired_count, residual in (
            ("1", 1, 0.3),
            ("all", 4, 0.4),
        ):
            points[key] = {
                "paired_patients": [
                    f"Q{fold}_{index}" for index in range(paired_count)
                ],
                "paired_specimens": [
                    f"S{fold}_{index}" for index in range(paired_count)
                ],
                "paired_patient_count": paired_count,
                "paired_specimen_count": paired_count,
                "selected_rare_counts": {
                    label: {
                        "cells": paired_count * 10,
                        "specimens": paired_count,
                        "patients": paired_count,
                    }
                    for label in ("T cell DN", "T cell DP")
                },
                "translation": {
                    "teacher_blocks": paired_count,
                    "selected_alphas": [1.0],
                },
                "resolutions": _curve_resolutions(patient, 0.2, residual),
            }
        output = root / f"fold_{fold}" / "seed_7"
        output.mkdir(parents=True)
        (output / "cell_type_probe.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "paired_count_unit": "patient",
                    "paired_count_order": ["1", "all"],
                    "paired_curve": points,
                }
            )
        )

    result = summarize_cell_type_probe(
        root,
        expected_folds=2,
        bootstrap_replicates=20,
        seed=11,
        result_seed=7,
    )

    assert result["paired_count_order"] == ["1", "all"]
    assert result["paired_curve"]["1"]["paired_patient_counts"] == [1, 1]
    assert result["paired_curve"]["all"]["selected_rare_counts"]["T cell DN"][
        "cells"
    ] == [40, 40]
    fine = result["paired_curve"]["all"]["resolutions"]["fine"]
    assert fine["patients"] == 2
    comparison = fine["comparisons"]["h_residual_ungated_minus_h_only"]
    assert comparison["rare_t_macro_auprc"]["delta"] == pytest.approx(0.2)


def test_single_point_summary_schema_remains_supported(tmp_path):
    root = tmp_path / "single"
    for fold in range(2):
        output = root / f"fold_{fold}" / "seed_7"
        output.mkdir(parents=True)
        (output / "cell_type_probe.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "resolutions": _curve_resolutions(f"P{fold}", 0.2, 0.3),
                }
            )
        )

    result = summarize_cell_type_probe(
        root,
        expected_folds=2,
        bootstrap_replicates=10,
        result_seed=7,
    )

    assert "paired_curve" not in result
    assert result["resolutions"]["fine"]["patients"] == 2
