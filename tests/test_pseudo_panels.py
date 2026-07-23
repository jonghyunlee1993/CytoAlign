import numpy as np
import pytest

from src.data.pseudo_panels import (
    build_two_sided_pseudo_panel_manifest,
    make_two_sided_pseudo_panel_views,
)
from src.data.sampling import patient_cell_type_balanced_indices


def test_two_sided_pseudo_panel_preserves_full_panel_order_and_exact_truth():
    markers = ("Y2", "H1", "X1", "H2", "Y1", "unused", "X2")
    manifest = build_two_sided_pseudo_panel_manifest(
        markers,
        common_markers=("H2", "H1"),
        source_exclusive_markers=("X2", "X1"),
        target_exclusive_markers=("Y1", "Y2"),
    )
    assert manifest.common_markers == ("H1", "H2")
    assert manifest.source_markers == ("H1", "X1", "H2", "X2")
    assert manifest.target_markers == ("Y2", "H1", "H2", "Y1")
    assert manifest.unused_markers == ("unused",)

    values = np.arange(21, dtype=np.float32).reshape(3, 7)
    views = make_two_sided_pseudo_panel_views(values, markers, manifest)
    np.testing.assert_array_equal(views.source_exclusive_values, values[:, [2, 6]])
    np.testing.assert_array_equal(views.target_exclusive_values, values[:, [0, 4]])
    np.testing.assert_array_equal(views.common_values, values[:, [1, 3]])


def test_two_sided_pseudo_panel_rejects_overlap_and_unassigned_markers():
    with pytest.raises(ValueError, match="overlap"):
        build_two_sided_pseudo_panel_manifest(
            ("H", "X", "Y"),
            common_markers=("H",),
            source_exclusive_markers=("X",),
            target_exclusive_markers=("X", "Y"),
        )
    with pytest.raises(ValueError, match="unassigned"):
        build_two_sided_pseudo_panel_manifest(
            ("H", "X", "Y", "unused"),
            common_markers=("H",),
            source_exclusive_markers=("X",),
            target_exclusive_markers=("Y",),
            require_complete_partition=True,
        )


def test_patient_cell_type_sampling_caps_every_stratum_deterministically():
    patients = np.asarray(["P1"] * 8 + ["P2"] * 5)
    labels = np.asarray(["T"] * 6 + ["B"] * 2 + ["T"] * 5)
    first = patient_cell_type_balanced_indices(
        patients, labels, maximum_per_stratum=3, random_state=9
    )
    second = patient_cell_type_balanced_indices(
        patients, labels, maximum_per_stratum=3, random_state=9
    )
    np.testing.assert_array_equal(first, second)
    selected_strata = list(zip(patients[first], labels[first]))
    assert selected_strata.count(("P1", "T")) == 3
    assert selected_strata.count(("P1", "B")) == 2
    assert selected_strata.count(("P2", "T")) == 3
