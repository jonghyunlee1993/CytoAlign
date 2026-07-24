import numpy as np
import pandas as pd
import pytest

from src.data.aml import coarsen_cell_types, load_specimen, load_specimen_reservoir


def test_coarse_cell_type_mapping():
    mapped = coarsen_cell_types(
        [
            "Blast CD34hi",
            "Monocyte",
            "T cell CD8",
            "B cell Kappa",
            "NK cell",
            "Debris",
        ]
    )
    assert mapped.tolist() == [
        "Blast",
        "Monocyte",
        "T cell",
        "B cell",
        "NK cell",
        None,
    ]


def test_load_specimen_keeps_original_row_indices_and_drops_unmapped(tmp_path):
    root = tmp_path / "spectral_flow"
    (root / "cells").mkdir(parents=True)
    (root / "labels").mkdir()
    pd.DataFrame({"CD3": [1.0, 2.0, 3.0], "CD4": [4.0, 5.0, 6.0]}).to_csv(
        root / "cells" / "R0001_A.csv", index=False
    )
    pd.DataFrame({"label": ["T cell CD4", "Debris", "Blast"]}).to_csv(
        root / "labels" / "R0001_A.csv", index=False
    )

    specimen = load_specimen(
        tmp_path, "spectral_flow", "R0001_A", marker_columns=["CD4", "CD3"]
    )
    assert specimen.markers == ("CD4", "CD3")
    assert specimen.values.tolist() == [[4.0, 1.0], [6.0, 3.0]]
    assert specimen.cell_types.tolist() == ["T cell", "Blast"]
    assert specimen.fine_cell_types.tolist() == ["T cell CD4", "Blast"]
    assert specimen.original_row_indices.tolist() == [0, 2]


def test_load_specimen_rejects_row_mismatch(tmp_path):
    root = tmp_path / "cytof"
    (root / "cells").mkdir(parents=True)
    (root / "labels").mkdir()
    pd.DataFrame({"CD3": [1.0, 2.0]}).to_csv(
        root / "cells" / "R0001_A.csv", index=False
    )
    pd.DataFrame({"label": ["T cell"]}).to_csv(
        root / "labels" / "R0001_A.csv", index=False
    )
    with pytest.raises(ValueError, match="row mismatch"):
        load_specimen(tmp_path, "cytof", "R0001_A")


def test_load_specimen_head_cap_keeps_cells_and_labels_aligned(tmp_path):
    root = tmp_path / "cytof"
    (root / "cells").mkdir(parents=True)
    (root / "labels").mkdir()
    pd.DataFrame({"CD3": [1.0, 2.0, 3.0], "CD4": [10.0, 20.0, 30.0]}).to_csv(
        root / "cells" / "R0001_A.csv", index=False
    )
    pd.DataFrame({"label": ["Blast", "T cell CD4", "B cell"]}).to_csv(
        root / "labels" / "R0001_A.csv", index=False
    )
    specimen = load_specimen(tmp_path, "cytof", "R0001_A", maximum_rows=2)
    assert specimen.values.tolist() == [[1.0, 10.0], [2.0, 20.0]]
    assert specimen.cell_types.tolist() == ["Blast", "T cell"]
    assert specimen.fine_cell_types.tolist() == ["Blast", "T cell CD4"]
    assert specimen.original_row_indices.tolist() == [0, 1]


def test_reservoir_sampler_is_uniform_order_independent_and_aligned(tmp_path):
    root = tmp_path / "cytof"
    (root / "cells").mkdir(parents=True)
    (root / "labels").mkdir()
    n_rows = 30
    pd.DataFrame({"CD3": np.arange(n_rows), "CD4": 100 + np.arange(n_rows)}).to_csv(
        root / "cells" / "R0001_A.csv", index=False
    )
    labels = np.asarray(["Blast", "T cell CD4", "Debris"] * 10)
    pd.DataFrame({"label": labels}).to_csv(root / "labels" / "R0001_A.csv", index=False)

    first = load_specimen_reservoir(
        tmp_path,
        "cytof",
        "R0001_A",
        marker_columns=("CD4", "CD3"),
        maximum_cells=7,
        chunk_size=4,
        random_state=19,
    )
    second = load_specimen_reservoir(
        tmp_path,
        "cytof",
        "R0001_A",
        marker_columns=("CD4", "CD3"),
        maximum_cells=7,
        chunk_size=11,
        random_state=19,
    )
    assert first.markers == ("CD4", "CD3")
    np.testing.assert_array_equal(
        first.original_row_indices, second.original_row_indices
    )
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.fine_cell_types, second.fine_cell_types)
    np.testing.assert_array_equal(first.values[:, 1], first.original_row_indices)
    assert not np.array_equal(first.original_row_indices, np.arange(7))
    assert all(labels[index] != "Debris" for index in first.original_row_indices)
