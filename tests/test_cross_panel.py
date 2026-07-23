import numpy as np
import pandas as pd

from src.data.cross_panel import load_cross_panel_dataset


def test_cross_panel_loader_aligns_aliases_and_excludes_technical_markers(tmp_path):
    for modality in ("spectral_flow", "cytof"):
        (tmp_path / modality / "cells").mkdir(parents=True)
        (tmp_path / modality / "labels").mkdir()
    for index in range(4):
        specimen = f"R{index:04d}_A"
        pd.DataFrame(
            {
                "CD3": np.arange(20),
                "PD-1": np.arange(20) + 1,
                "FSC-A": np.arange(20) + 2,
                "X": np.arange(20) + 3,
            }
        ).to_csv(tmp_path / "spectral_flow" / "cells" / f"{specimen}.csv", index=False)
        pd.DataFrame(
            {
                "CD3": np.arange(20),
                "CD279": np.arange(20) + 1,
                "Y": np.arange(20) + 4,
            }
        ).to_csv(tmp_path / "cytof" / "cells" / f"{specimen}.csv", index=False)
        labels = pd.DataFrame({"label": ["T cell"] * 20})
        labels.to_csv(
            tmp_path / "spectral_flow" / "labels" / f"{specimen}.csv", index=False
        )
        labels.to_csv(tmp_path / "cytof" / "labels" / f"{specimen}.csv", index=False)

    dataset = load_cross_panel_dataset(
        {
            "root": str(tmp_path),
            "source_modality": "spectral_flow",
            "target_modality": "cytof",
            "max_cells_per_specimen": 10,
            "chunk_size": 7,
            "sample_seed": 1,
            "split": {
                "n_splits": 2,
                "seed": 2,
                "validation_fraction": 0.5,
            },
        }
    )

    assert dataset.common_markers == ("CD3", "PD-1")
    assert dataset.source_exclusive_columns == ("X",)
    assert dataset.target_exclusive_columns == ("Y",)
    assert dataset.source["R0000_A"].values.shape == (10, 3)
