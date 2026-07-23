"""Immutable, pickle-free prediction artifacts shared by baseline methods."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass
class PredictionBundle:
    predictions: np.ndarray
    patient_ids: np.ndarray
    specimen_ids: np.ndarray
    source_row_indices: np.ndarray
    cell_types: np.ndarray
    target_markers: tuple[str, ...]
    metadata: dict = field(default_factory=dict)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> None:
        prediction = np.asarray(self.predictions)
        if prediction.ndim != 2:
            raise ValueError("predictions must be two-dimensional")
        if prediction.shape[1] != len(self.target_markers):
            raise ValueError("prediction columns do not match target_markers")
        n_rows = prediction.shape[0]
        for name, values in (
            ("patient_ids", self.patient_ids),
            ("specimen_ids", self.specimen_ids),
            ("source_row_indices", self.source_row_indices),
            ("cell_types", self.cell_types),
        ):
            if np.asarray(values).ndim != 1 or len(values) != n_rows:
                raise ValueError(f"{name} does not align with prediction rows")
        for name, values in self.diagnostics.items():
            if np.asarray(values).ndim != 1 or len(values) != n_rows:
                raise ValueError(f"Diagnostic {name} does not align with predictions")
        required_metadata = {"method", "direction", "fold", "label_source"}
        missing = required_metadata - set(self.metadata)
        if missing:
            raise ValueError(f"Prediction metadata is missing: {sorted(missing)}")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "predictions": np.asarray(self.predictions, dtype=np.float32),
            "patient_ids": np.asarray(self.patient_ids, dtype=str),
            "specimen_ids": np.asarray(self.specimen_ids, dtype=str),
            "source_row_indices": np.asarray(self.source_row_indices, dtype=np.int64),
            "cell_types": np.asarray(self.cell_types, dtype=str),
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "metadata": self.metadata,
                        "target_markers": list(self.target_markers),
                        "diagnostic_names": sorted(self.diagnostics),
                    },
                    sort_keys=True,
                )
            ),
        }
        arrays.update(
            {f"diagnostic__{name}": np.asarray(values) for name, values in self.diagnostics.items()}
        )
        np.savez_compressed(output, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "PredictionBundle":
        with np.load(Path(path), allow_pickle=False) as archive:
            description = json.loads(str(archive["metadata_json"]))
            diagnostics = {
                name: archive[f"diagnostic__{name}"]
                for name in description["diagnostic_names"]
            }
            bundle = cls(
                predictions=archive["predictions"],
                patient_ids=archive["patient_ids"],
                specimen_ids=archive["specimen_ids"],
                source_row_indices=archive["source_row_indices"],
                cell_types=archive["cell_types"],
                target_markers=tuple(description["target_markers"]),
                metadata=description["metadata"],
                diagnostics=diagnostics,
            )
        bundle.validate()
        return bundle

