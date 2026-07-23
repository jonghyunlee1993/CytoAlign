"""Null calibration for observable cross-platform concept fingerprints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from src.concepts.fingerprints import (
    ConceptFingerprintSet,
    fingerprint_cosine_similarity,
)


@dataclass(frozen=True)
class NullSimilarityCalibration:
    """Empirical cutoffs obtained after destroying shared semantic axes."""

    similarity_threshold: float
    margin_threshold: float
    quantile: float
    n_permutations: int
    null_best_scores: np.ndarray
    null_best_margins: np.ndarray

    def to_dict(self) -> dict:
        return {
            "similarity_threshold": float(self.similarity_threshold),
            "margin_threshold": float(self.margin_threshold),
            "quantile": float(self.quantile),
            "n_permutations": int(self.n_permutations),
            "null_best_score_quantiles": {
                str(value): float(np.quantile(self.null_best_scores, value))
                for value in (0.5, 0.9, 0.95, 0.99)
            },
            "null_best_margin_quantiles": {
                str(value): float(np.quantile(self.null_best_margins, value))
                for value in (0.5, 0.9, 0.95, 0.99)
            },
        }


def calibrate_semantic_axis_null(
    source: ConceptFingerprintSet,
    target: ConceptFingerprintSet,
    *,
    n_permutations: int = 100,
    quantile: float = 0.95,
    random_state: int = 42,
    block_weights: Mapping[str, float] | None = None,
) -> NullSimilarityCalibration:
    """Calibrate row-best similarity after permuting target semantic axes.

    Columns are independently permuted within each fingerprint block. This
    retains each block's value distribution and concept usage while breaking
    marker/cell-type semantics shared by the two platforms. The cutoff is
    based on the row-wise best null match, rather than all pairwise scores, so
    ordinary nearest-neighbor selection is included in the null hypothesis.
    """

    source.validate()
    target.validate()
    if source.common_markers != target.common_markers:
        raise ValueError("Source and target common-marker contracts differ")
    if source.cell_types != target.cell_types or source.blocks != target.blocks:
        raise ValueError("Source and target fingerprint layouts differ")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be positive")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")

    rng = np.random.RandomState(int(random_state))
    best_scores: list[np.ndarray] = []
    best_margins: list[np.ndarray] = []
    for _ in range(int(n_permutations)):
        permuted = np.asarray(target.values, dtype=np.float32).copy()
        for start, stop in target.blocks.values():
            order = rng.permutation(int(stop) - int(start))
            permuted[:, int(start) : int(stop)] = target.values[
                :, int(start) + order
            ]
        null_target = replace(target, values=permuted)
        similarity = fingerprint_cosine_similarity(
            source, null_target, block_weights=block_weights
        )
        ordered = np.sort(similarity, axis=1)
        best_scores.append(ordered[:, -1])
        if ordered.shape[1] == 1:
            best_margins.append(np.full(ordered.shape[0], np.inf))
        else:
            best_margins.append(ordered[:, -1] - ordered[:, -2])
    scores = np.concatenate(best_scores).astype(np.float32, copy=False)
    margins = np.concatenate(best_margins).astype(np.float32, copy=False)
    return NullSimilarityCalibration(
        similarity_threshold=float(np.quantile(scores, float(quantile))),
        margin_threshold=float(np.quantile(margins, float(quantile))),
        quantile=float(quantile),
        n_permutations=int(n_permutations),
        null_best_scores=scores,
        null_best_margins=margins,
    )
