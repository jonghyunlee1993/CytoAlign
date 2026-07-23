"""NumPy empirical-CDF transforms for a shared cross-panel marker space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return array


class EmpiricalPercentileTransformer:
    """Piecewise-linear, train-fitted marginal percentile transform."""

    def __init__(self, n_knots: int = 129):
        if int(n_knots) < 3:
            raise ValueError("n_knots must be at least 3")
        self.n_knots = int(n_knots)
        self.knots_: list[np.ndarray] | None = None
        self.probabilities_: list[np.ndarray] | None = None

    def fit(self, values: np.ndarray) -> "EmpiricalPercentileTransformer":
        array = _matrix(values, "values")
        if array.shape[0] < 2:
            raise ValueError("At least two training rows are required")
        probabilities = np.linspace(0.0, 1.0, self.n_knots)
        knots: list[np.ndarray] = []
        mapped_probabilities: list[np.ndarray] = []
        for marker_index in range(array.shape[1]):
            column = array[:, marker_index]
            column = column[np.isfinite(column)]
            if column.size < 2:
                raise ValueError(f"Marker {marker_index} has fewer than two finite rows")
            quantiles = np.quantile(column, probabilities)
            unique, inverse = np.unique(quantiles, return_inverse=True)
            if unique.size == 1:
                epsilon = max(1.0e-6, abs(float(unique[0])) * 1.0e-6)
                unique = np.asarray([unique[0] - epsilon, unique[0] + epsilon])
                current_probabilities = np.asarray([0.0, 1.0])
            else:
                sums = np.zeros(unique.size, dtype=np.float64)
                counts = np.zeros(unique.size, dtype=np.float64)
                np.add.at(sums, inverse, probabilities)
                np.add.at(counts, inverse, 1.0)
                current_probabilities = sums / counts
            knots.append(unique.astype(np.float64, copy=False))
            mapped_probabilities.append(
                current_probabilities.astype(np.float64, copy=False)
            )
        self.knots_ = knots
        self.probabilities_ = mapped_probabilities
        return self

    @property
    def n_features_in_(self) -> int:
        self._check_fitted()
        assert self.knots_ is not None
        return len(self.knots_)

    def _check_fitted(self) -> None:
        if self.knots_ is None or self.probabilities_ is None:
            raise RuntimeError("Transformer has not been fitted")

    def _validate(self, values: np.ndarray, name: str) -> np.ndarray:
        self._check_fitted()
        array = _matrix(values, name)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} has {array.shape[1]} columns; expected {self.n_features_in_}"
            )
        return array

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = self._validate(values, "values")
        assert self.knots_ is not None and self.probabilities_ is not None
        output = np.empty_like(array, dtype=np.float64)
        for index, (knots, probabilities) in enumerate(
            zip(self.knots_, self.probabilities_)
        ):
            output[:, index] = np.interp(
                array[:, index],
                knots,
                probabilities,
                left=probabilities[0],
                right=probabilities[-1],
            )
        return output.astype(np.float32)

    def inverse_transform(self, percentiles: np.ndarray) -> np.ndarray:
        array = self._validate(percentiles, "percentiles")
        assert self.knots_ is not None and self.probabilities_ is not None
        output = np.empty_like(array, dtype=np.float64)
        for index, (knots, probabilities) in enumerate(
            zip(self.knots_, self.probabilities_)
        ):
            output[:, index] = np.interp(
                np.clip(array[:, index], 0.0, 1.0),
                probabilities,
                knots,
                left=knots[0],
                right=knots[-1],
            )
        return output.astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


@dataclass
class CrossPanelCommonSpace:
    """Two train-only marginal transforms joined through percentile space."""

    source: EmpiricalPercentileTransformer
    target: EmpiricalPercentileTransformer

    @classmethod
    def fit(
        cls,
        source_training_common: np.ndarray,
        target_training_common: np.ndarray,
        *,
        n_knots: int = 129,
    ) -> "CrossPanelCommonSpace":
        source = _matrix(source_training_common, "source_training_common")
        target = _matrix(target_training_common, "target_training_common")
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target common-marker dimensions differ")
        return cls(
            source=EmpiricalPercentileTransformer(n_knots).fit(source),
            target=EmpiricalPercentileTransformer(n_knots).fit(target),
        )

    def source_percentiles(self, values: np.ndarray) -> np.ndarray:
        return self.source.transform(values)

    def target_percentiles(self, values: np.ndarray) -> np.ndarray:
        return self.target.transform(values)

    def source_to_target(self, values: np.ndarray) -> np.ndarray:
        return self.target.inverse_transform(self.source.transform(values))

