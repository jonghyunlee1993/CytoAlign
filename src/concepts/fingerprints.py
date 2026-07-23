"""Observable common-marker fingerprints for independently learned concepts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _safe_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Column-wise cross correlations, with constant columns mapped to zero."""

    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    numerator = left_centered.T @ right_centered
    left_norm = np.sqrt(np.sum(left_centered**2, axis=0))[:, None]
    right_norm = np.sqrt(np.sum(right_centered**2, axis=0))[None, :]
    denominator = left_norm * right_norm
    output = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator > 1.0e-12)
    return output


@dataclass(frozen=True)
class ConceptFingerprintSet:
    """One platform's concept fingerprints in an observable shared space."""

    values: np.ndarray
    activation_frequency: np.ndarray
    concept_ids: tuple[str, ...]
    common_markers: tuple[str, ...]
    cell_types: tuple[str, ...]
    blocks: dict[str, tuple[int, int]]
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        values = np.asarray(self.values)
        frequency = np.asarray(self.activation_frequency)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Fingerprint values must be a finite two-dimensional matrix")
        if len(self.concept_ids) != values.shape[0]:
            raise ValueError("concept_ids do not align with fingerprint rows")
        if frequency.shape != (values.shape[0], len(self.cell_types)):
            raise ValueError("activation_frequency has an unexpected shape")
        covered: list[int] = []
        for name, bounds in self.blocks.items():
            if len(bounds) != 2:
                raise ValueError(f"Fingerprint block {name} has invalid bounds")
            start, stop = map(int, bounds)
            if start < 0 or stop <= start or stop > values.shape[1]:
                raise ValueError(f"Fingerprint block {name} is out of bounds")
            covered.extend(range(start, stop))
        if sorted(covered) != list(range(values.shape[1])):
            raise ValueError("Fingerprint blocks must partition all columns exactly once")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        description = {
            "concept_ids": list(self.concept_ids),
            "common_markers": list(self.common_markers),
            "cell_types": list(self.cell_types),
            "blocks": {name: list(bounds) for name, bounds in self.blocks.items()},
            "metadata": self.metadata,
        }
        np.savez_compressed(
            output,
            values=np.asarray(self.values, dtype=np.float32),
            activation_frequency=np.asarray(self.activation_frequency, dtype=np.float32),
            description_json=np.asarray(json.dumps(description, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConceptFingerprintSet":
        with np.load(Path(path), allow_pickle=False) as archive:
            description = json.loads(str(archive["description_json"]))
            result = cls(
                values=archive["values"],
                activation_frequency=archive["activation_frequency"],
                concept_ids=tuple(description["concept_ids"]),
                common_markers=tuple(description["common_markers"]),
                cell_types=tuple(description["cell_types"]),
                blocks={
                    name: tuple(map(int, bounds))
                    for name, bounds in description["blocks"].items()
                },
                metadata=description["metadata"],
            )
        result.validate()
        return result


def build_concept_fingerprints(
    activations: np.ndarray,
    common_values: np.ndarray,
    *,
    common_markers: Sequence[str],
    cell_types: Sequence,
    cell_type_order: Sequence[str],
    intervention_effects: np.ndarray | None = None,
    activation_threshold: float = 0.0,
    minimum_cells: int = 3,
    concept_ids: Sequence[str] | None = None,
    metadata: Mapping | None = None,
) -> ConceptFingerprintSet:
    """Create conditional concept signatures that are comparable across panels.

    ``intervention_effects`` may be either ``concept x common_marker`` or
    ``cell x concept x common_marker``.  The latter is averaged within each
    declared cell type.  Missing/undersized strata receive zero fingerprints
    and zero activation frequency rather than borrowing information from test
    data or a different cell type.
    """

    concepts = _matrix(activations, "activations")
    common = _matrix(common_values, "common_values")
    labels = np.asarray(cell_types).astype(str)
    if concepts.shape[0] != common.shape[0] or labels.shape != (concepts.shape[0],):
        raise ValueError("activations, common_values, and cell_types are not aligned")
    markers = tuple(map(str, common_markers))
    if common.shape[1] != len(markers):
        raise ValueError("common_markers do not align with common_values")
    types = tuple(map(str, cell_type_order))
    if not types or len(set(types)) != len(types):
        raise ValueError("cell_type_order must contain unique declared strata")
    if int(minimum_cells) < 2:
        raise ValueError("minimum_cells must be at least two")

    n_concepts = concepts.shape[1]
    identifiers = (
        tuple(f"concept_{index:04d}" for index in range(n_concepts))
        if concept_ids is None
        else tuple(map(str, concept_ids))
    )
    if len(identifiers) != n_concepts or len(set(identifiers)) != n_concepts:
        raise ValueError("concept_ids must be unique and align with activation columns")

    intervention = None
    if intervention_effects is not None:
        intervention = np.asarray(intervention_effects, dtype=np.float64)
        valid_shapes = {
            (n_concepts, common.shape[1]),
            (concepts.shape[0], n_concepts, common.shape[1]),
        }
        if intervention.shape not in valid_shapes or not np.isfinite(intervention).all():
            raise ValueError("intervention_effects has an invalid shape or non-finite values")

    blocks: dict[str, tuple[int, int]] = {}
    pieces: list[np.ndarray] = []
    cursor = 0
    frequency = np.zeros((n_concepts, len(types)), dtype=np.float64)

    if intervention is not None:
        effect_pieces: list[np.ndarray] = []
        if intervention.ndim == 2:
            effect_pieces.append(intervention)
        else:
            for label in types:
                rows = labels == label
                if rows.sum() >= int(minimum_cells):
                    effect_pieces.append(intervention[rows].mean(axis=0))
                else:
                    effect_pieces.append(np.zeros((n_concepts, common.shape[1])))
        effect_block = np.concatenate(effect_pieces, axis=1)
        pieces.append(effect_block)
        blocks["intervention"] = (cursor, cursor + effect_block.shape[1])
        cursor += effect_block.shape[1]

    correlation_pieces: list[np.ndarray] = []
    for type_index, label in enumerate(types):
        rows = labels == label
        if rows.sum() >= int(minimum_cells):
            correlation_pieces.append(_safe_correlations(concepts[rows], common[rows]))
            frequency[:, type_index] = np.mean(
                concepts[rows] > float(activation_threshold), axis=0
            )
        else:
            correlation_pieces.append(np.zeros((n_concepts, common.shape[1])))
    correlation_block = np.concatenate(correlation_pieces, axis=1)
    pieces.append(correlation_block)
    blocks["conditional_correlation"] = (
        cursor,
        cursor + correlation_block.shape[1],
    )
    cursor += correlation_block.shape[1]
    pieces.append(frequency)
    blocks["activation_frequency"] = (cursor, cursor + frequency.shape[1])

    result = ConceptFingerprintSet(
        values=np.asarray(np.concatenate(pieces, axis=1), dtype=np.float32),
        activation_frequency=np.asarray(frequency, dtype=np.float32),
        concept_ids=identifiers,
        common_markers=markers,
        cell_types=types,
        blocks=blocks,
        metadata={} if metadata is None else dict(metadata),
    )
    result.validate()
    return result


def _row_normalized(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    output = np.zeros_like(values, dtype=np.float64)
    np.divide(values, norm, out=output, where=norm > 1.0e-12)
    return output


def fingerprint_cosine_similarity(
    source: ConceptFingerprintSet,
    target: ConceptFingerprintSet,
    *,
    block_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Compare concepts after separately normalizing each observable block."""

    source.validate()
    target.validate()
    if source.common_markers != target.common_markers:
        raise ValueError("Source and target common-marker contracts differ")
    if source.cell_types != target.cell_types or source.blocks != target.blocks:
        raise ValueError("Source and target fingerprint layouts differ")
    weights = {} if block_weights is None else dict(block_weights)
    source_pieces: list[np.ndarray] = []
    target_pieces: list[np.ndarray] = []
    for name, (start, stop) in source.blocks.items():
        weight = float(weights.get(name, 1.0))
        if weight < 0:
            raise ValueError("Fingerprint block weights must be non-negative")
        if weight == 0:
            continue
        scale = np.sqrt(weight)
        source_pieces.append(scale * _row_normalized(source.values[:, start:stop]))
        target_pieces.append(scale * _row_normalized(target.values[:, start:stop]))
    if not source_pieces:
        raise ValueError("At least one fingerprint block must have positive weight")
    source_vector = _row_normalized(np.concatenate(source_pieces, axis=1))
    target_vector = _row_normalized(np.concatenate(target_pieces, axis=1))
    return np.asarray(source_vector @ target_vector.T, dtype=np.float32)
