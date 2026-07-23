"""Sparse, auditable mappings between source and target concept dictionaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConceptGraph:
    """Row-sparse source-to-target concept map with explicit unmatched mass."""

    weights: np.ndarray
    unmatched_source_mass: np.ndarray
    source_concept_ids: tuple[str, ...]
    target_concept_ids: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        weights = np.asarray(self.weights)
        unmatched = np.asarray(self.unmatched_source_mass)
        expected = (len(self.source_concept_ids), len(self.target_concept_ids))
        if weights.shape != expected or unmatched.shape != (expected[0],):
            raise ValueError("Concept graph arrays do not match concept identifiers")
        if len(set(self.source_concept_ids)) != expected[0] or len(
            set(self.target_concept_ids)
        ) != expected[1]:
            raise ValueError("Concept graph identifiers must be unique")
        if not np.isfinite(weights).all() or not np.isfinite(unmatched).all():
            raise ValueError("Concept graph contains non-finite values")
        if np.any(weights < -1.0e-8) or np.any(unmatched < -1.0e-8):
            raise ValueError("Concept graph weights must be non-negative")
        total_mass = weights.sum(axis=1) + unmatched
        if not np.allclose(total_mass, np.ones(expected[0]), atol=1.0e-5):
            raise ValueError("Each source concept row must have total mass one")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        description = {
            "source_concept_ids": list(self.source_concept_ids),
            "target_concept_ids": list(self.target_concept_ids),
            "metadata": self.metadata,
        }
        np.savez_compressed(
            output,
            weights=np.asarray(self.weights, dtype=np.float32),
            unmatched_source_mass=np.asarray(
                self.unmatched_source_mass, dtype=np.float32
            ),
            description_json=np.asarray(json.dumps(description, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConceptGraph":
        with np.load(Path(path), allow_pickle=False) as archive:
            description = json.loads(str(archive["description_json"]))
            result = cls(
                weights=archive["weights"],
                unmatched_source_mass=archive["unmatched_source_mass"],
                source_concept_ids=tuple(description["source_concept_ids"]),
                target_concept_ids=tuple(description["target_concept_ids"]),
                metadata=description["metadata"],
            )
        result.validate()
        return result


def build_sparse_concept_graph(
    similarity: np.ndarray,
    *,
    source_concept_ids: Sequence[str] | None = None,
    target_concept_ids: Sequence[str] | None = None,
    minimum_similarity: float = 0.25,
    maximum_targets_per_source: int = 2,
    temperature: float = 0.1,
    metadata: dict | None = None,
) -> ConceptGraph:
    """Convert similarities into a many-to-one/one-to-many sparse soft graph.

    Concepts with no edge above ``minimum_similarity`` retain unit unmatched
    mass.  This is intentionally conservative: target-private biology is not
    forced into an unsupported cross-panel correspondence.
    """

    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("similarity must be a non-empty two-dimensional matrix")
    if not np.isfinite(scores).all():
        raise ValueError("similarity contains non-finite values")
    if int(maximum_targets_per_source) < 1:
        raise ValueError("maximum_targets_per_source must be positive")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    source_ids = (
        tuple(f"source_{index:04d}" for index in range(scores.shape[0]))
        if source_concept_ids is None
        else tuple(map(str, source_concept_ids))
    )
    target_ids = (
        tuple(f"target_{index:04d}" for index in range(scores.shape[1]))
        if target_concept_ids is None
        else tuple(map(str, target_concept_ids))
    )
    if len(source_ids) != scores.shape[0] or len(target_ids) != scores.shape[1]:
        raise ValueError("Concept identifiers do not align with similarity")

    weights = np.zeros_like(scores, dtype=np.float64)
    unmatched = np.ones(scores.shape[0], dtype=np.float64)
    maximum = min(int(maximum_targets_per_source), scores.shape[1])
    for source_index in range(scores.shape[0]):
        candidates = np.flatnonzero(scores[source_index] >= float(minimum_similarity))
        if candidates.size == 0:
            continue
        order = candidates[np.argsort(scores[source_index, candidates])[::-1][:maximum]]
        logits = scores[source_index, order] / float(temperature)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        weights[source_index, order] = probabilities
        unmatched[source_index] = 0.0
    result = ConceptGraph(
        weights=np.asarray(weights, dtype=np.float32),
        unmatched_source_mass=np.asarray(unmatched, dtype=np.float32),
        source_concept_ids=source_ids,
        target_concept_ids=target_ids,
        metadata={} if metadata is None else dict(metadata),
    )
    result.validate()
    return result


def build_mutual_nearest_concept_graph(
    similarity: np.ndarray,
    *,
    source_concept_ids: Sequence[str] | None = None,
    target_concept_ids: Sequence[str] | None = None,
    minimum_similarity: float,
    minimum_margin: float = 0.0,
    metadata: dict | None = None,
) -> ConceptGraph:
    """Build a conservative one-to-one graph from mutual best matches.

    A source concept is connected only when its best target also chooses that
    source, the similarity passes a calibrated cutoff, and the source's
    best-versus-second-best margin passes ``minimum_margin``. All other source
    concepts retain explicit unit unmatched mass.
    """

    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("similarity must be a non-empty finite matrix")
    if not np.isfinite(float(minimum_similarity)):
        raise ValueError("minimum_similarity must be finite")
    if float(minimum_margin) < 0 or not np.isfinite(float(minimum_margin)):
        raise ValueError("minimum_margin must be finite and non-negative")
    source_ids = (
        tuple(f"source_{index:04d}" for index in range(scores.shape[0]))
        if source_concept_ids is None
        else tuple(map(str, source_concept_ids))
    )
    target_ids = (
        tuple(f"target_{index:04d}" for index in range(scores.shape[1]))
        if target_concept_ids is None
        else tuple(map(str, target_concept_ids))
    )
    if len(source_ids) != scores.shape[0] or len(target_ids) != scores.shape[1]:
        raise ValueError("Concept identifiers do not align with similarity")

    source_best = np.argmax(scores, axis=1)
    target_best = np.argmax(scores, axis=0)
    if scores.shape[1] == 1:
        margins = np.full(scores.shape[0], np.inf)
    else:
        ordered = np.partition(scores, -2, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
    weights = np.zeros_like(scores, dtype=np.float32)
    unmatched = np.ones(scores.shape[0], dtype=np.float32)
    for source_index, target_index in enumerate(source_best):
        if (
            target_best[target_index] == source_index
            and scores[source_index, target_index] >= float(minimum_similarity)
            and margins[source_index] >= float(minimum_margin)
        ):
            weights[source_index, target_index] = 1.0
            unmatched[source_index] = 0.0
    graph_metadata = {} if metadata is None else dict(metadata)
    graph_metadata.update(
        {
            "matching": "mutual_nearest",
            "minimum_similarity": float(minimum_similarity),
            "minimum_margin": float(minimum_margin),
        }
    )
    result = ConceptGraph(
        weights=weights,
        unmatched_source_mass=unmatched,
        source_concept_ids=source_ids,
        target_concept_ids=target_ids,
        metadata=graph_metadata,
    )
    result.validate()
    return result


def map_concept_activations(
    source_activations: np.ndarray, graph: ConceptGraph
) -> np.ndarray:
    """Apply a source-to-target concept graph to a batch of activations."""

    graph.validate()
    values = np.asarray(source_activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != graph.weights.shape[0]:
        raise ValueError("source_activations do not match the concept graph")
    return np.asarray(values @ graph.weights, dtype=np.float32)


def shuffle_concept_graph(graph: ConceptGraph, *, random_state: int) -> ConceptGraph:
    """Permute target semantics while preserving row sparsity and edge weights."""

    graph.validate()
    rng = np.random.RandomState(int(random_state))
    permutation = rng.permutation(graph.weights.shape[1])
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "control": "shuffled_target_concepts",
            "shuffle_seed": int(random_state),
            "target_permutation": permutation.tolist(),
        }
    )
    result = ConceptGraph(
        weights=np.asarray(graph.weights[:, permutation], dtype=np.float32),
        unmatched_source_mass=np.asarray(
            graph.unmatched_source_mass, dtype=np.float32
        ).copy(),
        source_concept_ids=graph.source_concept_ids,
        target_concept_ids=graph.target_concept_ids,
        metadata=metadata,
    )
    result.validate()
    return result
