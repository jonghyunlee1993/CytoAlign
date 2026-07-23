import numpy as np

from src.concepts.fingerprints import (
    ConceptFingerprintSet,
    build_concept_fingerprints,
    fingerprint_cosine_similarity,
)
from src.concepts.calibration import calibrate_semantic_axis_null
from src.concepts.matching import (
    ConceptGraph,
    build_mutual_nearest_concept_graph,
    build_sparse_concept_graph,
    map_concept_activations,
    shuffle_concept_graph,
)


def _fingerprints():
    rng = np.random.RandomState(12)
    source_activation = rng.normal(size=(300, 2))
    common = source_activation + rng.normal(scale=0.03, size=(300, 2))
    target_activation = source_activation[:, [1, 0]]
    labels = np.asarray(["T"] * 150 + ["B"] * 150)
    source = build_concept_fingerprints(
        source_activation,
        common,
        common_markers=("H1", "H2"),
        cell_types=labels,
        cell_type_order=("T", "B"),
        intervention_effects=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        concept_ids=("s0", "s1"),
    )
    target = build_concept_fingerprints(
        target_activation,
        common,
        common_markers=("H1", "H2"),
        cell_types=labels,
        cell_type_order=("T", "B"),
        intervention_effects=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        concept_ids=("t0", "t1"),
    )
    return source, target


def test_common_marker_fingerprints_recover_permuted_concept_semantics(tmp_path):
    source, target = _fingerprints()
    similarity = fingerprint_cosine_similarity(source, target)
    assert tuple(np.argmax(similarity, axis=1)) == (1, 0)

    path = tmp_path / "source_fingerprints.npz"
    source.save(path)
    restored = ConceptFingerprintSet.load(path)
    np.testing.assert_array_equal(restored.values, source.values)
    assert restored.blocks == source.blocks


def test_sparse_graph_maps_matched_concepts_and_preserves_unmatched_mass(tmp_path):
    similarity = np.asarray([[0.1, 0.9], [0.8, 0.2], [0.0, 0.1]])
    graph = build_sparse_concept_graph(
        similarity,
        source_concept_ids=("s0", "s1", "s2"),
        target_concept_ids=("t0", "t1"),
        minimum_similarity=0.5,
        maximum_targets_per_source=1,
    )
    expected = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(graph.weights, expected)
    np.testing.assert_array_equal(graph.unmatched_source_mass, [0.0, 0.0, 1.0])
    mapped = map_concept_activations(np.asarray([[2.0, 3.0, 100.0]]), graph)
    np.testing.assert_array_equal(mapped, [[3.0, 2.0]])

    path = tmp_path / "graph.npz"
    graph.save(path)
    restored = ConceptGraph.load(path)
    np.testing.assert_array_equal(restored.weights, graph.weights)
    shuffled = shuffle_concept_graph(graph, random_state=3)
    np.testing.assert_array_equal(
        np.sort(shuffled.weights, axis=1), np.sort(graph.weights, axis=1)
    )


def test_null_calibration_and_mutual_graph_reject_ambiguous_concepts():
    source, target = _fingerprints()
    calibration = calibrate_semantic_axis_null(
        source, target, n_permutations=20, quantile=0.8, random_state=7
    )
    assert calibration.null_best_scores.shape == (40,)
    assert np.isfinite(calibration.similarity_threshold)
    similarity = np.asarray(
        [[0.1, 0.95, 0.2], [0.92, 0.1, 0.2], [0.3, 0.31, 0.1]],
        dtype=np.float32,
    )
    graph = build_mutual_nearest_concept_graph(
        similarity,
        minimum_similarity=0.5,
        minimum_margin=0.1,
    )
    np.testing.assert_array_equal(
        graph.weights,
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(graph.unmatched_source_mass, [0.0, 0.0, 1.0])
