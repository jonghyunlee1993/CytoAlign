"""Matched-versus-wrong-patient distribution evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import wasserstein_distance

from src.benchmark.artifacts import LockedFeatureScales


def _validate_unique_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    identifiers = tuple(values)
    if len(identifiers) < 2:
        raise ValueError(f"At least two {name} are required")
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError(f"{name} must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be unique")
    return tuple(sorted(identifiers))


def patient_derangement(
    patient_ids: Sequence[str], *, random_state: int
) -> dict[str, str]:
    """Return a deterministic patient permutation with no self matches."""

    patients = np.asarray(
        _validate_unique_ids(patient_ids, "patient IDs"),
        dtype=object,
    )
    rng = np.random.RandomState(int(random_state))
    order = patients[rng.permutation(patients.size)]
    shift = int(rng.randint(1, patients.size))
    wrong = np.roll(order, shift)
    return {str(patient): str(other) for patient, other in zip(order, wrong)}


def derangements_digest(derangements: Sequence[Mapping[str, str]]) -> str:
    """Return a stable digest for a frozen wrong-patient mapping manifest."""

    canonical = [
        {str(key): str(value) for key, value in sorted(mapping.items())}
        for mapping in derangements
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validated_specimen_matrices(
    values: Mapping[str, np.ndarray],
    *,
    name: str,
    n_markers: int,
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for specimen, matrix in values.items():
        if not isinstance(specimen, str) or not specimen.strip():
            raise ValueError(f"{name} specimen keys must be non-empty strings")
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != n_markers:
            raise ValueError(
                f"{name}[{specimen!r}] must be a non-empty matrix with "
                f"{n_markers} markers"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name}[{specimen!r}] contains non-finite values")
        matrices[specimen] = array
    if not matrices:
        raise ValueError(f"{name} must contain at least one specimen")
    return matrices


def _patient_mixtures(
    matrices: Mapping[str, np.ndarray],
    patient_by_specimen: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    specimens_by_patient: dict[str, list[str]] = {}
    for specimen in sorted(matrices):
        patient = patient_by_specimen[specimen]
        specimens_by_patient.setdefault(patient, []).append(specimen)

    patient_values = {}
    patient_weights = {}
    specimen_counts = {}
    for patient, specimens in sorted(specimens_by_patient.items()):
        specimen_count = len(specimens)
        arrays = [matrices[specimen] for specimen in specimens]
        weights = [
            np.full(
                array.shape[0],
                1.0 / (specimen_count * array.shape[0]),
                dtype=np.float64,
            )
            for array in arrays
        ]
        patient_values[patient] = np.concatenate(arrays, axis=0)
        patient_weights[patient] = np.concatenate(weights)
        specimen_counts[patient] = specimen_count
    return patient_values, patient_weights, specimen_counts


def _normalized_marker_wasserstein(
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_weights: np.ndarray,
    target_weights: np.ndarray,
    scales: np.ndarray,
) -> tuple[float, np.ndarray]:
    errors = np.asarray(
        [
            wasserstein_distance(
                prediction[:, marker],
                target[:, marker],
                u_weights=prediction_weights,
                v_weights=target_weights,
            )
            / scales[marker]
            for marker in range(scales.size)
        ],
        dtype=np.float64,
    )
    return float(errors.mean()), errors


def _validate_derangements(
    derangements: Sequence[Mapping[str, str]],
    patients: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    expected = set(patients)
    validated = []
    if not derangements:
        raise ValueError("At least one frozen derangement is required")
    for index, mapping in enumerate(derangements):
        normalized = {str(key): str(value) for key, value in mapping.items()}
        if set(normalized) != expected or set(normalized.values()) != expected:
            raise ValueError(f"Derangement {index} is not a patient bijection")
        if any(patient == wrong for patient, wrong in normalized.items()):
            raise ValueError(f"Derangement {index} contains a self match")
        validated.append(normalized)
    return tuple(validated)


def evaluate_matched_patient_gain(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    patient_by_specimen: Mapping[str, str],
    feature_scales: LockedFeatureScales,
    *,
    marker_names: Sequence[str],
    derangements: Sequence[Mapping[str, str]],
    derangement_manifest_digest: str,
) -> dict:
    """Evaluate patient-specific signal with equal-specimen patient mixtures."""

    if not isinstance(feature_scales, LockedFeatureScales):
        raise TypeError("feature_scales must be a LockedFeatureScales artifact")
    markers = tuple(marker_names)
    if markers != feature_scales.marker_names:
        raise ValueError("marker_names do not match the locked scale artifact")

    predicted = _validated_specimen_matrices(
        predictions,
        name="predictions",
        n_markers=len(markers),
    )
    observed = _validated_specimen_matrices(
        targets,
        name="targets",
        n_markers=len(markers),
    )
    if set(predicted) != set(observed):
        missing_predictions = sorted(set(observed) - set(predicted))
        missing_targets = sorted(set(predicted) - set(observed))
        raise ValueError(
            "Prediction and target specimen keys differ: "
            f"missing_predictions={missing_predictions}, "
            f"missing_targets={missing_targets}"
        )
    if set(patient_by_specimen) != set(predicted):
        raise ValueError("patient_by_specimen keys must exactly match specimen keys")
    if any(
        not isinstance(patient, str) or not patient.strip()
        for patient in patient_by_specimen.values()
    ):
        raise ValueError("Patient IDs must be non-empty strings")

    prediction_values, prediction_weights, specimen_counts = _patient_mixtures(
        predicted, patient_by_specimen
    )
    target_values, target_weights, target_specimen_counts = _patient_mixtures(
        observed, patient_by_specimen
    )
    patients = _validate_unique_ids(tuple(prediction_values), "patients")
    frozen_derangements = _validate_derangements(derangements, patients)
    actual_digest = derangements_digest(frozen_derangements)
    if actual_digest != derangement_manifest_digest:
        raise ValueError("Derangement manifest digest does not match its content")

    matched: dict[str, float] = {}
    matched_by_marker: dict[str, np.ndarray] = {}
    for patient in patients:
        matched[patient], matched_by_marker[patient] = (
            _normalized_marker_wasserstein(
                prediction_values[patient],
                target_values[patient],
                prediction_weights[patient],
                target_weights[patient],
                feature_scales.values,
            )
        )

    wrong_values = {patient: [] for patient in patients}
    wrong_marker_values = {patient: [] for patient in patients}
    per_derangement = []
    for mapping in frozen_derangements:
        scalar_errors = []
        marker_errors = []
        for patient in patients:
            wrong_patient = mapping[patient]
            scalar, per_marker = _normalized_marker_wasserstein(
                prediction_values[patient],
                target_values[wrong_patient],
                prediction_weights[patient],
                target_weights[wrong_patient],
                feature_scales.values,
            )
            wrong_values[patient].append(scalar)
            wrong_marker_values[patient].append(per_marker)
            scalar_errors.append(scalar)
            marker_errors.append(per_marker)
        per_derangement.append(
            {
                "wrong_patient_error": float(np.mean(scalar_errors)),
                "wrong_patient_error_by_marker": np.mean(
                    np.stack(marker_errors), axis=0
                ).tolist(),
            }
        )

    per_patient = {}
    for patient in patients:
        wrong = float(np.mean(wrong_values[patient]))
        wrong_by_marker = np.mean(np.stack(wrong_marker_values[patient]), axis=0)
        per_patient[patient] = {
            "matched_error": matched[patient],
            "wrong_patient_error": wrong,
            "individualization_gain": wrong - matched[patient],
            "matched_error_by_marker": matched_by_marker[patient].tolist(),
            "wrong_patient_error_by_marker": wrong_by_marker.tolist(),
            "individualization_gain_by_marker": (
                wrong_by_marker - matched_by_marker[patient]
            ).tolist(),
            "prediction_specimen_count": specimen_counts[patient],
            "target_specimen_count": target_specimen_counts[patient],
        }

    matched_mean = float(np.mean([per_patient[p]["matched_error"] for p in patients]))
    wrong_mean = float(
        np.mean([per_patient[p]["wrong_patient_error"] for p in patients])
    )
    return {
        "n_patients": len(patients),
        "n_markers": len(markers),
        "n_derangements": len(frozen_derangements),
        "scale_id": feature_scales.scale_id,
        "derangement_manifest_digest": actual_digest,
        "matched_error": matched_mean,
        "wrong_patient_error": wrong_mean,
        "individualization_gain": wrong_mean - matched_mean,
        "per_patient": per_patient,
        "per_derangement": per_derangement,
    }
