"""Efficient clinical10-to-H19 single-marker add-back screening."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.benchmark.self_recoverability_cache import load_cached_specimen
from src.data.markers import canonical_marker_name
from src.data.splits import patient_id_from_specimen
from src.models.gpu_knn import TorchKNNMedianRegressor
from src.training.self_recoverability import (
    PLATFORM_FINE_CLASSES,
    _aligned_probabilities,
    _atomic_json,
    _class_patient_balanced_rows,
    _classifier,
    _digest,
    _hardware,
    _load_patient_balanced_training,
    _patient_balanced_bank,
    _robust_location_scale,
    _scale,
    biology_metrics,
    marker_metrics,
    panel_indices,
)


def addback_panel_definitions(
    config: Mapping,
    full_markers: Sequence[str],
    *,
    modality: str | None = None,
) -> tuple[
    dict[str, dict],
    np.ndarray,
    tuple[str, ...],
]:
    """Resolve base, single-add-back, and upper-anchor panels.

    The returned target indices are fixed to ``full - upper`` for every panel,
    so candidate effects are not confounded by a changing target set.
    """

    sweep = config["sweep"]
    base_name = str(sweep["base_panel"])
    upper_name = str(sweep["upper_panel"])
    base_markers = tuple(config["panels"][base_name]["markers"])
    upper_markers = tuple(config["panels"][upper_name]["markers"])
    _, target_indices, _, target_markers = panel_indices(
        full_markers,
        upper_markers,
    )

    base_canonical = tuple(canonical_marker_name(x) for x in base_markers)
    upper_canonical = tuple(canonical_marker_name(x) for x in upper_markers)
    if not set(base_canonical).issubset(upper_canonical):
        raise ValueError("The add-back base panel must be a subset of upper")
    expected_candidates = set(upper_canonical) - set(base_canonical)

    configured_candidates = {
        str(name): str(marker)
        for name, marker in sweep["addback_panels"].items()
    }
    candidate_canonical = {
        name: canonical_marker_name(marker)
        for name, marker in configured_candidates.items()
    }
    if set(candidate_canonical.values()) != expected_candidates:
        raise ValueError(
            "Configured add-back markers must equal upper minus base: "
            f"expected={sorted(expected_candidates)}, "
            f"configured={sorted(candidate_canonical.values())}"
        )
    if len(candidate_canonical.values()) != len(
        set(candidate_canonical.values())
    ):
        raise ValueError("Configured add-back markers are not unique")

    definitions: dict[str, dict] = {}

    def add_definition(
        name: str,
        markers: Sequence[str],
        *,
        added_marker: str | None,
        kind: str,
    ) -> None:
        observed_indices, hidden_indices, observed_names, hidden_names = (
            panel_indices(full_markers, markers)
        )
        definitions[name] = {
            "kind": kind,
            "added_marker": added_marker,
            "observed_indices": observed_indices,
            "hidden_indices": hidden_indices,
            "observed_markers": observed_names,
            "hidden_markers": hidden_names,
        }

    add_definition(
        base_name,
        base_markers,
        added_marker=None,
        kind="base",
    )
    if bool(sweep.get("include_single_addbacks", True)):
        for name, marker in configured_candidates.items():
            add_definition(
                name,
                (*base_markers, marker),
                added_marker=marker,
                kind="single_addback",
            )

    combination_by_modality = sweep.get("combination_panels", {})
    if combination_by_modality:
        if modality is None:
            raise ValueError(
                "modality is required when combination panels are configured"
            )
        combinations = combination_by_modality.get(str(modality), {})
        for name, markers in combinations.items():
            markers = tuple(map(str, markers))
            canonical = tuple(canonical_marker_name(x) for x in markers)
            if len(canonical) < 2 or len(set(canonical)) != len(canonical):
                raise ValueError(
                    f"Combination panel {name} must contain unique markers"
                )
            if not set(canonical).issubset(expected_candidates):
                raise ValueError(
                    f"Combination panel {name} contains non-candidate markers"
                )
            add_definition(
                str(name),
                (*base_markers, *markers),
                added_marker="+".join(markers),
                kind="targeted_pair",
            )

    custom_by_modality = sweep.get("custom_panels", {})
    if custom_by_modality:
        if modality is None:
            raise ValueError(
                "modality is required when custom panels are configured"
            )
        custom_panels = custom_by_modality.get(str(modality), {})
        for name, payload in custom_panels.items():
            operation = str(payload["operation"])
            markers = tuple(map(str, payload["markers"]))
            canonical = tuple(canonical_marker_name(x) for x in markers)
            if not canonical or len(set(canonical)) != len(canonical):
                raise ValueError(
                    f"Custom panel {name} must contain unique markers"
                )
            if operation == "add_to_base":
                if not set(canonical).issubset(expected_candidates):
                    raise ValueError(
                        f"Custom panel {name} contains non-candidate additions"
                    )
                panel_markers = (*base_markers, *markers)
                change = "+".join(markers)
            elif operation == "remove_from_upper":
                if not set(canonical).issubset(set(upper_canonical)):
                    raise ValueError(
                        f"Custom panel {name} removes unavailable markers"
                    )
                removed = set(canonical)
                panel_markers = tuple(
                    marker
                    for marker in upper_markers
                    if canonical_marker_name(marker) not in removed
                )
                change = "REMOVE:" + "+".join(markers)
            else:
                raise ValueError(
                    f"Unsupported custom panel operation: {operation}"
                )
            add_definition(
                str(name),
                panel_markers,
                added_marker=change,
                kind=str(payload["kind"]),
            )
    add_definition(
        upper_name,
        upper_markers,
        added_marker=None,
        kind="upper",
    )
    return definitions, target_indices, target_markers


def _fixed_full_biology(
    model,
    values: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
) -> dict:
    predicted, probabilities = _aligned_probabilities(
        model,
        values,
        classes,
    )
    return biology_metrics(
        labels,
        predicted,
        probabilities,
        classes,
    )


def _hybrid_biology(
    model,
    hybrid: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
) -> dict:
    predicted, probabilities = _aligned_probabilities(
        model,
        hybrid,
        classes,
    )
    return biology_metrics(
        labels,
        predicted,
        probabilities,
        classes,
    )


def run_addback_sweep(
    config: dict,
    *,
    modality: str,
    fold_index: int,
    seed: int,
) -> dict:
    """Run all single-marker add-back panels for one modality and fold."""

    import torch

    started = time.time()
    modality = str(modality)
    if modality not in PLATFORM_FINE_CLASSES:
        raise ValueError(f"Unsupported modality: {modality}")
    device = str(config["training"]["device"])
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    hardware = _hardware(device)

    split_path = Path(config["data"]["split_manifest"])
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    fold = next(
        (
            current
            for current in split_manifest["folds"]
            if int(current["fold_index"]) == int(fold_index)
        ),
        None,
    )
    if fold is None:
        raise ValueError(f"Fold {fold_index} is unavailable")
    fit_specimens = tuple(
        list(fold["train_specimens"]) + list(fold["validation_specimens"])
    )
    test_specimens = tuple(fold["test_specimens"])
    fit_values, fit_labels, fit_patients, full_markers = (
        _load_patient_balanced_training(
            cache_root=config["data"]["cache_root"],
            modality=modality,
            specimens=fit_specimens,
            cells_per_patient=int(config["training"]["cells_per_fit_patient"]),
            seed=int(config["training"]["fit_sample_seed"]),
        )
    )
    panel_definitions, target_indices, target_markers = (
        addback_panel_definitions(
            config,
            full_markers,
            modality=modality,
        )
    )
    full_median, full_scale = _robust_location_scale(fit_values)

    knn_config = config["training"]["knn"]
    bank_rows = _patient_balanced_bank(
        fit_values,
        fit_patients,
        maximum=int(knn_config["max_reference_cells"]),
        seed=int(config["training"]["fit_sample_seed"]),
    )

    classes = PLATFORM_FINE_CLASSES[modality]
    unknown_fit = sorted(set(np.unique(fit_labels)) - set(classes))
    if unknown_fit:
        raise ValueError(f"Unexpected {modality} training labels: {unknown_fit}")
    classifier_config = config["classifier"]
    classifier_rows = _class_patient_balanced_rows(
        fit_labels,
        fit_patients,
        classes,
        maximum_per_class=int(classifier_config["maximum_cells_per_class"]),
        seed=int(classifier_config["sample_seed"]),
    )
    full_classifier = _classifier(
        fit_values[classifier_rows],
        fit_labels[classifier_rows],
        c_value=float(classifier_config["c"]),
        maximum_iterations=int(classifier_config["max_iter"]),
        seed=int(classifier_config["seed"]),
    )

    test_payloads = {}
    for specimen in test_specimens:
        cached = load_cached_specimen(
            config["data"]["cache_root"],
            modality,
            specimen,
        )
        if cached["markers"] != full_markers:
            raise ValueError(f"Test marker mismatch for {modality}/{specimen}")
        labels = cached["labels"]
        unknown_test = sorted(set(np.unique(labels)) - set(classes))
        if unknown_test:
            raise ValueError(f"Unexpected {modality} test labels: {unknown_test}")
        test_payloads[specimen] = {
            "values": cached["values"],
            "labels": labels,
            "full_true": _fixed_full_biology(
                full_classifier,
                cached["values"],
                labels,
                classes,
            ),
        }

    marker_rows = []
    biology_by_panel = {}
    panel_timings = {}
    for panel_name, definition in panel_definitions.items():
        panel_started = time.time()
        observed_indices = definition["observed_indices"]
        hidden_indices = definition["hidden_indices"]
        observed_fit = fit_values[:, observed_indices]
        observed_location, observed_scale = _robust_location_scale(
            observed_fit
        )
        scaled_observed_fit = _scale(
            observed_fit,
            observed_location,
            observed_scale,
        )
        knn = TorchKNNMedianRegressor(
            k=int(knn_config["k"]),
            device=device,
            query_chunk_size=knn_config.get("query_chunk_size"),
            distance_memory_fraction=float(
                knn_config.get("distance_memory_fraction", 0.08)
            ),
        ).fit(
            scaled_observed_fit[bank_rows],
            fit_values[bank_rows],
        )
        panel_specimens = {}
        for specimen, payload in test_payloads.items():
            values = payload["values"]
            labels = payload["labels"]
            scaled_observed = _scale(
                values[:, observed_indices],
                observed_location,
                observed_scale,
            )
            predicted_full = knn.predict(scaled_observed)
            median_full = np.repeat(
                full_median[None, :],
                len(values),
                axis=0,
            ).astype(np.float32)
            median_hybrid = np.asarray(values, dtype=np.float32).copy()
            median_hybrid[:, hidden_indices] = median_full[:, hidden_indices]
            knn_hybrid = np.asarray(values, dtype=np.float32).copy()
            knn_hybrid[:, hidden_indices] = predicted_full[:, hidden_indices]

            prediction_by_name = {
                "median": median_full[:, target_indices],
                "knn": predicted_full[:, target_indices],
            }
            for representation, prediction in prediction_by_name.items():
                current_rows = marker_metrics(
                    values[:, target_indices],
                    prediction,
                    median_full[:, target_indices],
                    full_scale[target_indices],
                    target_markers,
                )
                for row in current_rows:
                    row.update(
                        {
                            "modality": modality,
                            "panel": panel_name,
                            "panel_kind": definition["kind"],
                            "added_marker": definition["added_marker"],
                            "fold": int(fold_index),
                            "seed": int(seed),
                            "patient": patient_id_from_specimen(specimen),
                            "specimen": specimen,
                            "representation": representation,
                            "n_cells": int(len(values)),
                        }
                    )
                marker_rows.extend(current_rows)

            panel_specimens[specimen] = {
                "patient": patient_id_from_specimen(specimen),
                "n_cells": int(len(values)),
                "metrics": {
                    "full_true": payload["full_true"],
                    "full_hybrid_median": _hybrid_biology(
                        full_classifier,
                        median_hybrid,
                        labels,
                        classes,
                    ),
                    "full_hybrid_knn": _hybrid_biology(
                        full_classifier,
                        knn_hybrid,
                        labels,
                        classes,
                    ),
                },
            }
        biology_by_panel[panel_name] = {
            "kind": definition["kind"],
            "added_marker": definition["added_marker"],
            "observed_markers": list(definition["observed_markers"]),
            "hidden_markers": list(definition["hidden_markers"]),
            "specimens": panel_specimens,
        }
        panel_timings[panel_name] = time.time() - panel_started
        print(
            json.dumps(
                {
                    "event": "panel_complete",
                    "modality": modality,
                    "fold": int(fold_index),
                    "panel": panel_name,
                    "elapsed_seconds": panel_timings[panel_name],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del knn
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = (
        Path(config["output"]["root"])
        / str(config["experiment"]["name"])
        / modality
        / f"fold_{int(fold_index)}"
        / f"seed_{int(seed)}"
    )
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / "marker_metrics.csv"
    pd.DataFrame(marker_rows).to_csv(marker_path, index=False)
    biology_path = output / "biology_metrics.json"
    _atomic_json(
        {
            "classes": list(classes),
            "classifier": {
                "type": "fixed_class_balanced_multinomial_logistic_regression",
                "trained_on": "outer_fit_true_full_markers_and_existing_labels",
                "representation_specific_refit": False,
                "training_cells": int(len(classifier_rows)),
            },
            "primary_target_markers": list(target_markers),
            "panels": biology_by_panel,
        },
        biology_path,
    )
    result = {
        "status": "ok",
        "experiment": str(config["experiment"]["name"]),
        "modality": modality,
        "fold": int(fold_index),
        "seed": int(seed),
        "claim_scope": "processed_upstream_pregated_conditional_sensitivity",
        "split_manifest": str(split_path),
        "split_manifest_sha256": _digest(split_path),
        "fit_patients": int(len(np.unique(fit_patients))),
        "fit_cells": int(len(fit_values)),
        "test_patients": int(len(fold["test_patients"])),
        "test_specimens": int(len(test_specimens)),
        "primary_target_markers": list(target_markers),
        "knn_reference_cells": int(len(bank_rows)),
        "panels": {
            name: {
                "kind": definition["kind"],
                "added_marker": definition["added_marker"],
                "observed_markers": list(definition["observed_markers"]),
                "hidden_markers": list(definition["hidden_markers"]),
            }
            for name, definition in panel_definitions.items()
        },
        "panel_timings_seconds": panel_timings,
        "hardware": hardware,
        "artifacts": {
            "marker_metrics": str(marker_path),
            "biology_metrics": str(biology_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    result_path = output / "run_summary.json"
    result["artifacts"]["run_summary"] = str(result_path)
    _atomic_json(result, result_path)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        result["hardware"]["peak_gpu_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        _atomic_json(result, result_path)
        torch.cuda.empty_cache()
    return result
