"""Run one literature marker-imputation baseline on the frozen same-cell split."""

from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.self_recoverability_cache import load_cached_specimen
from src.data.splits import patient_id_from_specimen
from src.models.literature_imputation import (
    predict_cycombine,
    predict_cytovi,
    predict_uvae,
)
from src.training.self_recoverability import (
    PLATFORM_FINE_CLASSES,
    _atomic_json,
    _class_patient_balanced_rows,
    _classifier,
    _digest,
    _hardware,
    _load_patient_balanced_training,
    _patient_balanced_bank,
    _representation_biology,
    _robust_location_scale,
    _scale,
    marker_metrics,
    panel_indices,
)


SUPPORTED_METHODS = ("cycombine", "cytovi", "uvae")


def _git_commit(path: str | Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_queries(
    *,
    cache_root: str | Path,
    modality: str,
    specimens: tuple[str, ...],
    expected_markers: tuple[str, ...],
    classes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int, int]]]:
    values = []
    labels = []
    samples = []
    offsets = []
    start = 0
    for specimen in specimens:
        cached = load_cached_specimen(cache_root, modality, specimen)
        if cached["markers"] != expected_markers:
            raise ValueError(f"Test marker mismatch for {modality}/{specimen}")
        unknown = sorted(set(np.unique(cached["labels"])) - set(classes))
        if unknown:
            raise ValueError(f"Unexpected {modality} test labels: {unknown}")
        stop = start + len(cached["values"])
        offsets.append((str(specimen), start, stop))
        values.append(cached["values"])
        labels.append(cached["labels"])
        samples.append(np.repeat(str(specimen), stop - start))
        start = stop
    return (
        np.concatenate(values).astype(np.float32, copy=False),
        np.concatenate(labels).astype(str),
        np.concatenate(samples).astype(str),
        offsets,
    )


def run_literature_baseline(
    config: dict,
    *,
    method: str,
    modality: str,
    panel_name: str,
    fold_index: int,
    seed: int,
) -> dict:
    """Fit and evaluate one transductive literature baseline."""

    import torch

    started = time.time()
    method = str(method).lower()
    modality = str(modality)
    panel_name = str(panel_name)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported literature baseline: {method}")
    if modality not in PLATFORM_FINE_CLASSES:
        raise ValueError(f"Unsupported modality: {modality}")
    panel_config = config["panels"].get(panel_name)
    if panel_config is None:
        raise ValueError(f"Unknown panel: {panel_name}")
    method_config = config["methods"][method]

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = str(method_config.get("device", config["training"]["device"]))
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
    test_specimens = tuple(map(str, fold["test_specimens"]))
    fit_values, fit_labels, fit_patients, full_markers = (
        _load_patient_balanced_training(
            cache_root=config["data"]["cache_root"],
            modality=modality,
            specimens=fit_specimens,
            cells_per_patient=int(config["training"]["cells_per_fit_patient"]),
            seed=int(config["training"]["fit_sample_seed"]),
        )
    )
    (
        observed_indices,
        hidden_indices,
        observed_markers,
        hidden_markers,
    ) = panel_indices(full_markers, panel_config["markers"])
    classes = PLATFORM_FINE_CLASSES[modality]
    unknown_fit = sorted(set(np.unique(fit_labels)) - set(classes))
    if unknown_fit:
        raise ValueError(f"Unexpected {modality} training labels: {unknown_fit}")

    full_location, full_scale = _robust_location_scale(fit_values)
    scaled_fit = _scale(fit_values, full_location, full_scale)
    reference_config = config["training"]["reference_bank"]
    bank_rows = _patient_balanced_bank(
        scaled_fit,
        fit_patients,
        maximum=int(reference_config["max_reference_cells"]),
        seed=int(reference_config["seed"]),
    )
    query_values, query_labels, query_samples, offsets = _load_queries(
        cache_root=config["data"]["cache_root"],
        modality=modality,
        specimens=test_specimens,
        expected_markers=full_markers,
        classes=classes,
    )
    scaled_query_observed = _scale(
        query_values[:, observed_indices],
        full_location[observed_indices],
        full_scale[observed_indices],
    )
    if method == "cycombine":
        scaled_prediction, method_metadata = predict_cycombine(
            reference_full=scaled_fit[bank_rows],
            query_observed=scaled_query_observed,
            full_markers=full_markers,
            observed_markers=observed_markers,
            hidden_markers=hidden_markers,
            fallback_hidden=np.zeros(len(hidden_indices), dtype=np.float32),
            script=method_config["script"],
            rscript=method_config["rscript"],
            seed=int(seed),
            xdim=int(method_config["xdim"]),
            ydim=int(method_config["ydim"]),
            rlen=int(method_config["rlen"]),
            minimum_reference_cells=int(
                method_config["minimum_reference_cells"]
            ),
            distance=str(method_config.get("distance", "sumofsquares")),
        )
    elif method == "uvae":
        scaled_prediction, method_metadata = predict_uvae(
            reference_full=scaled_fit[bank_rows],
            query_observed=scaled_query_observed,
            full_markers=full_markers,
            observed_markers=observed_markers,
            query_samples=query_samples,
            hidden_indices=hidden_indices,
            runner=method_config["runner"],
            python=method_config["python"],
            external_root=method_config["external_root"],
            seed=int(seed),
            epochs=int(method_config["epochs"]),
            batch_size=int(method_config["batch_size"]),
            latent_dim=int(method_config["latent_dim"]),
            hidden=int(method_config["hidden"]),
            width=int(method_config["width"]),
            pull=float(method_config["pull"]),
            early_stop_epochs=int(method_config["early_stop_epochs"]),
            samples_per_epoch=int(method_config["samples_per_epoch"]),
            max_query_training_cells=int(
                method_config["max_query_training_cells"]
            ),
        )
        method_metadata["external_commit"] = _git_commit(
            method_config["external_root"]
        )
    else:
        scaled_prediction, method_metadata = predict_cytovi(
            reference_full=scaled_fit[bank_rows],
            query_observed=scaled_query_observed,
            full_markers=full_markers,
            observed_markers=observed_markers,
            reference_samples=fit_patients[bank_rows],
            query_samples=query_samples,
            hidden_indices=hidden_indices,
            seed=int(seed),
            max_epochs=int(method_config["max_epochs"]),
            batch_size=int(method_config["batch_size"]),
            learning_rate=float(method_config["learning_rate"]),
            n_hidden=int(method_config["n_hidden"]),
            n_latent=int(method_config["n_latent"]),
            n_layers=int(method_config["n_layers"]),
            prior_mixture=bool(method_config["prior_mixture"]),
            early_stopping_patience=int(
                method_config["early_stopping_patience"]
            ),
            n_samples=int(method_config["n_samples"]),
            max_training_cells_per_epoch=int(
                method_config["max_training_cells_per_epoch"]
            ),
            n_epochs_kl_warmup=int(method_config["n_epochs_kl_warmup"]),
            max_query_training_cells=int(
                method_config["max_query_training_cells"]
            ),
        )

    if scaled_prediction.shape != (len(query_values), len(hidden_indices)):
        raise RuntimeError("Literature adapter returned an invalid prediction shape")
    prediction = (
        scaled_prediction * full_scale[hidden_indices][None, :]
        + full_location[hidden_indices][None, :]
    ).astype(np.float32)
    median_prediction = np.repeat(
        full_location[hidden_indices][None, :],
        len(query_values),
        axis=0,
    ).astype(np.float32)

    classifier_config = config["classifier"]
    classifier_rows = _class_patient_balanced_rows(
        fit_labels,
        fit_patients,
        classes,
        maximum_per_class=int(classifier_config["maximum_cells_per_class"]),
        seed=int(classifier_config["sample_seed"]),
    )
    classifier_kwargs = {
        "c_value": float(classifier_config["c"]),
        "maximum_iterations": int(classifier_config["max_iter"]),
        "seed": int(classifier_config["seed"]),
    }
    full_classifier = _classifier(
        fit_values[classifier_rows],
        fit_labels[classifier_rows],
        **classifier_kwargs,
    )
    hidden_classifier = _classifier(
        fit_values[classifier_rows][:, hidden_indices],
        fit_labels[classifier_rows],
        **classifier_kwargs,
    )
    observed_classifier = _classifier(
        fit_values[classifier_rows][:, observed_indices],
        fit_labels[classifier_rows],
        **classifier_kwargs,
    )

    marker_rows = []
    biology_by_specimen = {}
    representation = method
    for specimen, start, stop in offsets:
        current_values = query_values[start:stop]
        current_labels = query_labels[start:stop]
        current_prediction = prediction[start:stop]
        current_median = median_prediction[start:stop]
        rows = marker_metrics(
            current_values[:, hidden_indices],
            current_prediction,
            current_median,
            full_scale[hidden_indices],
            hidden_markers,
        )
        for row in rows:
            row.update(
                {
                    "modality": modality,
                    "panel": panel_name,
                    "fold": int(fold_index),
                    "seed": int(seed),
                    "patient": patient_id_from_specimen(specimen),
                    "specimen": specimen,
                    "representation": representation,
                    "n_cells": int(stop - start),
                }
            )
        marker_rows.extend(rows)
        biology_by_specimen[specimen] = {
            "patient": patient_id_from_specimen(specimen),
            "n_cells": int(stop - start),
            "metrics": _representation_biology(
                full_model=full_classifier,
                hidden_model=hidden_classifier,
                observed_model=observed_classifier,
                full_values=current_values,
                labels=current_labels,
                observed_indices=observed_indices,
                hidden_indices=hidden_indices,
                predictions={representation: current_prediction},
                classes=classes,
            ),
        }

    output = (
        Path(config["output"]["root"])
        / str(config["experiment"]["name"])
        / method
        / modality
        / panel_name
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
                "trained_on": "outer_fit_true_markers_and_existing_labels",
                "representation_specific_refit": False,
                "training_cells": int(len(classifier_rows)),
            },
            "method": method,
            "method_metadata": method_metadata,
            "specimens": biology_by_specimen,
        },
        biology_path,
    )
    result = {
        "status": "ok",
        "experiment": str(config["experiment"]["name"]),
        "method": method,
        "modality": modality,
        "panel": panel_name,
        "fold": int(fold_index),
        "seed": int(seed),
        "claim_scope": "processed_upstream_pregated_conditional_sensitivity",
        "information_access": {
            "fit_complete_markers": True,
            "query_shared_markers": True,
            "query_hidden_markers": False,
            "query_cell_labels": False,
            "mode": "transductive_query_H",
        },
        "split_manifest": str(split_path),
        "split_manifest_sha256": _digest(split_path),
        "fit_patients": int(len(np.unique(fit_patients))),
        "fit_cells": int(len(fit_values)),
        "reference_cells": int(len(bank_rows)),
        "test_patients": int(len(fold["test_patients"])),
        "test_specimens": int(len(test_specimens)),
        "test_cells": int(len(query_values)),
        "observed_markers": list(observed_markers),
        "hidden_markers": list(hidden_markers),
        "method_metadata": method_metadata,
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
