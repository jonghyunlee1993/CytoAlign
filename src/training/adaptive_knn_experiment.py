"""Fold-level experiment for panel-aware target-specific kNN metrics."""

from __future__ import annotations

import json
import os
import random
import socket
import time
from pathlib import Path

import numpy as np

from src.data.cross_panel import load_cross_panel_dataset
from src.data.splits import patient_id_from_specimen
from src.models.adaptive_knn import (
    AdaptiveMetricLearner,
    MetricFit,
    predict_with_candidate_reranking,
)
from src.models.cytofmerge import patient_balanced_indices
from src.training.experiment import (
    _evaluate,
    _fit_common_space,
    _marker_scale,
    _split_view,
    _training_arrays,
    _without_cell_type_strata,
)


def _panel_masks(common_markers, configured) -> dict[str, np.ndarray]:
    lookup = {marker: index for index, marker in enumerate(common_markers)}
    masks = {}
    for name, markers in configured.items():
        unknown = sorted(set(markers) - set(lookup))
        if unknown:
            raise ValueError(f"Panel {name!r} contains unknown markers: {unknown}")
        mask = np.zeros(len(common_markers), dtype=bool)
        mask[[lookup[marker] for marker in markers]] = True
        if not mask.any():
            raise ValueError(f"Panel {name!r} is empty")
        masks[str(name)] = mask
    if not masks:
        raise ValueError("At least one panel mask is required")
    return masks


def _fit_key(family: str, temperature: float, regularization: float) -> str:
    return f"{family}.temp_{temperature:g}.reg_{regularization:g}"


def _metrics(predictions, view, scales) -> dict:
    return {
        "cell_type_stratified": _evaluate(predictions, view, scales),
        "pooled": _evaluate(
            predictions, _without_cell_type_strata(view), scales
        ),
    }


def _predict_view(
    view,
    reference_common,
    reference_targets,
    weights,
    mask,
    *,
    k,
    candidate_k,
    batch_size,
    device,
):
    predictions = {
        "plain_knn": {},
        **{name: {} for name in weights},
    }
    for specimen in sorted(view["source_h"]):
        current = predict_with_candidate_reranking(
            reference_common,
            reference_targets,
            view["source_h"][specimen],
            weights,
            mask,
            k=k,
            candidate_k=candidate_k,
            batch_size=batch_size,
            device=device,
        )
        for name, values in current.items():
            predictions[name][specimen] = values
    return predictions


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(current) for key, current in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(current) for current in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_adaptive_knn_experiment(config: dict) -> dict:
    """Train and evaluate global, target-specific, and panel-dropout metrics."""

    import torch

    started = time.time()
    fold_index = int(config["experiment"]["fold"])
    seed = int(config["experiment"]["seed"])
    training = config["training"]["adaptive_knn"]
    device = str(config["training"]["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    hardware = {"host": socket.gethostname(), "device": device}
    if device == "cuda":
        probe = torch.randn(1024, 1024, device=device) @ torch.randn(
            1024, 1024, device=device
        )
        torch.cuda.synchronize()
        if not torch.isfinite(probe).all():
            raise RuntimeError("CUDA matrix multiplication failed")
        properties = torch.cuda.get_device_properties(0)
        hardware.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": int(properties.total_memory),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = load_cross_panel_dataset(config["data"])
    fold = dataset.splits["folds"][fold_index]
    train_specimens = fold["train_specimens"]
    validation_specimens = fold["validation_specimens"]
    test_specimens = fold["test_specimens"]
    common_space = _fit_common_space(
        dataset,
        train_specimens,
        n_knots=int(config["preprocessing"]["n_knots"]),
        maximum_cells=int(config["preprocessing"]["max_fit_cells"]),
        seed=seed,
    )
    n_common = len(dataset.common_markers)
    target_h_raw, target_y, _, specimen_groups = _training_arrays(
        dataset.target, train_specimens, n_common
    )
    target_h = common_space.target_percentiles(target_h_raw)
    patient_groups = np.asarray(
        [patient_id_from_specimen(group) for group in specimen_groups]
    )
    fit_rows = patient_balanced_indices(
        patient_groups,
        len(target_h),
        int(training["max_fit_cells"]),
        seed + 1,
    )
    target_h = target_h[fit_rows].astype(np.float32)
    target_y = target_y[fit_rows].astype(np.float32)
    patient_groups = patient_groups[fit_rows]
    scales = _marker_scale(target_y).astype(np.float32)
    panel_masks = _panel_masks(
        dataset.common_markers, training["panel_masks"]
    )

    common_learner = {
        "epochs": int(training["epochs"]),
        "steps_per_epoch": int(training["steps_per_epoch"]),
        "query_batch_size": int(training["query_batch_size"]),
        "reference_batch_size": int(training["reference_batch_size"]),
        "learning_rate": float(training["learning_rate"]),
    }
    fits: dict[str, MetricFit] = {}
    families = {
        "global_metric": ("global", False),
        "target_metric": ("target", False),
        "target_metric_panel_dropout": ("target", True),
    }
    temperatures = tuple(map(float, training["temperatures"]))
    regularizations = tuple(map(float, training["regularizations"]))
    for family_index, (family, (mode, dropout)) in enumerate(families.items()):
        for temperature in temperatures:
            for regularization in regularizations:
                key = _fit_key(family, temperature, regularization)
                learner = AdaptiveMetricLearner(
                    mode=mode,
                    temperature=temperature,
                    regularization=regularization,
                    random_state=seed + 1000 * family_index,
                    **common_learner,
                )
                fits[key] = learner.fit(
                    target_h,
                    target_y,
                    patient_groups,
                    scales,
                    panel_masks=list(panel_masks.values()),
                    mask_dropout=dropout,
                    device=device,
                )

    reference_rows = patient_balanced_indices(
        patient_groups,
        len(target_h),
        int(training["max_reference_cells"]),
        seed + 2,
    )
    reference_h = target_h[reference_rows]
    reference_y = target_y[reference_rows]
    validation = _split_view(dataset, validation_specimens, common_space)
    test = _split_view(dataset, test_specimens, common_space)
    inference = {
        "k": int(training["k"]),
        "candidate_k": int(training["candidate_k"]),
        "batch_size": int(training["inference_batch_size"]),
        "device": device,
    }

    panel_results = {}
    selected_keys = {}
    all_weights = {key: fit.weights for key, fit in fits.items()}
    for panel, mask in panel_masks.items():
        validation_predictions = _predict_view(
            validation,
            reference_h,
            reference_y,
            all_weights,
            mask,
            **inference,
        )
        validation_metrics = {
            name: _metrics(predictions, validation, scales)
            for name, predictions in validation_predictions.items()
        }
        selected = {}
        for family in families:
            candidates = [key for key in fits if key.startswith(f"{family}.")]
            selected[family] = min(
                candidates,
                key=lambda key: validation_metrics[key]["pooled"][
                    "patient_first_normalized_wasserstein"
                ],
            )
        selected_keys[panel] = selected
        test_weights = {
            family: fits[key].weights for family, key in selected.items()
        }
        test_predictions = _predict_view(
            test,
            reference_h,
            reference_y,
            test_weights,
            mask,
            **inference,
        )
        test_metrics = {
            name: _metrics(predictions, test, scales)
            for name, predictions in test_predictions.items()
        }
        plain_wasserstein = test_metrics["plain_knn"]["cell_type_stratified"][
            "patient_first_normalized_wasserstein"
        ]
        methods = {
            "plain_knn": {
                "validation": validation_metrics["plain_knn"],
                "test": test_metrics["plain_knn"],
            }
        }
        for family, key in selected.items():
            current_wasserstein = test_metrics[family]["cell_type_stratified"][
                "patient_first_normalized_wasserstein"
            ]
            methods[family] = {
                "selected_fit": key,
                "temperature": fits[key].temperature,
                "regularization": fits[key].regularization,
                "validation": validation_metrics[key],
                "test": test_metrics[family],
                "relative_wasserstein_improvement": (
                    plain_wasserstein - current_wasserstein
                )
                / plain_wasserstein,
            }
        panel_results[panel] = {
            "common_markers": np.asarray(dataset.common_markers)[mask].tolist(),
            "validation_candidate_wasserstein": {
                name: values["pooled"][
                    "patient_first_normalized_wasserstein"
                ]
                for name, values in validation_metrics.items()
            },
            "methods": methods,
        }

    result = {
        "status": "completed",
        "runner": "adaptive_knn",
        "experiment": config["experiment"]["name"],
        "fold": fold_index,
        "seed": seed,
        "source_modality": dataset.source_modality,
        "target_modality": dataset.target_modality,
        "common_markers": list(dataset.common_markers),
        "target_markers": list(dataset.target_exclusive_columns),
        "split": {
            "train_specimens": list(train_specimens),
            "validation_specimens": list(validation_specimens),
            "test_specimens": list(test_specimens),
        },
        "sample_sizes": {
            "metric_fit_cells": int(len(target_h)),
            "reference_cells": int(len(reference_h)),
        },
        "search": {
            "k": inference["k"],
            "candidate_k": inference["candidate_k"],
            "candidate_rule": (
                "exact Euclidean candidate search within each panel, followed by "
                "target-marker-specific reranking"
            ),
        },
        "selection": {
            "scope": "pooled validation populations without cell-type labels",
            "metric": "patient_first_normalized_wasserstein",
            "fits": selected_keys,
        },
        "fit_diagnostics": {
            key: {
                "mode": fit.mode,
                "temperature": fit.temperature,
                "regularization": fit.regularization,
                "epoch_losses": fit.losses,
                "weights": fit.weights,
            }
            for key, fit in fits.items()
        },
        "panel_results": panel_results,
        "hardware": hardware,
        "runtime_seconds": time.time() - started,
    }
    output = (
        Path(config["output"]["root"])
        / config["experiment"]["name"]
        / f"fold_{fold_index}"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "adaptive_knn_metrics.json"
    metrics_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["metrics_path"] = str(metrics_path)
    return _jsonable(result)
