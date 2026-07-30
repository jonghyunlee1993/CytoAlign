"""Adapters for literature methods used in same-cell pseudo-masking.

All adapters receive fit-only reference cells with a complete marker panel and
held-out query cells containing only shared markers.  They never receive query
hidden-marker values or cell labels.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def marker_column_indices(
    available_markers: Sequence[str],
    requested_markers: Sequence[str],
) -> np.ndarray:
    """Resolve marker columns by name without assuming panel-union order."""

    available = tuple(map(str, available_markers))
    if len(set(available)) != len(available):
        raise ValueError("Available marker names must be unique")
    lookup = {marker: index for index, marker in enumerate(available)}
    requested = tuple(map(str, requested_markers))
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"Requested markers are absent: {missing}")
    return np.asarray([lookup[marker] for marker in requested], dtype=np.int64)


def balanced_group_rows(
    groups: Sequence[str],
    maximum: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample at most ``maximum`` rows with approximately equal group quotas."""

    groups = np.asarray(groups).astype(str)
    if int(maximum) <= 0:
        raise ValueError("maximum must be positive")
    if len(groups) <= int(maximum):
        return np.arange(len(groups), dtype=np.int64)
    unique = np.unique(groups)
    quota = max(1, int(np.ceil(int(maximum) / len(unique))))
    rng = np.random.RandomState(int(seed))
    pieces = []
    for group in unique:
        candidates = np.flatnonzero(groups == group)
        if len(candidates) > quota:
            candidates = rng.choice(candidates, quota, replace=False)
        pieces.append(candidates)
    selected = np.concatenate(pieces)
    if len(selected) > int(maximum):
        selected = rng.choice(selected, int(maximum), replace=False)
    return np.sort(selected.astype(np.int64))


def _write_markers(path: Path, markers: Sequence[str]) -> None:
    path.write_text(
        "".join(f"{str(marker)}\n" for marker in markers),
        encoding="utf-8",
    )


def predict_cycombine(
    *,
    reference_full: np.ndarray,
    query_observed: np.ndarray,
    full_markers: Sequence[str],
    observed_markers: Sequence[str],
    hidden_markers: Sequence[str],
    fallback_hidden: np.ndarray,
    script: str | Path,
    rscript: str | Path,
    seed: int,
    xdim: int = 8,
    ydim: int = 8,
    rlen: int = 10,
    minimum_reference_cells: int = 50,
    distance: str = "sumofsquares",
) -> tuple[np.ndarray, dict]:
    """Run the SOM plus within-node KDE draw from cyCombine panel merging."""

    reference_full = np.asarray(reference_full, dtype=np.float32)
    query_observed = np.asarray(query_observed, dtype=np.float32)
    fallback_hidden = np.asarray(fallback_hidden, dtype=np.float32)
    if reference_full.shape[1] != len(full_markers):
        raise ValueError("reference_full columns do not match full_markers")
    if query_observed.shape[1] != len(observed_markers):
        raise ValueError("query_observed columns do not match observed_markers")
    if fallback_hidden.shape != (len(hidden_markers),):
        raise ValueError("fallback_hidden does not match hidden_markers")

    with tempfile.TemporaryDirectory(prefix="cytoalign_cycombine_") as directory:
        root = Path(directory)
        reference_path = root / "reference.csv"
        query_path = root / "query.csv"
        observed_path = root / "observed_markers.txt"
        hidden_path = root / "hidden_markers.txt"
        output_path = root / "prediction.csv"
        pd.DataFrame(reference_full, columns=full_markers).to_csv(
            reference_path,
            index=False,
        )
        pd.DataFrame(query_observed, columns=observed_markers).to_csv(
            query_path,
            index=False,
        )
        _write_markers(observed_path, observed_markers)
        _write_markers(hidden_path, hidden_markers)
        command = [
            str(rscript),
            str(script),
            str(reference_path),
            str(query_path),
            str(observed_path),
            str(hidden_path),
            str(output_path),
            str(int(seed)),
            str(int(xdim)),
            str(int(ydim)),
            str(int(rlen)),
            str(int(minimum_reference_cells)),
            str(distance),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        result = pd.read_csv(output_path)

    if result[".row_id"].tolist() != list(range(len(query_observed))):
        raise RuntimeError("cyCombine adapter changed query row order")
    raw = result.loc[:, list(hidden_markers)].to_numpy(dtype=np.float32)
    finite = np.all(np.isfinite(raw), axis=1)
    prediction = raw.copy()
    prediction[~finite] = fallback_hidden[None, :]
    node_sizes = result[".som_node"].value_counts().sort_index()
    metadata = {
        "adapter": "cycombine_impute_across_panels_function_level",
        "output_mode": "distribution_draw",
        "query_access": "transductive_shared_markers",
        "coverage_fraction": float(np.mean(finite)),
        "covered_cells": int(np.sum(finite)),
        "fallback_cells": int(np.sum(~finite)),
        "fallback": "fit_marker_median",
        "som_nodes_with_query": int(len(node_sizes)),
        "som_query_node_sizes": {
            str(int(key)): int(value) for key, value in node_sizes.items()
        },
        "r_stdout": completed.stdout[-4000:],
        "r_stderr": completed.stderr[-4000:],
    }
    return prediction, metadata


def predict_uvae(
    *,
    reference_full: np.ndarray,
    query_observed: np.ndarray,
    full_markers: Sequence[str],
    observed_markers: Sequence[str],
    query_samples: Sequence[str],
    hidden_indices: np.ndarray,
    runner: str | Path,
    python: str | Path,
    external_root: str | Path,
    seed: int,
    epochs: int = 30,
    batch_size: int = 512,
    latent_dim: int = 50,
    hidden: int = 2,
    width: int = 256,
    pull: float = 1.0,
    early_stop_epochs: int = 0,
    samples_per_epoch: int = 0,
    max_query_training_cells: int = 50000,
) -> tuple[np.ndarray, dict]:
    """Run the official UVAE cross-panel decoder in an isolated process."""

    reference_full = np.asarray(reference_full, dtype=np.float32)
    query_observed = np.asarray(query_observed, dtype=np.float32)
    query_training_rows = balanced_group_rows(
        query_samples,
        int(max_query_training_cells),
        seed=int(seed) + 7919,
    )
    with tempfile.TemporaryDirectory(prefix="cytoalign_uvae_") as directory:
        root = Path(directory)
        input_path = root / "input.npz"
        output_path = root / "output.npz"
        model_path = root / "model.uv"
        np.savez_compressed(
            input_path,
            reference=reference_full,
            query_training=query_observed[query_training_rows],
            query_all=query_observed,
            reference_markers=np.asarray(full_markers, dtype=str),
            query_markers=np.asarray(observed_markers, dtype=str),
        )
        command = [
            str(python),
            str(runner),
            "--external-root",
            str(external_root),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model-path",
            str(model_path),
            "--epochs",
            str(int(epochs)),
            "--batch-size",
            str(int(batch_size)),
            "--latent-dim",
            str(int(latent_dim)),
            "--hidden",
            str(int(hidden)),
            "--width",
            str(int(width)),
            "--pull",
            str(float(pull)),
            "--early-stop-epochs",
            str(int(early_stop_epochs)),
            "--samples-per-epoch",
            str(int(samples_per_epoch)),
            "--seed",
            str(int(seed)),
        ]
        environment = dict(os.environ)
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        output = np.load(output_path, allow_pickle=False)
        full_prediction = np.asarray(output["prediction"], dtype=np.float32)
        external_metadata = json.loads(
            output_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            and not external_metadata.get("gpu_devices")
        ):
            raise RuntimeError(
                "UVAE TensorFlow subprocess did not detect its allocated GPU"
            )

    expected = (len(query_observed), len(full_markers))
    if full_prediction.shape != expected:
        raise RuntimeError(
            f"UVAE prediction shape {full_prediction.shape} != {expected}"
        )
    hidden_prediction = full_prediction[:, np.asarray(hidden_indices, dtype=int)]
    metadata = {
        "adapter": "official_uvae_subspace_reference_decoder",
        "output_mode": "point_decoder_mean",
        "query_access": "transductive_shared_markers",
        "query_training_cells": int(len(query_training_rows)),
        "query_total_cells": int(len(query_observed)),
        "query_training_sampling": "specimen_balanced_without_hidden_markers",
        "external": external_metadata,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
    }
    return hidden_prediction, metadata


def predict_cytovi(
    *,
    reference_full: np.ndarray,
    query_observed: np.ndarray,
    full_markers: Sequence[str],
    observed_markers: Sequence[str],
    reference_samples: Sequence[str],
    query_samples: Sequence[str],
    hidden_indices: np.ndarray,
    seed: int,
    max_epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1.0e-3,
    n_hidden: int = 128,
    n_latent: int = 10,
    n_layers: int = 1,
    prior_mixture: bool = True,
    early_stopping_patience: int = 10,
    n_samples: int = 1,
    max_training_cells_per_epoch: int = 0,
    n_epochs_kl_warmup: int = 30,
    max_query_training_cells: int = 50000,
) -> tuple[np.ndarray, dict]:
    """Fit CytoVI to a complete reference panel and shared-only query panel."""

    import anndata as ad
    import scvi
    import torch
    from scvi.external import cytovi

    scvi.settings.seed = int(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    reference = ad.AnnData(
        X=np.asarray(reference_full, dtype=np.float32),
        obs=pd.DataFrame(
            {
                "_sample": np.asarray(reference_samples).astype(str),
                "_source": "reference_full",
            },
            index=[f"reference_{index}" for index in range(len(reference_full))],
        ),
        var=pd.DataFrame(index=list(map(str, full_markers))),
    )
    query_training_rows = balanced_group_rows(
        query_samples,
        int(max_query_training_cells),
        seed=int(seed) + 7919,
    )
    query_training = ad.AnnData(
        X=np.asarray(query_observed[query_training_rows], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "_sample": np.asarray(query_samples)[query_training_rows].astype(str),
                "_source": "heldout_shared",
            },
            index=[
                f"query_training_{index}"
                for index in range(len(query_training_rows))
            ],
        ),
        var=pd.DataFrame(index=list(map(str, observed_markers))),
    )
    reference.layers["scaled"] = reference.X.copy()
    query_training.layers["scaled"] = query_training.X.copy()
    merged = cytovi.merge_batches(
        [reference, query_training],
        batch_key="_panel",
        scaled_layer_key="scaled",
        mask_layer_key="_nan_mask",
    )
    cytovi.CYTOVI.setup_anndata(
        merged,
        layer="scaled",
        nan_layer="_nan_mask",
    )
    model = cytovi.CYTOVI(
        merged,
        n_hidden=int(n_hidden),
        n_latent=int(n_latent),
        n_layers=int(n_layers),
        prior_mixture=bool(prior_mixture),
    )
    trainer_kwargs = {}
    if int(max_training_cells_per_epoch) > 0:
        trainer_kwargs["limit_train_batches"] = max(
            1,
            int(
                np.ceil(
                    int(max_training_cells_per_epoch) / float(batch_size)
                )
            ),
        )
        trainer_kwargs["limit_val_batches"] = max(
            1,
            int(
                np.ceil(
                    max(1, int(max_training_cells_per_epoch) // 5)
                    / float(batch_size)
                )
            ),
        )
    model.train(
        max_epochs=int(max_epochs),
        batch_size=int(batch_size),
        lr=float(learning_rate),
        early_stopping=True,
        early_stopping_patience=int(early_stopping_patience),
        n_epochs_kl_warmup=int(n_epochs_kl_warmup),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        **trainer_kwargs,
    )
    reference_template = reference[:1].copy()
    all_query = ad.AnnData(
        X=np.asarray(query_observed, dtype=np.float32),
        obs=pd.DataFrame(
            {"_source": "heldout_shared"},
            index=[f"query_{index}" for index in range(len(query_observed))],
        ),
        var=pd.DataFrame(index=list(map(str, observed_markers))),
    )
    all_query.layers["scaled"] = all_query.X.copy()
    prediction_adata = cytovi.merge_batches(
        [reference_template, all_query],
        batch_key="_panel",
        scaled_layer_key="scaled",
        mask_layer_key="_nan_mask",
    )
    if tuple(map(str, prediction_adata.var_names)) != tuple(
        map(str, merged.var_names)
    ):
        raise RuntimeError("CytoVI train and prediction marker unions differ")
    query_indices = np.arange(
        1,
        len(query_observed) + 1,
        dtype=np.int64,
    )
    full_prediction = model.get_normalized_expression(
        adata=prediction_adata,
        indices=query_indices,
        n_samples=int(n_samples),
        batch_size=int(batch_size),
        return_numpy=True,
    )
    full_prediction = np.asarray(full_prediction, dtype=np.float32)
    if full_prediction.ndim == 3:
        full_prediction = np.mean(full_prediction, axis=0)
    expected = (len(query_observed), len(full_markers))
    if full_prediction.shape != expected:
        raise RuntimeError(
            f"CytoVI prediction shape {full_prediction.shape} != {expected}"
        )
    decoded_markers = tuple(map(str, merged.var_names))
    requested_hidden = tuple(
        str(full_markers[int(index)]) for index in hidden_indices
    )
    decoded_hidden_indices = marker_column_indices(
        decoded_markers,
        requested_hidden,
    )
    hidden_prediction = full_prediction[
        :,
        decoded_hidden_indices,
    ]
    history = getattr(model, "history", None)
    metadata = {
        "adapter": "scvi_tools_cytovi_masked_vae",
        "output_mode": "decoder_mean" if int(n_samples) == 1 else "posterior_mean",
        "query_access": "transductive_shared_markers",
        "scvi_tools": scvi.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "epochs_trained": (
            int(len(history["elbo_train"])) if history is not None else None
        ),
        "max_training_cells_per_epoch": int(max_training_cells_per_epoch),
        "query_training_cells": int(len(query_training_rows)),
        "query_total_cells": int(len(query_observed)),
        "query_training_sampling": "specimen_balanced_without_hidden_markers",
        "decoded_marker_order": list(decoded_markers),
        "returned_hidden_marker_order": list(requested_hidden),
    }
    return hidden_prediction, metadata
