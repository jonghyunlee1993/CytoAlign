#!/usr/bin/env python
"""Isolated official-UVAE runner used by the benchmark adapter.

The external repository and this project both expose a top-level ``src``
package.  Running UVAE in a fresh process with its repository first on
``sys.path`` avoids importing a mixture of the two packages.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--pull", type=float, default=1.0)
    parser.add_argument("--early-stop-epochs", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    external_root = args.external_root.resolve()
    if not (external_root / "src" / "UVAE.py").is_file():
        raise FileNotFoundError(f"Official UVAE source is absent: {external_root}")
    sys.path.insert(0, str(external_root))

    import tensorflow as tf
    from src.UVAE import UVAE
    from src.UVAE_classes import Data, Subspace

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gpu_devices = tf.config.list_physical_devices("GPU")
    if "CUDA_VISIBLE_DEVICES" in os.environ and not gpu_devices:
        raise RuntimeError(
            "TensorFlow did not detect the GPU allocated through "
            "CUDA_VISIBLE_DEVICES"
        )

    payload = np.load(args.input, allow_pickle=False)
    reference = np.asarray(payload["reference"], dtype=np.float32)
    query_training = np.asarray(payload["query_training"], dtype=np.float32)
    query_all = np.asarray(payload["query_all"], dtype=np.float32)
    reference_markers = payload["reference_markers"].astype(str).tolist()
    query_markers = payload["query_markers"].astype(str).tolist()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    uv = UVAE(str(args.model_path))
    uv.hyper.update(
        {
            "latent_dim": int(args.latent_dim),
            "hidden": int(args.hidden),
            "width": int(args.width),
            "batch_size": int(args.batch_size),
        }
    )
    reference_panel = uv + Data(
        reference,
        channels=reference_markers,
        name="reference_full_panel",
    )
    query_panel = uv + Data(
        query_training,
        channels=query_markers,
        name="heldout_shared_panel",
    )
    uv + Subspace(
        masks=[reference_panel, query_panel],
        name="shared_markers",
        pull=float(args.pull),
    )
    history = uv.train(
        maxEpochs=int(args.epochs),
        batchSize=int(args.batch_size),
        samplesPerEpoch=int(args.samples_per_epoch),
        earlyStopEpochs=int(args.early_stop_epochs),
        saveBest=bool(args.early_stop_epochs),
        verbose=True,
    )
    # The encoder is amortized and accepts unseen rows with the same shared
    # channels. Replace the training subset after fitting so every query cell
    # is reconstructed without adding another panel-specific autoencoder.
    query_panel.X = query_all
    query_panel.normed = None
    query_panel.predictions.clear()
    prediction = uv.reconstruct(
        {
            query_panel: np.arange(
                len(query_all),
                dtype=np.int64,
            )
        },
        channels=reference_markers,
        decoderPanels=[reference_panel],
        bs=int(args.batch_size),
        stacked=True,
        mean=True,
    )
    prediction = np.asarray(prediction, dtype=np.float32)
    np.savez_compressed(args.output, prediction=prediction)
    summary = {
        "tensorflow": tf.__version__,
        "reference_cells": int(len(reference)),
        "query_training_cells": int(len(query_training)),
        "query_cells": int(len(query_all)),
        "epochs_requested": int(args.epochs),
        "samples_per_epoch": int(args.samples_per_epoch),
        "history_type": type(history).__name__,
        "gpu_devices": [device.name for device in gpu_devices],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
