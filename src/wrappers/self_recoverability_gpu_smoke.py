"""Small application-level CUDA probe for the self-recoverability stack."""

from __future__ import annotations

import json

import numpy as np

from src.models.gpu_knn import TorchKNNMedianRegressor
from src.models.mlp import MLPRegressor


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    rng = np.random.RandomState(4207)
    features = rng.normal(size=(2048, 12)).astype(np.float32)
    targets = np.column_stack(
        (
            features[:, 0] + features[:, 1],
            features[:, 2] * features[:, 3],
        )
    ).astype(np.float32)
    query = features[:256]
    knn = TorchKNNMedianRegressor(
        k=15,
        device="cuda",
        query_chunk_size=64,
    ).fit(features, targets)
    knn_prediction = knn.predict(query)
    mlp = MLPRegressor(
        hidden_dims=(32,),
        epochs=3,
        batch_size=256,
        learning_rate=0.01,
        patience=2,
        random_state=4207,
    ).fit(
        features,
        targets,
        groups=np.repeat(np.arange(32), 64),
        device="cuda",
    )
    mlp_prediction = mlp.predict(query, device="cuda")
    if (
        not np.isfinite(knn_prediction).all()
        or not np.isfinite(mlp_prediction).all()
    ):
        raise RuntimeError("Non-finite CUDA prediction")
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "ok",
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "knn_shape": list(knn_prediction.shape),
                "mlp_shape": list(mlp_prediction.shape),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
