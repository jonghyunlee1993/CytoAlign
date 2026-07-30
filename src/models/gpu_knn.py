"""Exact median kNN regression with a torch distance backend."""

from __future__ import annotations

import numpy as np


class TorchKNNMedianRegressor:
    """Predict marker-wise medians among exact Euclidean nearest neighbors.

    The implementation intentionally keeps model semantics identical across CPU
    and CUDA devices. CUDA is only a compute backend for the distance search.
    """

    def __init__(
        self,
        *,
        k: int = 50,
        device: str = "cuda",
        query_chunk_size: int | None = None,
        distance_memory_fraction: float = 0.08,
    ):
        if int(k) < 1:
            raise ValueError("k must be positive")
        if query_chunk_size is not None and int(query_chunk_size) < 1:
            raise ValueError("query_chunk_size must be positive when provided")
        if not 0 < float(distance_memory_fraction) <= 0.5:
            raise ValueError("distance_memory_fraction must be in (0, 0.5]")
        self.k = int(k)
        self.device = str(device)
        self.query_chunk_size = (
            None if query_chunk_size is None else int(query_chunk_size)
        )
        self.distance_memory_fraction = float(distance_memory_fraction)

    def fit(
        self,
        reference_features: np.ndarray,
        reference_targets: np.ndarray,
    ) -> "TorchKNNMedianRegressor":
        features = np.asarray(reference_features, dtype=np.float32)
        targets = np.asarray(reference_targets, dtype=np.float32)
        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("reference features and targets must be matrices")
        if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
            raise ValueError("reference rows must be equal and non-empty")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("reference arrays must be finite")
        self.reference_features_ = np.ascontiguousarray(features)
        self.reference_targets_ = np.ascontiguousarray(targets)
        self.n_features_in_ = features.shape[1]
        self.n_targets_ = targets.shape[1]
        self._torch_reference = None
        self._torch_targets = None
        return self

    def _chunk_size(self, torch_module) -> int:
        if self.query_chunk_size is not None:
            return self.query_chunk_size
        if self.device.startswith("cuda"):
            free_bytes, _ = torch_module.cuda.mem_get_info(self.device)
            budget = max(
                64 * 1024**2,
                int(free_bytes * self.distance_memory_fraction),
            )
        else:
            budget = 256 * 1024**2
        # Distances are float32 and top-k retains values plus int64 indices.
        bytes_per_query = max(1, 4 * self.reference_features_.shape[0])
        estimated = int(budget / bytes_per_query / 3)
        return max(16, min(2048, estimated))

    def predict(self, query_features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "reference_features_"):
            raise RuntimeError("Regressor has not been fitted")
        import torch

        query = np.asarray(query_features, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != self.n_features_in_:
            raise ValueError("query_features have an unexpected shape")
        if not np.isfinite(query).all():
            raise ValueError("query_features must be finite")
        if query.shape[0] == 0:
            return np.empty((0, self.n_targets_), dtype=np.float32)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        device = torch.device(self.device)
        if (
            self._torch_reference is None
            or self._torch_reference.device != device
        ):
            self._torch_reference = torch.from_numpy(
                self.reference_features_
            ).to(device)
            self._torch_targets = torch.from_numpy(
                self.reference_targets_
            ).to(device)
        reference = self._torch_reference
        targets = self._torch_targets
        k = min(self.k, reference.shape[0])
        chunk_size = self._chunk_size(torch)
        output = []

        previous_tf32 = None
        if device.type == "cuda":
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.no_grad():
                for left in range(0, query.shape[0], chunk_size):
                    right = min(query.shape[0], left + chunk_size)
                    current = torch.from_numpy(query[left:right]).to(device)
                    distances = torch.cdist(current, reference, p=2)
                    neighbors = torch.topk(
                        distances,
                        k=k,
                        dim=1,
                        largest=False,
                        sorted=False,
                    ).indices
                    prediction = torch.median(targets[neighbors], dim=1).values
                    output.append(prediction.cpu().numpy())
                    del current, distances, neighbors, prediction
        finally:
            if previous_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        return np.ascontiguousarray(np.concatenate(output), dtype=np.float32)

    @property
    def reference_size(self) -> int:
        if not hasattr(self, "reference_features_"):
            raise RuntimeError("Regressor has not been fitted")
        return int(self.reference_features_.shape[0])
