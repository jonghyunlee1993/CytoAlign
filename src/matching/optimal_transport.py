"""Small-batch entropic optimal transport for pseudo-cell coupling."""

from __future__ import annotations

try:  # pragma: no cover - exercised in the neural environment
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for optimal transport")


def squared_euclidean_cost(source, target):
    """Mean squared Euclidean distance between two cell matrices."""

    _require_torch()
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be two-dimensional tensors")
    if source.shape[1] != target.shape[1] or source.shape[1] < 1:
        raise ValueError("source and target feature dimensions must match")
    return torch.cdist(source, target).square() / source.shape[1]


def balanced_sinkhorn(
    cost,
    *,
    epsilon: float,
    iterations: int = 200,
):
    """Return a log-domain entropic OT plan with uniform marginals."""

    _require_torch()
    if cost.ndim != 2 or min(cost.shape) < 1:
        raise ValueError("cost must be a non-empty matrix")
    if not torch.isfinite(cost).all() or torch.any(cost < 0):
        raise ValueError("cost must be finite and non-negative")
    if float(epsilon) <= 0 or int(iterations) < 1:
        raise ValueError("epsilon and iterations must be positive")
    rows, columns = cost.shape
    log_a = cost.new_full((rows,), -torch.log(cost.new_tensor(float(rows))))
    log_b = cost.new_full((columns,), -torch.log(cost.new_tensor(float(columns))))
    log_kernel = -cost / float(epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(int(iterations)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
    return plan / plan.sum()


def barycentric_projection(plan, target):
    """Project target values through row-normalized transport weights."""

    _require_torch()
    if plan.ndim != 2 or target.ndim != 2 or plan.shape[1] != target.shape[0]:
        raise ValueError("plan and target shapes do not align")
    conditional = plan / torch.clamp(plan.sum(dim=1, keepdim=True), min=1.0e-12)
    return conditional @ target


def coupling_diagnostics(plan) -> dict:
    """Summarize uncertainty and marginal accuracy of a coupling."""

    _require_torch()
    if plan.ndim != 2 or min(plan.shape) < 1:
        raise ValueError("plan must be a non-empty matrix")
    conditional = plan / torch.clamp(plan.sum(dim=1, keepdim=True), min=1.0e-12)
    entropy = -(conditional * torch.log(torch.clamp(conditional, min=1.0e-12))).sum(
        dim=1
    )
    if plan.shape[1] > 1:
        normalized_entropy = entropy / torch.log(plan.new_tensor(float(plan.shape[1])))
    else:
        normalized_entropy = torch.zeros_like(entropy)
    expected_rows = plan.new_full((plan.shape[0],), 1.0 / float(plan.shape[0]))
    expected_columns = plan.new_full((plan.shape[1],), 1.0 / float(plan.shape[1]))
    return {
        "normalized_entropy_mean": float(normalized_entropy.mean()),
        "effective_targets_mean": float(torch.exp(entropy).mean()),
        "top_probability_mean": float(conditional.max(dim=1).values.mean()),
        "row_marginal_max_error": float(
            torch.max(torch.abs(plan.sum(dim=1) - expected_rows))
        ),
        "column_marginal_max_error": float(
            torch.max(torch.abs(plan.sum(dim=0) - expected_columns))
        ),
    }
