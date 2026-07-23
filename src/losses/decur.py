"""Barlow Twins and DeCUR losses for paired population prototypes."""

from __future__ import annotations

try:  # pragma: no cover - exercised in the neural environment
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional neural dependency
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for DeCUR losses")


def normalized_cross_correlation(left, right, *, epsilon: float = 1.0e-4):
    """Dimension-wise correlation after batch centering and scaling."""

    _require_torch()
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left and right must be aligned two-dimensional tensors")
    if left.shape[0] < 2:
        raise ValueError("At least two paired prototype rows are required")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    left = left / torch.sqrt(left.var(dim=0, unbiased=False, keepdim=True) + epsilon)
    right = right / torch.sqrt(right.var(dim=0, unbiased=False, keepdim=True) + epsilon)
    return left.T @ right / left.shape[0]


def weighted_normalized_cross_correlation(
    left,
    right,
    plan,
    *,
    epsilon: float = 1.0e-4,
):
    """Cross-correlation under a non-negative soft coupling matrix."""

    _require_torch()
    if left.ndim != 2 or right.ndim != 2 or plan.ndim != 2:
        raise ValueError("left, right, and plan must be two-dimensional")
    if (
        left.shape[1] != right.shape[1]
        or plan.shape != (left.shape[0], right.shape[0])
        or min(plan.shape) < 2
    ):
        raise ValueError("Soft-coupling shapes do not align")
    if not torch.isfinite(plan).all() or torch.any(plan < 0):
        raise ValueError("plan must be finite and non-negative")
    mass = plan.sum()
    if not torch.isfinite(mass) or float(mass.detach()) <= 0:
        raise ValueError("plan must have positive finite mass")
    coupling = plan / mass
    left_weight = coupling.sum(dim=1)
    right_weight = coupling.sum(dim=0)
    left_centered = left - (left_weight[:, None] * left).sum(
        dim=0, keepdim=True
    )
    right_centered = right - (right_weight[:, None] * right).sum(
        dim=0, keepdim=True
    )
    left_scale = torch.sqrt(
        (left_weight[:, None] * left_centered.square()).sum(
            dim=0, keepdim=True
        )
        + float(epsilon)
    )
    right_scale = torch.sqrt(
        (right_weight[:, None] * right_centered.square()).sum(
            dim=0, keepdim=True
        )
        + float(epsilon)
    )
    return (
        (left_centered / left_scale).T
        @ coupling
        @ (right_centered / right_scale)
    )


def _off_diagonal(matrix):
    size = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[1] != size:
        raise ValueError("matrix must be square")
    if size == 1:
        return matrix.new_empty((0,))
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def barlow_identity_loss(correlation, *, lambda_off_diagonal: float = 0.005):
    """Identity-target redundancy reduction, normalized by dimension."""

    _require_torch()
    dimension = correlation.shape[0]
    if correlation.ndim != 2 or correlation.shape[1] != dimension or dimension < 1:
        raise ValueError("correlation must be a non-empty square tensor")
    diagonal = (torch.diagonal(correlation) - 1.0).square().sum()
    off_diagonal = _off_diagonal(correlation).square().sum()
    return (diagonal + float(lambda_off_diagonal) * off_diagonal) / dimension


def barlow_zero_loss(correlation, *, lambda_off_diagonal: float = 0.005):
    """DeCUR cross-modal unique target: diagonal and off-diagonal are zero."""

    _require_torch()
    dimension = correlation.shape[0]
    if correlation.ndim != 2 or correlation.shape[1] != dimension or dimension < 1:
        raise ValueError("correlation must be a non-empty square tensor")
    diagonal = torch.diagonal(correlation).square().sum()
    off_diagonal = _off_diagonal(correlation).square().sum()
    return (diagonal + float(lambda_off_diagonal) * off_diagonal) / dimension


def prototype_decur_loss(
    source,
    target,
    *,
    n_common: int,
    lambda_off_diagonal: float = 0.005,
    cross_block_weight: float = 0.1,
):
    """Split prototype axes into common identity and unique zero targets."""

    _require_torch()
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target prototypes must be aligned matrices")
    total = source.shape[1]
    common = int(n_common)
    if not 1 <= common <= total:
        raise ValueError("n_common must lie between one and the total dimension")
    correlation = normalized_cross_correlation(source, target)
    common_correlation = correlation[:common, :common]
    common_loss = barlow_identity_loss(
        common_correlation, lambda_off_diagonal=lambda_off_diagonal
    )
    zero = common_loss.new_zeros(())
    if common == total:
        unique_loss = zero
        cross_block_loss = zero
    else:
        unique_correlation = correlation[common:, common:]
        unique_loss = barlow_zero_loss(
            unique_correlation, lambda_off_diagonal=lambda_off_diagonal
        )
        cross_block_loss = 0.5 * (
            correlation[:common, common:].square().mean()
            + correlation[common:, :common].square().mean()
        )
    total_loss = common_loss + unique_loss + float(cross_block_weight) * cross_block_loss
    return {
        "loss": total_loss,
        "common": common_loss,
        "unique": unique_loss,
        "cross_block": cross_block_loss,
        "correlation": correlation,
    }


def weighted_decur_loss(
    source,
    target,
    plan,
    *,
    n_common: int,
    lambda_off_diagonal: float = 0.005,
    cross_block_weight: float = 0.1,
):
    """DeCUR common/unique targets under a soft cell-level coupling."""

    _require_torch()
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target concepts must be matrices")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target concept dimensions differ")
    total = source.shape[1]
    common = int(n_common)
    if not 1 <= common <= total:
        raise ValueError("n_common must lie between one and the total dimension")
    correlation = weighted_normalized_cross_correlation(source, target, plan)
    common_correlation = correlation[:common, :common]
    common_loss = barlow_identity_loss(
        common_correlation, lambda_off_diagonal=lambda_off_diagonal
    )
    zero = common_loss.new_zeros(())
    if common == total:
        unique_loss = zero
        cross_block_loss = zero
    else:
        unique_loss = barlow_zero_loss(
            correlation[common:, common:],
            lambda_off_diagonal=lambda_off_diagonal,
        )
        cross_block_loss = 0.5 * (
            correlation[:common, common:].square().mean()
            + correlation[common:, :common].square().mean()
        )
    total_loss = (
        common_loss
        + unique_loss
        + float(cross_block_weight) * cross_block_loss
    )
    return {
        "loss": total_loss,
        "common": common_loss,
        "unique": unique_loss,
        "cross_block": cross_block_loss,
        "correlation": correlation,
    }


def intra_modal_barlow_loss(
    first, second, *, lambda_off_diagonal: float = 0.005
):
    """Identity-target Barlow Twins loss over all modality dimensions."""

    correlation = normalized_cross_correlation(first, second)
    return barlow_identity_loss(
        correlation, lambda_off_diagonal=lambda_off_diagonal
    )


def decoded_population_summary_loss(
    prediction,
    target,
    *,
    n_groups: int,
    cells_per_group: int,
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
):
    """Match decoded residual distributions with differentiable group summaries."""

    _require_torch()
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be aligned two-dimensional tensors")
    groups = int(n_groups)
    cells = int(cells_per_group)
    if groups < 1 or cells < 2 or prediction.shape[0] != groups * cells:
        raise ValueError("rows must equal n_groups times cells_per_group")
    requested_quantiles = tuple(float(value) for value in quantiles)
    if not requested_quantiles or any(
        value <= 0.0 or value >= 1.0 for value in requested_quantiles
    ):
        raise ValueError("quantiles must lie strictly between zero and one")
    predicted_groups = prediction.reshape(groups, cells, -1)
    target_groups = target.reshape(groups, cells, -1)
    mean_loss = torch.nn.functional.mse_loss(
        predicted_groups.mean(dim=1), target_groups.mean(dim=1)
    )
    prediction_std = torch.sqrt(
        predicted_groups.var(dim=1, unbiased=False) + 1.0e-4
    )
    target_std = torch.sqrt(target_groups.var(dim=1, unbiased=False) + 1.0e-4)
    standard_deviation_loss = torch.nn.functional.mse_loss(
        prediction_std, target_std
    )
    levels = prediction.new_tensor(requested_quantiles)
    quantile_loss = torch.nn.functional.mse_loss(
        torch.quantile(predicted_groups, levels, dim=1),
        torch.quantile(target_groups, levels, dim=1),
    )
    return {
        "loss": mean_loss + standard_deviation_loss + quantile_loss,
        "mean": mean_loss,
        "standard_deviation": standard_deviation_loss,
        "quantile": quantile_loss,
    }
