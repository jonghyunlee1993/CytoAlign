import pytest

torch = pytest.importorskip("torch")

from src.matching.optimal_transport import (
    balanced_sinkhorn,
    barycentric_projection,
    coupling_diagnostics,
    squared_euclidean_cost,
)


def test_sinkhorn_prefers_diagonal_and_has_uniform_marginals():
    values = torch.tensor([[0.0], [3.0], [7.0]])
    cost = squared_euclidean_cost(values, values)
    plan = balanced_sinkhorn(cost, epsilon=0.05, iterations=300)
    assert float(torch.diagonal(plan).sum()) > 0.99
    assert torch.allclose(plan.sum(dim=1), torch.full((3,), 1 / 3), atol=1e-5)
    assert torch.allclose(plan.sum(dim=0), torch.full((3,), 1 / 3), atol=1e-5)
    projection = barycentric_projection(plan, values)
    assert torch.allclose(projection, values, atol=1e-3)


def test_rectangular_sinkhorn_and_diagnostics_are_finite():
    source = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[0.0], [0.5], [1.0]])
    plan = balanced_sinkhorn(
        squared_euclidean_cost(source, target), epsilon=0.2, iterations=200
    )
    diagnostic = coupling_diagnostics(plan)
    assert plan.shape == (2, 3)
    assert diagnostic["row_marginal_max_error"] < 1e-5
    assert diagnostic["column_marginal_max_error"] < 1e-5
    assert 0 <= diagnostic["normalized_entropy_mean"] <= 1
