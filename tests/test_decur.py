import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.losses.decur import (
    decoded_population_summary_loss,
    intra_modal_barlow_loss,
    normalized_cross_correlation,
    prototype_decur_loss,
    weighted_decur_loss,
    weighted_normalized_cross_correlation,
)


def test_prototype_decur_prefers_aligned_common_and_decorrelated_unique():
    generator = torch.Generator().manual_seed(11)
    common = torch.randn(4000, 3, generator=generator)
    source_unique = torch.randn(4000, 2, generator=generator)
    target_unique = torch.randn(4000, 2, generator=generator)
    source = torch.cat([common, source_unique], dim=1)
    target = torch.cat([common + 0.01 * torch.randn(common.shape, generator=generator), target_unique], dim=1)
    good = prototype_decur_loss(source, target, n_common=3)["loss"]
    shuffled = prototype_decur_loss(
        source, target[torch.randperm(target.shape[0], generator=generator)], n_common=3
    )["loss"]
    assert float(good) < float(shuffled)
    assert normalized_cross_correlation(source, target).shape == (5, 5)


def test_intra_modal_barlow_rejects_shape_mismatch():
    values = torch.randn(20, 4)
    assert float(intra_modal_barlow_loss(values, values)) >= 0
    with pytest.raises(ValueError, match="aligned"):
        intra_modal_barlow_loss(values, values[:, :3])


def test_weighted_decur_uses_soft_coupling_and_backpropagates():
    generator = torch.Generator().manual_seed(17)
    common = torch.randn(64, 2, generator=generator)
    source = torch.cat(
        [common, torch.randn(64, 2, generator=generator)], dim=1
    ).requires_grad_()
    target = torch.cat(
        [
            common + 0.01 * torch.randn(64, 2, generator=generator),
            torch.randn(64, 2, generator=generator),
        ],
        dim=1,
    )
    identity = torch.eye(64) / 64
    uniform = torch.full((64, 64), 1.0 / (64 * 64))
    aligned = weighted_decur_loss(
        source, target, identity, n_common=2
    )["loss"]
    unpaired = weighted_decur_loss(
        source, target, uniform, n_common=2
    )["loss"]
    assert float(aligned) < float(unpaired)
    aligned.backward()
    assert torch.isfinite(source.grad).all()
    correlation = weighted_normalized_cross_correlation(
        source.detach(), target, identity
    )
    assert correlation.shape == (4, 4)


def test_weighted_correlation_rejects_bad_plan_shape():
    with pytest.raises(ValueError, match="align"):
        weighted_normalized_cross_correlation(
            torch.randn(8, 3),
            torch.randn(7, 3),
            torch.ones(8, 8),
        )


def test_single_common_axis_has_no_off_diagonal_terms():
    values = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    result = prototype_decur_loss(values, values, n_common=1)
    assert torch.isfinite(result["loss"])


def test_population_summary_loss_prefers_matching_group_distributions_and_backpropagates():
    generator = torch.Generator().manual_seed(23)
    target = torch.randn(12, 3, generator=generator)
    good = (target + 0.01 * torch.randn(target.shape, generator=generator)).requires_grad_()
    bad = 1.7 * target + 1.0
    good_loss = decoded_population_summary_loss(
        good, target, n_groups=3, cells_per_group=4
    )["loss"]
    bad_loss = decoded_population_summary_loss(
        bad, target, n_groups=3, cells_per_group=4
    )["loss"]
    assert float(good_loss) < float(bad_loss)
    good_loss.backward()
    assert torch.isfinite(good.grad).all()


def test_population_summary_loss_validates_group_shape():
    with pytest.raises(ValueError, match="rows"):
        decoded_population_summary_loss(
            torch.zeros(12, 2),
            torch.zeros(12, 2),
            n_groups=2,
            cells_per_group=4,
        )
