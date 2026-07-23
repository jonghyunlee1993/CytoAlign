import numpy as np

from src.data.prototypes import (
    build_disjoint_prototype_pools,
    shuffled_target_group_order,
)


def test_prototype_pools_are_disjoint_and_shuffle_only_within_cell_type():
    specimens = np.repeat(["P1", "P2", "P3"], 24)
    labels = np.tile(np.repeat(["T", "B"], 12), 3)
    groups = build_disjoint_prototype_pools(
        specimens, labels, minimum_cells_per_side=4, random_state=3
    )
    assert len(groups) == 6
    for group in groups:
        assert not set(group.source_indices) & set(group.target_indices)
    order = shuffled_target_group_order(groups, random_state=9)
    for source_index, target_index in enumerate(order):
        assert source_index != target_index
        assert groups[source_index].cell_type == groups[target_index].cell_type
