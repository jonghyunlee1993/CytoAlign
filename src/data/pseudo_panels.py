"""Leakage-safe two-sided pseudo-panel definitions.

The existing one-sided pseudo-panel benchmark masks target-exclusive markers
and therefore measures ``H -> Y`` predictability.  A concept bridge also reads
source-exclusive markers, so its exact-truth benchmark needs two disjoint views
of the same full-panel cell: ``[H, X]`` and ``[H, Y]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _ordered_subset(
    requested: Sequence[str], full_markers: tuple[str, ...], name: str
) -> tuple[str, ...]:
    values = tuple(map(str, requested))
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{name} contains duplicate markers: {duplicates}")
    missing = sorted(set(values) - set(full_markers))
    if missing:
        raise ValueError(f"{name} contains markers absent from the full panel: {missing}")
    requested_set = set(values)
    return tuple(marker for marker in full_markers if marker in requested_set)


@dataclass(frozen=True)
class TwoSidedPseudoPanelManifest:
    """Immutable marker contract for an exact-truth two-view experiment."""

    full_markers: tuple[str, ...]
    common_markers: tuple[str, ...]
    source_exclusive_markers: tuple[str, ...]
    target_exclusive_markers: tuple[str, ...]
    source_markers: tuple[str, ...]
    target_markers: tuple[str, ...]
    unused_markers: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def build_two_sided_pseudo_panel_manifest(
    full_markers: Sequence[str],
    *,
    common_markers: Sequence[str],
    source_exclusive_markers: Sequence[str],
    target_exclusive_markers: Sequence[str],
    require_complete_partition: bool = False,
) -> TwoSidedPseudoPanelManifest:
    """Partition one full panel into explicit ``H``, ``X``, and ``Y`` views.

    Marker order always follows the original full-panel order, even when the
    requested marker lists use a different order.  This mirrors the immutable
    ordering guarantees used by :mod:`src.data.markers`.
    """

    full = tuple(map(str, full_markers))
    if not full:
        raise ValueError("full_markers must not be empty")
    duplicates = sorted({value for value in full if full.count(value) > 1})
    if duplicates:
        raise ValueError(f"full_markers contains duplicate markers: {duplicates}")

    common = _ordered_subset(common_markers, full, "common_markers")
    source_only = _ordered_subset(
        source_exclusive_markers, full, "source_exclusive_markers"
    )
    target_only = _ordered_subset(
        target_exclusive_markers, full, "target_exclusive_markers"
    )
    if not common or not source_only or not target_only:
        raise ValueError("H, X, and Y must each contain at least one marker")

    groups = {
        "H": set(common),
        "X": set(source_only),
        "Y": set(target_only),
    }
    overlaps = {
        f"{left}/{right}": sorted(groups[left] & groups[right])
        for left, right in (("H", "X"), ("H", "Y"), ("X", "Y"))
        if groups[left] & groups[right]
    }
    if overlaps:
        raise ValueError(f"Pseudo-panel marker groups overlap: {overlaps}")

    used = groups["H"] | groups["X"] | groups["Y"]
    unused = tuple(marker for marker in full if marker not in used)
    if require_complete_partition and unused:
        raise ValueError(f"Full panel contains unassigned markers: {list(unused)}")
    source = tuple(marker for marker in full if marker in groups["H"] | groups["X"])
    target = tuple(marker for marker in full if marker in groups["H"] | groups["Y"])
    return TwoSidedPseudoPanelManifest(
        full_markers=full,
        common_markers=common,
        source_exclusive_markers=source_only,
        target_exclusive_markers=target_only,
        source_markers=source,
        target_markers=target,
        unused_markers=unused,
    )


@dataclass(frozen=True)
class TwoSidedPseudoPanelViews:
    """Same-cell source and target views retained only for exact-truth evaluation."""

    source_values: np.ndarray
    target_values: np.ndarray
    common_values: np.ndarray
    source_exclusive_values: np.ndarray
    target_exclusive_values: np.ndarray


def make_two_sided_pseudo_panel_views(
    values: np.ndarray,
    markers: Sequence[str],
    manifest: TwoSidedPseudoPanelManifest,
) -> TwoSidedPseudoPanelViews:
    """Project a full-panel matrix into aligned source/target pseudo views."""

    array = np.asarray(values)
    marker_order = tuple(map(str, markers))
    if array.ndim != 2 or array.shape[1] != len(marker_order):
        raise ValueError("values and markers do not define an aligned matrix")
    if marker_order != manifest.full_markers:
        raise ValueError("markers do not match the pseudo-panel full marker contract")
    lookup = {marker: index for index, marker in enumerate(marker_order)}

    def select(names: tuple[str, ...]) -> np.ndarray:
        return np.asarray(array[:, [lookup[name] for name in names]], dtype=np.float32)

    return TwoSidedPseudoPanelViews(
        source_values=select(manifest.source_markers),
        target_values=select(manifest.target_markers),
        common_values=select(manifest.common_markers),
        source_exclusive_values=select(manifest.source_exclusive_markers),
        target_exclusive_values=select(manifest.target_exclusive_markers),
    )
