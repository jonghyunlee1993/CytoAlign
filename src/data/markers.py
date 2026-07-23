"""Explicit marker canonicalization for cross-panel experiments.

The original CSV column names are retained for loading data.  Canonical names
are used only to decide which biological measurements are shared between two
panels.  This prevents a silent column-order change from altering an
experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


def _token(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


DEFAULT_MARKER_ALIASES: dict[str, str] = {
    "PD-1": "PD-1",
    "PD1": "PD-1",
    "CD279": "PD-1",
    "CCR7": "CCR7",
    "CD197": "CCR7",
}

DEFAULT_TECHNICAL_MARKERS: frozenset[str] = frozenset(
    {"FSC-A", "SSC-A", "SSC-B-A", "AF-A"}
)


def _normalized_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    if aliases is None:
        aliases = DEFAULT_MARKER_ALIASES
    return {_token(key): str(value).strip() for key, value in aliases.items()}


def canonical_marker_name(name: str, aliases: Mapping[str, str] | None = None) -> str:
    """Return a deterministic comparison name for one marker column."""
    stripped = str(name).strip()
    return _normalized_aliases(aliases).get(_token(stripped), _token(stripped))


def _canonical_lookup(
    markers: Sequence[str], aliases: Mapping[str, str] | None
) -> tuple[tuple[str, ...], dict[str, str]]:
    canonical = tuple(canonical_marker_name(marker, aliases) for marker in markers)
    duplicates = sorted({name for name in canonical if canonical.count(name) > 1})
    if duplicates:
        raise ValueError(
            "A panel contains multiple columns for the same canonical marker: "
            f"{duplicates}"
        )
    return canonical, dict(zip(canonical, map(str, markers)))


@dataclass(frozen=True)
class PairMarkerManifest:
    """Ordered source/target column contract for one translation direction."""

    source_markers: tuple[str, ...]
    target_markers: tuple[str, ...]
    common_markers: tuple[str, ...]
    source_common_columns: tuple[str, ...]
    target_common_columns: tuple[str, ...]
    source_exclusive_columns: tuple[str, ...]
    target_exclusive_columns: tuple[str, ...]
    target_primary_exclusive_columns: tuple[str, ...]
    target_technical_exclusive_columns: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def build_pair_marker_manifest(
    source_markers: Sequence[str],
    target_markers: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
    technical_markers: Sequence[str] = tuple(DEFAULT_TECHNICAL_MARKERS),
) -> PairMarkerManifest:
    """Build an aligned marker manifest while preserving source common order."""
    source = tuple(map(str, source_markers))
    target = tuple(map(str, target_markers))
    source_canonical, source_lookup = _canonical_lookup(source, aliases)
    target_canonical, target_lookup = _canonical_lookup(target, aliases)
    target_set = set(target_canonical)
    common = tuple(name for name in source_canonical if name in target_set)
    common_set = set(common)

    source_exclusive = tuple(
        original
        for canonical, original in zip(source_canonical, source)
        if canonical not in common_set
    )
    target_exclusive = tuple(
        original
        for canonical, original in zip(target_canonical, target)
        if canonical not in common_set
    )
    technical = {_token(marker) for marker in technical_markers}
    target_technical = tuple(
        marker for marker in target_exclusive if _token(marker) in technical
    )
    target_primary = tuple(
        marker for marker in target_exclusive if _token(marker) not in technical
    )

    return PairMarkerManifest(
        source_markers=source,
        target_markers=target,
        common_markers=common,
        source_common_columns=tuple(source_lookup[name] for name in common),
        target_common_columns=tuple(target_lookup[name] for name in common),
        source_exclusive_columns=source_exclusive,
        target_exclusive_columns=target_exclusive,
        target_primary_exclusive_columns=target_primary,
        target_technical_exclusive_columns=target_technical,
    )
