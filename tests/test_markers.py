import pytest

from src.data.markers import build_pair_marker_manifest


SPECTRAL = [
    "SSC-A",
    "FSC-A",
    "SSC-B-A",
    "CD38",
    "CD4",
    "CD123",
    "CD20",
    "CD56",
    "HLA-DR",
    "CD14",
    "CTLA-4",
    "CD45",
    "CD33",
    "CD3",
    "CD117",
    "CD27",
    "CD95",
    "Ki-67",
    "PD-1",
    "CD25",
    "TIGIT",
    "CCR7",
    "FOXP3",
    "CD34",
    "CD19",
    "EOMES",
    "T-bet",
    "CD8",
    "CD45RA",
    "AF-A",
]

CYTOF = [
    "CD3",
    "CD4",
    "CD8",
    "CD11c",
    "CD13",
    "CD14",
    "CD15",
    "CD16",
    "CD19",
    "CD20",
    "CD25",
    "CD27",
    "CD28",
    "CD33",
    "CD34",
    "CD38",
    "CD45",
    "CD47",
    "CD56",
    "CD57",
    "CD64",
    "CD66b",
    "CD69",
    "CD96",
    "CD117",
    "CD123",
    "CD127",
    "CD161",
    "CD183",
    "CD185",
    "CD194",
    "CD196",
    "CD197",
    "CD200",
    "CD274",
    "CD276",
    "CD279",
    "CD294",
    "CD366",
    "CD45RA",
    "CD45RO",
    "HLA-DR",
    "IgD",
    "TCRgd",
]


def test_aml_marker_manifest_has_17_exact_and_19_canonical_shared_markers():
    exact = build_pair_marker_manifest(SPECTRAL, CYTOF, aliases={})
    canonical = build_pair_marker_manifest(SPECTRAL, CYTOF)

    assert len(exact.common_markers) == 17
    assert len(canonical.common_markers) == 19
    assert "PD-1" in canonical.common_markers
    assert "CCR7" in canonical.common_markers
    assert (
        canonical.source_common_columns[canonical.common_markers.index("PD-1")]
        == "PD-1"
    )
    assert (
        canonical.target_common_columns[canonical.common_markers.index("PD-1")]
        == "CD279"
    )


def test_target_technical_channels_are_not_primary_endpoints():
    manifest = build_pair_marker_manifest(CYTOF, SPECTRAL)
    assert set(manifest.target_technical_exclusive_columns) == {
        "FSC-A",
        "SSC-A",
        "SSC-B-A",
        "AF-A",
    }
    assert not set(manifest.target_primary_exclusive_columns) & set(
        manifest.target_technical_exclusive_columns
    )


def test_alias_collision_is_rejected():
    with pytest.raises(ValueError, match="multiple columns"):
        build_pair_marker_manifest(["PD-1", "CD279"], ["PD-1"])
