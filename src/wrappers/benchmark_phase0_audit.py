"""Generate the draft AML SF--CyTOF Phase 0 audit artifacts."""

from __future__ import annotations

import argparse
import json

from src.benchmark.phase0_audit import build_phase0_audit, write_phase0_audit
from src.benchmark.reference_rows import materialize_reference_row_indices
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/benchmark/protocol_v1.yaml",
    )
    parser.add_argument("--data-root", default="data/AML")
    parser.add_argument(
        "--aml-metadata",
        help="Path to the protocol-locked AML_meta_111224.csv",
    )
    parser.add_argument(
        "--output",
        default="benchmark/audits/aml_sf_cytof_phase0_draft_v1",
    )
    parser.add_argument(
        "--hash-source-files",
        action="store_true",
        help="Hash the ~37 GB paired expression CSVs; omitted in quick draft mode",
    )
    parser.add_argument(
        "--cohort-policy",
        choices=("primary", "low_event_exclusion_sensitivity"),
        default="primary",
    )
    parser.add_argument(
        "--base-split-manifest",
        help=(
            "Primary split_manifest.json required for exclusion sensitivities"
        ),
    )
    parser.add_argument(
        "--materialize-reference-rows",
        action="store_true",
        help="Create and validate shared label-free uint32 reservoir indices",
    )
    args = parser.parse_args()
    if args.materialize_reference_rows and not args.hash_source_files:
        parser.error(
            "--materialize-reference-rows requires --hash-source-files"
        )
    audit = build_phase0_audit(
        args.data_root,
        load_config(args.protocol),
        hash_source_files=args.hash_source_files,
        authoritative_metadata_path=args.aml_metadata,
        cohort_policy=args.cohort_policy,
        base_split_manifest_path=args.base_split_manifest,
    )
    if args.materialize_reference_rows:
        materialize_reference_row_indices(audit, args.output)
    result = write_phase0_audit(audit, args.output)
    print(
        json.dumps(
            {
                **result,
                "audit_summary": audit["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
