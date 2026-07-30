# Processed AML session closeout

날짜: 2026-07-30

Checkpoint commit: `8fece8b`

검증: `90 passed, 6 skipped`

## 목적

이 문서는 processed AML hypothesis-generation session을 닫으면서 삭제한
disposable run state와 Git에 남긴 durable evidence를 기록한다. 다음 세션은
AML–Nuñez paired cytometry marker-information landscape를 별도 protocol로
시작한다.

## 보존된 과학적 결론

### Within-modality same-cell recovery

- AML SF/H19: kNN50 skill 0.213, MLP skill 0.242
- AML SF/clinical10: kNN50 skill 0.178, MLP skill 0.173
- AML CyTOF/H19: kNN50 skill 0.125, MLP skill 약 0
- AML CyTOF/clinical10: kNN50 skill 0.116, MLP skill 음수
- Point, distribution과 fixed-classifier biology는 서로 대체할 수 없었다.

### Shared-marker information

- SF에서는 CCR7이 clinical10→H19 global skill gap의 40.5%를 단독 복원했다.
- CyTOF에서는 CD45RA 55.2%, HLA-DR 32.8%, CD38 25.7%, CCR7 25.3%,
  CD123 19.3%였다.
- Equal-weight kNN에서는 marker 수 증가와 성능이 비단조적이었다.
- Processed-data compact candidates는 adaptive shortlist이며 independent
  sufficient-panel evidence가 아니다.

### Literature methods

- SF/H19에서는 CytoVI와 UVAE가 양수 skill을 보였지만 simple MLP보다
  우수하지 않았다.
- CyTOF에서는 CytoVI와 UVAE의 mean skill이 음수였다.
- cyCombine은 distribution 폭을 보존하면서도 same-cell skill이 모든
  조건에서 음수였다.

### Rare biology와 pairing triage

- CyTOF `T cell gd` full-marker recall은 0.986이었다.
- TCRgd를 숨기면 kNN/MLP/literature method 모두 full recall을 회복하지
  못했다.
- 초기 SF→CyTOF residual experiment에서 correct-X가 shuffled-X 또는
  H-only residual보다 안정적인 rare-T gain을 보이지 않았다.
- 과거 paired-count와 `unpaired_control` 설계는 pairing identity와
  teacher-data budget을 분리하지 못했으므로 pairing evidence로 사용하지
  않는다.

상세 marker, population과 method 표는 아래 세 문서에 보존한다.

- `same_cell_recoverability_v0_results.md`
- `h19_shared_marker_ablation_v0_results.md`
- `literature_imputation_baselines_v0.md`

## 정리한 disposable state

| 경로 | 정리 전 크기 | 내용 | 처분 |
|---|---:|---|---|
| `outputs/aml_same_cell_recoverability_v0/` | 122 MB | fold/seed predictions와 patient summaries | 삭제 |
| `outputs/aml_h19_addback_screen_v0/` | 67 MB | add-back runs | 삭제 |
| `outputs/aml_h19_targeted_pairs_v0/` | 46 MB | targeted pair runs | 삭제 |
| `outputs/aml_h19_compact_and_removal_v0/` | 66 MB | compact/removal runs | 삭제 |
| `outputs/aml_literature_baselines_v0/` | 70 MB | cyCombine/CytoVI/UVAE runs | 삭제 |
| `outputs/sf_to_cytof_rare_population_ablation_cv/` | 19 MB | legacy residual probe | 삭제 |
| `outputs/sf_to_cytof_rare_population_paired_count_cv/` | 139 MB | confounded paired-count curve | 삭제 |
| `outputs/sf_to_cytof_rare_population_unpaired_control_cv/` | 6.6 MB | invalid unpaired-control path | 삭제 |
| `logs/` | 19 MB | completed/retried LSF stdout/stderr | 삭제 |
| `benchmark/audits/*/reference_rows/` | 139 MB | generated processed row-index banks | 삭제 |
| `benchmark/audits/aml_sf_cytof_*` | 약 19 MB | superseded phase-0 generated audit tables | 삭제 |

삭제된 output과 log는 Git에 포함되지 않았으며 원자료가 아니다. 필요한 핵심
수치와 해석은 `docs/research_record.md`와 위 detailed report에 보존했다.
실행 code와 AML v0 config는 checkpoint commit에서 재현 가능하다.

## 유지한 local reusable state

- AML processed expression/label data
- AML `self_recoverability_cache/processed_seed4207` 1.6 GB
- Nuñez public raw FCS, metadata, download/checksum manifest
- Nuñez analysis-ready HDF5
- source code, tests와
  `configs/archive/2026-07-27_processed_aml_v0/`의 completed AML v0 configs

AML cache는 다음 landscape의 동일-row smoke test에 재사용할 수 있어
삭제하지 않았다. 새 frozen row policy가 달라지면 그때 명시적으로
재생성한다.

## Active handoff

- Scientific plan: `docs/benchmark_plan.md`
- Machine-readable design: `configs/benchmark/landscape_v1.yaml`
- First task: AML/Nuñez marker alias와 H19/H20/H9 manifest freeze
