# CytoAlign benchmark plan

마지막 갱신: 2026-07-28  
상태: processed-AML hypothesis-generation 완료, raw primary benchmark 준비  
Canonical 연구 기록: `docs/research_record.md`

## 1. 목적

CytoAlign의 목표는 새 imputation architecture를 제안하는 것이 아니다.
서로 다른 cytometry panel에서 공통 marker `H`만 관찰될 때
target-exclusive marker `Y`가 다음 중 어느 상태인지 판정하는 benchmark를
만드는 것이다.

- `recoverable`
- `conditionally_recoverable`
- `non_individualized_prior_reproduction`
- `unrecoverable_under_tested_methods`
- `unsupported_h_domain`
- `unsupported_target_reference`
- `not_evaluable`

최종 산출물은 marker·population·direction별 recoverability map, 검증된
abstention rule, method-selection decision tree와 재현 가능한 benchmark
harness다.

핵심 메시지는 다음과 같다.

> Recoverability is determined by backbone information, conditional
> ambiguity and support—not by model complexity alone.

## 2. 현재 evidence가 고정한 결정

Processed AML 실험은 다음 운영 결정을 지지한다.

1. **추가 architecture 탐색을 우선하지 않는다.** CytoVI, UVAE, cyCombine은
   simple kNN50/MLP를 일관되게 넘지 못했다.
2. **Same-platform self ceiling을 먼저 측정한다.** 같은 modality에서도
   shared `H`가 `Y`를 결정하지 못하면 cross-platform model로 구제할 수
   있다는 claim을 하지 않는다.
3. **세 endpoint를 분리한다.**
   same-cell point fidelity, distribution fidelity, fixed downstream biology는
   서로 대체할 수 없다.
4. **Shared-set sufficiency는 target-specific이다.** Endpoint-free minimum
   panel을 찾지 않는다.
5. **Rare population defining marker를 숨기지 않는다.** CyTOF T-cell-gd가
   required endpoint이면 TCRgd를 직접 측정한다.
6. **Panel size를 정보량으로 간주하지 않는다.** Equal-weight kNN은 noisy
   distance dimension 때문에 shared-marker 수에 대해 비단조적이었다.
7. **Processed compact panels는 frozen raw 후보일 뿐이다.** 같은 processed
   data에서 adaptive하게 선택됐으므로 독립 확인 전에는 sufficient set으로
   부르지 않는다.

상세 근거와 모든 수치는 `docs/research_record.md`에 있다.

## 3. 데이터와 claim 범위

| 역할 | 데이터 | 허용 범위 |
|---|---|---|
| Primary | AML spectral flow ↔ CyTOF | raw FCS, technical-QC-only events, 93 paired specimens/83 patients |
| External | Nuñez FACSymphony ↔ CyTOF | released pre-gated FCS의 모든 row, 164 pairs/51 patients |
| Secondary | AML clinical flow H4 | raw detector-level label-free audit 통과 시에만 |
| Sensitivity | 현재 processed AML | upstream pre-gated compartment에 조건부 |

Processed AML은 raw 대비 SF 37.68%, CyTOF 67.72%의 서로 다른 event만
남긴다. 따라서 processed 결과를 full-population prevalence,
unseen-population discovery 또는 primary cross-platform evidence로 사용하지
않는다.

## 4. Frozen candidate panels

Raw confirmatory screen에서 평가할 후보를 더 늘리지 않는다.

| Modality | 역할 | Panel |
|---|---|---|
| SF | lower bound | clinical10 |
| SF | compact candidate | clinical10 + CCR7 + CD117 + CD123 |
| SF | upper reference | H19 |
| CyTOF | lower bound | clinical10 |
| CyTOF | compact candidate A | clinical10 + CD45RA + HLA-DR + CD38 |
| CyTOF | compact candidate B | clinical10 + CD45RA + CD38 + CD123 |
| CyTOF | upper reference | H19 |

Clinical10은
`CD3, CD4, CD8, CD14, CD19, CD20, CD33, CD34, CD45, CD56`이다.

SF에서 CCR7은 hidden functional-marker recovery, CD117/CD123은 fixed
classifier biology를 위해 선택했다. CyTOF에서는 CD45RA, HLA-DR, CD38,
CD123이 서로 다른 target 정보를 제공했다.

H4는 위 후보와 같은 단계가 아니다. Raw label-free detector/channel audit와
row reconstruction을 통과한 경우에만 secondary stress test로 추가한다.

## 5. 공통 평가 계약

### 5.1 Split과 정보 격리

- Biological inference unit은 patient다.
- Longitudinal/multiple specimens는 patient 단위로 함께 이동한다.
- Patient-grouped outer folds를 사용한다.
- Scaling, calibration, reference bank와 threshold는 fit/validation에서만
  결정한다.
- Test hidden marker, label, outcome 또는 target statistic으로 method,
  retry, threshold를 선택하지 않는다.
- Event inclusion에 biological H/Y, cell label 또는 outcome을 사용하지
  않는다.
- 모든 방법은 같은 frozen split과 data-matched reference budget을 쓴다.

### 5.2 Marker endpoints

- normalized MAE와 fit-median null-relative skill
- within-specimen Spearman correlation
- marker median, positive fraction, upper quantile
- IQR/dynamic-range retention
- H–Y 및 Y–Y joint fidelity

Marker recovery call:

- 95% patient-bootstrap CI가 0보다 크면 `recoverable`
- CI가 0을 포함하면 `unresolved`
- CI가 0보다 작으면 `worse_than_null`

이 call은 self same-cell truth에서만 사용한다. Cross-platform에는 exact
cell correspondence가 없으므로 cell-level coverage를 주장하지 않는다.

### 5.3 Biology endpoints

- Fixed target-domain/full-marker classifier replay
- balanced accuracy, macro-F1, macro-AUPRC
- class별 recall, precision, AUPRC, prevalence error
- defining marker가 hidden인 rare population stress test

Representation마다 classifier를 새로 학습하는 평가는 supplementary다.
Observed `H`만으로 쉬운 label을 유지하는 것은 hidden `Y` recovery의 증거가
아니다.

### 5.4 Nulls

- fit-marker median
- row-shuffled observed markers
- target-prior/source-blind prediction
- patient-level wrong/deranged source
- H-only target-domain ceiling

Source-conditioned claim은 target-prior와 wrong-patient null을 모두 이겨야
한다.

## 6. 실행 gate

### Gate 0 — Raw data freeze

필수 산출물:

1. Raw SF/CytoF FCS inventory와 checksum
2. Technical-QC-only event mask
3. Raw expression matrix와 original row index
4. Row-label alignment 검증용 label artifact
5. Portable patient/specimen/visit metadata
6. Marker/channel mapping과 transform manifest
7. Patient-grouped split manifest
8. Fixed reference-row bank
9. Protocol, code, config, environment digest

Full preflight와 forbidden-information test가 모두 통과하기 전에는 primary
run을 제출하지 않는다.

### Gate 1 — Raw within-platform confirmatory ceiling

SF와 CyTOF를 별도로 평가한다.

- Panels: Section 4의 frozen 후보만 사용
- Targets: 우선 `full panel - H19`로 고정하여 panel 간 target을 동일하게
  유지
- Methods: median, exact kNN50, simple MLP
- Primary comparison: H19 vs compact candidates vs clinical10
- Biology: processed experiment와 동일한 fixed-classifier replay

확인할 가설:

1. SF/H19의 functional markers가 raw event에서도 양수 skill을 갖는가?
2. SF compact panel이 H19 macro-F1을 유지하면서 quantitative gap 일부를
   복원하는가?
3. CyTOF compact panels가 H19과 global skill에서 동급인가?
4. Ki-67, CD25, CD96, CD57, CD66b와 rare T-cell-gd failure가 재현되는가?
5. Panel-size non-monotonicity가 raw에서도 유지되는가?

Stage-2/3 후보를 raw 결과를 보고 다시 바꾸지 않는다. 후보 변경이 필요하면
새 protocol ID로 exploratory cycle을 분리한다.

### Gate 2 — Failure-mechanism positive controls

Same-platform ceiling이 재현된 marker를 대상으로 다음을 분리한다.

1. `conditional ambiguity`: 비슷한 H에서 Y가 얼마나 다중적인가?
2. `H-support risk`: query H가 target-fit support 밖에 있는가?
3. `target-reference support`: phenotype/reference가 충분한가?
4. `method instability`: fold/seed/model 차이가 큰가?

Semi-synthetic positive controls에서 support removal과 ambiguity 증가가
예상한 error/risk 변화를 만드는지 먼저 검증한다. 이 gate에서 새 deep
architecture를 개발하지 않는다.

### Gate 3 — Cross-platform H calibration and H→Y

Gate 1 ceiling이 충분한 marker/regime만 진행한다.

```text
same-platform H→Y ceiling
  → cross-platform H→H calibration
  → source H→target Y
  → target-prior / wrong-patient individualization
  → fixed downstream replay
```

Primary methods는 kNN50과 simple MLP다. Literature model은 simple baseline이
실패하지 않는 중간 regime에서 한 개의 reference comparator로만 추가한다.

Pairing pilot은 source/target bank, row 수와 총 data budget을 고정한
`{0, 8, all}` 조건으로 수행한다.

- identity-blind pooled
- patient-level deranged
- patient-matched

Pairing identity와 teacher-data 양을 동시에 바꾸는 기존 paired-count 설계는
사용하지 않는다.

### Gate 4 — Validation-locked abstention

Validation patients에서 risk formula와 rejection threshold를 고정한다.

- held-out realized error 예측력
- risk–coverage curve
- random rejection 대비 improvement
- marker/population별 coverage와 residual risk

큰 uncertainty 값만으로 calibrated uncertainty라 부르지 않는다.

### Gate 5 — External validation

AML에서 panel rule, method choice, calibration과 rejection threshold를 모두
lock한 뒤 Nuñez cohort에 적용한다. External direction이 재현되지 않으면
general recoverability rule을 주장하지 않는다.

## 7. GO/NO-GO

### GO

- Raw manifest와 row alignment가 완전하고 digest가 고정됨
- Self recoverability가 patient level에서 재현됨
- Cross-platform source gain이 target-prior와 wrong-patient null보다 우수함
- Support/ambiguity risk가 held-out error를 예측함
- Locked rejection이 random rejection보다 우수함
- Marker/regime rule이 external cohort로 이전됨

### NO-GO 또는 claim 제한

- Raw event mask나 row alignment를 검증하지 못함
- Defining marker가 hidden일 때 rare population을 복원하지 못함
- Self ceiling이 null과 같거나 낮음
- Source를 shuffle/derange해도 결과가 유지됨
- Distribution만 맞고 same-cell/biology evidence가 없음
- Adaptive panel/method 선택을 독립 확인 없이 confirmatory로 사용함

NO-GO는 프로젝트 실패가 아니라 `unrecoverable`, `not_evaluable` 또는
`direct_measurement_required`라는 actionable result다.

## 8. 지금 하지 않을 일

- 9-marker 이상의 exhaustive panel combination search
- 같은 processed data에서 compact panel 재선택
- VAE/OT/residual architecture hyperparameter sweep
- UMAP 또는 marginal distribution만으로 성공 판정
- Representation-specific classifier refit을 primary biology endpoint로 사용
- Raw audit 전 clinical H4 primary run
- Wrong-patient control 없는 pairing claim

## 9. 다음 작업 순서

1. Raw AML SF/CytoF FCS inventory와 checksum 갱신
2. Technical-QC-only row mask와 expression/row-index artifact 생성
3. Portable metadata, marker mapping과 patient split 동결
4. Fixed reference-row bank와 endpoint manifest 동결
5. Full preflight 및 forbidden-information test
6. Gate-1 one-fold/one-seed resource pilot
7. Frozen candidate panels의 full raw OOF confirmatory run
8. Gate-2 support/ambiguity positive controls
9. Gate-3 양방향 AML cross-platform pilot
10. Locked abstention 및 Nuñez external validation

## 10. 재현 artifact

- Canonical research record: `docs/research_record.md`
- Active plan: `docs/benchmark_plan.md`
- Machine-readable protocol: `configs/benchmark/protocol_v1.yaml`
- Audit drafts: `benchmark/audits/`
- Processed same-cell outputs:
  `outputs/aml_same_cell_recoverability_v0/`
- Processed ablation outputs:
  `outputs/aml_h19_addback_screen_v0/`,
  `outputs/aml_h19_targeted_pairs_v0/`,
  `outputs/aml_h19_compact_and_removal_v0/`
- Literature outputs:
  `outputs/aml_literature_baselines_v0/`
- Archived detailed reports:
  `docs/archive/2026-07-27_processed_aml_v0/`
- Current full test suite: `100 passed`

## 11. 실행 환경 주의

현재 LPC shell 초기화에서는 login shell이 `no direct access allowed`를
출력할 수 있다. Repository 검사, local wrapper와 test는 non-login shell로
실행한다.

예:

```bash
bash --noprofile --norc -c 'pytest -q'
```

LSF job은 scheduler가 제공하는 execution environment와 명시적 conda
executable을 사용한다. Application 시작 전 `EXIT 255` job-file staging
failure는 동일 command로 재제출하고 failure artifact와 replacement job ID를
함께 보존한다.
