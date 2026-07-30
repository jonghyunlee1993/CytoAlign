# Paired cytometry marker-information landscape

마지막 갱신: 2026-07-30

상태: processed AML hypothesis-generation 종료, AML–Nuñez landscape benchmark 준비

Canonical 연구 기록: `docs/research_record.md`

## 1. 연구 목적

이 프로젝트의 1차 목표는 새로운 imputation architecture를 개발하는 것이
아니다. 두 paired clinical cytometry cohort에서 다음을 체계적으로
지도화하는 benchmark를 만드는 것이다.

1. shared backbone으로 복원하기 쉬운 marker와 어려운 marker
2. shared marker와 target marker 사이의 predictive information relationship
3. marker 수와 보존 정보량 사이의 Pareto frontier
4. within-modality ceiling에서 cross-modality로 이동할 때의 성능 저하
5. source-exclusive marker와 patient pairing이 추가하는 정보
6. marker-level 성공이 major/rare population biology를 보존하는 조건

핵심 주장은 다음과 같다.

> Marker recoverability is a property of the observed marker set, target
> marker and biological state. Paired cytometry makes it possible to measure
> how this information degrades across modalities and whether source-exclusive
> markers add patient-specific information.

모델은 landscape를 측정하는 probe다. 모델 순위나 새 architecture는 중심
산출물이 아니다.

## 2. 데이터와 notation

Flow-like modality를 `A`, CyTOF를 `B`라고 둔다.

- `H`: A와 B가 공유하는 marker
- `X`: A에만 관찰되는 marker
- `Y`: B에만 관찰되는 marker

| Cohort | A | B | Native H | X | Y | 분석 단위 |
|---|---|---|---:|---:|---:|---|
| AML | spectral flow | CyTOF | 19 | 7 | 25 | 93 specimens, 83 patients |
| Nuñez | FACSymphony | CyTOF | 20 | 6 | 17 | 164 visits, 51 patients |

AML과 Nuñez의 source–target은 같은 환자·방문의 aliquot-level pair이며 동일
세포 pair가 아니다. 각 modality 내부에서 marker를 숨기는 실험에만 exact
same-cell truth가 있다.

현재 processed AML은 upstream-gated compartment에 조건부다. Nuñez 공개 FCS
역시 pre-gated data다. 따라서 primary estimand는 다음으로 제한한다.

> Marker recovery within analysis-ready, pre-gated clinical cytometry events.

AML raw FCS는 primary 선결조건이 아니다. Current processed selection에 대한
robustness/sensitivity track으로만 유지하고, 결론이 바뀔 때만 raw를 primary로
승격한다.

## 3. Marker canonicalization

두 cohort를 합치기 전에 최소한 다음 alias를 고정한다.

- `PD-1`, `PD1`, `CD279` → `PD-1`
- `CCR7`, `CD197` → `CCR7`
- `CTLA-4`, `CTLA4` → `CTLA-4`
- `Ki-67`, `KI67`, `Ki67` → `Ki-67`
- `TCRgd`, `TCRGD`, `PANGD`, `PANGT` → `TCRgd`
- `IgD`, `IGD` → `IgD`
- `GranzymeB`, `GYMB`, `GZMB` → `GZMB`
- `IgG4`, `IGG4` → `IgG4`

Canonical alias는 이름을 맞추는 규칙일 뿐 reagent clone, staining chemistry
또는 dynamic range가 같다는 가정이 아니다. Cross-cohort result에는
measurement provenance를 함께 기록한다.

## 4. 세 shared-marker level

### 4.1 Native H

- AML: H19
- Nuñez: H20

각 cohort가 실제로 제공하는 최대 shared information ceiling이다.

### 4.2 Universal H9

네 panel 모두에 존재하는 공통 backbone은 다음 9개다.

`CD3, CD4, CD8, CD19, CD27, CD45RA, CD56, HLA-DR, PD-1`

H9는 cohort 간 marker difficulty와 relationship을 같은 입력 조건에서
비교하는 표준 panel이다.

### 4.3 Minimal S*

H9의 subset 중 AML outer-training/validation에서 선택한 하나의 universal
operating point다. Nuñez에서 panel을 다시 선택하지 않는다.

Native H, H9과 S*의 비교는 서로 다른 손실을 측정한다.

```text
Native H19/H20
  → H9: cross-cohort standardization loss
  → S*: marker minimization loss
```

## 5. Primary task matrix

### 5.1 Within-modality same-cell recovery

| Cohort | Modality | Task |
|---|---|---|
| AML | spectral flow | `H → X` |
| AML | CyTOF | `H → Y` |
| Nuñez | FACSymphony | `H → X` |
| Nuñez | CyTOF | `H → Y` |

Native H, H9, S*를 각각 사용한다. H9/S* track에서는 각 modality의
`full panel - observed set` 전체를 target으로 하여 동일 marker의
cross-cohort 비교 범위를 넓힌다.

### 5.2 Cross-modality paired recovery

두 cohort에서 양방향을 수행한다.

- A→B: `H → Y`와 `H + X → Y`
- B→A: `H → X`와 `H + Y → X`

세 비교량을 분리한다.

- `platform penalty`: within-modality distribution ceiling 대비 cross-modality
  error 증가
- `exclusive gain`: cross H-only 대비 H+exclusive error 감소
- `pairing gain`: wrong-patient 대비 patient-matched error 감소

## 6. Primary methods

| 역할 | 방법 | 해석 |
|---|---|---|
| Null | fit-patient-balanced median / target prior | 관측 정보가 없는 기준 |
| Linear ML | ridge | 선형 marker relationship |
| Local ML | exact kNN50 | local phenotype relationship |
| Nonlinear | two-layer MLP | 추가 비선형성의 가치 |

Primary landscape를 이 네 family로 고정한다. MLP는 단순 fixed architecture를
사용하고 architecture/hyperparameter search를 하지 않는다.

CytoVI, UVAE, cyCombine과 다른 literature method는 landscape가 완성된 뒤
easy/intermediate/impossible regime의 reference comparator로만 사용한다.
S* 선택에는 관여하지 않는다.

## 7. Split과 정보 격리

- Biological inference unit은 patient다.
- 동일 patient의 모든 specimen/visit는 같은 fold에 둔다.
- 각 cohort에서 5 patient-grouped outer folds를 사용한다.
- Scaling, reference bank, model selection과 threshold는 outer fit/validation
  data에서만 결정한다.
- Test hidden marker, cell label, outcome, density 또는 target statistic으로
  panel, model, retry나 threshold를 선택하지 않는다.
- Cell label은 imputer와 cross-modality matching에 사용하지 않는다.
- 모든 방법은 같은 frozen row bank와 data budget을 사용한다.
- Deterministic method는 seed를 replicate로 세지 않는다.
- MLP는 final candidate panel에서만 3 seeds를 평가한다.

AML은 panel discovery cohort다. S*와 landscape definition을 AML에서 lock한
후 Nuñez에 적용한다. Nuñez model 자체는 Nuñez outer-training patients에서
학습하지만 panel, metric, target grouping과 claim rule을 다시 선택하지 않는다.

## 8. Within-modality endpoints

Exact same-cell truth가 있으므로 다음을 모두 평가한다.

### Marker point fidelity

- fit-median null-relative normalized MAE skill
- within-specimen Spearman correlation
- practical effect size와 patient-bootstrap confidence interval

### Distribution fidelity

- normalized 1D Wasserstein distance
- median, positive fraction, IQR와 upper-quantile error
- H–target 및 target–target joint structure

### Cell representation fidelity

- full-marker 대 imputed kNN-neighborhood Jaccard
- neighborhood cell-label purity
- full-marker fixed classifier replay

`recoverable`은 단일 binary label로 축약하지 않는다. Point, distribution,
major biology와 rare biology를 별도 축으로 보고한다.

## 9. Marker-information relationship

각 observed marker `H_j`와 target `T_k`에 대해 다음을 계산한다.

1. marginal Spearman/association
2. `H_j` single-marker recoverability
3. core panel에 `H_j`를 추가한 add-back gain
4. full H에서 `H_j`를 제거한 leave-one-out loss
5. model family 간 edge 방향의 concordance
6. AML–Nuñez에서 공통 target edge의 concordance

H9의 non-empty subset은 511개다. AML inner folds에서 ridge를 이용해 전체
subset utility를 screen하고 marker별 Shapley-style contribution과 pairwise
interaction을 계산할 수 있다. 이는 causal relationship이 아니라
predictive information relationship으로 해석한다.

## 10. Minimal S* selection

### Stage 1 — Cheap screen

- AML outer-training patients만 사용
- H9의 511개 subset
- ridge와 frozen downsample
- cardinality별 Pareto candidates 보존

### Stage 2 — Local confirmation

- Stage 1 Pareto candidate만 exact kNN50으로 재평가
- fold 간 panel inclusion stability와 lower-quartile target performance 평가

### Stage 3 — Lock

S*는 다음을 만족하는 가장 작은 panel로 정의한다.

1. H9 multi-target utility의 95% 이상 유지
2. lower-quartile target utility가 사전 허용 범위 안에 있음
3. prespecified rare endpoint가 허용 범위 이상 악화되지 않음
4. major-cell preservation이 H9 대비 non-inferior

Tie는 marker 수, lower-quartile utility, mean utility 순으로 결정한다.
MLP test 결과로 S*를 바꾸지 않는다.

Primary 결과는 Pareto frontier 전체다. S*는 그 위의 한 operating point이지
모든 endpoint에 보편적으로 최적인 panel이라고 주장하지 않는다.

## 11. Cross-modality bridge

Cross-modality에는 same-cell pair가 없으므로 다음 fixed bridge를 사용한다.

1. outer-fit patients에서 modality별 H percentile calibration
2. paired training specimen 안에서 label-free H-only mass-preserving matching
3. target-exclusive marker의 barycentric pseudo-target 생성
4. 같은 pseudo-target으로 ridge, kNN50, MLP의 H-only/H+exclusive 비교

Bridge 자체를 제안 방법으로 내세우지 않는다. Matching method와 regularization
값은 validation에서 한 번 고정하고 모든 cohort/direction에 같은 규칙을 쓴다.

## 12. Pairing controls

각 cohort, direction, shared panel과 reference budget에서 다음을 비교한다.

1. target prior
2. identity-blind pooled/unpaired
3. patient-level deranged source
4. patient-matched source

Patient identity 비교에서는 source/target patient 수, cell 수와 총 reference
budget을 고정한다. 여러 frozen derangement를 사용하고 patient-first
uncertainty를 보고한다.

Matched가 wrong-patient와 target prior를 이기지 못하면 individualized
imputation이라고 부르지 않는다.

## 13. Cross-modality endpoints

Cross-modality source와 target은 서로 다른 세포이므로 same-cell MAE,
same-cell correlation 또는 neighborhood identity를 주장하지 않는다.

### Sample-level exact endpoints

- marker별 normalized Wasserstein distance
- median, IQR, positive fraction과 tail error
- joint MMD/energy distance
- evaluation-only cell-state conditional density
- patient-first equal-visit summary

### Cell-level surrogate endpoints

- target-domain fixed classifier를 imputed source cells에 replay
- source evaluation label 기준 major/rare AUPRC와 prevalence
- neighborhood label purity
- imputed-vs-target kNN two-sample discrimination
- local density와 support

Cell-level surrogate는 cell-wise target-marker truth가 아니다.

## 14. Major와 rare biology

Major-cell classification은 structure가 파괴되지 않았는지 확인하는 safety
endpoint다. Nuñez broad labels가 `CD3/CD4/CD8/CD19/CD56`에서 파생되므로 높은
major-cell score를 hidden-marker recovery의 증거로 사용하지 않는다.

Prespecified rare endpoints:

- AML CyTOF manual `T cell gd`
- Nuñez full-truth `CD3+ TCRgd+` state

추가 functional state는 marker rule, threshold와 minimum support를 결과를
보기 전에 고정한 경우에만 secondary endpoint로 추가한다.

Rare failure는 다음을 분리한다.

- H-only ceiling failure
- finite reference-count failure
- algorithmic failure
- defining-marker omission

## 15. 실행 단계

### Phase 0 — Freeze

1. AML/Nuñez canonical marker manifest
2. H19, H20, H9 manifest
3. patient-grouped split와 frozen row bank
4. target and rare-endpoint manifest
5. method hyperparameters와 metric contract
6. information-access/preflight test

### Phase 1 — Within-modality landscape

1. 2 cohorts × 2 modalities
2. Native H와 H9
3. median, ridge, kNN50, MLP
4. marker difficulty와 cross-cohort concordance

### Phase 2 — Relationship and minimum panel

1. AML H9 subset screen
2. add-back/removal/interaction landscape
3. Pareto frontier와 S* lock
4. Nuñez S* external evaluation

### Phase 3 — Cross-modality degradation

1. 2 cohorts × 2 directions
2. Native H, H9, S*
3. H-only 대 H+exclusive
4. matched 대 deranged 대 pooled
5. sample distribution과 cell-level surrogate biology

### Phase 4 — Final synthesis

1. marker recoverability heatmap
2. observed→target information network
3. marker-count–information Pareto curve
4. within→cross degradation map
5. exclusive/pairing gain map
6. major/rare preservation matrix

## 16. Raw AML sensitivity

Raw 전체 재구축은 Phase 1의 선결조건이 아니다. 다음 세 event view의 작은
selection-dependence audit를 별도 sensitivity로 수행한다.

1. current processed events
2. raw expression restricted to processed-mappable rows
3. raw technical-QC-only events

결론이 안정적이면 processed primary를 유지한다. Marker difficulty, rare
failure 또는 relationship 방향이 바뀌면 raw track을 확장하고 claim 범위를
수정한다.

## 17. 지금 하지 않을 일

- 새 VAE/OT/residual architecture 개발
- deep-model hyperparameter sweep
- native H19/H20의 exhaustive subset search
- Nunez 결과를 본 뒤 S* 재선택
- label-conditioned matching 또는 imputation
- UMAP, marginal density 또는 major-cell accuracy만으로 성공 판정
- cross-modality same-cell recovery claim
- raw full-population claim을 primary로 사용

## 18. 새 세션의 첫 작업

1. Cross-cohort marker alias와 panel manifest 구현
2. Nuñez patient-grouped split 및 row-bank manifest 생성
3. Existing within-modality runner를 cohort-agnostic하게 연결
4. H9 AML one-fold ridge/kNN smoke test
5. H9 Nuñez one-fold ridge/kNN smoke test
6. 결과 schema가 동일한지 확인한 뒤 Phase 1 full run 제출

Active machine-readable design은
`configs/benchmark/landscape_v1.yaml`에 둔다. 기존
`configs/benchmark/protocol_v1.yaml`은 processed-AML phase-0 audit와 그
테스트를 재현하기 위한 legacy contract로 유지한다.
