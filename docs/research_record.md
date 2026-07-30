# CytoAlign research record

마지막 갱신: 2026-07-30

범위: processed AML conditional experiments, 완료된 방법 triage, AML–Nuñez
marker-information landscape로의 전환

## 1. 이 문서의 역할

이 문서는 현재까지의 대화와 실험을 하나의 연구 기록으로 통합한다.
개별 실행의 상세 표는 `docs/archive/2026-07-27_processed_aml_v0/`에
보존하며, 앞으로의 실행 계약은 `docs/benchmark_plan.md`에서 관리한다.

현재 수치 결과는 upstream에서 pre-gating된 processed AML 83 patients,
93 specimens에 조건부다. 이를 full-population prevalence,
unseen-population discovery 또는 보편적인 clinical panel sufficiency
claim으로 확장하지 않는다. 후속 primary estimand는 AML과 Nuñez의
analysis-ready pre-gated events에서 marker information과 recoverability를
비교하는 것이다. Raw AML은 event-selection sensitivity로 둔다.

## 2. Processed AML 단계에서 확정된 연구 방향

연구의 초점은 다음 순서로 수렴했다.

1. 기존 세션처럼 새 방법론 개발로 흐르지 않고, 먼저 현상을 규명한다.
2. Spectral flow(SF)와 CyTOF를 분리하여 같은 cell의 shared marker만으로
   pseudo-unobserved marker를 복원할 수 있는지 측정한다.
3. 첫 기준선은 exact kNN50과 simple MLP로 제한한다. 복잡한 모델은 이
   ceiling과 failure pattern을 확인한 뒤에만 비교한다.
4. H19뿐 아니라 clinical-flow overlap을 모사한 clinical10까지 shared
   marker를 줄여, 어떤 공통 marker를 보장해야 하는지 조사한다.
5. 성공 여부는 marginal distribution만으로 정하지 않는다. Full manual
   labels로 outer-fit에서 한 번 학습한 fixed classifier를 imputed test
   representation에 replay하여 biology preservation을 평가한다.
6. Shared-marker 필요성은 모든 조합을 열거하지 않고 single add-back,
   targeted pair, compact triple, H19-minus-one 순서로 효율적으로
   screening한다.
7. 마지막으로 cyCombine, CytoVI, UVAE와 CyTOFmerge 계열을 같은 fold와
   endpoint에 올려, 모델 복잡도가 missing information을 구제하는지
   확인한다.

이 방향 전환의 핵심은 “최고 성능 모델 개발”이 아니라 다음 질문에
답하는 것이다.

> 어떤 target marker와 population이 현재 shared set에서 식별 가능하며,
> 어떤 경우에는 직접 측정 또는 예측 거부가 필요한가?

## 3. 공통 과학적 계약

- Split은 83 patients를 분리한 5 outer folds다. 같은 patient의 여러
  specimen은 같은 fold에 둔다.
- Primary inference와 bootstrap 단위는 cell이 아니라 patient다.
- Test hidden marker와 test label은 imputer에 제공하지 않는다.
- Scaling과 reference bank는 outer-fit data로만 만든다.
- Same-cell marker endpoint는 normalized MAE의 fit-median null 대비 skill과
  within-specimen Spearman correlation이다.
- Distribution endpoint는 IQR retention, median 및 upper-quantile error를
  별도로 기록한다.
- Biology endpoint는 representation마다 classifier를 다시 학습하지 않는
  fixed full-marker classifier replay다.
- Rare population은 recall만이 아니라 precision, AUPRC, prevalence error를
  함께 본다.
- Row-shuffle, fit-median, wrong-patient 또는 target-prior control을 이기지
  못하면 individualized recovery로 부르지 않는다.

`null-relative skill > 0`은 fit-patient marker median보다 same-cell MAE가
작다는 뜻이다. Distribution 폭이 비슷하거나 classifier 성능이 높다는
사실과 동일한 판정이 아니다.

## 4. 초기 cross-platform neural residual triage

Processed SF→CyTOF H19에서 neural `H+X` residual을 5 folds × 3 seeds로
검토했다.

- `correct-X − shuffled-X` rare-T macro-AUPRC:
  `-0.006881` (patient-bootstrap 95% CI `-0.011353` to `-0.002708`)
- `correct-X − H-residual`:
  `-0.001884` (95% CI `-0.008861` to `0.005171`)
- Pair-count curve에 positive X-specific dose-response가 없었다.
- H19만으로 rare-T AUPRC가 약 0.94여서 기존 cell-type probe가 포화됐다.

결정: 이 neural residual 방향은 GO 기준을 통과하지 못했다. 또한 기존
`unpaired_control`은 실제 unpaired path가 아니고, paired-count는 pairing
identity와 teacher-data 양을 동시에 바꾸므로 pairing evidence로 사용하지
않는다. 이 실패가 same-cell phenomenon characterization으로 전환한 직접적
이유다.

## 5. Same-cell recoverability v0

### 5.1 설계

- Modality: SF, CyTOF 각각 별도
- Observed panels: H19, clinical10
- Methods: fit-median null, exact kNN50, 2-layer 128-unit MLP
- Runs: 2 modalities × 2 panels × 5 folds × 3 seeds = 60/60
- Clinical10:
  `CD3, CD4, CD8, CD14, CD19, CD20, CD33, CD34, CD45, CD56`
- Clinical10은 각 modality 내부 pseudo-mask이며 실제 clinical-flow
  cross-platform experiment가 아니다.

### 5.2 전체 성능

Patient-marker mean skill과 fixed-classifier patient-mean macro-F1:

| Modality | Panel | Method | Skill | Spearman | Macro-F1 |
|---|---|---|---:|---:|---:|
| SF | H19 | kNN50 | 0.213 | 0.512 | 0.910 |
| SF | H19 | MLP | 0.242 | 0.591 | 0.913 |
| SF | clinical10 | kNN50 | 0.178 | 0.428 | 0.878 |
| SF | clinical10 | MLP | 0.173 | 0.463 | 0.884 |
| CyTOF | H19 | kNN50 | 0.125 | 0.327 | 0.786 |
| CyTOF | H19 | MLP | -0.003 | 0.360 | 0.836 |
| CyTOF | clinical10 | kNN50 | 0.116 | 0.313 | 0.799 |
| CyTOF | clinical10 | MLP | -0.053 | 0.323 | 0.810 |

Full-truth macro-F1은 SF 0.931, CyTOF 0.928이다. CyTOF MLP의 음수 skill과
양수 correlation/biology는 모순이 아니다. Cell ordering은 일부 보존하지만
절대값과 tail calibration error가 커서 median MAE를 이기지 못한다.

### 5.3 Marker별 결론

SF/H19의 hidden 7 markers는 kNN 기준 모두 recoverable이었다.

| Marker | kNN skill | 해석 |
|---|---:|---|
| CD95 | 0.354 | 강한 recovery |
| CTLA-4 | 0.318 | 강한 recovery |
| T-bet | 0.208 | 중간 recovery |
| EOMES | 0.200 | 중간 recovery |
| FOXP3 | 0.185 | 중간 recovery |
| TIGIT | 0.173 | 중간 recovery |
| Ki-67 | 0.050 | 통계적으로 양수지만 약한 recovery |

CyTOF/H19에서는 25개 중 18개만 recoverable이었다. CD64, CD11c, CD28,
CD45RO, CD127, CD16, TCRgd, CD294, CD69가 상대적으로 강했다. CD200,
CD15, CD57, CD66b, CD276, CD366은 unresolved였고 CD96은 median null보다
나빴다.

Clinical10에서는 SF 16 targets 중 15개가 recoverable이었고 Ki-67은
unresolved였다. CyTOF 34 targets 중 25개가 recoverable, 7개 unresolved,
CD96과 CD25는 null보다 나빴다.

### 5.4 H19에서 clinical10으로 축소

두 panel 모두에서 hidden인 동일 target만 paired 비교했다.

| Modality | Method | Skill 변화 | Spearman 변화 |
|---|---|---:|---:|
| SF | kNN50 | -0.065 | -0.085 |
| SF | MLP | -0.107 | -0.127 |
| CyTOF | kNN50 | -0.019 | -0.030 |
| CyTOF | MLP | -0.063 | -0.048 |

SF의 가장 큰 kNN 손실은 CTLA-4(-0.118), EOMES(-0.071)였고, CyTOF는
CD45RO(-0.123), CD64(-0.053), CD15(-0.050), IgD(-0.047)였다.

Clinical10에서 새로 hidden이 되는 H19 markers도 동일하지 않았다. SF의
CD117, HLA-DR, CD27, CD123은 비교적 잘 복원됐지만 CD25와 PD-1은 약했다.
CyTOF에서 CD25는 null보다 나빴고 PD-1/CD279도 약했다.

## 6. Shared-marker ablation v0

### 6.1 효율적 screen

Target은 모든 후보에서 `full panel - H19`로 고정했다. Exact kNN50과
seed 4207을 사용해 다음만 평가했다.

1. `H19 - clinical10`의 9개 single add-back
2. Stage 1 결과에 따른 modality별 targeted pairs
3. Modality별 두 compact triples
4. 정보량이 클 것으로 예상된 H19-minus-one panels

총 30 modality-fold runs가 완료됐다. Stage 2/3 후보는 같은 processed
dataset에서 adaptively 선택됐으므로 confirmatory가 아니라 descriptive
shortlist다.

### 6.2 Single-marker 정보량

SF에서는 CCR7이 clinical10→H19 skill gap의 40.5%를 단독 복원하여 가장
강했다. HLA-DR, CD117, CD27, CD45RA, PD-1, CD25, CD123은 각각 약
10–15%였고 CD38은 거의 중립이었다.

CyTOF에서는 CD45RA가 55.2%, HLA-DR 32.8%, CD38 25.7%, CCR7 25.3%,
CD123 19.3%를 복원했다. CD27, PD-1, CD25는 equal-weight kNN의 global
distance coordinate로 추가했을 때 작게 악화됐다.

Marker-to-target link는 구조적이었다.

- SF CCR7 → CTLA-4
- SF/CyTOF CD45RA → EOMES/CD45RO
- HLA-DR → T-bet/IgD
- CyTOF CD38 → CD47/TCRgd
- CyTOF CD123 → CD274

### 6.3 Compact shared sets

| Modality | Provisional panel | Skill | H19 gap rescue | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| SF | clinical10 + CCR7 + CD117 + CD123 | 0.1890 | 63.6% | 0.9593 | 0.9103 |
| SF | H19 | 0.2125 | 100% | 0.9667 | 0.9100 |
| CyTOF | clinical10 + CD45RA + HLA-DR + CD38 | 0.1268 | 109.1% | 0.8441 | 0.8112 |
| CyTOF | clinical10 + CD45RA + CD38 + CD123 | 0.1234 | 91.9% | 0.8527 | 0.8035 |
| CyTOF | H19 | 0.1250 | 100% | 0.8448 | 0.7857 |

SF compact panel은 H19과 macro-F1이 같았지만 quantitative skill과 balanced
accuracy는 낮았다. CyTOF compact panels가 H19 일부 지표를 넘은 것은 덜
측정하는 것이 우월하다는 뜻이 아니다. Equal-weight kNN에서 irrelevant/noisy
distance dimension이 neighbor selection을 악화시켜 panel-size effect가
비단조적이라는 뜻이다.

### 6.4 필요성

- SF CCR7은 hidden-marker recovery에 가장 필요했다.
- SF CD117과 CD123은 평균 skill보다 fixed-classifier biology에 더 중요했다.
- CyTOF CD45RA, HLA-DR, CD38, CD123은 서로 다른 target 정보를 제공했다.
- CyTOF CD27, PD-1, CD25는 현재 kNN distance에는 도움이 되지 않았지만,
  이것이 protein 자체의 생물학적 불필요성을 뜻하지는 않는다.

결론적으로 endpoint-free minimum shared set은 없다. Marker가 downstream
population을 직접 정의하거나 prespecified target에서 재현 가능한
leave-one-out loss를 보이면 shared set에 보장해야 한다.

## 7. Literature baseline comparison

### 7.1 포함 및 제외

- 기존 kNN50은 [CyTOFmerge](https://pmc.ncbi.nlm.nih.gov/articles/PMC6792069/)
  core rule과 같아 중복 실행하지 않았다.
- [cyCombine](https://www.nature.com/articles/s41467-022-29383-5)의
  SOM + within-node reference/KDE panel merging을 적용했다.
- 공식 [CytoVI](https://docs.scvi-tools.org/en/stable/user_guide/models/cytovi.html)
  및 [UVAE](https://github.com/mikephn/UVAE)를 적용했다.
- CytoBackBone은 선택적 matching/coverage 방법이라 모든 query cell에 값을
  주는 primary full-cell comparison에서는 제외했다.
- InfinityFlow 계열의 supervised prediction family는 simple MLP가 대표했다.

CytoVI, cyCombine, UVAE는 test shared-marker panel 전체를 unlabeled query로
보는 transductive methods다. kNN50/MLP는 inductive fit-only methods이므로
information-access mode를 결과에 명시했다.

### 7.2 실행

- Seed 4207, 2 modalities × 2 panels × 5 folds
- cyCombine, CytoVI, UVAE 각각 20/20, 총 60 literature runs
- CytoVI와 UVAE는 H200 MIG에서 실행
- Deep models: fit reference 최대 50,000 cells, specimen-balanced query
  training 최대 50,000 cells, 전체 query reconstruction
- 최종 통합 summary: 2,000 patient bootstrap replicates

### 7.3 Same-cell skill

| Modality | Panel | kNN50 | MLP | cyCombine | CytoVI | UVAE |
|---|---|---:|---:|---:|---:|---:|
| SF | H19 | 0.213 | **0.242** | -0.335 | 0.151 | 0.155 |
| SF | clinical10 | **0.178** | 0.172 | -0.305 | 0.120 | 0.012 |
| CyTOF | H19 | **0.125** | -0.002 | -0.554 | -0.162 | -0.203 |
| CyTOF | clinical10 | **0.116** | -0.055 | -0.545 | -0.173 | -0.311 |

복잡한 literature model은 정보 접근의 이점에도 simple baseline을 넘지
못했다. SF/H19에서는 deep model도 일부 유용했지만 MLP보다 낫지 않았다.
CyTOF에서는 CytoVI와 UVAE가 평균적으로 median null보다 나빴다.

cyCombine은 query coverage가 약 98.5–99.9%이고 distribution 폭을
유지/팽창시켰지만 same-cell skill은 모든 조건에서 음수였다. CyTOF에서
CytoVI/UVAE IQR retention도 약 1이었으나 MAE skill은 음수였다. 따라서
distributional realism은 correct cell-wise coupling의 증거가 아니다.

### 7.4 Fixed-classifier biology

Balanced accuracy:

| Modality | Panel | Median | kNN50 | MLP | cyCombine | CytoVI | UVAE | Full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SF | H19 | 0.957 | 0.967 | **0.971** | 0.948 | 0.970 | 0.969 | 0.982 |
| SF | clinical10 | 0.911 | 0.933 | **0.944** | 0.871 | 0.939 | 0.939 | 0.982 |
| CyTOF | H19 | 0.782 | 0.845 | **0.879** | 0.775 | 0.821 | 0.836 | 0.976 |
| CyTOF | clinical10 | 0.724 | 0.820 | **0.827** | 0.741 | 0.781 | 0.780 | 0.976 |

SF clinical10은 broad quantitative imputation에는 부족하지만 coarse
classifier biology 상당 부분을 유지한다. 이는 classifier가 hidden-marker
정보 전체가 아니라 일부 결정 경계만 사용하기 때문이다.

## 8. Rare-cell stress test

CyTOF T-cell-gd가 가장 진단적인 endpoint다. Defining TCRgd marker가
hidden이기 때문이다.

Three-seed same-cell sensitivity에서:

| Panel | Full recall | Median | kNN50 | MLP |
|---|---:|---:|---:|---:|
| H19 | 0.986 | 0.000 | 0.253 | 0.583 |
| clinical10 | 0.986 | 0.000 | 0.478 | 0.615 |

Seed-4207 literature comparison에서는 H19 recall이 kNN50 0.253, MLP 0.614,
cyCombine 0.123, CytoVI 0.009, UVAE 0.225였다. Clinical10은 각각 0.478,
0.645, 0.248, 0.012, 0.114였다.

Ablation compact panels의 kNN recall은 0.446–0.449로 H19보다 높았지만,
full 0.986을 복원하지 못했다. TCRgd mean marker skill이 약하게 양수여도
CytoVI replay는 population의 약 1%만 찾았다.

결정: T-cell-gd가 required endpoint이면 TCRgd를 직접 측정해야 한다.
Marker-average error, correlation, IQR retention 또는 coarse classifier
성능만으로 rare-cell preservation을 승인하지 않는다.

## 9. 통합된 현상 해석

### 9.1 Recoverability는 marker와 modality의 성질이다

SF functional-marker recovery는 H19에서 실제로 존재하지만 균일하지 않다.
CD95와 CTLA-4는 강하고 Ki-67은 약하다. CyTOF는 local structure가 있는
marker와 null보다도 못한 marker가 함께 존재한다.

### 9.2 Shared-marker 수는 충분조건이 아니다

H19가 clinical10보다 평균적으로 안전하지만, kNN 성능은 feature 수에 대해
비단조적이다. 필요한 것은 marker 수가 아니라 target과 population을
식별하는 좌표다.

### 9.3 모델은 없는 정보를 만들지 못한다

Transductive VAE와 distributional panel-merging도 simple MLP/kNN을 넘지
못했다. 복잡도 증가는 conditional ambiguity, missing phenotype anchor,
reference-support 부족을 구제하지 않았다.

### 9.4 세 평가 축은 분리해야 한다

1. Same-cell point fidelity
2. Marginal/distribution fidelity
3. Fixed downstream biology

세 축은 서로 대체할 수 없다. Row-shuffle은 marginal distribution을
유지하면서 same-cell correlation과 biology를 무너뜨렸고, cyCombine은
distribution 폭을 보존하면서 point skill이 음수였으며, clinical10 UVAE는
평균 marker skill이 거의 0이어도 SF coarse classifier 성능은 비교적 높았다.

### 9.5 “성공”은 endpoint-specific이다

- SF coarse annotation: clinical10도 provisional하게 가능
- SF broad functional-marker recovery: H19 또는 검증된 compact anchor 필요
- CyTOF broad quantitative imputation: 현재 방법으로 보장 불가
- CyTOF T-cell-gd: TCRgd 직접 측정 필요
- Distribution synthesis: same-cell imputation claim과 분리

## 10. 현재 허용되는 claim

### 말할 수 있는 것

- Processed AML에서 same-cell recoverability는 marker/modality-dependent다.
- SF/H19 hidden functional markers는 정도 차이는 있으나 예측 가능하다.
- Clinical10 축소는 동일 target의 recoverability를 낮춘다.
- Provisional compact shared sets가 H19 정보의 상당 부분을 보존한다.
- Shared marker를 더 많이 쓰는 것이 equal-weight kNN에 항상 유리하지 않다.
- Deep integration model이 simple baseline을 일관되게 개선하지 않는다.
- Distribution 보존은 individualized recovery나 rare-cell 복원을 보장하지
  않는다.

### 아직 말할 수 없는 것

- Raw full-population에서 같은 효과가 성립한다.
- 하나의 endpoint-free minimum sufficient panel이 존재한다.
- Compact panels가 독립 cohort에서도 H19과 동등하다.
- Cross-platform source-conditioned recovery가 입증됐다.
- Pairing이 필요하거나 불필요하다는 결론
- 현재 uncertainty가 calibrated abstention을 제공한다.
- Clinical-flow H4가 label-free cross-platform reconstruction에 적합하다.

## 11. 후속 연구 질문

1. Processed 결과가 raw technical-QC-only events에서 재현되는가?
2. Fixed target마다 conditional ambiguity와 achievable self ceiling은
   얼마인가?
3. Provisional compact panels가 frozen confirmatory test에서도 유지되는가?
4. Query H-support와 reference phenotype support가 failure를 예측하는가?
5. Locked risk score와 rejection threshold가 random abstention보다 나은가?
6. Cross-platform H calibration 후에도 source-conditioned gain이 남는가?
7. Wrong-patient와 target-prior control을 이기는가?
8. AML에서 고정한 marker/regime rule이 Nuñez cohort로 이전되는가?

구체적 실행 순서와 GO/NO-GO 기준은 `docs/benchmark_plan.md`에 둔다.

## 12. 재현 artifact

완료된 processed AML run의 disposable output과 scheduler log는
2026-07-30 session closeout에서 정리했다. 다음 durable artifact가 결과와
실행 계약을 보존한다.

- Checkpoint commit: `8fece8b`
- Same-cell detailed report:
  `docs/archive/2026-07-27_processed_aml_v0/same_cell_recoverability_v0_results.md`
- Shared-marker ablation detailed report:
  `docs/archive/2026-07-27_processed_aml_v0/h19_shared_marker_ablation_v0_results.md`
- Literature comparison detailed report:
  `docs/archive/2026-07-27_processed_aml_v0/literature_imputation_baselines_v0.md`
- Completed experiment configs:
  `configs/archive/2026-07-27_processed_aml_v0/`
- Legacy audit contract: `configs/benchmark/protocol_v1.yaml`
- Active landscape design: `configs/benchmark/landscape_v1.yaml`
- Session closeout inventory:
  `docs/archive/2026-07-30_processed_aml_session_closeout.md`

Patient-level output은 Git에 보존하지 않는다. 필요하면 checkpoint의 config,
code와 frozen sampling rule로 재생성한다.

## 13. 실행 및 provenance 주의사항

- Same-cell three-seed 결과는 patient-first sensitivity evidence다. Seed를
  replicate 수로 세지 않는다.
- Stage 2/3 ablation 후보는 같은 dataset에서 adaptive하게 선택됐다.
- Literature deep methods는 transductive, simple baselines는 inductive다.
- Accepted GPU 결과는 H200 CUDA probe와 method-specific metadata를
  검증했다.
- LSF `EXIT 255` 중 application 시작 전 job-file staging failure는 동일
  command로 재제출하고 failed artifact를 보존했다.
- Checkpoint `8fece8b`에서 test suite는 `90 passed, 6 skipped`였다.
- 이 환경에서는 login shell 초기화가 `no direct access allowed`를 출력할
  수 있다. Repository 검사와 로컬 실행은 명시적으로 non-login shell
  (`login=false`, 또는 동등한 `bash --noprofile --norc`)을 사용한다.

## 14. AML–Nuñez marker-information landscape 전환

2026-07-30에 다음 방향을 active plan으로 고정했다.

1. 새 imputation architecture를 개발하지 않는다.
2. AML과 Nuñez를 처음부터 같은 landscape benchmark에 포함한다.
3. 각 cohort의 native shared panel(H19/H20), 네 panel의 universal H9,
   AML에서 선택한 minimal S*를 비교한다.
4. Within-modality `H→X/Y`에서 exact same-cell marker difficulty를 측정한다.
5. Cross-modality에서는 양방향 `H` 대 `H+exclusive`를 비교하되
   aliquot-level pairing에 맞게 sample distribution과 cell-level surrogate
   biology만 평가한다.
6. Marker relationship은 marginal association, single-marker skill,
   add-back gain과 leave-one-out loss를 분리한다.
7. H9 subset의 marker-count–information Pareto frontier를 primary panel
   result로 삼고, 하나의 S*는 operating point로만 보고한다.
8. Pairing은 matched, patient-deranged, pooled/unpaired와 target prior의
   fixed-budget 비교로 검증한다.
9. Major-cell classification은 structural safety endpoint, AML `T cell gd`와
   Nuñez `CD3+ TCRgd+`는 prespecified rare endpoint로 둔다.

이 전환 이후 processed compact panel과 deep literature comparison은
hypothesis-generation provenance다. 새 benchmark의 panel이나 model을
결정하는 confirmatory evidence로 재사용하지 않는다.
