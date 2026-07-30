# CytoAlign: cross-platform cytometry recoverability benchmark

Archive note: superseded by the current AML–Nuñez landscape plan and retained
for protocol-history provenance. The disposable runtime outputs referenced
below were removed at the 2026-07-30 session closeout; see
`../2026-07-30_processed_aml_session_closeout.md` for durable results and
recovery details.

마지막 갱신: 2026-07-27  
상태: protocol draft, processed same-cell conditional sensitivity와
shared-marker ablation 완료, raw primary benchmark 실행 전  
목표 저널: *Briefings in Bioinformatics*

## 1. 프로젝트 정의

CytoAlign은 새로운 대규모 imputation architecture를 제안하는 프로젝트가
아니다. 서로 다른 cytometry panel 사이에서 공통 marker `H`만 관찰될 때
target-exclusive marker `Y`가 **무엇은 복원 가능하고, 무엇은 구조적으로
복원할 수 없으며, 언제 예측을 거부해야 하는지** 밝히는 benchmark다.

핵심 질문은 “어떤 모델이 평균적으로 1등인가?”가 아니다.

> Cross-panel recoverability가 backbone 정보량, modality shift,
> H-domain support, target-reference support, conditional ambiguity와
> patient pairing에 의해 어떻게 결정되는가?

최종 산출물은 marker·population·direction별 recoverability map, 검증된
uncertainty와 abstention rule, 실무적인 method-selection decision tree,
재사용 가능한 benchmark harness다. Simple imputation은 결론이 아니라
recoverability가 충분한 영역에서 선택되는 운영 도구다.

## 2. 왜 필요한가

Cross-platform cytometry에는 동일 cell의 source–target ground truth가 없다.
따라서 그럴듯한 marginal distribution을 생성해도 다음 실패를 숨길 수 있다.

- source `H`를 사용하지 않고 target population prior만 재현
- wrong patient를 사용해도 같은 결과가 나오는 non-individualized prediction
- 같은 platform 안에서도 `H`로 결정되지 않는 intrinsically ambiguous `Y`
- source `H`가 target-training support 밖에 있는 domain extrapolation
- target reference에 희귀 population이 없어 생기는 hallucination
- upstream biological gating 차이를 imputation 성능으로 오인

따라서 평균 RMSE나 UMAP 정렬만으로는 성공을 판단할 수 없다. 이 프로젝트는
intrinsic predictability, cross-platform transfer, individualization,
uncertainty와 downstream preservation을 순서대로 분리한다.

## 3. 검증할 중심 가설

1. Same-platform `H→Y` predictability는 cross-platform recoverability의
   필요조건이다.
2. Self performance가 비슷할 때 H calibration error, H-support deficit과
   target-reference deficit이 cross-platform penalty를 설명한다.
3. 복잡한 방법의 이점은 중간 recoverability regime에 제한되며, 정보나
   support가 부족한 영역은 모델 복잡도로 안정적으로 구제되지 않는다.
4. H-support와 target conditional ambiguity를 분리한 risk score는 held-out
   failure를 예측하고 selective rejection을 가능하게 한다.

`Simple methods are sufficient`나 `pairing is unnecessary`는 사전 결론이
아니다. Source-blind null과 wrong-patient control을 이긴 결과만
source-conditioned recovery로 인정한다.

## 4. 데이터와 claim 범위

| 역할 | 데이터 | 범위 |
|---|---|---|
| Primary | AML spectral flow ↔ CyTOF | raw FCS, technical-QC-only event set, 93 pairs/83 patients |
| External | Nuñez FACSymphony ↔ CyTOF | released pre-gated FCS의 모든 row, 164 pairs/51 patients |
| Secondary | AML clinical flow 관련 H4 | raw detector-level label-free audit 통과 시에만 |
| Sensitivity | 현재 processed AML | upstream pre-gated compartment에 조건부 |

Primary는 양방향 H19와 controlled H4 backbone을 평가한다. Clinical pair는
label 기반 channel splitting과 row selection 문제를 해결하지 못하면
제외한다.

현재 processed AML은 raw 대비 spectral flow 37.68%, CyTOF 67.72%의 서로
다른 event만 남긴다. 따라서 processed 결과로 full-population prevalence,
unseen-population discovery 또는 BIB primary claim을 하지 않는다.

## 5. Benchmark 구조

분석 순서는 다음과 같다.

```text
same-platform H→Y
  → cross-platform H→H calibration
  → cross-platform H→Y
  → target-prior / wrong-patient individualization
  → downstream preservation
  → uncertainty-calibrated rejection
```

| 단계 | 목적 | 핵심 비교 |
|---|---|---|
| Self recoverability | H가 Y를 담고 있는지 확인 | exact same-cell masking truth |
| H calibration | modality shift 분리 | source H vs calibrated target H |
| Cross H→Y | 실제 transfer 평가 | median/ridge/kNN/tree/MLP 및 대표 도구 |
| Individualization | source 사용 여부 확인 | target prior, pooled, wrong patient, matched |
| Downstream | target semantics 보존 확인 | 고정 target-domain classifier |
| Uncertainty | 위험 예측 탐지 | locked risk–coverage vs random rejection |

Point prediction과 distributional prediction은 별도 track으로 평가한다.
Distributional 방법도 test 결과가 좋은 sample을 사후 선택하지 않는다.

## 6. 평가와 통계 단위

Biological inference unit은 patient다.

- longitudinal specimen은 patient 안에서 먼저 동일 가중 집계
- cell은 distribution 추정 observation이지 replicate가 아님
- fold와 seed를 표본 수에 더하지 않음
- 모든 method comparison은 같은 patient의 paired difference 사용
- bootstrap unit은 patient

주요 평가는 다음을 포함한다.

- marker별 median, positive fraction, upper quantile와 distribution error
- H–Y 및 Y–Y joint-fidelity
- target-prior 대비 source gain과 matched-vs-wrong-patient gain
- major/minor population 비율과 고정 target-domain downstream 성능
- runtime, memory, software failure와 fold/seed stability

Representation마다 새 classifier를 학습하는 평가는 supplementary다.
`H + predicted Y`가 H-only로 이미 쉬운 label을 유지한다는 이유만으로
Y recovery를 주장하지 않는다.

## 7. Marker-wise uncertainty와 최종 판정

Uncertainty는 다음 원인을 분리한다.

1. `conditional ambiguity`: target에서 비슷한 H가 서로 다른 Y를 갖는 정도
2. `H-support risk`: query H가 target-training H domain 밖에 있는 정도
3. `target-reference support`: 해당 phenotype/population의 reference 부족
4. `method instability`: fold/seed/model에 따른 불안정성

Uncertainty 값이 큰 것만으로 calibrated uncertainty라고 부르지 않는다.
Validation patient에서 formula와 rejection threshold를 고정하고, held-out
error 예측력과 risk–coverage가 random rejection보다 좋은지 검증한다.

Cross-modal data에는 true cell correspondence가 없으므로 cross-modal
cell-level interval coverage를 주장하지 않는다. Marker × patient × population
summary 수준의 interval과 realized error를 사용한다.

최종 marker status:

- `recoverable`
- `conditionally_recoverable`
- `non_individualized_prior_reproduction`
- `unrecoverable_under_tested_methods`
- `unsupported_h_domain`
- `unsupported_target_reference`
- `not_evaluable`

Method deployability는 별도 축으로 `deployable`,
`deployable_with_abstention`, `not_deployable`, `not_assessed`를 부여한다.

## 8. 고정해야 할 과학적 규칙

- Event inclusion에 biological H/Y, cell label, outcome을 사용하지 않음
- Translator와 calibrator에 cell/endpoint label 사용 금지
- Test target statistic으로 transform, method, threshold 또는 retry 선택 금지
- Patient-grouped fold와 train/validation/test patient isolation
- 모든 방법이 같은 frozen split과 data-matched reference bank 사용
- Pairing curve에서 source/target bank, row와 총 data budget 고정
- Longitudinal visit는 patient 단위로 함께 이동
- 결과를 본 뒤 같은 protocol ID로 threshold나 endpoint를 변경하지 않음
- Protocol, manifest, code/config digest와 failure artifact 보존

## 9. 현재까지 밝혀진 것

### Data audit

- AML spectral-flow와 CyTOF raw FCS coverage는 각각 95/95, 93/93 확인
- Current processed 93 paired specimens와 83 patients의 exact mapping 확인
- Patient-grouped 5-fold split과 shared reservoir artifact 초안 생성
- Raw technical-QC mask/expression, portable metadata와 raw frozen manifests는
  아직 미완료

### Processed AML conditional smoke

SF→CyTOF H19, 5 folds × 3 seeds에서 current neural H+X residual을 검증했다.

- `correct-X − shuffled-X` rare-T macro-AUPRC:
  `-0.006881`, patient-bootstrap 95% CI `-0.011353~-0.002708`
- `correct-X − H-residual` rare-T macro-AUPRC:
  `-0.001884`, 95% CI `-0.008861~0.005171`
- Pair-count curve는 비단조적이며 positive X-specific dose-response가 없음
- Target-only H→Y MLP는 H-only kNN보다 높았지만 patient-specific gain은
  아직 증명되지 않음
- H19 alone rare-T AUPRC가 약 0.94여서 현재 cell-type probe는 포화됨

결론: 현재 neural X-residual 방향은 사전 GO 기준을 통과하지 못했으므로
중단/deprioritize한다. 이 결과는 benchmark 방향 전환을 지지하는 method
triage이지 BIB primary evidence가 아니다.

또한 기존 `unpaired_control` config는 실제 unpaired path가 아니며, 현재
paired-count는 pairing identity와 teacher data 양을 함께 바꾼다. 두 결과는
pairing evidence로 사용하지 않는다.

### Processed AML same-cell recoverability v0

사용자 요청에 따라 cross-platform 방법 개발보다 먼저, modality별 exact
same-cell pseudo-masking 현상을 확인했다. Spectral flow와 CyTOF를 분리하고
H19/clinical10, 5 patient folds, 3 seeds에서 median, kNN50, simple MLP를
비교했다. Clinical10은 SF/CyTOF 내부의 clinical-overlap-inspired
pseudo-mask이며 clinical-flow dataset을 이용한 cross-platform run은 아니다.
기존 full annotation으로 학습한 classifier는 representation마다 재학습하지
않고 고정했다.

- 총 60/60 run 완료, 83 patients/93 specimens, patient-first bootstrap
- Spectral-flow H19 hidden 7 markers는 kNN null-relative skill의 95% CI가
  모두 0보다 높음. Ki-67은 skill 0.050으로 약한 recovery
- CyTOF H19 hidden 25 markers 중 kNN 기준 18개 recoverable, 6개 unresolved,
  CD96은 median null보다 낮음
- H19에서 clinical10으로 줄이면 두 panel에서 공통으로 hidden인 marker의
  skill이 SF kNN `-0.065`, MLP `-0.107`; CyTOF kNN `-0.019`,
  MLP `-0.063` 감소
- Fixed full-panel classifier macro-F1:
  SF H19 `0.910/0.913` (kNN/MLP), SF clinical10 `0.878/0.884`,
  CyTOF H19 `0.786/0.836`, CyTOF clinical10 `0.799/0.810`
- CyTOF rare T-cell-gd는 true recall `0.986`에 비해 H19
  kNN/MLP `0.253/0.583`으로 부분 복원에 그침
- H를 row-shuffle하면 marginal/dynamic range는 유지되지만 same-cell
  correlation과 downstream 성능이 무너짐. Distribution matching만으로
  성공을 판정할 수 없다는 직접적인 positive control

상세 판정과 한계:
`docs/same_cell_recoverability_v0_results.md`.

이 결과도 processed upstream-pregated event set에 조건부이며 raw primary
evidence가 아니다. H19와 clinical10 두 점만으로 unique minimal shared set을
정할 수 없으므로, raw benchmark에서는 같은 endpoint를 유지한 nested-panel
또는 leave-one-shared-marker-out 실험이 필요하다.

### Processed AML shared-marker ablation v0

전체 조합을 탐색하지 않고 clinical10에서 시작한 9개 single add-back,
modality별 5개 targeted pair, 2개 compact three-marker panel과 선택된
H19-minus-one panel을 순차 평가했다. 비교 target은 모든 panel에서
`full - H19`로 고정했고, biology는 같은 full-label classifier를 고정한 채
평가했다.

- SF `clinical10 + CCR7 + CD117 + CD123`은 H19 hidden-target kNN skill
  gap의 63.6%를 복원했고 fixed-classifier macro-F1은 H19과 같았다
  (`0.9103` vs `0.9100`). 정량 skill과 balanced accuracy는 H19보다 낮았다.
- CyTOF `clinical10 + CD45RA + HLA-DR + CD38`은 global skill과 Spearman에서
  H19과 동급이었다. `CD45RA + CD38 + CD123` 후보도 skill gap의 91.9%를
  복원했다.
- Leave-one-out에서 SF CCR7은 hidden-target recovery, CD117/CD123은
  classifier biology에 중요했다. CyTOF CD45RA, HLA-DR, CD38, CD123은
  서로 다른 target에 필요한 정보를 제공했다.
- CyTOF CD27, PD-1, CD25를 equal-weight kNN distance에 추가하면 일부
  endpoint가 악화되었다. 따라서 shared marker 수와 성능은 단조 관계가
  아니며, “많을수록 안전하다”는 panel rule은 성립하지 않는다.
- CyTOF T-cell-gd는 compact panel에서 H19보다 좋아졌지만 recall
  `0.446~0.449`, AUPRC `0.678~0.683`으로 true full
  `0.986/0.994`를 복원하지 못했다. TCRgd 직접 측정이 필요하다.

상세 결과와 artifact:
`docs/h19_shared_marker_ablation_v0_results.md`.

이 screen은 exact kNN, seed 4207의 exploratory shortlist이며 stage 2/3
후보를 같은 processed data에서 adaptive하게 선택했다. 따라서 interval은
descriptive이고, 최종 sufficient-set claim은 raw event에서 frozen 후보를
독립 확인한 뒤에만 한다.

## 10. 다음 실행 순서

1. AML raw FCS technical-QC-only mask, expression과 row-label alignment 생성
2. Portable metadata와 raw data/marker/split/endpoint/reference-bank manifest 동결
3. Full preflight와 forbidden-information test 통과
4. Raw AML self H19/H4 one-fold/one-seed pilot
5. Same-platform ceiling과 semi-synthetic support/ambiguity positive control 검증
6. Simple methods의 full OOF 양방향 benchmark
7. Fixed-budget `{0, 8, all}` pairing pilot:
   identity-blind pooled, patient-level deranged, matched 비교
8. Validation-locked uncertainty/rejection과 fixed target downstream 평가
9. AML에서 모든 rule을 lock한 뒤 Nuñez external validation
10. 필요한 regime에서만 third-party/reference method 추가

Clinical H4는 raw label-free reconstruction을 통과할 때만 secondary로 진행한다.

## 11. GO/NO-GO 기준

Benchmark가 성공하려면 다음이 필요하다.

- Self recoverability와 cross penalty의 관계가 patient-level에서 재현됨
- Source-conditioned prediction이 target-prior와 wrong-patient null보다 우수함
- Support/ambiguity risk가 held-out error를 예측함
- Locked rejection이 random rejection보다 우수함
- Marker/regime rule이 AML에서 Nuñez로 이전됨
- Simple/complex method의 이점 또는 robust null result가 actionable rule을 제공

Raw event set, patient mapping, row alignment 또는 forbidden-information
contract를 검증하지 못하면 primary run을 중단한다. External validation이
실패하면 general recoverability rule을 주장하지 않는다.

## 12. 논문 메시지와 BIB 포지셔닝

목표 메시지:

> Cross-panel recoverability is a predictable property of backbone
> information, calibrated domain overlap and explicit support—not merely
> model complexity.

논문은 새 imputation 모델보다 “언제 어떤 도구를 사용하고 언제 거부해야
하는가”를 설명하는 benchmark/problem-solving protocol로 작성한다. Simple
method는 Pareto-efficient baseline 또는 최종 decision tree의 선택지다.

*Briefings in Bioinformatics*는 benchmarking, reproducibility, bottleneck
analysis와 practical guidance를 scope에 포함하지만, 공식 manuscript 안내는
pure original research에 제한을 둔다. 따라서 review/protocol/case-study
형태와 empirical benchmark의 허용 범위를 원고 작성 전에 편집부에
pre-submission inquiry로 확인한다.

- Author guidance: https://academic.oup.com/bib/pages/author-guidelines
- Manuscript preparation: https://academic.oup.com/bib/pages/msprep_submission

## 13. 재현 artifact

- Machine-readable protocol: `configs/benchmark/protocol_v1.yaml`
- Draft audits: `benchmark/audits/`
- Conditional summaries:
  `outputs/sf_to_cytof_rare_population_paired_count_cv/`
- Same-cell conditional run:
  `outputs/aml_same_cell_recoverability_v0/`
- Same-cell result report:
  `docs/same_cell_recoverability_v0_results.md`
- Literature-baseline runs and patient-bootstrap summaries:
  `outputs/aml_literature_baselines_v0/`
- Literature-baseline protocol and result report:
  `docs/literature_imputation_baselines_v0.md`
- Effective validation jobs:
  `48223884–48223891`, `48223893`, `48223912`
- LSF staging failure `48223892`는 application 실행 전 종료되어
  `48223912`로 대체
- Current full test suite: `100 passed`

Seed 4207 fold payload에는 code/config digest가 없어 seeds 4208–4209와 같은
dirty snapshot임을 artifact만으로 증명할 수 없다. 세 seed 결합은 conditional
stability evidence이며 frozen-protocol evidence가 아니다.
## 14. Literature baseline extension

The concrete CytoVI, cyCombine, UVAE, and CyTOFmerge comparison contract is
documented in `docs/literature_imputation_baselines_v0.md`.  The extension uses
the same patient folds, H19/clinical10 panels, marker endpoints, and fixed
biology classifier as the same-cell benchmark.  CytoVI/cyCombine/UVAE are
explicitly tagged as transductive shared-marker methods; the existing kNN50 is
treated as the CyTOFmerge core rule rather than duplicated.

The seed-4207 screen completed all 60 literature-method runs. No deep
integration model exceeded the simple MLP overall. Spectral-flow/H19 remained
the most recoverable setting, whereas CyTOF deep-model mean skill was negative
and fixed-classifier gamma-delta T-cell recall remained severely degraded.
Final point, distribution, biology, and per-class results are recorded in
`docs/literature_imputation_baselines_v0.md`.
