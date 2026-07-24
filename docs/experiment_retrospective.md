# CytoAlign 실험 회고: 무엇을 했고 어디에서 막혔는가

마지막 정리: 2026-07-24

## 문서의 목적

이 문서는 삭제된 raw outputs, LSF logs, compact JSON records, prototype
scripts와 archive slide를 대신하는 영구 기록이다. 수치는 성공 사례만
모은 것이 아니라, 각 방향의 질문·결과·한계·중단 이유를 함께 남긴다.

공통적으로 Spectral Flow의 공통 marker를 `H`, source-exclusive marker를
`X`, CyTOF-exclusive marker를 `Y`로 표기한다.

## 전체 흐름

```text
대규모 cyclic representation learning
  -> alignment failure와 해석 어려움
  -> pseudo-panel에서 X incremental signal 탐색
  -> OT soft teacher + H baseline residual
  -> 강한 H-only kNN 확인
  -> marker gate / paired-count / panel-mask stress test
  -> adaptive neighbor metric과 low-rank anchor 탐색
  -> point imputation의 식별 한계 확인
  -> mutation compatibility와 rare fine-cell diagnostic
  -> 현재: rare X-value를 capacity controls로 검증
```

## 1. Archive: MoCA/CycleAlign preliminary representation learning

### 무엇을 시도했는가

2026-03-25 preliminary slide의 목표는 clinical flow, Spectral Flow,
CyTOF처럼 panel과 platform이 다른 cytometry 자료를 하나의 latent space에
정렬하는 것이었다. 당시 이름은 “Multi-modal Cytometry Cyclic
Alignment”였고 다음 구성요소를 사용했다.

- modality별 encoder와 decoder
- cell-type classifier와 cell-type consistency
- latent consistency
- adversarial generator loss
- masked-input reconstruction
- paired cell identity를 요구하지 않는 self-supervised translation

자료 설명은 95 samples, 85 AML patients, 83 paired와 2 unpaired
patients였고, CyTOF 44 markers와 Spectral Flow 30 markers를 사용했다.
train/validation/test는 67/18/10 samples, 61/15/9 patients로 제시되었고,
specimen당 최대 200,000 cells와 5개 coarse cell type을 사용했다.

### 무엇이 관찰되었는가

UMAP 사례는 다음 세 실패 모드를 순차적으로 보여주었다.

1. modality와 cell type이 모두 정렬되지 않음
2. cell type은 정렬되지만 modality가 분리됨
3. 일부 설정에서만 partial alignment

마지막 slide의 mutation downstream 결과도 조건별로 일관되지 않았다.
예를 들어 Spectral Flow NPM AUROC는 original 0.847, generated-only
0.879, cross 0.915였지만, CyTOF RAS AUROC는 0.875에서 generated-only
0.726, cross 0.589로 크게 감소했다. TP53와 NPM의 여러 generated/cross
조건에서 F1이 0이었고, modality와 endpoint에 따라 개선 방향이
뒤집혔다.

### 한계와 결정

- 결과가 주로 UMAP 정성 평가에 의존했다.
- 많은 loss가 동시에 작동해 무엇이 alignment를 만들거나 깨뜨렸는지
  분리하기 어려웠다.
- downstream의 “generated”와 “cross” 조건이 어떤 train/test domain을
  뜻하는지 slide만으로 완전히 재구성하기 어려웠다.
- single split의 preliminary 결과이고 uncertainty가 제시되지 않았다.
- coarse cell-type alignment가 fine/rare identity 보존을 보장하지 않았다.

**결정:** 대규모 cyclic architecture를 현재 코드베이스로 이어가지
않는다. 강한 단순 baseline과 falsifiable control을 먼저 두는 방향으로
전환했다.

## 2. Exact-truth pseudo-panel과 DeCUR 계열

### 50k pseudo-panel screen

full CyTOF를 인위적으로 H/X/Y panel로 나누어 같은 cell의 hidden Y를
truth로 사용할 수 있게 했다. 초기 sparse autoencoder 계열은 latent
explained variance 0.935와 paired-cell oracle 기준 약 6.46%의 relative
MAE potential을 보였다.

그러나 semantic graph의 seed 간 edge Jaccard는 평균 0.024에 불과했고,
correct semantic graph가 shuffle을 모든 seed에서 이기지 못했다.
validation gate는 residual alpha를 일관되게 0으로 선택했다.

### DeCUR prototype

correct specimen pairing과 shuffled pairing을 비교하는 pre-registered
gate에서 correct 조건은 세 seed 모두 alpha 0을 선택했다. validation
exact error와 population error 모두 baseline과 shuffle을 동시에 이기지
못했다.

**결정:** X에 incremental signal 가능성은 있었지만 semantic latent
구조가 불안정했다. DeCUR를 중단했다.

## 3. OT pseudo-pair와 soft teacher

H-space optimal transport로 cell-unpaired source와 target을 연결했다.
hard argmax pseudo-pair, random hard pair, barycentric soft target을
비교했다.

- hard argmax는 불안정하고 exact error에서 열세여서 중단했다.
- soft OT는 H-neighborhood보다 일부 exact-cell diagnostic에서 나았고
  teacher candidate로 유지했다.
- pooled-unpaired OT는 paired specimen OT보다 distribution shift에
  취약했다.

여기서 얻은 핵심 교훈은 OT coupling 자체를 true cell pair로 해석하지
않고, residual model을 위한 soft training target으로만 쓰는 것이었다.

## 4. Ridge residual과 강한 kNN baseline

paired-count `0, 1, 2, 4, 8, 16, 32`에서 Ridge H baseline 위에 OT
residual을 추가했다. best count 16의 normalized Wasserstein은
1.087505였는데, plain kNN H baseline은 이미 0.966242였다.

baseline을 kNN으로 바꾼 residual curve에서 count 32는 0.964044로,
kNN보다 0.002197만 개선했다.

**결정:** parametric H baseline 위의 큰 개선은 약한 baseline을 고친
것에 가까웠다. 이후 모든 주장은 strong H-only kNN을 기준으로 해야
한다.

## 5. Pairing과 marker-adaptive residual gate

### Label-assisted 결과

coarse cell-type label을 kNN conditioning, OT block, residual feature,
validation selection에 사용했을 때 paired marker gate는 count 32에서
0.955849를 기록해 kNN 0.966242보다 0.010393 개선했다. raw paired-count
response `R²`는 0.886이었다.

matched/shuffled/pooled-unpaired one-seed 비교에서도 matched condition의
이점이 관찰되었다.

### Strict-unpaired와 label-free 결과

cell-type annotation을 predictor와 selector에서 모두 제거하자 효과가
거의 사라졌다.

- strict-unpaired count 32: 약 0.9663, 유의한 kNN 개선 없음
- count-response `R²`: 약 0.005
- global residual selector는 거의 항상 alpha 0
- marker-wise selector는 paired에서만 매우 작은 개선, unpaired에서 악화

이는 paired biological information의 일부가 존재할 수 있지만,
coarse label이 없으면 global population objective가 그 신호를 안정적으로
선택하지 못한다는 뜻이다.

## 6. Realistic common-panel stress test

25개 Y endpoint를 고정하고 common H marker를 19, 15, 12, 8개로 줄였다.

| H 수 | Plain kNN | Paired gate, 32 | Gain | Strict-unpaired gate, 32 | Gain |
|---:|---:|---:|---:|---:|---:|
| 19 | 0.965553 | 0.963647 | +0.001906 | 0.967853 | -0.002300 |
| 15 | 0.964198 | 0.963556 | +0.000642 | 0.979724 | -0.015526 |
| 12 | 0.989344 | 0.992057 | -0.002714 | 0.995028 | -0.005684 |
| 8 | 1.051058 | 1.051391 | -0.000333 | 1.053479 | -0.002421 |

19개에서 activation marker 네 개를 제거한 15-marker panel은 kNN 성능을
거의 유지했다. `CD20`, `CD123`, `CD45RA` 등이 더 빠진 12-marker
backbone부터 명확한 열화가 시작되었다.

**결정:** residual correction을 global point-imputation 주력 방법으로
확대하지 않았다. H baseline 자체의 metric을 marker별로 바꾸는 방향을
시험했다.

## 7. Five-specimen cross-platform H-only quick check

five patient-unique exact pairs, specimen당 5,000 cells,
leave-one-specimen-out으로 최소 cross-platform feasibility를 확인했다.

- global median 대비 pooled Wasserstein 20.9% 개선
- 5/5 specimens, 23/25 markers에서 개선
- same-platform target-H kNN보다는 15.1% 열세
- pooled median error는 6.5% 악화

**결정:** H-only kNN은 strong baseline으로 확정했다. 다만 5-specimen
pilot이므로 cell-wise 또는 generalization 주장은 하지 않았다.

## 8. Target-conditioned adaptive kNN

각 Y marker마다 H distance의 diagonal weight를 학습하고, panel mask에
따라 weight를 재정규화했다. target-training cells만 사용했으며 label과
OT teacher는 쓰지 않았다. fold 0, seeds 4207–4209에서 19/15/12/8 marker
panel을 평가했다.

아래는 seed 평균이며, 각 panel에서 Wasserstein이 가장 낮은 adaptive
variant를 plain kNN과 비교한 것이다.

| H panel | Plain W | Best adaptive W | W 개선 | Plain Spearman | Adaptive Spearman | Median-error 방향 |
|---:|---:|---:|---:|---:|---:|---|
| 19 | 0.986661 | 0.951478 | 3.57% | 0.718767 | 0.693927 | 소폭 악화 |
| 15 | 0.980904 | 0.948161 | 3.34% | 0.700857 | 0.682892 | 거의 동일/악화 |
| 12 | 1.009935 | 0.974775 | 3.48% | 0.688303 | 0.673725 | 개선 |
| 8 | 1.065240 | 1.045915 | 1.81% | 0.647573 | 0.647616 | 개선 |

Wasserstein은 개선되었지만 target-specific metric은 marker-median
Spearman을 19/15/12 panels에서 낮췄다. 가장 안정적인 global metric도
Spearman +0.01이라는 사전 기준을 넘지 못했고 8-marker setting에서는
Wasserstein이 악화되었다. fold 0만 평가했고 official CyTOFmerge/CytoVI
비교도 이 screen 안에서 완료되지 않았다.

**결정:** 사전 go/no-go를 통과하지 못했다. query-adaptive neural metric
같은 더 복잡한 후속 모델로 확대하지 않았다.

## 9. Shared-anchor factor model

source `[H]` 또는 `[H,X]`와 target `[H,Y]`에 factor analyzer를 독립적으로
fit한 뒤 common-marker loading으로 latent axes를 정렬했다. label을
사용하지 않았고 source/target training rows도 분리했다.

Fold 0, specimen당 head 500 cells, seeds 4207–4209, rank 2:

| 방법 | Exact normalized MAE |
|---|---:|
| Label-free kNN H | 0.630201 |
| Anchor H | 0.788037 |
| Anchor H+X | 0.775455 |
| Anchor H+shuffled X | 0.794941 |
| Paired-cell Ridge H+X oracle | 0.710610 |

correct X는 H와 shuffled X보다 나아 작은 X signal을 보였지만 kNN에
크게 뒤졌다. unpaired likelihood가 선택한 rank 6에서는 H+X가 H보다
0.014751 나빴다.

permutation-null anchor-strength gate는 여섯 축 중 다섯 축을 남겼지만
결론을 바꾸지 못했다.

| 방법 | Gated rank-6 MAE |
|---|---:|
| kNN H | 0.630201 |
| Anchor H | 0.764984 |
| Anchor H+X | 0.768442 |
| Anchor H+shuffled X | 0.792753 |

**결정:** X signal diagnostic으로는 흥미롭지만 standalone imputation
method로 no-go. common loading strength만으로 어떤 latent direction이
Y에 올바르게 transfer되는지 식별할 수 없었다.

## 10. Partial-identification coupling bounds

Point imputation 대신, 관측된 두 marginal과 H-cost constraint를
만족하는 모든 coupling에서 joint-positive population의 최솟값과
최댓값을 계산했다.

```text
Gamma(delta) = {
  Pi >= 0,
  Pi 1 = source marginal,
  Pi^T 1 = target marginal,
  <C_H, Pi> <= c_min + delta (c_independent - c_min)
}
```

same-platform full-CyTOF pseudo-panel, disjoint 80x80 supports, 6 marker pairs,
80/90th percentile thresholds, 3 seeds의 36 endpoints를 평가했다.

| Slack | Truth coverage | Median Fréchet contraction |
|---:|---:|---:|
| 0.005 | 0.9167 | 0.6056 |
| 0.010 | 0.9722 | 0.4535 |
| 0.020 | 1.0000 | 0.2796 |
| 0.100 | 1.0000 | 0.0142 |

minimum-cost OT, 즉 slack 0은 거의 point interval을 만들었지만 coverage가
0이었다. slack 0.01은 35/36 endpoints를 cover하면서 Fréchet width를
median 45.35% 줄였다.

### 한계

- held-out specimen의 target Y marginal을 ambiguity set에 사용했다.
  따라서 prospective source-only imputation이 아니라 same-specimen
  unpaired-panel fusion이다.
- slack 0.01은 coverage-width curve를 본 뒤 고른 oracle-tuned 값이다.
- empirical marginal을 정확한 것으로 취급해 finite-cell uncertainty를
  포함하지 않았다.
- 40-cell support에서는 seed sensitivity가 컸다.

**결정:** finite coupling bounds의 계산 가능성은 GO였지만 deployable
method는 아니다. prospective source-only 목표와는 다른 문제이므로 현재
rare translation 실험과 분리한다.

## 11. Mutation-prediction translation

93 exact pairs, 83 patients, patient-grouped 5-fold CV, 최대 50,000
cells/specimen에서 observed SF, real CyTOF, H-only translation, gated H+X
translation의 specimen mean을 비교했다.

| View | Macro AUROC | Macro AUPRC | Balanced accuracy | Brier |
|---|---:|---:|---:|---:|
| Original SF | 0.739 | 0.529 | 0.673 | 0.202 |
| CyTOF oracle | 0.800 | 0.686 | 0.751 | 0.158 |
| H-only translated | 0.736 | 0.566 | 0.635 | 0.221 |
| H+X gated translated | 0.738 | 0.571 | 0.632 | 0.213 |

모든 translated-vs-SF와 H+X-vs-H patient-bootstrap AUROC interval이 0을
포함했다. CyTOF oracle은 NPM과 RAS에서 translated model보다 유의하게
좋았다.

**결정:** CyTOF-domain model에 translated SF를 넣는 compatibility는
유망하지만 mutation performance 개선 전략으로는 no-go. mean pooling이
rare-cell signal을 평균내는 한계도 확인했다.

## 12. Fine-cell-type rare-population probe

translator에는 label을 주지 않고 source의 original 8-class fine label을
held-out probe target으로만 사용했다. 4.60M OOF cells 중 DN은 0.392%,
DP는 0.098%였다.

Y-only H-only translation의 rare DN/DP mean AUPRC는 0.2721이었고,
ungated H+X는 0.3067로 +0.0346
([0.0248, 0.0437]) 개선했다. DN/DP 각각의 AUPRC와 recall도 모두
positive patient-bootstrap interval을 보였다.

global Wasserstein으로 고른 marker gate는 이 효과의 대부분을
제거했다. Full H+Y probe는 rare AUPRC 약 0.93을 유지했지만, 이는
observed H가 label을 전달했기 때문이므로 Y imputation 성공으로
해석하지 않았다.

**결정:** rare-population-focused H+X ablation으로 GO. correct X,
capacity-matched H-only residual, shuffled X를 비교하기 전에는 X의
인과적 가치를 주장하지 않는다.

## 13. External baseline bundle에서 확인된 것

삭제 전 baseline workspace에는 다음이 있었다.

- vendored UVAE와 UVAE-COVID19 repositories
- cyCombine source와 local R packages
- 별도 UVAE Python environment
- CytoVI/scvi-tools source, workflow, models, h5ad results와 LSF logs
- local xgboost installation

CytoVI bundle은 label-informed Y1 held-out setting으로 `cytof_clinical`,
`sf_clinical`, `sf_cytof` 세 panel pair의 folds 0–4 결과를 만들었다.
LSF logs는 완료와 output path를 확인했지만, bundle 안에 최종
patient-first comparison table이나 현재 rare endpoint와 직접 비교할
수 있는 aggregate가 없었다. 따라서 저장된 h5ad와 model의 존재 자체를
성능 증거로 해석하지 않는다.

UVAE/cyCombine 설치와 source tree도 현재 repository에서 import되거나
호출되지 않았다. 외부 코드는 원 upstream에서 다시 받을 수 있고,
local environment와 trained binary는 재현 가능한 source 기록이
아니므로 정리 대상이 되었다.

## 14. 반복해서 드러난 구조적 한계

### Strong H baseline

H-only kNN이 강해 overall distribution metric의 개선 여지가 작다. 약한
Ridge baseline을 이긴 결과는 실제 method value를 과대평가했다.

### Objective mismatch

global Wasserstein은 큰 population과 흔한 marker를 우선한다. rare
subtype-specific correction은 전체 loss에서 거의 보이지 않으며,
marker gate가 이를 제거할 수 있다.

### Non-identification

cell-unpaired panels은 X/Y joint를 직접 식별하지 않는다. OT, latent
alignment, adaptive metric 모두 additional assumption이며 true cell
pair를 증명하지 않는다.

### Cells are not patients

specimen당 50,000 cells는 정밀도를 높이지만 biological replicate를
늘리지 않는다. rare effect의 유효 표본수는 rare-positive patient 수에
의존한다.

### Capacity confounding

H+X 모델이 H-only보다 복잡하면 개선을 X 정보로 귀속할 수 없다.
capacity-matched H residual과 X-shuffle은 선택 사항이 아니라 필수
control이다.

### Qualitative alignment is insufficient

좋은 UMAP, marginal Wasserstein, full H+Y label probe 중 어느 하나도
생성된 Y가 fine/rare identity를 올바르게 표현한다는 충분조건이 아니다.

## 15. 현재 남긴 결정

1. Point-imputation baseline은 H-only patient-balanced median kNN이다.
2. 현재 H+X OT residual은 rare-X hypothesis를 검증하는 최소 모델로만
   유지한다.
3. global marker gate를 final selector로 사용하지 않는다.
4. 다음 실험은 correct X, H-only residual, shuffled X와 paired-count
   sensitivity를 비교한다.
5. 아키텍처 확장은 위 control을 통과한 뒤에만 고려한다.
6. partial identification은 same-specimen panel fusion이라는 별도
   scientific path로 구분한다.

세부 다음 실험 계약과 현재 주장 가능한 범위는
[`current_findings_and_next_experiment.md`](current_findings_and_next_experiment.md)에
정리되어 있다.

## 16. Provenance

정리 전 주요 Git source commits:

- `c607a4b`: kNN residual experiment
- `40c83be`: marker-adaptive paired curve
- `058ceb9`: strict-unpaired selection
- `933ce14`: label-free realistic panels
- `ee680b3`: target-conditioned adaptive kNN

LSF job groups:

- initial paired curves: `48175176–48175178`
- kNN residual curves: `48179983–48179985`
- matched/shuffled/unpaired go/no-go: `48183273`, `48183274`, `48183630`
- marker-gate grids: `48186129–48186283`
- realistic-panel grid: `48186708–48186731`
- mutation 5-fold: `48216985–48216989`
- fine-cell probe 5-fold: `48218067–48218071`

이 문서의 aggregate 수치가 영구 기록이다. raw outputs, scheduler logs,
compact records, model binaries, third-party environments와 preliminary
archive는 repository의 새 실험 상태에 포함하지 않는다.
