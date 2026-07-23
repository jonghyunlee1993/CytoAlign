# CycleAlign 연구 요약

업데이트: 2026-07-23  
목표: paired-specimen, cell-unpaired cytometry cross-panel marker prediction  
현재 결론: **H+cell type baseline을 유지하고, 같은 training specimen 안에서 common-marker OT로 만든
soft target을 H+X MLP에 직접 distillation하는 방법을 주 모델로 채택한다.**

## 1. 문제 설정

두 panel을 다음과 같이 쓴다.

- `H`: 양쪽 panel에서 관측되는 common markers
- `X`: source-exclusive markers
- `Y`: target-exclusive markers
- `L`: coarse cell type
- `S_i,T_i`: 같은 patient/specimen에서 측정됐지만 cell은 대응되지 않는 두 population

목표는 새로운 source specimen에서 `H,X,L`만으로 `Y`를 예측하는 것이다. Validation/test
inference에는 target cells, OT plan, target encoder를 사용할 수 없다. Pair의 의미는 동일 cell이
아니라 **same patient + same specimen/timepoint**다.

핵심 질문은 다음 두 가지다.

1. `H,L`이 설명하는 강한 conditional-mean baseline을 넘을 수 있는가?
2. Training specimen pairing을 공개했을 때 `X`의 추가 정보가 새로운 patient로 전달되는가?

PICASSO의 Top-K SAE concept와 cyDM식 cross-decoding은 후보 표현으로 검토했다. 그러나 독립
AE/SAE axis에는 rotation·permutation·scale 자유도가 있고, cycle/reconstruction만으로
`X→Y` semantics가 식별되지 않는다. 따라서 명시적인 relation supervision이 필요하다는 가설을
검증했다.

## 2. 데이터와 평가 계약

### AML 데이터

| Modality | Specimens | Unique patients | Channels | Transform |
|---|---:|---:|---:|---|
| CyTOF | 93 | 83 | 44 | `asinh(x/5)` |
| Spectral Flow | 95 | 85 | 30 | upstream `asinh(x/6000)` |
| Clinical Flow | 46 | 40 | 17 | marker `asinh(x/5)` |

SF–CyTOF exact specimen pair는 93개다. Clinical Flow는 multiplex channel에 cell-type label이
직접 반영되고 일부 longitudinal specimen에서 label vector 재사용 정황이 있어 primary benchmark에서
제외했다. 모든 split은 `Rxxxx` patient 단위이며 longitudinal specimen은 같은 fold에 둔다.

SF–CyTOF canonical common markers는 exact-name 17개에 `PD-1↔CD279`, `CCR7↔CD197`을 더한
19개다. 실제 cross-platform 방향의 target-exclusive marker는 SF→CyTOF 25개,
CyTOF→SF 11개이며 SF의 technical channels 4개는 biological result와 분리해야 한다.

### Two-sided exact-truth benchmark

CyTOF full panel을 biology를 보지 않고 immutable panel order로 분할했다.

- `H=19`: CD3, CD4, CD8, CD14, CD19, CD20, CD25, CD27, CD33, CD34, CD38, CD45,
  CD56, CD117, CD123, CD197, CD279, CD45RA, HLA-DR
- `X=13`: CD11c, CD15, CD28, CD57, CD66b, CD96, CD161, CD185, CD196, CD274,
  CD294, CD45RO, TCRgd
- `Y=12`: CD13, CD16, CD47, CD64, CD69, CD127, CD183, CD194, CD200, CD276,
  CD366, IgD

93 specimens에서 acquisition-order bias가 없는 reservoir sampling으로 specimen당 최대 50,000 cells,
총 4,358,476 cells를 사용했다. Fold 0은 train/validation/test 62/13/18 specimens이며 test는
17 patients, 855,002 cells다. Seeds는 4207/4208/4209다.

평가 원칙:

- exact-cell: patient-first normalized MAE/RMSE와 marker-wise Spearman
- population: patient×cell-type×marker Wasserstein, median/quantile error, Spearman
- marker scale: train target의 `max(IQR, 0.1·SD, 1e-3)`
- alpha와 architecture는 validation에서만 선택
- hidden source `Y`는 diagnostic에만 쓰고 model fitting에는 사용하지 않음
- source/target training cells는 동일 stratum 안에서도 서로 disjoint

## 3. 최종 후보: paired soft-OT direct distillation

### 학습

1. Train target cells로 deployable baseline `b(H,L)=Ridge-H+L`을 학습한다.
2. 각 training specimen×coarse-cell-type을 disjoint source/target half로 나눈다.
3. Train-fitted common percentile space의 `H`만으로 balanced Sinkhorn plan `P`를 계산한다.
4. Target `Y`의 row-conditional barycentric projection을 soft teacher로 사용한다.
5. `fθ(H,X,L)`가 `P` teacher와 baseline의 residual을 예측하도록 2×128 MLP를 학습한다.
6. 최종 예측은 `Y_hat=b(H,L)+alpha·fθ(H,X,L)`이며 alpha는 validation에서 고른다.

고정값은 `k≤256`, `k_min=32`, Sinkhorn 300 iterations, paired epsilon/median-cost ratio 0.1,
batch 4096, 최대 30 epochs, train-teacher 10% early-stopping holdout,
`alpha∈{0,0.1,0.25,0.5,1}`이다. OT cost에는 `X,Y`를 넣지 않는다.

### 추론

새 specimen에서는 `H,X,L`, target-percentile mapping과 학습된 baseline/MLP만 사용한다.
Target population, OT, projection head는 필요 없다. 따라서 OT는 inference algorithm이 아니라
**train-only weak supervision generator**다.

## 4. 실험 결과

### 4.1 초기 unpaired/SAE 결과

Acquisition-head 20K screening에서는 classical H/kNN 방법이 AE/SAE보다 강했다. 예를 들어
SF→CyTOF에서 kNN+L은 population `W=0.5843`, SAE graph는 `1.2547`,
SAE shuffled graph는 `1.1468`이었다. CyTOF→SF에서도 MLP-H `W=0.4288` 대비
SAE graph `1.3734`였다. 이 결과는 final-analysis eligible 수치가 아니라 실패 원인 분리용
screening이다.

50K exact-truth benchmark에서는 `X` signal의 존재와 unpaired mapping 실패가 동시에 확인됐다.

| Method | alpha | Exact MAE ↓ | RMSE ↓ | Spearman ↑ | Population W ↓ |
|---|---:|---:|---:|---:|---:|
| Ridge-H | 0 | 0.7719 | 1.0983 | 0.3328 | 0.8758 |
| Direct AE / SAE(H) / SAE(H+X) | 모두 0 | 0.7719 | 1.0983 | 0.3328 | 0.8758 |
| Paired-cell Ridge(H+X) oracle | — | **0.7220** | **1.0335** | **0.3689** | **0.7492** |

Oracle은 MAE를 6.46% 개선했으므로 `X`에는 `Y` 정보가 있다. SAE reconstruction explained
variance도 최소 0.935였으므로 단순 representation collapse가 아니다. 실패는 concept relation이다.
Null-calibrated matched edge는 seed별 3/0/2개뿐이고 edge Jaccard 평균은 0.0236,
real graph가 shuffled보다 낫다는 empirical p는 0.93–0.96이었다.

### 4.2 Prototype DeCUR

Specimen×cell-type prototype DeCUR은 cell pair 없이 common axes correlation을 identity,
unique axes를 zero로 학습했다. Axis warm start 후 correct-pair common diagonal은 0.64–0.72까지
형성됐지만 validation은 3/3 seeds에서 residual alpha=0을 선택했다. Shuffled pairing도 비슷한
correlation을 만들었다. 결론은 다음과 같다.

> Activation correlation은 target decoder direction의 호환성을 보장하지 않는다.

따라서 prototype DeCUR은 `stop_decur`로 판정했다.

### 4.3 OT pseudo-target의 품질

Specimen×type별 `k≤256` cells에서 common-marker OT를 직접 평가했다. 아래 수치는 동일하게
balanced sampling된 source cells의 transductive diagnostic이며 target pool을 사용한다.

| Donor condition | Method | Exact MAE ↓ |
|---|---|---:|
| Source only | Ridge-H | 0.983741 ± 0.001833 |
| Same specimen×type | H 1-NN | 0.865658 ± 0.002838 |
| Same specimen×type | hard OT argmax | 0.894416 ± 0.004162 |
| Same specimen×type | **soft OT barycenter** | **0.720088 ± 0.003009** |
| Other-specimen same-type pool | Ridge/group reference | 0.960867 |
| Other-specimen same-type pool | **soft OT barycenter** | **0.904130 ± 0.003551** |

Soft OT는 paired H-only 대비 26.8%, pooled H-only 대비 8.1% 개선했다. Hard argmax는 실패했고
pooled plan의 top probability는 0.057, effective targets는 약 124개였다. 따라서 `P`는 hard cell
pair가 아니라 diffuse soft coupling으로 사용해야 한다. Barycenter는 평균 방향으로 shrink되므로
cell MAE는 좋지만 distribution W를 악화시킬 수 있다.

### 4.4 Train-only OT와 new-specimen inference

Train 297 blocks, source/target 각각 66,885 cells로 teacher를 만들고 validation/test에서는 target
reference를 제거했다.

| Method | Selected alpha | Test exact MAE ↓ | Population W ↓ |
|---|---:|---:|---:|
| Ridge-H | — | 0.766106 ± 0.000764 | 0.638040 |
| Ridge-H+L | — | 0.622729 ± 0.001141 | **0.511320** |
| Paired OT, MLP H+L | 0.5 / 0.5 / 0.5 | 0.607280 ± 0.001263 | 0.520163 |
| **Paired OT, MLP H+X+L** | **1 / 1 / 1** | **0.595547 ± 0.002476** | 0.520510 |
| Paired OT, Ridge H+X+L | 1 / 0.5 / 1 | 0.611810 ± 0.002665 | 0.538771 |
| Pooled OT, MLP H+X+L | 0.1 / 0.1 / 0.1 | 0.622082 ± 0.000897 | 0.523091 |
| Pooled uniform, MLP H+X+L | 0.1 / 0.1 / 0.1 | 0.622276 ± 0.001081 | 0.530647 |

Paired direct OT는 사전 gate를 통과했다.

- Ridge-H+L 대비 exact MAE 4.37% 개선
- 동일 teacher의 H-only distiller 대비 1.93% 개선
- 두 control을 validation/test에서 모두 이김: 3/3 seeds
- RMSE `0.85745→0.83443`, population median error `0.43211→0.39630`
- alpha=1이 3/3이므로 bridge signal은 validation에서 제거되지 않음

반면 pooled-unpaired H+X와 H-only는 test MAE가 각각 0.622023/0.622023으로 동일했다.
Transductive pooled OT의 이득은 target pool을 제거하면 source `X`에 남지 않는다.

### 4.5 Sparse OT-DeCUR

`X-E[X|H,L]`, `Y-E[Y|H,L]`에 AE latent 16 + Top-K SAE 64/Top-8을 학습하고,
soft `P`-weighted DeCUR과 cross-decoding을 결합했다.

| Pairing | Method | Test exact MAE ↓ | Population W ↓ |
|---|---|---:|---:|
| Paired | Direct OT MLP | **0.595547** | 0.520510 |
| Paired | OT crossdecode | 0.621351 | **0.500494** |
| Paired | OT+DeCUR | 0.621335 | 0.500723 |
| Pooled | OT crossdecode | 0.622331 | 0.504544 |
| Pooled | OT+DeCUR | 0.622035 | **0.501980** |

Paired DeCUR common diagonal은 0.029, pooled는 0.0056이었고 common loss도 감소하지 않았다.
Distribution loss는 W를 개선했지만 DeCUR 고유 효과는 아니었다. Sparse axis swapping은 direct
predictor보다 MAE가 4.33% 높아 최종 모델에서 제외한다.

### 4.6 Direct MLP의 auxiliary-only DeCUR

SAE 병목과 DeCUR objective를 분리하기 위해 predictor hidden에 별도 projection head를 달았다.
Target encoder와 projection은 train-only이며 decoder 입력이나 inference에는 사용하지 않는다.
Projection 32 dimensions 중 common/unique는 16/16, DeCUR weight는 0.05로 고정했다.

| Method | Validation MAE ↓ | Test MAE ↓ | Population W ↓ | Spearman ↑ |
|---|---:|---:|---:|---:|
| Direct OT rerun | 0.602028 | 0.595622 | 0.520279 | 0.387567 |
| **Paired OT + DeCUR auxiliary** | **0.601277** | **0.595083** | **0.519914** | **0.388045** |
| Uniform DeCUR control | 0.602028 | 0.595622 | 0.520279 | 0.387567 |

Directional gate는 validation/test 각 2/3 seeds로 통과했다. Common diagonal은 약 0에서
0.753–0.759로 증가하고 loss는 0.46–0.49에서 0.106–0.110으로 감소했다. 즉 SAE가 없으면
DeCUR signal은 학습된다. 그러나 common off-diagonal도 0.597–0.757로 높고 예측 이득은
MAE `-0.000539`, 상대 0.0905%뿐이다. Seed 평균 뒤 12/17 patients가 개선됐지만 patient-cluster
bootstrap 95% CI `[-0.00162,+0.00075]`는 0을 포함한다.

결론은 **SAE가 representation failure의 주원인이지만 DeCUR의 추가 predictive value도 작다**다.
Auxiliary DeCUR은 원인 분리 ablation으로만 유지한다.

## 5. 최종 해석과 모델 결정

### 유지

- `H+L`: 강한 deployable baseline이자 residual skip
- Same-specimen×type common-marker soft OT: explicit training relation
- Direct `H+X+L→Y residual` MLP: 현재 최종 후보
- Validation-only alpha gate: 실패한 bridge가 baseline을 망가뜨리지 않도록 유지

### 중단 또는 supplementary

- Independent AE decoder cross-use와 latent cycle
- Common-fingerprint SAE graph
- Prototype/Top-K SAE DeCUR cross-decoding
- Fully pooled-unpaired OT distillation
- Auxiliary DeCUR의 main-method claim

왜 H를 제거하지 않는가:

1. H+L 자체가 강한 conditional-mean predictor다.
2. OT cost도 H에서 계산된다.
3. Paired relation이 약하거나 없는 경우 alpha gate가 정확히 H-only로 돌아간다.
4. SAE는 H를 대체하지 못했고, direct paired OT도 H baseline 위의 residual로 가장 안정적이었다.

현재 positive claim은 제한적이지만 명확하다.

> Training specimen identity를 알고 있으면, common-marker soft OT가 exact cell pair 없이도
> source-exclusive `X`의 정보를 새로운 specimen의 `Y` 예측으로 전달한다.

완전히 unpaired한 cohort에는 이 claim을 적용하지 않는다. 그 경우 specimen-level outer matching
또는 다른 식별 가정이 추가로 필요하다.

## 6. Publication readiness와 다음 단계

현재 pseudo-panel fold 0만으로 BIB 제출 수준은 아니다. 알고리즘보다 paired-specimen/cell-unpaired
문제 설정과 실제 검증 범위가 논문 임팩트를 결정한다.

우선순위:

1. Direct paired OT의 distribution/quantile regularizer를 작은 fixed ablation으로 시험
2. Real AML SF↔CyTOF fold 0에서 target-free population generalization 확인
3. `H-shuffled`, specimen-pair shuffled, H-only distiller, CyTOFmerge/kNN control 완성
4. AML patient-disjoint 5-fold와 patient-cluster bootstrap
5. Marker/cell-type/rare-population 및 downstream AML phenotype 분석
6. AML에서 protocol을 동결한 뒤 Nunez external validation

Real AML과 Nunez에서 재현되면 main story는 “specimen-paired but cell-unpaired soft-OT supervision”으로
BIB를 노릴 수 있다. DeCUR은 이 스토리의 핵심이 아니다.

## 7. 재현 자산

Active scripts:

- [`prepare_pseudopanel_cache.py`](../scripts/prepare_pseudopanel_cache.py): raw CyTOF에서 deterministic
  50K cache 생성
- [`ot_new_sample_experiment.py`](../scripts/ot_new_sample_experiment.py): direct paired/pooled OT 비교
- [`aggregate_ot_new_sample.py`](../scripts/aggregate_ot_new_sample.py): 3-seed gate
- [`direct_ot_decur_aux_experiment.py`](../scripts/direct_ot_decur_aux_experiment.py): auxiliary DeCUR ablation
- [`aggregate_direct_ot_decur_aux.py`](../scripts/aggregate_direct_ot_decur_aux.py): auxiliary 결과 집계

Retained results:

- [`artifacts/ot_new_sample/aggregate.json`](../artifacts/ot_new_sample/aggregate.json)
- [`artifacts/direct_ot_decur_aux/aggregate.json`](../artifacts/direct_ot_decur_aux/aggregate.json)
- [`artifacts/pseudopanel_50k/aggregate.json`](../artifacts/pseudopanel_50k/aggregate.json)
- [`artifacts/prototype_decur/final_gate/aggregate.json`](../artifacts/prototype_decur/final_gate/aggregate.json)
- [`artifacts/ot_pseudopair/aggregate.json`](../artifacts/ot_pseudopair/aggregate.json)
- [`artifacts/ot_decur_new_sample/aggregate.json`](../artifacts/ot_decur_new_sample/aggregate.json)

Main GPU runs used the `cytometry` environment and H200 `1/1` MIG
(`NVIDIA H200 MIG 1g.18gb`, visible 16 GiB). Direct OT jobs were 48158599–48158601,
auxiliary DeCUR jobs were 48160484–48160486. CUDA matmul, training, JSON/checkpoint generation을
확인했고 stderr는 비어 있었다. 정리 전 전체 test suite는 54 tests를 통과했다.
