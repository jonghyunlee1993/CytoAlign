# Cross-platform cytometry translation: 현재 결론과 다음 실험

마지막 정리: 2026-07-24

## 한 문장 결론

현재 결과는 **“Spectral Flow의 source-exclusive marker `X`가 희귀한
fine cell type 정보를 CyTOF target marker `Y`로 옮기는 데 실제로 도움이
될 가능성”**을 지지한다. 하지만 아직 `X`의 생물학적 정보가 원인인지,
단순히 H+X 모델의 추가 용량이 원인인지 분리하지 못했다. 따라서 지금은
아키텍처를 크게 바꾸기보다, 같은 모델에서 capacity-matched H-only
residual과 X-shuffle control을 추가해 이 차이를 먼저 확인하는 것이 맞다.

## 1. 풀고자 하는 문제

같은 specimen에서 두 플랫폼이 다음처럼 서로 다른 marker를 측정한다고
하자.

```text
Spectral Flow: H + X
CyTOF:         H + Y
```

- `H`: 두 플랫폼에 공통인 marker
- `X`: Spectral Flow에만 있는 marker
- `Y`: CyTOF에만 있는 marker

cell identity는 플랫폼 사이에 paired되어 있지 않다. 따라서 관측 자료는
`P(H, X)`와 `P(H, Y)`를 주지만, 우리가 원하는 `P(Y | H, X)`는 직접
식별되지 않는다. 같은 `H`를 가진 세포라도 `X`가 다르면 `Y`가 다를 수
있지만, cell-unpaired 자료만으로는 어떤 `X`와 어떤 `Y`가 실제로
연결되는지 알 수 없기 때문이다.

이 프로젝트에서 point imputation이 답해야 하는 질문은 다음과 같다.

> 새 Spectral Flow 세포의 `H+X`를 보고, CyTOF에서 관측했을 법한 `Y`를
> 생성할 때 `X`가 `H`만 사용한 강한 비모수 baseline보다 추가 정보를
> 제공하는가?

이 질문은 단순한 marginal distribution matching보다 강하다. 전체
marker 분포를 비슷하게 만드는 것만으로는 한 세포의 rare subtype
identity가 생성된 `Y`에 보존되었다고 말할 수 없다.

## 2. 현재 translation 방식

현재 파이프라인은 의도적으로 단순한 baseline과 작은 residual model로
구성되어 있다.

1. 19개 공통 marker `H`를 train-patient 자료에서 플랫폼별 empirical
   CDF로 변환한다.
2. CyTOF training reference에서 H-only, `k=50` median kNN으로 `Y`를
   예측한다.
3. paired specimen 안에서 cell-unpaired H-space OT를 계산해 soft
   `Y` teacher를 만든다.
4. Spectral Flow의 `H+X`로 kNN의 오차를 예측하는 residual MLP를
   학습한다.
5. 각 Y marker의 residual multiplier를 validation population
   Wasserstein으로 선택한다.

이를 식으로 쓰면 다음과 같다.

```text
Y_hat_H    = kNN_Y(H)
Y_hat_HX   = kNN_Y(H) + alpha_marker * residual(H, X)
```

translator 학습과 gate 선택에는 cell-type label이나 mutation label을
사용하지 않았다. label은 held-out 평가에만 사용했다.

## 3. 지금까지 가장 중요한 결과

### 3.1 먼저 확인된 것: H-only cross-platform kNN은 유효한 baseline이다

5개 patient-unique specimen의 leave-one-specimen-out quick check에서
H-only kNN은 target-training global median보다 pooled normalized
Wasserstein을 20.9% 줄였다.

| 방법 | Pooled Wasserstein | Cell-type-stratified Wasserstein |
|---|---:|---:|
| Global target median | 1.0794 | 2.0676 |
| Cross-platform H-only kNN | 0.8540 | 1.5017 |
| Same-platform target-H kNN diagnostic | 0.7418 | 1.3142 |

다섯 specimen 모두에서 Wasserstein이 개선되었고 25개 Y marker 중
23개가 개선되었다. 반면 pooled median error는 6.5% 나빠졌다. 즉 H-only
kNN은 분포 모양을 옮기는 데는 유용하지만 specimen 중심값, cell-wise
identity, X/Y joint structure까지 회복한다고 볼 수는 없다.

### 3.2 Mutation prediction: compatibility는 유지했지만 성능을 높이지는 못했다

전체 93 paired specimens, 83 patient groups에 대해 patient-grouped
5-fold CV와 specimen당 최대 50,000 cells를 사용했다. 각 cell
representation을 specimen mean으로 pooling한 뒤 mutation classifier를
평가했다.

| Representation | Macro AUROC | Macro AUPRC | Balanced accuracy | Brier |
|---|---:|---:|---:|---:|
| Original Spectral Flow | 0.739 | 0.529 | 0.673 | 0.202 |
| Original CyTOF oracle | 0.800 | 0.686 | 0.751 | 0.158 |
| Translated H-only | 0.736 | 0.566 | 0.635 | 0.221 |
| Translated H+X gated | 0.738 | 0.571 | 0.632 | 0.213 |

Translated representation의 AUROC는 original Spectral Flow와 거의
같았고, 모든 endpoint의 patient-bootstrap interval이 0을 포함했다.
이는 CyTOF-domain classifier가 translated Spectral Flow를 입력으로
받는 cross-platform compatibility에는 긍정적이다.

그러나 H+X와 H-only 차이 역시 모든 endpoint에서 0을 포함했다. mutation
prediction이 목적이라면 observed Spectral Flow를 직접 쓰는 것이 더
간단하고 적어도 동등하다. 또한 mean pooling은 0.1% 수준의 rare
population 신호를 거의 지워버릴 수 있으므로, 이 negative result가
cell-level X 신호의 부재를 의미하지는 않는다.

### 3.3 Fine cell type: 전체 평균보다 rare population에서 X의 가치가 보였다

같은 5-fold translation에서 source의 원래 8개 fine label을 held-out
evaluation target으로 사용했다. 총 4,599,694 OOF source cells 가운데
rare T-cell prevalence는 다음과 같았다.

| Fine type | Cell 수 | Prevalence |
|---|---:|---:|
| T cell DN | 18,040 | 0.392% |
| T cell DP | 4,521 | 0.098% |

먼저 observed source H 자체에는 fine label 정보가 충분했다.

| Representation | Balanced accuracy | Macro-F1 | Macro-AUPRC | Rare DN/DP AUPRC |
|---|---:|---:|---:|---:|
| Observed source H | 0.964 | 0.871 | 0.965 | 0.937 |
| Observed source H+X | 0.978 | 0.895 | 0.977 | 0.944 |
| Imputed Y, H-only | 0.787 | 0.667 | 0.746 | 0.272 |
| Imputed Y, H+X ungated | 0.816 | 0.683 | 0.760 | 0.307 |
| Imputed Y, H+X gated | 0.792 | 0.675 | 0.751 | 0.281 |

중요한 현상은 `Y_hat`만 probe했을 때 나타난다. H-only imputation은 큰
population에서는 그럴듯하지만 rare subtype을 표현하는 능력이 크게
떨어졌다. H+X ungated residual은 그 손실의 일부를 일관되게 회복했다.

| Rare endpoint | H-only | H+X ungated | 차이와 95% patient-bootstrap CI |
|---|---:|---:|---:|
| DN AUPRC | 0.4488 | 0.4884 | +0.0397 [0.0286, 0.0510] |
| DN recall | 0.6351 | 0.6740 | +0.0389 [0.0266, 0.0520] |
| DP AUPRC | 0.0907 | 0.1207 | +0.0300 [0.0158, 0.0432] |
| DP recall | 0.6019 | 0.6687 | +0.0668 [0.0280, 0.1060] |
| DN/DP mean AUPRC | 0.2721 | 0.3067 | +0.0346 [0.0248, 0.0437] |

반면 기존 marker-wise gate는 DP AUPRC 증가를 +0.0032로 줄였고 그
interval은 0을 포함했다. DP recall은 오히려 -0.0061이었다. 전체
population Wasserstein으로 alpha를 고르면 57% Blast와 31% Monocyte가
목적함수를 지배하고, 0.1% DP에만 중요한 residual은 제거되기 쉽다.

### 3.4 Full H+Y panel 결과를 과대해석하면 안 된다

Mapped observed H를 imputed Y와 다시 합치면 rare DN/DP AUPRC는
H-only에서도 0.933이었다. 이 수치는 translation 성공의 증거가 아니다.
변하지 않은 H만으로 source fine label을 거의 완벽히 분류할 수 있기
때문이다.

```text
H + Y_hat가 label을 보존함
    └─ H가 원래 label 정보를 보존했기 때문일 수 있음

Y_hat만 label을 표현함
    └─ 새로 생성된 marker가 subtype 정보를 담았는지 직접 진단
```

따라서 다음 실험의 primary diagnostic은 계속 Y-only여야 한다. Full
H+Y는 실제 downstream usability를 보는 secondary endpoint로 두면 된다.

## 4. 현재 결과가 설명할 수 있는 것

현재 자료로 비교적 강하게 말할 수 있는 것은 다음 네 가지다.

1. **H-only kNN은 무시할 수 없는 baseline이다.** 단순 global median보다
   cross-platform marginal distribution을 안정적으로 잘 맞춘다.
2. **전체 평균은 rare failure를 숨긴다.** coarse label과 full-panel
   metric만 보면 H-only는 충분해 보이지만 Y-only fine probe에서는 rare
   subtype 정보가 크게 무너진다.
3. **현재 H+X path에는 rare-population correction이 있다.** DN과 DP의
   AUPRC와 recall이 patient-first evaluation에서 일관되게 증가했다.
4. **현재 global gate는 rare 목적에 맞지 않는다.** marker 전체 분포를
   평균한 objective가 rare-specific correction을 대부분 제거했다.

이 결과는 “복잡한 모델이 전체 평균을 조금 더 좋게 한다”보다 더
흥미롭다. 단순 H-only 모델이 구조적으로 잃기 쉬운 정보가 어느
population에서 드러나는지를 특정했기 때문이다.

## 5. 아직 설명할 수 없는 것

### 5.1 X의 정보인지 추가 model capacity인지

H-only는 kNN 하나이고 H+X는 kNN 위에 MLP residual이 추가된다. 따라서
현재 차이는 다음 두 설명을 모두 허용한다.

```text
설명 A: X가 H로는 알 수 없는 rare identity를 제공한다.
설명 B: residual MLP의 비선형 용량만으로도 H를 더 잘 사용할 수 있다.
```

capacity-matched H-only residual과 X-shuffle이 없으므로 아직 A를
인과적으로 지지할 수 없다.

### 5.2 생성된 Y가 실제 CyTOF biological state와 맞는지

fine label probe는 source label이 `Y_hat`에서 선형 분리되는지를 본다.
그 label을 표현하는 임의의 synthetic code도 probe를 통과할 수 있다.
따라서 다음 단계에서는 가능한 범위에서 real CyTOF Y의 subtype별
distribution, marker ordering, positive fraction과 비교해야 한다.

### 5.3 Point identification

cell-unpaired `P(H,X)`와 `P(H,Y)`만으로 `P(Y|H,X)`는 일반적으로
식별되지 않는다. OT teacher와 neural residual은 한 가지 inductive
bias를 선택한 것이지, 참 cell-wise coupling을 증명한 것이 아니다.
현재 결과는 predictive utility의 증거이지 cell-wise ground truth
recovery의 증거가 아니다.

### 5.4 Generalization 범위

- 한 AML cohort의 Spectral Flow → CyTOF 방향만 평가했다.
- fine labels는 source annotation이므로 cross-platform label taxonomy가
  완전히 같은지 별도 확인이 필요하다.
- DN/DP는 cell 수는 수천 개지만 patient 내 cell들이 독립 표본은 아니다.
  유효 표본수는 cell 수보다 rare population을 가진 patient 수에 가깝다.
- current intervals는 patient resampling을 반영하지만 cohort shift,
  label error, marker batch shift를 포함하지 않는다.

## 6. 현재 아키텍처를 고도화해야 하는가

### 지금 당장의 답: 큰 고도화는 보류한다

현재 residual architecture는 다음 질문에 답하기에 충분하다.

> correct X가 capacity-matched H-only와 shuffled X보다 rare Y identity를
> 더 잘 보존하는가?

이 질문을 통과하기 전에 mixture-of-experts, attention, conditional
diffusion, adversarial alignment 같은 큰 모델을 추가하면 해석이 더
어려워진다. 성능이 좋아져도 X signal, capacity, regularization 중 무엇
때문인지 분리하기 어렵다.

### 먼저 필요한 작은 변경

1. **H-only residual MLP:** 동일한 OT teacher, hidden width, epoch,
   optimizer를 사용하되 입력에서
   X만 제외한다.
2. **H+shuffled-X residual MLP:** train patient/specimen 안에서 X row를
   섞어 X의 marginal과 model
   capacity는 유지하고 cell-level association만 제거한다.
3. **세 가지 gate를 분리 보고:** `alpha=1` ungated, 기존 label-free
   global gate, fine-label-balanced
   supervised gate를 함께 둔다. 마지막 것은 deployable method가 아니라
   attainable upper bound로 명시한다.
4. **Rare-aware metric:** DN/DP AUPRC와 recall을 primary로,
   common-population non-inferiority를
   safety constraint로 둔다.

### 그 다음에만 고려할 고도화

correct X 효과가 두 control을 통과하면 다음 순서가 합리적이다.

1. marker-global alpha 대신 population- 또는 query-adaptive shrinkage;
2. common population과 rare population을 함께 다루는 hierarchical
   residual/expert;
3. 환자별 X reliability를 반영하는 uncertainty-aware gate;
4. point estimate와 함께 specimen-level uncertainty 또는 identified
   range 제공.

고도화의 목적은 전체 Wasserstein을 조금 더 낮추는 것이 아니라,
확인된 rare-X signal을 보존하면서 큰 population을 해치지 않는 것이다.

## 7. 기존 baseline 대비 가질 수 있는 장점

### H-only kNN의 장점

- 학습과 해석이 단순하고 안정적이다.
- paired specimen identity가 없어도 target reference만 있으면 된다.
- 전체 marginal distribution에서 이미 강하다.
- 추가 모델이 실패해도 자연스러운 fallback이다.

따라서 새 모델은 단순히 kNN과 비슷한 overall score를 내는 것으로는
가치가 없다.

### H+X residual이 가질 수 있는 고유한 장점

가능한 장점은 정확히 하나로 좁혀진다.

> H가 유사한 세포들 사이에서 X가 제공하는 조건부 정보를 이용해,
> H-only neighborhood가 평균내어 버리는 rare subtype-specific Y를
> 복원한다.

이 장점은 현재 DN/DP 결과와 방향이 맞는다. 최종적으로는 다음 조건을
동시에 만족해야 baseline 대비 우위가 성립한다.

- correct H+X가 H-only residual과 shuffled X보다 낫다.
- rare endpoint 개선이 patient 단위로 재현된다.
- common population과 global Wasserstein을 실질적으로 악화시키지 않는다.
- real target-domain subtype structure와 일치한다.
- paired specimen 수가 현실적으로 줄어도 효과가 유지된다.

## 8. Paired specimen 수가 줄어들 때 예상되는 것

현재 rare result는 93 exact pairs, 83 patient groups를 모두 이용한
translation에서 얻었다. 따라서 paired 수 감소에 대한 직접 증거는 아직
없다.

과거 전체-population paired-count curve에서는 다음 현상이 있었다.

- label-assisted marker gate는 32 pairs에서 kNN 대비 0.0104 Wasserstein
  개선과 강한 count response(`R²=0.886`)를 보였다.
- labels를 제거한 현실적 19-marker 설정에서는 32 pairs의 개선이
  0.0019에 불과했고 count response가 거의 사라졌다.
- strict-unpaired 설정은 오히려 kNN보다 나빴다.

이는 global metric 관점에서 pairing의 이점이 작다는 뜻이지, rare-X
효과가 적은 pairs에서도 유지되거나 사라진다는 답은 아니다. rare
signal은 더 적은 patient에 집중될 수 있어 오히려 paired 수에 더
민감할 가능성이 크다.

paired 수 감소 시 세 구성요소는 다르게 반응한다.

| 구성요소 | Paired 수 의존성 |
|---|---|
| H-only kNN | specimen pairing 불필요; target reference의 다양성이 중요 |
| H+X OT residual | paired specimen의 수와 다양성에 직접 의존 |
| Rare-aware gate | rare type을 가진 validation patient 수에 특히 의존 |

50,000 cells/specimen은 cell sampling noise를 줄이지만 독립 biological
replicate를 늘리지는 않는다. rare cell이 많은 한 patient를 더 많이
sampling하는 것보다 rare population을 가진 patient를 더 많이 포함하는
것이 중요하다.

## 9. 다음 실험의 최소 설계

### 고정할 것

- 동일한 83 patient-grouped 5 folds
- specimen당 최대 50,000-cell uniform reservoir
- 동일한 H common-space transform, H-only kNN, OT teacher
- 동일한 residual MLP capacity와 optimizer
- fine label은 translator 입력에서 제외
- natural-prevalence test와 patient-first aggregation

### 비교할 것

```text
1. kNN(H)
2. kNN(H) + residual(H)                 capacity control
3. kNN(H) + residual(H, shuffled X)     association-null control
4. kNN(H) + residual(H, correct X)      proposed path
```

각 residual에 대해 최소한 `alpha=1`과 label-free global gate를
보고한다. correct-X에 대해서는 fine-label-balanced supervised gate를
upper-bound diagnostic으로 추가한다.

### Paired-count curve

각 outer fold 안에서 patient/specimen pair를 seed-fixed nested subset으로
선택한다.

```text
4, 8, 16, 32, 64, all available pairs
```

0-pair condition은 plain H-only kNN이다. 핵심 x축은 cell 수가 아니라
paired patient 수이며, 각 count에서 rare-positive patient 수와 DN/DP
training cell 수를 함께 기록한다. 한 random ordering에 의존하지 않도록
최소 세 개의 nested-subset seed를 사용한다.

### Primary endpoints

1. Y-only DN/DP macro AUPRC
2. Y-only DN AUPRC와 recall
3. Y-only DP AUPRC와 recall

Secondary endpoints:

- fine macro-AUPRC와 T-subtype AUPRC
- coarse macro-AUPRC
- global/cell-type-stratified normalized Wasserstein
- marker별 target-domain median, positive fraction, upper quantile
- full H+Y downstream probe

### 사전에 정할 판단 기준

방법 수준의 GO는 다음을 모두 요구한다.

1. all-pairs에서 correct H+X의 rare macro-AUPRC가 H-only residual과
   shuffled X를 모두 patient-bootstrap CI 기준으로 상회한다.
2. 효과가 DN과 DP 중 하나에만 의존하지 않고 두 subtype에서 같은
   방향이다.
3. coarse macro-AUPRC와 global Wasserstein이 사전 정의한 허용 범위
   이상 악화되지 않는다.
4. paired-count curve에서 최소 하나의 현실적인 count가 all-pairs 효과의
   사전 정의 비율을 유지한다.
5. generated Y의 subtype별 target-domain marker 구조가 실제 CyTOF와
   모순되지 않는다.

correct X가 control을 이기지 못하면 현재 neural residual direction은
중단하고 H-only kNN을 point-imputation baseline으로 남기는 것이 맞다.

## 10. 최종 종합 판단

현재 결과는 유망하지만 그 이유를 좁게 해석해야 한다.

- 유망한 부분은 overall metric의 작은 향상이 아니다.
- 진짜 신호는 H-only가 rare subtype의 Y 표현을 잃고, H+X path가 그
  일부를 patient-consistent하게 회복했다는 점이다.
- 현재 gate가 그 효과를 제거한다는 사실은 모델 전체의 실패가 아니라
  objective mismatch를 보여준다.
- 동시에 capacity confounding과 point-identification 한계 때문에 아직
  “X가 true Y를 복원한다”는 결론은 이르다.

따라서 현재 아키텍처는 **확장 대상이 아니라 검증 대상**이다. 다음
실험에서 correct X의 조건부 가치가 두 control과 reduced-pair setting을
통과할 때만 rare-aware architecture로 고도화하는 것이 가장 정보
효율적이다.
