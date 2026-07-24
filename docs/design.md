# 현재 구현 설계

## Data contract

```text
source Spectral Flow = H + X
target CyTOF         = H + Y
```

specimen identity는 paired되어 있지만 cell identity는 paired되어 있지
않다. split unit은 patient이며 test patient는 common-space calibration,
kNN reference, OT teacher, gate와 probe training에서 제외된다.

## Translation

1. train patients에서 source와 target H의 empirical CDF transform을
   각각 fit한다.
2. target-training cells의 patient-balanced reference bank로 median
   `k=50` H-only kNN을 fit한다.
3. paired train specimens 안에서 H만 사용한 balanced Sinkhorn OT로 soft
   Y teacher를 만든다.
4. kNN prediction과 teacher Y의 차이를 동일한 MLP 세 개로 학습한다.

```text
r_H       = residual(H)
r_shuffle = residual(H, within-specimen shuffled X)
r_HX      = residual(H, correct X)
```

inference에서 OT나 target specimen은 사용하지 않는다.

## 왜 세 residual이 필요한가

- `r_H`는 비선형 residual capacity만 추가한 control이다.
- `r_shuffle`은 X marginal과 input dimension은 유지하지만 row-level
  H/X association을 제거한 null이다.
- `r_HX`가 두 control을 모두 이길 때만 X의 incremental value를
  주장할 수 있다.

기존 label-free marker gate는 validation population Wasserstein으로
`r_HX`의 marker별 alpha를 선택한다. ungated `alpha=1`도 항상 함께
평가해 global objective가 rare signal을 제거하는지 확인한다.

## Probe

source original fine labels 8개와 coarse labels 5개를 translator와
분리된 linear probe target으로 사용한다. probe training은
specimen/class cap과 total class cap으로 균형화하고, test는 natural
prevalence의 모든 reservoir-sampled cells를 평가한다.

primary representations:

```text
Y_hat from kNN(H)
Y_hat from kNN(H) + r_H
Y_hat from kNN(H) + r_shuffle
Y_hat from kNN(H) + r_HX
```

`H+Y_hat`는 downstream usability를 보는 secondary diagnostic이다. H가
원래 label을 강하게 보존하므로 translation fidelity의 primary
evidence로 사용하지 않는다.

## Evaluation

- patient-first balanced accuracy, macro-F1, macro-AUPRC
- T-subtype AUPRC
- DN/DP macro AUPRC
- DN과 DP 각각의 AUPRC와 recall
- representation 간 paired patient-bootstrap interval

다음 paired-count 확장에서는 H-only kNN reference는 고정하고 OT
teacher에 공개되는 paired patient 수만 nested subset으로 줄인다.
