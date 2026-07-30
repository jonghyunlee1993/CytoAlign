# Same-cell marker recoverability v0

Archive note: superseded as the active synthesis by `docs/research_record.md`;
retained for detailed provenance.
The disposable `outputs/` tree was removed at the 2026-07-30 session closeout;
the tables below are the durable result record.

Date: 2026-07-27

## Question

Within each modality, can markers hidden from a cell be predicted from the
markers that remain observed on that same cell? If prediction is possible,
which markers are recoverable, how much is lost when the observed panel is
reduced from H19 to a clinical 10-marker panel, and does a fixed classifier
trained from the original manual labels still recover rare populations?

This is a phenomenon-characterization benchmark. It does not introduce a new
imputation method.

## Scope and design

- Modalities are evaluated separately: spectral flow and CyTOF.
- The split is patient-grouped, with 5 outer folds covering 83 patients and 93
  specimens.
- Every condition was run with seeds 4207, 4208, and 4209. There are 60
  completed runs: 2 modalities x 2 observed panels x 5 folds x 3 seeds.
- H19 and clinical10 are the two observed panels. Clinical10 contains CD3, CD4,
  CD8, CD14, CD19, CD20, CD33, CD34, CD45, and CD56.
- Clinical10 is a clinical-overlap-inspired pseudo-mask applied inside the SF
  and CyTOF data. It is not a cross-platform run on the clinical-flow dataset.
- Predictors are exact median kNN with k=50 and a fixed two-layer 128-unit MLP.
  A train-patient median is the null predictor.
- Marker evaluation is performed against the true value of the same cell.
  Primary metrics are null-relative MAE skill and within-specimen Spearman
  correlation. A shuffled-observed-marker control tests whether performance
  depends on the same-cell correspondence.
- Biology replay uses the existing manual annotation labels. A class-balanced
  multinomial logistic classifier is trained once on true markers from the
  outer-fit patients and is never refit for an imputed representation.
- Patient is the bootstrap unit. Seeds and multiple specimens from one patient
  are averaged before the patient bootstrap.

The current input is the processed, upstream-pregated data. The results are
therefore conditional sensitivity results and are not yet full-population
claims.

## Main result

Same-cell prediction is possible, but it is marker- and modality-specific.
Spectral-flow functional markers are consistently recoverable from H19. CyTOF
is mixed: local kNN is useful for many markers, while this simple MLP often
preserves ordering but has poor absolute calibration. Reducing H19 to
clinical10 causes a statistically clear loss for the markers that are hidden
in both panels.

The table reports patient-marker means for marker metrics and patient means for
the fixed full-panel classifier.

| Modality | Observed panel | Method | Null-relative skill | Spearman | Full macro-F1 |
|---|---|---:|---:|---:|---:|
| Spectral flow | H19 | kNN | 0.213 | 0.512 | 0.910 |
| Spectral flow | H19 | MLP | 0.242 | 0.591 | 0.913 |
| Spectral flow | clinical10 | kNN | 0.178 | 0.428 | 0.878 |
| Spectral flow | clinical10 | MLP | 0.173 | 0.463 | 0.884 |
| CyTOF | H19 | kNN | 0.125 | 0.327 | 0.786 |
| CyTOF | H19 | MLP | -0.003 | 0.360 | 0.836 |
| CyTOF | clinical10 | kNN | 0.116 | 0.313 | 0.799 |
| CyTOF | clinical10 | MLP | -0.053 | 0.323 | 0.810 |

The true full-panel macro-F1 is 0.931 for spectral flow and 0.928 for CyTOF.
Median-fill macro-F1 is 0.891/0.845 for spectral-flow H19/clinical10 and
0.651/0.636 for CyTOF H19/clinical10.

The negative MLP skill in CyTOF is not a contradiction with its positive
Spearman correlation or classifier performance. It means the fixed MLP has
cell-level ordering information but makes sufficiently large magnitude or tail
errors to lose against the median in MAE. It should not be called a successful
quantitative imputer for CyTOF in this configuration.

## Marker-level recoverability

The operational call below uses the kNN null-relative skill:

- recoverable: patient-bootstrap 95% CI is entirely above zero;
- unresolved: the CI includes zero;
- worse than null: the CI is entirely below zero.

For spectral flow with H19, all 7 hidden markers are recoverable:

| Marker | kNN skill | 95% CI |
|---|---:|---:|
| CD95 | 0.354 | 0.304 to 0.401 |
| CTLA-4 | 0.318 | 0.215 to 0.403 |
| T-bet | 0.208 | 0.152 to 0.252 |
| EOMES | 0.200 | 0.178 to 0.223 |
| FOXP3 | 0.185 | 0.149 to 0.224 |
| TIGIT | 0.173 | 0.149 to 0.196 |
| Ki-67 | 0.050 | 0.021 to 0.077 |

Ki-67 is statistically above the null, but the gain is small. It should be
described as weakly recoverable rather than interchangeable with measurement.

For CyTOF with H19, 18 of 25 hidden markers are recoverable by this criterion.
The strongest are CD64 (0.388), CD11c (0.348), CD28 (0.290), CD45RO (0.231),
CD127 (0.199), CD16 (0.197), TCRgd (0.189), CD294 (0.183), and CD69 (0.179).
CD200, CD15, CD57, CD66b, CD276, and CD366 are unresolved. CD96 is worse than
the median null: -0.028, 95% CI -0.055 to -0.006.

With clinical10, kNN calls 15 of 16 spectral-flow targets recoverable; Ki-67 is
unresolved. For CyTOF, 25 of 34 targets are recoverable, 7 are unresolved, and
CD96 and CD25 are worse than the null. Thus neither H19 nor clinical10
guarantees recoverability of every hidden CyTOF marker.

## Effect of reducing H19 to clinical10

The clean comparison uses only markers hidden under both panels. This avoids
confounding panel reduction with the nine additional targets created when H19
markers are hidden.

| Modality | Method | Paired skill change | 95% CI | Paired Spearman change |
|---|---|---:|---:|---:|
| Spectral flow | kNN | -0.065 | -0.075 to -0.055 | -0.085 |
| Spectral flow | MLP | -0.107 | -0.129 to -0.087 | -0.127 |
| CyTOF | kNN | -0.019 | -0.028 to -0.012 | -0.030 |
| CyTOF | MLP | -0.063 | -0.076 to -0.052 | -0.048 |

Every spectral-flow H19-hidden marker loses kNN skill under clinical10. CTLA-4
has the largest loss (-0.118), followed by EOMES (-0.071). In CyTOF, the
largest kNN losses are CD45RO (-0.123), CD64 (-0.053), CD15 (-0.050), and IgD
(-0.047).

The nine H19 markers that become targets under clinical10 are not equally
recoverable. In spectral flow, kNN skill is high for CD117 (0.397), HLA-DR
(0.329), CD27 (0.314), and CD123 (0.244), but weak for CD25 (0.047) and PD-1
(0.093). In CyTOF, CD25 is worse than null (-0.075), CD279/PD-1 is weak
(0.038), while CD197/CCR7 (0.263), CD123 (0.222), CD117 (0.213), and CD27
(0.208) are more recoverable.

These two panel points do not identify a unique minimal sufficient set. They do
support the following guardrails:

1. Clinical10 is a viable lower bound for coarse lineage replay, not for
   faithful recovery of every hidden protein.
2. CD25 should remain directly observed when its quantitative value matters,
   especially in CyTOF. PD-1/CD279 is also too weak to treat as safely
   recoverable.
3. If CD45RO, IgD, CTLA-4, EOMES, or related functional states are primary
   endpoints, H19 is materially safer than clinical10.
4. A claim that a shared panel “guarantees success” requires targeted
   leave-one-shared-marker-out or nested-panel experiments; it cannot be made
   from H19 versus clinical10 alone.

## Fixed-label biology replay

| Modality | Panel | True full | Median fill | kNN hybrid | MLP hybrid |
|---|---|---:|---:|---:|---:|
| Spectral flow | H19 | 0.931 | 0.891 | 0.910 | 0.913 |
| Spectral flow | clinical10 | 0.931 | 0.845 | 0.878 | 0.884 |
| CyTOF | H19 | 0.928 | 0.651 | 0.786 | 0.836 |
| CyTOF | clinical10 | 0.928 | 0.636 | 0.799 | 0.810 |

The entries are patient-mean macro-F1 from the fixed full-panel classifier.
Relative to median fill, the MLP recovers 54% and 46% of the true-full gap in
spectral-flow H19 and clinical10, and 67% and 60% in CyTOF H19 and clinical10.
This is meaningful but incomplete biology preservation.

Panel reduction lowers spectral-flow full-hybrid macro-F1 by 0.032 for kNN and
0.028 for MLP. In CyTOF it lowers balanced accuracy by 0.025 for kNN and 0.051
for MLP. CyTOF kNN macro-F1 rises by 0.013 because of class-specific tradeoffs,
so macro-F1 alone must not be used to claim that clinical10 is better.

Shuffling the observed-marker rows keeps the predicted marginal distribution
and dynamic range unchanged, but destroys same-cell correlation. It also
reduces full-panel macro-F1 to 0.839-0.846 in spectral-flow H19 and
0.638-0.652 in CyTOF H19. This directly shows why distribution matching is not
an adequate success criterion.

## Rare populations

Rare populations defined mainly by markers already in the observed panel can
look almost perfectly preserved without proving that hidden functional biology
was recovered. For example, spectral-flow T-cell DP (mean prevalence 0.094%)
and DN (0.377%) retain approximately 0.99 recall because CD3/CD4/CD8 remain
observed.

CyTOF T-cell gd is the more diagnostic stress test because TCRgd is hidden in
both panels:

| Panel | True recall | Median recall | kNN recall | MLP recall | MLP AUPRC |
|---|---:|---:|---:|---:|---:|
| H19 | 0.986 | 0.000 | 0.253 | 0.583 | 0.732 |
| clinical10 | 0.986 | 0.000 | 0.478 | 0.615 | 0.725 |

The rare population is partially recovered but not restored to the measured
full-panel result. Its true AUPRC is 0.994. If reliable T-cell-gd recovery is a
requirement, TCRgd should remain measured rather than imputed.

CyTOF T-cell DN illustrates a different failure mode. Under H19, kNN recall is
0.941 but its absolute prevalence error is 0.457 percentage points for a
population whose mean prevalence is only 0.402%. The MLP has lower recall
(0.794) but better prevalence error (0.237 percentage points). Rare-cell
assessment therefore needs recall, precision/AUPRC, and prevalence error
together.

## Artifacts

The machine-readable patient-first tables are under
`outputs/aml_same_cell_recoverability_v0/summary/`:

- `marker_summary.csv`: per-marker estimates and patient-bootstrap intervals;
- `marker_patient.csv`: patient-level marker metrics;
- `biology_summary.csv`: fixed-classifier overall metrics;
- `biology_class_summary.csv`: class-specific rare-cell metrics;
- `biology_retention.csv`: recovery of the median-to-true biology gap;
- `summary.json`: run coverage and aggregation metadata.
