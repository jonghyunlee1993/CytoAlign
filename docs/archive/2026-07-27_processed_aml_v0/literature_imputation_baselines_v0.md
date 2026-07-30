# Literature marker-imputation baselines v0

Archive note: superseded as the active synthesis by `docs/research_record.md`;
retained for detailed provenance.
The disposable `outputs/` tree was removed at the 2026-07-30 session closeout;
the tables below are the durable result record.

## Question

On the already frozen same-cell pseudo-masking folds, can a method recover each
pseudo-unobserved marker from a specified shared-marker set, and do those
imputed markers preserve the output of the fixed biology classifier trained on
the complete outer-fit data?

This experiment is a method comparison, not method development.  H19 and
clinical10 are evaluated separately in spectral flow and CyTOF.  Hidden-marker
truth and held-out cell labels are never supplied to an imputer.

## Literature triage

| Method | Relevant mechanism | Decision in v0 |
|---|---|---|
| CyTOFmerge | Median target vector of the 50 nearest reference cells in shared-marker space | Do not run as a duplicate. The existing `knn`/`knn50` result is the CyTOFmerge core rule and is relabeled as such in comparison tables. |
| cyCombine | One SOM on combined shared-marker data, followed by within-node reference-event sampling plus marker-wise KDE noise | Include. It supplies a distributional rather than same-cell-optimal baseline. Record the fraction of query cells whose SOM node has at least 50 reference cells. |
| CytoVI | Mask-aware VAE whose encoder uses the shared backbone and whose decoder models the marker union | Include through scvi-tools `1.5.0.post1`. Use a complete fit reference panel plus the held-out shared-only panel, without label conditioning. |
| UVAE | Panel-specific autoencoders aligned by a shared-marker `Subspace`; a reference-panel decoder reconstructs the marker union | Include from the official repository at commit `37bf8a587bcfc931f3a704e19bb310be3a662e03`. Do not add cell-type classification, MMD, or label-based resampling constraints. |
| CytoBackBone | Mutual one-to-one nearest-neighbor matches retained only below a distance threshold | Exclude from the primary full-cell comparison. It is a selective matching/abstention method and does not define an imputed value for every held-out cell. It can be assessed later with an explicit coverage-risk endpoint. |
| InfinityFlow/pyInfinityFlow | Supervised extrapolation from backbone markers in a one-marker-per-aliquot screening design | Related but not an exact panel-merging baseline for this paired pseudo-mask design. Its supervised prediction family is already represented by the simple MLP; a tree baseline can be added separately. |

Primary sources:

- CyTOFmerge paper and code: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6792069/>,
  <https://github.com/tabdelaal/CyTOFmerge>
- cyCombine paper, panel-merging vignette, and code:
  <https://www.nature.com/articles/s41467-022-29383-5>,
  <https://biosurf.org/cyCombine_panel_merging.html>,
  <https://github.com/biosurf/cyCombine>
- CytoVI documentation and tutorial:
  <https://docs.scvi-tools.org/en/stable/user_guide/models/cytovi.html>,
  <https://docs.scvi-tools.org/en/stable/tutorials/notebooks/cytometry/CytoVI_advanced_tutorial.html>
- UVAE paper and code:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC12964232/>,
  <https://github.com/mikephn/UVAE>
- CytoBackBone paper and code:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6792066/>,
  <https://github.com/tchitchek-lab/CytoBackBone>
- pyInfinityFlow paper:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10166583/>

## Frozen comparison contract

For each modality, shared panel, fold, and seed:

1. The outer-fit set is the original train plus validation specimens. The
   original patient-grouped test specimens remain held out.
2. At most 5,000 fit cells per patient enter the fit pool. One
   patient-balanced 50,000-cell complete reference bank is shared by all
   literature methods and the existing kNN rule.
3. Fit-only marker medians and IQRs define robust scaling. The same transform is
   applied to held-out shared markers; held-out hidden-marker statistics are
   inaccessible.
4. For tractability and to prevent a roughly 0.9-million-cell query panel from
   overwhelming the 50,000-cell reference bank, deep panel-integration models
   train with a specimen-balanced maximum of 50,000 query cells. All query
   cells are still reconstructed and evaluated. CytoVI and UVAE process at
   most 100,000 training cells per epoch. These caps were fixed during adapter
   validation, before the accepted results were inspected.
5. CytoVI, cyCombine, and UVAE may see all held-out shared-marker values as an
   unlabeled query panel. Therefore their information-access mode is
   `transductive_query_H`. They never see held-out hidden markers or labels.
   Existing kNN/MLP results are inductive and this distinction must remain in
   every comparison table.
6. The primary point endpoints remain per-specimen/per-marker normalized MAE,
   null-relative skill, and Spearman correlation. Distributional endpoints
   include IQR retention and median/q90 errors.
7. Biology is the fixed classifier replay already agreed upon: fit a
   class-balanced multinomial logistic model once on complete outer-fit
   truth, replace only held-out hidden markers with each method's output, and
   measure balanced accuracy, macro-F1/AUPRC, per-class recall, and prevalence
   error.
8. cyCombine nodes with fewer than 50 reference events are true abstentions.
   Coverage is reported. To keep the biology denominator fixed, only those
   missing values receive the outer-fit marker median; this is recorded as a
   fallback and is not presented as native cyCombine coverage.

## Implementations

- Experiment config:
  `configs/archive/2026-07-27_processed_aml_v0/aml_literature_baselines_v0.yaml`
- Common evaluator:
  `src/training/literature_baseline.py`
- Method adapters:
  `src/models/literature_imputation.py`
- cyCombine function-level adapter:
  `scripts/cycombine_panel_impute.R`
- isolated official UVAE runner:
  `scripts/run_uvae_external.py`
- LSF runners:
  `scripts/run_literature_baseline.sh` and
  `scripts/submit_literature_baselines.sh`

The cyCombine adapter copies the behavior of the official
`create_som`/`impute_across_panels` path using `kohonen` `3.0.13`: online
8-by-8 SOM, `rlen=10`, sum-of-squares distance, within-node sampled reference
vectors, density-bandwidth Gaussian noise, and node-range clamping. This
function-level route is used because the package namespace imports unrelated
Bioconductor batch-correction dependencies that the panel-merging function does
not call.

UVAE runs in a fresh process because both its official checkout and this
project use a top-level Python package named `src`. The environment uses
TensorFlow `2.21.0` with its CUDA 12 extra; user site-packages are disabled.
CytoVI runs in its existing isolated environment with scvi-tools
`1.5.0.post1`, PyTorch `2.6.0+cu124`, and AnnData `0.13.2`.

## Execution validation

Accepted H200 pilots were CytoVI job `48230740` and UVAE job `48230793`.
UVAE's TensorFlow/CUDA path was independently checked on the allocated H200 MIG
slice in job `48230751`. The accepted UVAE summary records
`/physical_device:GPU:0`, 50,000 specimen-balanced query training cells,
50,000 complete reference cells, and reconstruction of all 869,534 held-out
query cells.

Earlier provisional GPU runs were excluded and overwritten: the first CytoVI
adapter assumed the pre-merge marker order, and the first UVAE environment did
not expose its allocated GPU libraries. Scheduler `EXIT 255` events caused by
LSF failing to open its own temporary job file were also resubmitted. A run is
accepted only when its `run_summary.json` has status `ok`, the complete
modality/panel/fold/seed grid is present, and the method-specific hardware and
sampling metadata pass validation.

## cyCombine five-fold result

All 20 modality/panel/fold runs at seed 4207 completed successfully. Native
coverage was high but not complete:

| Modality | Shared panel | Mean coverage | Minimum fold coverage |
|---|---:|---:|---:|
| spectral flow | H19 | 0.9850 | 0.9597 |
| spectral flow | clinical10 | 0.9909 | 0.9776 |
| CyTOF | H19 | 0.9941 | 0.9871 |
| CyTOF | clinical10 | 0.9985 | 0.9925 |

Despite that coverage, cyCombine did not beat the fit-median null on same-cell
MAE for any hidden marker in spectral-flow/H19, CyTOF/H19, or
CyTOF/clinical10; it did so for only 1 of 16 spectral-flow/clinical10 hidden
markers. Mean null-relative skill was -0.335, -0.305, -0.554, and -0.545 for
spectral-flow/H19, spectral-flow/clinical10, CyTOF/H19, and CyTOF/clinical10,
respectively.

The output nevertheless retained or inflated distribution width: mean IQR
ratios were 1.31 and 1.46 in spectral flow, and about 2.05 in both CyTOF
panels. This is the expected behavior of a within-node density draw, but it is
not evidence of correct cell-wise coupling.

The fixed full-marker classifier reached the following patient-mean balanced
accuracies:

| Modality | Shared panel | cyCombine | CyTOFmerge-core kNN50 | simple MLP | Full truth |
|---|---:|---:|---:|---:|---:|
| spectral flow | H19 | 0.948 | 0.967 | 0.971 | 0.982 |
| spectral flow | clinical10 | 0.871 | 0.933 | 0.944 | 0.982 |
| CyTOF | H19 | 0.775 | 0.845 | 0.879 | 0.976 |
| CyTOF | clinical10 | 0.741 | 0.820 | 0.827 | 0.976 |

The most visible class failures were CyTOF gamma-delta T-cell recall (0.123
with H19 and 0.248 with clinical10) and monocyte prevalence error (absolute
mean 0.060 and 0.085). Thus high event coverage and plausible marginal marker
distributions did not imply preservation of the fixed biological decision
boundary.

## Full five-fold comparison

All 60 literature-method runs completed: 20 each for cyCombine, CytoVI, and
UVAE. The integrated summary additionally includes the seed-matched median,
CyTOFmerge-core kNN50, and simple-MLP results from the same frozen 20
modality/panel/fold combinations. Point estimates below first average markers
within patient and then average patients; confidence intervals use 2,000
patient bootstrap replicates and are available in the summary artifacts.

The primary null-relative same-cell skill was:

| Modality | Shared panel | kNN50 | simple MLP | cyCombine | CytoVI | UVAE |
|---|---|---:|---:|---:|---:|---:|
| spectral flow | H19 | 0.213 | **0.242** | -0.335 | 0.151 | 0.155 |
| spectral flow | clinical10 | **0.178** | 0.172 | -0.305 | 0.120 | 0.012 |
| CyTOF | H19 | **0.125** | -0.002 | -0.554 | -0.162 | -0.203 |
| CyTOF | clinical10 | **0.116** | -0.055 | -0.545 | -0.173 | -0.311 |

kNN50 and the MLP are inductive fit-only methods. cyCombine, CytoVI, and UVAE
are transductive: they see the full held-out panel of shared-marker values,
although never its hidden markers or labels. The information advantage did
not make any of the literature integration models the best same-cell
predictor.

The number of hidden markers with positive mean null-relative skill was:

| Modality | Shared panel | kNN50 | simple MLP | cyCombine | CytoVI | UVAE |
|---|---|---:|---:|---:|---:|---:|
| spectral flow | H19 | 7/7 | 7/7 | 0/7 | 6/7 | 7/7 |
| spectral flow | clinical10 | 16/16 | 15/16 | 1/16 | 14/16 | 10/16 |
| CyTOF | H19 | 23/25 | 13/25 | 0/25 | 9/25 | 9/25 |
| CyTOF | clinical10 | 30/34 | 16/34 | 0/34 | 13/34 | 13/34 |

For spectral-flow/H19, UVAE recovered all seven markers weakly or better, but
its mean skill still did not exceed the MLP. Ki-67 remained the limiting
marker: CytoVI skill was -0.055 and UVAE was only 0.016. Reducing the shared
set to clinical10 exposed method-specific failures. UVAE's mean skill fell by
0.144, driven especially by CD25 (-0.893) and EOMES (-0.132); CytoVI retained
14/16 positive markers but still failed CD25 and Ki-67.

CyTOF was qualitatively different. kNN50 was positive for 23/25 H19 targets
and 30/34 clinical10 targets, whereas both deep models were negative on
average. Difficult targets were consistent across architectures, including
CD57, CD66b, CD25, CD161, and several checkpoint/chemokine-receptor markers.
CytoVI and UVAE nevertheless retained approximately the observed mean IQR in
CyTOF (ratios near 1.0), while cyCombine inflated it to about 2.1. Thus
plausible distribution width coexisted with poor same-cell coupling.

## Fixed-classifier biology replay

Patient-mean balanced accuracy from the fixed classifier trained only on
complete outer-fit truth was:

| Modality | Shared panel | Median null | kNN50 | simple MLP | cyCombine | CytoVI | UVAE | Full truth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| spectral flow | H19 | 0.957 | 0.967 | **0.971** | 0.948 | 0.970 | 0.969 | 0.982 |
| spectral flow | clinical10 | 0.911 | 0.933 | **0.944** | 0.871 | 0.939 | 0.939 | 0.982 |
| CyTOF | H19 | 0.782 | 0.845 | **0.879** | 0.775 | 0.821 | 0.836 | 0.976 |
| CyTOF | clinical10 | 0.724 | 0.820 | **0.827** | 0.741 | 0.781 | 0.780 | 0.976 |

For coarse spectral-flow cell types, H19 preserved most classifier behavior
and clinical10 cost approximately three balanced-accuracy points for kNN50,
MLP, CytoVI, and UVAE. This does not mean that clinical10 preserved every
marker: UVAE's near-zero aggregate marker skill and reasonable classifier
accuracy show that the classifier depends on only part of the hidden-marker
information.

The rare-population endpoint was more stringent. CyTOF gamma-delta T-cell
recall was:

| Shared panel | kNN50 | simple MLP | cyCombine | CytoVI | UVAE | Full truth |
|---|---:|---:|---:|---:|---:|---:|
| H19 | 0.253 | **0.614** | 0.123 | 0.009 | 0.225 | 0.986 |
| clinical10 | 0.478 | **0.645** | 0.248 | 0.012 | 0.114 | 0.986 |

TCRgd itself had weakly positive mean skill under CytoVI and UVAE, yet the
CytoVI replay recovered only about 1% of gamma-delta T cells. This is direct
evidence that marker-average error, correlation, and distribution retention
are insufficient acceptance criteria for rare-cell reconstruction.

## Interpretation

1. Spectral-flow/H19 pseudo-unobserved markers are recoverable to a useful but
   marker-dependent degree. Neither CytoVI nor UVAE improves on the simple MLP
   overall.
2. clinical10 can be adequate for coarse spectral-flow annotation but is not a
   guaranteed shared set for broad marker recovery. Its effect must be stated
   against a target marker or biological population, not as one global score.
3. For CyTOF, neither H19 nor clinical10 supports reliable full-panel
   same-cell reconstruction or gamma-delta T-cell recovery. A target-specific
   observed anchor or an explicit abstention rule is needed; changing to a
   more complex panel-integration architecture does not solve the missing
   information.
4. Distributional realism and biological replay answer different questions.
   The benchmark should keep null-relative same-cell skill, distribution
   metrics, and fixed-classifier rare-population metrics as separate required
   endpoints.

Final machine-readable artifacts are under
`outputs/aml_literature_baselines_v0/summary/`, including panel-level and
marker-level patient-bootstrap summaries, biology summaries, per-class
metrics, and cyCombine coverage.
