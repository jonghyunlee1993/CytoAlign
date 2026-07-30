# H19 shared-marker ablation screen v0

Archive note: superseded as the active synthesis by `docs/research_record.md`;
retained for detailed provenance.
The disposable `outputs/` tree was removed at the 2026-07-30 session closeout;
the tables below are the durable result record.

Date: 2026-07-27

## Question

Which markers added to a clinical10 backbone make same-cell prediction of the
H19-hidden targets possible, and which H19 markers are actually necessary?
The endpoint is not distribution matching. Marker recovery is evaluated
against the true value of the same cell, and biology is evaluated by replaying
a fixed classifier trained from the original full-panel labels.

This is an exploratory phenomenon-characterization screen, not a search for a
new imputation method or a confirmatory minimal-panel claim.

## Design

- Spectral flow and CyTOF were evaluated separately.
- The evaluation used the same patient-grouped five folds, 83 patients and 93
  specimens as the same-cell recoverability v0 experiment.
- Exact kNN50 was used throughout. The screen used seed 4207 because exact kNN
  is deterministic for a fixed event reservoir and split.
- The marker endpoint was fixed before the sweep:
  `full panel - H19`, namely 7 SF targets and 25 CyTOF targets. Thus every
  candidate is compared on the same targets even when its observed panel is
  smaller than H19.
- The biology endpoint was a class-balanced multinomial classifier trained
  once per modality/fold on true full-panel markers and existing labels. It was
  not refit for individual imputed representations.
- All reported effects are paired within patient; bootstrap resampling uses
  patient as the unit.

The screen had three stages.

1. Add each of the nine `H19 - clinical10` markers separately.
2. Evaluate five targeted pairs per modality, selected from stage 1 rather
   than enumerating all 36 pairs.
3. Evaluate two three-marker compact candidates per modality and selected
   H19-minus-one panels.

There were 30 completed modality-fold runs across the three stages. All valid
runs used an NVIDIA H200 MIG GPU and passed the CUDA probe. One stage-3 CyTOF
job had an LSF dispatch failure before the program started and completed
normally after one exact retry.

## Stage 1: single-marker add-backs

The table reports the change in mean kNN null-relative skill over the fixed
H19-hidden targets. Gap rescue is the fraction of the
`clinical10 -> H19` skill difference recovered by one marker.

| Modality | Added marker | Skill change | 95% CI | H19 gap rescue |
|---|---|---:|---:|---:|
| SF | CCR7 | +0.0261 | +0.0177 to +0.0345 | 40.5% |
| SF | HLA-DR | +0.0093 | +0.0056 to +0.0142 | 14.5% |
| SF | CD117 | +0.0092 | approximately 0 to +0.0172 | 14.3% |
| SF | CD27 | +0.0091 | +0.0073 to +0.0111 | 14.2% |
| SF | CD45RA | +0.0082 | +0.0032 to +0.0126 | 12.7% |
| SF | PD-1 | +0.0076 | +0.0050 to +0.0104 | 11.8% |
| SF | CD25 | +0.0072 | +0.0032 to +0.0117 | 11.2% |
| SF | CD123 | +0.0064 | +0.0036 to +0.0096 | 10.0% |
| SF | CD38 | +0.0018 | -0.0022 to +0.0053 | 2.7% |
| CyTOF | CD45RA | +0.0107 | +0.0063 to +0.0158 | 55.2% |
| CyTOF | HLA-DR | +0.0064 | +0.0033 to +0.0104 | 32.8% |
| CyTOF | CD38 | +0.0050 | +0.0023 to +0.0070 | 25.7% |
| CyTOF | CCR7 | +0.0049 | +0.0028 to +0.0068 | 25.3% |
| CyTOF | CD123 | +0.0037 | +0.0010 to +0.0068 | 19.3% |

CyTOF CD117 was neutral, while CD27, PD-1 and CD25 had small negative global
effects whose intervals were not used as confirmatory exclusion tests. The
marker-to-target links were biologically structured rather than uniform:
CCR7 most improved SF CTLA-4, CD45RA improved SF EOMES and CyTOF CD45RO,
HLA-DR improved SF T-bet and CyTOF IgD, CD38 improved CyTOF CD47/TCRgd, and
CD123 improved CyTOF CD274.

## Stage 2: targeted pairs

The best pairs were mostly additive or mildly redundant; there was no strong
positive pair interaction that justified an exhaustive combinatorial sweep.

| Modality | Pair | Skill change | H19 gap rescue | Biology BA change | Biology F1 change |
|---|---|---:|---:|---:|---:|
| SF | CCR7 + CD123 | +0.0344 | 53.3% | +0.0168 | +0.0151 |
| SF | CCR7 + HLA-DR | +0.0334 | 51.8% | — | +0.0065 |
| SF | CCR7 + CD45RA | +0.0323 | 50.0% | — | +0.0069 |
| SF | CD117 + CD123 | +0.0158 | 24.5% | +0.0261 | +0.0332 |
| CyTOF | CD45RA + HLA-DR | +0.0177 | 91.2% | — | +0.0104 |
| CyTOF | CD45RA + CD38 | +0.0149 | 76.8% | +0.0177 | +0.0049 |
| CyTOF | CD45RA + CD123 | +0.0137 | 70.6% | +0.0214 | +0.0089 |
| CyTOF | HLA-DR + CD38 | +0.0110 | 56.9% | +0.0192 | +0.0116 |
| CyTOF | CD38 + CD123 | +0.0081 | 41.8% | +0.0275 | +0.0058 |

Here biology changes are relative to clinical10 using the fixed full-panel
classifier and kNN hybrid. A dash means that the metric was not the reason to
select that pair; the machine-readable table contains all endpoints.

## Stage 3: compact candidates

Only three markers were added to clinical10, giving 13-marker observed panels.

| Modality | Added markers | Skill | Skill gap rescue | Skill vs H19 | BA | Macro-F1 | Macro-AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| SF | CCR7 + CD117 + CD123 | 0.1890 | 63.6% | -0.0235 | 0.9593 | 0.9103 | 0.9701 |
| SF | CCR7 + CD45RA + CD123 | 0.1876 | 61.5% | -0.0249 | 0.9518 | 0.8921 | 0.9590 |
| SF | H19 | 0.2125 | 100% | reference | 0.9667 | 0.9100 | 0.9750 |
| CyTOF | CD45RA + CD38 + CD123 | 0.1234 | 91.9% | -0.0016 | 0.8527 | 0.8035 | 0.8595 |
| CyTOF | CD45RA + HLA-DR + CD38 | 0.1268 | 109.1% | +0.0018 | 0.8441 | 0.8112 | 0.8584 |
| CyTOF | H19 | 0.1250 | 100% | reference | 0.8448 | 0.7857 | 0.8449 |

For SF, `clinical10 + CCR7 + CD117 + CD123` is the stronger of the two tested
compact panels. It recovers about 64% of the H19 hidden-target skill gap while
its fixed-classifier macro-F1 is indistinguishable from H19
(`+0.0003` versus H19; 95% CI `-0.0046 to +0.0052`). It does not reproduce all
quantitative marker information: skill and balanced accuracy remain below
H19.

For CyTOF, both compact panels are statistically indistinguishable from H19 in
global null-relative skill. `CD45RA + HLA-DR + CD38` also matches H19 in
Spearman correlation. Both compact panels exceed H19 on some classifier
metrics. This is not evidence that imputation is better than measuring the
markers. It shows that equal-weight kNN is non-monotonic in the observed
feature set: extra, irrelevant or noisy distance dimensions can worsen local
neighbour selection.

## H19-minus-one necessity test

The skill effect is `H19-minus-marker - H19` on the fixed H19-hidden targets.
Negative values mean that removing the marker damaged target recovery.
Biology effects use the same direction.

| Modality | Removed marker | Skill effect (95% CI) | BA effect | F1 effect | AUPRC effect |
|---|---|---:|---:|---:|---:|
| SF | CCR7 | -0.0150 (-0.0196, -0.0104) | -0.0002 | +0.0007 | 0.0000 |
| SF | CD117 | -0.0052 (-0.0088, -0.0019) | -0.0121 | -0.0181 | -0.0128 |
| SF | CD123 | -0.0066 (-0.0130, -0.0017) | -0.0098 | -0.0060 | -0.0064 |
| SF | CD45RA | -0.0021 (-0.0049, +0.0015) | -0.0019 | +0.0026 | -0.0012 |
| SF | HLA-DR | -0.0043 (-0.0064, -0.0025) | -0.0034 | -0.0051 | -0.0025 |
| CyTOF | CD45RA | -0.0049 (-0.0076, -0.0027) | -0.0082 | -0.0014 | -0.0065 |
| CyTOF | HLA-DR | -0.0044 (-0.0069, -0.0026) | -0.0082 | -0.0104 | -0.0076 |
| CyTOF | CD38 | -0.0038 (-0.0049, -0.0028) | -0.0169 | +0.0023 | -0.0034 |
| CyTOF | CD123 | -0.0024 (-0.0041, -0.0011) | -0.0146 | -0.0002 | -0.0086 |
| CyTOF | CD27 | +0.0010 (-0.0002, +0.0023) | +0.0069 | +0.0116 | +0.0048 |
| CyTOF | PD-1 | +0.0028 (+0.0018, +0.0038) | +0.0041 | +0.0068 | +0.0164 |
| CyTOF | CD25 | +0.0027 (+0.0004, +0.0053) | +0.0024 | +0.0015 | -0.0005 |

The results separate two different notions of necessity.

- SF CCR7 is most necessary for predicting the seven H19-hidden targets,
  especially CTLA-4, but removing it barely changes the coarse label replay.
- SF CD117 and CD123 are more important for classifier biology than their
  average target-skill effects suggest.
- CyTOF CD45RA, HLA-DR, CD38 and CD123 each carry distinct target information:
  their largest losses are CD45RO/CD294, IgD/CD64, CD47/TCRgd and
  CD274/CD15, respectively.
- CyTOF CD27, PD-1 and CD25 are not helpful as equal-weight kNN distance
  coordinates in this screen. This does not make the proteins biologically
  unimportant, nor does it justify imputing them when they are themselves an
  endpoint.

## Rare-cell replay

CyTOF T-cell gd remains the most stringent rare-cell test because its defining
TCRgd marker is hidden.

| Observed panel | Recall | Precision | AUPRC |
|---|---:|---:|---:|
| True full markers | 0.986 | 0.942 | 0.994 |
| clinical10 kNN hybrid | 0.478 | 0.800 | 0.683 |
| clinical10 + CD45RA + CD38 + CD123 | 0.449 | 0.847 | 0.683 |
| clinical10 + CD45RA + HLA-DR + CD38 | 0.446 | 0.847 | 0.678 |
| H19 kNN hybrid | 0.253 | 0.865 | 0.564 |

The compact panels improve T-cell-gd recall and AUPRC substantially relative
to H19, but do not restore the true full-panel result. Clinical10 has the
highest recall among these kNN panels, at lower precision, and no tested panel
recovers the population faithfully. If T-cell-gd is a required endpoint,
TCRgd should remain directly measured.

The rare-cell result reinforces the non-monotonicity finding: a larger shared
panel can make a fixed unweighted kNN worse. The result should drive panel
choice and model diagnostics, not be reinterpreted as proof that less
measurement is intrinsically better.

## Practical conclusion

There is no single endpoint-free “minimum sufficient shared set.”

- For SF hidden functional-marker recovery, CCR7 is the primary information
  coordinate, while CD117 and CD123 are needed to preserve the tested
  classifier biology. The provisional compact set is
  `clinical10 + CCR7 + CD117 + CD123`.
- For CyTOF, CD45RA and CD38 form the strongest tested core. HLA-DR is the
  better third marker for global skill and macro-F1; CD123 is useful for
  balanced accuracy and CD274-related biology. Both 13-marker candidates
  approximate H19 better than expected.
- A marker should be guaranteed in the shared set when it either defines the
  downstream population directly or has a reproducible leave-one-out loss for
  the prespecified target. Average imputation skill alone is insufficient.

These are shortlist rules for the current processed, upstream-pregated AML
data. They require confirmation on raw technical-QC-only events. Because
stages 2 and 3 were adaptively selected using the same dataset, their
bootstrap intervals are descriptive rather than independent confirmatory
evidence.

## Artifacts

- Single add-backs:
  `outputs/aml_h19_addback_screen_v0/summary/`
- Targeted pairs:
  `outputs/aml_h19_targeted_pairs_v0/summary/`
- Compact panels and H19 removals:
  `outputs/aml_h19_compact_and_removal_v0/summary/`
- Configurations:
  `configs/archive/2026-07-27_processed_aml_v0/aml_h19_addback_screen_v0.yaml`,
  `configs/archive/2026-07-27_processed_aml_v0/aml_h19_targeted_pairs_v0.yaml`,
  `configs/archive/2026-07-27_processed_aml_v0/aml_h19_compact_and_removal_v0.yaml`
