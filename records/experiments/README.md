# Experiment record

This directory is the compact, Git-tracked record of experiments completed
before the target-conditioned adaptive-kNN direction. Generated `outputs/`,
`logs/`, and ignored `artifacts/` were removed after the records below were
copied and verified.

## Provenance

- Direction: spectral flow to CyTOF
- Primary evaluation: patient-grouped fold 0
- Seeds: 4207, 4208, 4209 unless explicitly noted
- Primary distribution metric: patient-first normalized Wasserstein distance
  (lower is better)
- Source commits:
  - `c607a4b`: kNN-baseline residual experiment
  - `40c83be`: marker-adaptive paired curves
  - `058ceb9`: specimen-agnostic strict-unpaired selection
  - `933ce14`: label-free realistic-panel experiments
- The experiment YAML files remain under `configs/experiments/`.

## Legacy prototypes

Compact aggregate JSON is retained under `legacy_artifacts/`.

| Experiment | Recorded decision |
|---|---|
| `pseudopanel_50k` | Initial exact-truth SAE gate; latent explained variance 0.935 and oracle relative MAE gain 6.46% |
| `prototype_decur/final_gate` | Stop DeCUR; pre-registered gate failed |
| `ot_pseudopair` | Retain soft-OT teacher pilot; reject hard argmax pseudo-pairs |
| `ot_new_sample` | Retain paired OT candidate; stop pooled-unpaired OT distillation |
| `ot_decur_new_sample` | Retain paired OT-DeCUR only as an ablation; stop pooled OT-DeCUR |
| `direct_ot_decur_aux` | Directional but immaterial auxiliary signal: relative exact-MAE gain 0.09%, bootstrap CI crossed zero |

## Fold-0 experiment progression

The complete aggregate JSON is retained under `fold0_summaries/`.

| Experiment | Main result |
|---|---|
| Ridge paired-count curve | Best count 16: 1.087505; plain kNN was already much stronger at 0.966242 |
| kNN residual curve | Count 32: 0.964044, a 0.002197 gain over kNN |
| One-seed matched go/no-go | Marker gate: 0.956726 versus kNN 0.969859 |
| One-seed shuffled go/no-go | Marker gate: 0.967310; matched-pair advantage mostly disappeared |
| One-seed pooled-unpaired go/no-go | Marker gate: 0.967403; matched-pair advantage mostly disappeared |
| Label-assisted paired marker gate | Count 32: 0.955849, gain 0.010393, count-response R² 0.886 |
| Strict-unpaired marker gate | Count 32: 0.966334, no gain, count-response R² 0.005 |

The label-assisted experiments used the stored coarse cell-type annotations in
kNN conditioning, OT blocks, residual features, and validation selection. The
later label-free experiments removed annotations from all four roles and used
them only for final stratified reporting.

## Label-free realistic-panel experiment

All methods used the same 25 target-exclusive endpoints.

| Common markers | kNN | Paired gate, count 32 | Gain | Strict-unpaired gate, count 32 | Gain |
|---:|---:|---:|---:|---:|---:|
| 19 | 0.965553 | 0.963647 | +0.001906 | 0.967853 | -0.002300 |
| 15 | 0.964198 | 0.963556 | +0.000642 | 0.979724 | -0.015526 |
| 12 | 0.989344 | 0.992057 | -0.002714 | 0.995028 | -0.005684 |
| 8 | 1.051058 | 1.051391 | -0.000333 | 1.053479 | -0.002421 |

There was no meaningful paired-count dose response after labels were removed.
Global residual selection chose alpha zero in nearly every run. Marker-wise
selection produced only a small paired gain and overfit badly in strict-unpaired
settings. This stopped the residual-correction direction.

The panel result motivates the next direction: learn the neighbor metric itself.
Removing four activation markers preserved marginal Wasserstein, while removing
the `CD20`, `CD123`, and `CD45RA` group caused the first clear degradation.

## LSF jobs

- Initial paired curves: `48175176-48175178`
- kNN residual curves: `48179983-48179985`
- matched/shuffled/pooled go/no-go: `48183273`, `48183274`, `48183630`
- label-assisted paired marker gate: `48186129-48186131`
- preliminary and corrected strict-unpaired runs:
  `48186132-48186134`, `48186281-48186283`
- label-free 19/15/12/8 panel grid: `48186708-48186731`

The final 24-job panel grid completed successfully on `dbeigpu`, host
`dbei-ai1`, using H200 MIG 1/1, four CPUs, 12 GB requested host RAM, and a
four-hour limit. All stderr files were empty, CUDA matrix multiplication was
verified by the application, maximum observed host memory was 5.894 GB, and
runtime ranged from 1,427 to 2,842 seconds.

