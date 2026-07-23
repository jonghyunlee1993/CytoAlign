# CytoAlign

CytoAlign predicts target-exclusive cytometry markers from a paired,
cell-unpaired source panel. Training uses common-marker optimal transport only
to create soft targets; inference uses source common markers, source-exclusive
markers, and coarse cell type.

## Setup

```bash
python -m pip install -e '.[neural,dev]'
```

Install the PyTorch build compatible with the LPC CUDA runtime when the default
wheel is not appropriate.

## Configuration

Experiment YAML files use a small `extends` key and recursively override
`configs/base.yaml`. No configuration framework is required.

```yaml
extends: ../base.yaml
experiment:
  name: sf_to_cytof
data:
  source_modality: spectral_flow
  target_modality: cytof
```

## Local run

```bash
python -m src.wrappers.train \
  --config configs/experiments/smoke_sf_to_cytof.yaml \
  --fold 0 \
  --seed 4207
```

Production runs write a deployable `model.pkl` and `metrics.json` below
`outputs/<experiment>/fold_<fold>/seed_<seed>/`.

Predict one source specimen with a saved model:

```bash
python -m src.wrappers.predict \
  --model outputs/sf_to_cytof/fold_0/seed_4207/model.pkl \
  --data-root data/AML \
  --specimen R0263_175 \
  --output outputs/predictions/R0263_175.npz
```

## LPC submission

The submission wrapper launches independent fold/seed jobs:

```bash
FOLDS="0 1 2 3 4" SEEDS="4207 4208 4209" \
  QUEUE=kimgpu \
  scripts/submit_train.sh configs/experiments/sf_to_cytof.yaml
```

Queue, GPU request, CPU, memory, walltime, and Python executable can be changed
with `QUEUE`, `GPU_REQUEST`, `CPU_CORES`, `MEMORY_MB`, `WALLTIME`, and
`PYTHON_BIN`.

After all fold/seed jobs finish:

```bash
python -m src.wrappers.summarize --experiment sf_to_cytof
```

The summary reports whether CytoAlign beats Ridge, kNN, direct MLP, and the
H-only OT distiller on mean test population Wasserstein, and whether source
exclusive markers add value over the `ot_hl` control.

## Paired-specimen curve

Fold 0 can evaluate nested paired training sets of size
`0, 1, 2, 4, 8, 16, 32` while fitting the baselines only once per seed:

```bash
QUEUE=dbeigpu \
GPU_REQUEST='num=1:mig=1/1:mode=shared:gmodel=NVIDIAH200' \
FOLDS=0 \
SEEDS='4207 4208 4209' \
scripts/submit_train.sh configs/experiments/sf_to_cytof_paired_curve.yaml
```

Aggregate the three seeds with:

```bash
python -m src.wrappers.summarize_paired_curve \
  --experiment sf_to_cytof_paired_curve
```
