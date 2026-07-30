#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <modality> <fold> <seed> [config]" >&2
  exit 2
fi

modality=$1
fold=$2
seed=$3
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/project/kimlab_tcga/JH_workspace/conda_envs/cytometry/bin/python}
config=${4:-${ADDBACK_CONFIG:-$project_dir/configs/experiments/aml_h19_addback_screen_v0.yaml}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

echo "HOST=$(hostname) JOB=${LSB_JOBID:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$project_dir"
exec "$python_bin" -m src.wrappers.addback_sweep \
  --config "$config" \
  --modality "$modality" \
  --fold "$fold" \
  --seed "$seed"
