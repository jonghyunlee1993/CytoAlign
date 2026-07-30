#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <modality> <panel> <fold> <seed>" >&2
  exit 2
fi

modality=$1
panel=$2
fold=$3
seed=$4
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/project/kimlab_tcga/JH_workspace/conda_envs/cytometry/bin/python}
config=${SELFREC_CONFIG:-$project_dir/configs/experiments/aml_same_cell_recoverability_v0.yaml}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

host=$(hostname)
scheduler_gpu=""
if [[ $host == "jawn" && -n ${LSB_JOBID:-} ]]; then
  # LSF remaps an allocated device to logical CUDA device 0.  Read the
  # scheduler's external message to recover the physical jawn GPU number.
  for _ in 1 2 3 4 5; do
    scheduler_gpu=$(
      bjobs -l "$LSB_JOBID" 2>/dev/null |
        grep -o 'jawn:gpus=[0-9]\+' |
        head -n 1 |
        cut -d= -f2 || true
    )
    [[ -n $scheduler_gpu ]] && break
    sleep 1
  done
fi

echo "HOST=$host JOB=${LSB_JOBID:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} PHYSICAL_GPU=${scheduler_gpu:-unknown}"
if [[ $host == "jawn" && $scheduler_gpu == "5" ]]; then
  echo "REJECTED_KNOWN_BAD_PHYSICAL_GPU=5" >&2
  exit 75
fi

cd "$project_dir"
exec "$python_bin" -m src.wrappers.self_recoverability \
  --config "$config" \
  --modality "$modality" \
  --panel "$panel" \
  --fold "$fold" \
  --seed "$seed"
