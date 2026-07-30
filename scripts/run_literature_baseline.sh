#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <method> <modality> <panel> <fold> <seed>" >&2
  exit 2
fi

method=$1
modality=$2
panel=$3
fold=$4
seed=$5
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${LITERATURE_CONFIG:-$project_dir/configs/experiments/aml_literature_baselines_v0.yaml}
case "$method" in
  cytovi)
    python_bin=${CYTOVI_PYTHON:-/project/kimlab_tcga/JH_workspace/conda_envs/cytovi/bin/python}
    ;;
  cycombine|uvae)
    python_bin=${CYTOMETRY_PYTHON:-/project/kimlab_tcga/JH_workspace/conda_envs/cytometry/bin/python}
    ;;
  *)
    echo "unsupported method: $method" >&2
    exit 2
    ;;
esac

if [[ "$method" == uvae ]]; then
  uvae_site=/project/kimlab_tcga/JH_workspace/conda_envs/uvae_baselines/lib/python3.12/site-packages
  cuda_library_path=
  for library_dir in "$uvae_site"/nvidia/*/lib; do
    [[ -d "$library_dir" ]] || continue
    if [[ -z "$cuda_library_path" ]]; then
      cuda_library_path=$library_dir
    else
      cuda_library_path="$cuda_library_path:$library_dir"
    fi
  done
  if [[ -z "$cuda_library_path" ]]; then
    echo "UVAE CUDA wheel libraries are absent under $uvae_site/nvidia" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="$uvae_site/nvidia/cuda_nvcc/bin:$PATH"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

echo "HOST=$(hostname) JOB=${LSB_JOBID:-unset} METHOD=$method CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$project_dir"
exec "$python_bin" -m src.wrappers.literature_baseline \
  --config "$config" \
  --method "$method" \
  --modality "$modality" \
  --panel "$panel" \
  --fold "$fold" \
  --seed "$seed"
