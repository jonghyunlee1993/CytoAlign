#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${1:-configs/experiments/aml_same_cell_recoverability_v0.yaml}
if [[ "$config" != /* ]]; then
  config="$project_dir/$config"
fi
test -f "$config"

queue=${QUEUE:-dbeigpu}
gpu_request=${GPU_REQUEST:-num=1:mig=1/1:mode=shared:gmodel=NVIDIAH200}
cpu_cores=${CPU_CORES:-8}
memory_mb=${MEMORY_MB:-32000}
walltime=${WALLTIME:-12:00}
python_bin=${PYTHON_BIN:-/project/kimlab_tcga/JH_workspace/conda_envs/cytometry/bin/python}
modalities=${MODALITIES:-"spectral_flow cytof"}
panels=${PANELS:-"h19 clinical10"}
folds=${FOLDS:-"0 1 2 3 4"}
seeds=${SEEDS:-"4207"}
dependency=${DEPENDENCY:-}
log_dir="$project_dir/logs"
mkdir -p "$log_dir"

for modality in $modalities; do
  for panel in $panels; do
    for fold in $folds; do
      for seed in $seeds; do
        [[ "$fold" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]]
        job_name="selfrec.${modality}.${panel}.f${fold}.s${seed}"
        command=(
          env
          "OMP_NUM_THREADS=$cpu_cores"
          "MKL_NUM_THREADS=$cpu_cores"
          "OPENBLAS_NUM_THREADS=$cpu_cores"
          "NUMEXPR_NUM_THREADS=$cpu_cores"
          "$python_bin" -m src.wrappers.self_recoverability
          --config "$config"
          --modality "$modality"
          --panel "$panel"
          --fold "$fold"
          --seed "$seed"
        )
        args=(
          -J "$job_name"
          -q "$queue"
          -gpu "$gpu_request"
          -n "$cpu_cores"
          -W "$walltime"
          -R "rusage[mem=${memory_mb}MB]"
          -R "span[hosts=1]"
          -cwd "$project_dir"
          -o "$log_dir/$job_name.%J.out"
          -e "$log_dir/$job_name.%J.err"
        )
        if [[ -n "$dependency" ]]; then
          args+=(-w "$dependency")
        fi
        bsub "${args[@]}" "${command[@]}"
      done
    done
  done
done
