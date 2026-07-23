#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${1:?usage: scripts/submit_train.sh <config.yaml>}
if [[ "$config" != /* ]]; then
  config="$project_dir/$config"
fi
test -f "$config"

queue=${QUEUE:?set QUEUE after checking current LPC GPU inventory}
gpu_request=${GPU_REQUEST:-num=1:mode=shared}
cpu_cores=${CPU_CORES:-4}
memory_mb=${MEMORY_MB:-32000}
walltime=${WALLTIME:-24:00}
python_bin=${PYTHON_BIN:-/project/kimlab_tcga/JH_workspace/conda_envs/cytometry/bin/python}
folds=${FOLDS:-"0 1 2 3 4"}
seeds=${SEEDS:-"4207 4208 4209"}
experiment=$(basename "$config" .yaml)
case "$experiment" in
  ""|*[!A-Za-z0-9_.-]*) echo "invalid experiment name" >&2; exit 2 ;;
esac
log_dir="$project_dir/logs"
mkdir -p "$log_dir"

for fold in $folds; do
  for seed in $seeds; do
    [[ "$fold" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]]
    job_name="${experiment}.f${fold}.s${seed}"
    command=(
      "$python_bin" -m src.wrappers.train
      --config "$config"
      --fold "$fold"
      --seed "$seed"
    )
    bsub \
      -J "$job_name" \
      -q "$queue" \
      -gpu "$gpu_request" \
      -n "$cpu_cores" \
      -W "$walltime" \
      -R "rusage[mem=${memory_mb}MB]" \
      -R "span[hosts=1]" \
      -cwd "$project_dir" \
      -o "$log_dir/$job_name.%J.out" \
      -e "$log_dir/$job_name.%J.err" \
      "${command[@]}"
  done
done
