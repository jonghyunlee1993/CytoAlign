#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${1:-configs/experiments/aml_literature_baselines_v0.yaml}
if [[ "$config" != /* ]]; then
  config="$project_dir/$config"
fi
test -f "$config"

methods=${METHODS:-"cytovi uvae cycombine"}
modalities=${MODALITIES:-"spectral_flow cytof"}
panels=${PANELS:-"h19 clinical10"}
folds=${FOLDS:-"0 1 2 3 4"}
seeds=${SEEDS:-"4207"}
gpu_queue=${GPU_QUEUE:-dbeigpu}
cpu_queue=${CPU_QUEUE:-i2c2_normal}
gpu_request=${GPU_REQUEST:-num=1:mig=1/1:mode=shared:gmodel=NVIDIAH200}
cpu_cores=${CPU_CORES:-8}
gpu_cores=${GPU_CORES:-4}
gpu_memory_mb=${GPU_MEMORY_MB:-4000}
cpu_memory_mb=${CPU_MEMORY_MB:-32000}
gpu_walltime=${GPU_WALLTIME:-12:00}
cpu_walltime=${CPU_WALLTIME:-08:00}
force=${FORCE:-0}
log_dir="$project_dir/logs/literature_baselines"
mkdir -p "$log_dir"

for method in $methods; do
  case "$method" in
    cytovi|uvae) resource_type=gpu; job_cores=$gpu_cores ;;
    cycombine) resource_type=cpu; job_cores=$cpu_cores ;;
    *) echo "unsupported method: $method" >&2; exit 2 ;;
  esac
  for modality in $modalities; do
    for panel in $panels; do
      for fold in $folds; do
        for seed in $seeds; do
          [[ "$fold" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]]
          summary="$project_dir/outputs/aml_literature_baselines_v0/$method/$modality/$panel/fold_$fold/seed_$seed/run_summary.json"
          if [[ "$force" != 1 && -s "$summary" ]]; then
            echo "SKIP complete $method $modality $panel fold=$fold seed=$seed"
            continue
          fi
          job_name="litimp.${method}.${modality}.${panel}.f${fold}.s${seed}"
          command=(
            env
            "LITERATURE_CONFIG=$config"
            "OMP_NUM_THREADS=$job_cores"
            "MKL_NUM_THREADS=$job_cores"
            "OPENBLAS_NUM_THREADS=$job_cores"
            "NUMEXPR_NUM_THREADS=$job_cores"
            "$project_dir/scripts/run_literature_baseline.sh"
            "$method" "$modality" "$panel" "$fold" "$seed"
          )
          common=(
            -J "$job_name"
            -n "$job_cores"
            -R "span[hosts=1]"
            -cwd "$project_dir"
            -o "$log_dir/$job_name.%J.out"
            -e "$log_dir/$job_name.%J.err"
          )
          if [[ "$resource_type" == gpu ]]; then
            bsub "${common[@]}" \
              -q "$gpu_queue" \
              -gpu "$gpu_request" \
              -W "$gpu_walltime" \
              -R "rusage[mem=${gpu_memory_mb}MB]" \
              "${command[@]}"
          else
            bsub "${common[@]}" \
              -q "$cpu_queue" \
              -W "$cpu_walltime" \
              -R "rusage[mem=${cpu_memory_mb}MB]" \
              "${command[@]}"
          fi
        done
      done
    done
  done
done
