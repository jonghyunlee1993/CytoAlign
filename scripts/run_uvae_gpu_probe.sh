#!/usr/bin/env bash
set -euo pipefail

uvae_env=/project/kimlab_tcga/JH_workspace/conda_envs/uvae_baselines
uvae_site="$uvae_env/lib/python3.12/site-packages"
cuda_library_path=
for library_dir in "$uvae_site"/nvidia/*/lib; do
  [[ -d "$library_dir" ]] || continue
  if [[ -z "$cuda_library_path" ]]; then
    cuda_library_path=$library_dir
  else
    cuda_library_path="$cuda_library_path:$library_dir"
  fi
done
test -n "$cuda_library_path"
export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$uvae_site/nvidia/cuda_nvcc/bin:$PATH"
export PYTHONNOUSERSITE=1

"$uvae_env/bin/python" - <<'PY'
import json
import os

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("TensorFlow did not detect the allocated GPU")
with tf.device("/GPU:0"):
    left = tf.random.normal((2048, 2048), seed=7)
    right = tf.random.normal((2048, 2048), seed=11)
    result = tf.linalg.matmul(left, right)
    checksum = float(tf.reduce_sum(result).numpy())
print(
    json.dumps(
        {
            "status": "ok",
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": [device.name for device in gpus],
            "checksum": checksum,
        },
        sort_keys=True,
    )
)
PY
