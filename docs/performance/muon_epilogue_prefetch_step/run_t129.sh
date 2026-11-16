#!/usr/bin/env bash
set -eo pipefail
# shellcheck source=/dev/null
source /home/cann_custom_path/8.5.0/cann-8.5.0/set_env.sh
set -u
root=/home/townwish/workspace/projects/hp-prefetch-step-20261116
base=/home/townwish/workspace/projects/hp-epilogue-20261116
envdir=/home/townwish/workspace/projects/hp-prefetch-step-env-20261116
base_python=/home/townwish/workspace/projects/hp-epilogue-env-20261116/bin/python
mkdir -p "$root"
tar -xf /home/townwish/workspace/projects/hp-prefetch-step-20261116.tar -C "$root" 2>/dev/null
cd "$root"
mkdir -p validation
/home/miniconda3/envs/twx_pt29/bin/python -m venv --system-site-packages "$envdir"
python="$envdir/bin/python"
"$python" -m pip install -e . --no-deps > validation/build.log 2>&1
cp tests/torch/optimizer/benchmark_muon_models.py "$base/tests/torch/optimizer/benchmark_muon_models.py"
export CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1
"$python" -m pytest -q tests/ut/core/optimizer/test_muon.py tests/ut/core/optimizer/test_muon_symm_gemm.py \
  tests/ut/components/functional/test_symmetric_matmul.py tests/ut/test_setup_ascendc.py > validation/ut.log 2>&1
ASCEND_RT_VISIBLE_DEVICES=1 "$python" tests/torch/functional/symmetric_matmul_smoke.py \
  --expected-root "$root" --output validation/smoke.json > validation/smoke.log 2>&1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for model in qwen3-30b-a3b qwen3.5-27b; do
  for run in 1 2 3 4; do
    flavor=baseline
    workdir="$base"
    executable="$base_python"
    if [[ "$run" == 2 || "$run" == 3 ]]; then
      flavor=prefetch
      workdir="$root"
      executable="$python"
    fi
    fingerprints=()
    if [[ "$run" == 1 || "$run" == 2 ]]; then
      fingerprints=(--fingerprint-steps 1 5)
    fi
    prefix="$root/validation/${model}-${run}-${flavor}"
    npu-smi info > "${prefix}-before.txt"
    cd "$workdir"
    timeout 2400 "$executable" -m torch.distributed.run --standalone --nproc-per-node=8 \
      -m tests.torch.optimizer.benchmark_muon_models --model "$model" --include-epilogue \
      --warmup 2 --steps 5 --cycles 1 --accuracy-steps "${fingerprints[@]}" \
      --output "${prefix}.json" > "${prefix}.log" 2>&1
    date -Is >> "$root/validation/completed.txt"
    echo "$model $run $flavor" >> "$root/validation/completed.txt"
  done
done
