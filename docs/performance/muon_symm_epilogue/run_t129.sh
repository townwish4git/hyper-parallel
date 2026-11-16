#!/usr/bin/env bash
set -eo pipefail
# shellcheck source=/dev/null
source /home/cann_custom_path/8.5.0/cann-8.5.0/set_env.sh
set -u
root=/home/townwish/workspace/projects/hp-epilogue-20261116
mkdir -p "$root"
tar -xzf /home/townwish/workspace/projects/hp-epilogue-20261116.tar.gz -C "$root" 2>/dev/null
cd "$root"
mkdir -p validation
/home/miniconda3/envs/twx_pt29/bin/python -m venv --system-site-packages /home/townwish/workspace/projects/hp-epilogue-env-20261116
python=/home/townwish/workspace/projects/hp-epilogue-env-20261116/bin/python
"$python" -m pip install -e . --no-deps > validation/build.log 2>&1
export CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1
"$python" -m pytest -q tests/ut/core/optimizer/test_muon.py tests/ut/core/optimizer/test_muon_symm_gemm.py tests/ut/components/functional/test_symmetric_matmul.py tests/ut/test_setup_ascendc.py > validation/ut.log 2>&1
ASCEND_RT_VISIBLE_DEVICES=0 "$python" tests/torch/functional/symmetric_matmul_smoke.py --expected-root "$root" --output validation/smoke.json > validation/smoke.log 2>&1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for model in qwen3-30b-a3b qwen3.5-27b; do
  npu-smi info > "validation/${model}-before.txt"
  timeout 2400 "$python" -m torch.distributed.run --standalone --nproc-per-node=8 -m tests.torch.optimizer.benchmark_muon_models \
    --model "$model" --include-epilogue --warmup 2 --steps 5 --cycles 2 --accuracy-steps 1 5 --metric-threads 16 \
    --output "validation/${model}.json" > "validation/${model}.log" 2>&1
  date -Is >> validation/completed.txt
  echo "$model" >> validation/completed.txt
done
