#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=/dev/null
. /home/cann_custom_path/8.5.0/cann-8.5.0/set_env.sh
cd /home/townwish/workspace/projects/hp-symm-only-20261113
export CLOSE_MATMUL_K_SHIFT=1
export OMP_NUM_THREADS=1
python=/home/townwish/workspace/projects/hp-symm-only-env/bin/python
"$python" -m pytest -q tests/ut/core/optimizer/test_muon.py tests/ut/core/optimizer/test_muon_symm_gemm.py tests/ut/components/functional/test_symmetric_matmul.py tests/ut/test_setup_ascendc.py > validation/ut-final.log 2>&1
ASCEND_RT_VISIBLE_DEVICES=0 "$python" tests/torch/functional/symmetric_matmul_smoke.py --expected-root "$PWD" --output validation/editable-smoke-final.json > validation/editable-smoke-final.log 2>&1
sha256sum hyper_parallel/core/optimizer/muon.py hyper_parallel/components/functional/symmetric_matmul.py hyper_parallel/components/functional/_ascendc/symmetric_matmul/*.so hyper_parallel/components/functional/_ascendc/symmetric_matmul/*.h hyper_parallel/components/functional/_ascendc/symmetric_matmul/*.cpp hyper_parallel/components/functional/_ascendc/symmetric_matmul/launch.py tests/torch/optimizer/benchmark_muon_models.py tests/torch/optimizer/muon_model_inputs.py tests/torch/optimizer/muon_step_metrics.py > validation/source-hashes-before.txt
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for model in qwen3-30b-a3b qwen3.5-27b; do
    npu-smi info > "validation/${model}-before.txt"
    timeout 1800 "$python" -m torch.distributed.run --standalone --nproc-per-node=8 \
        -m tests.torch.optimizer.benchmark_muon_models --model "$model" \
        --warmup 2 --steps 5 --cycles 2 --accuracy-steps 1 5 --metric-threads 16 \
        --output "validation/${model}.json" > "validation/${model}.log" 2>&1
    npu-smi info > "validation/${model}-after.txt"
    date -Is >> validation/completed.txt
    echo "$model" >> validation/completed.txt
done
sha256sum hyper_parallel/core/optimizer/muon.py hyper_parallel/components/functional/symmetric_matmul.py hyper_parallel/components/functional/_ascendc/symmetric_matmul/*.so hyper_parallel/components/functional/_ascendc/symmetric_matmul/*.h hyper_parallel/components/functional/_ascendc/symmetric_matmul/*.cpp hyper_parallel/components/functional/_ascendc/symmetric_matmul/launch.py tests/torch/optimizer/benchmark_muon_models.py tests/torch/optimizer/muon_model_inputs.py tests/torch/optimizer/muon_step_metrics.py > validation/source-hashes-after.txt
