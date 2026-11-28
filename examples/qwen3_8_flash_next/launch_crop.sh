#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 LOCAL_FLASH_NEXT_SOURCE IMAGE_FILE [Trainer overrides]" >&2
    exit 2
fi
model_dir="$(cd "$1" && pwd)"
image_path="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
shift 2
if [[ ! -f "$model_dir/config.json" || ! -f "$model_dir/preprocessor_config.json" || ! -f "$model_dir/tokenizer.json" || ! -f "$image_path" ]]; then
    echo "The local Flash-Next configuration, image processor configuration, tokenizer and image file are required." >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
exec python -m torch.distributed.run --standalone --nproc-per-node=4 --module scripts.train_vl \
    hyper_parallel/models/qwen4_exp/recipes/train_crop.yaml \
    --model.config_path="$model_dir" \
    --dataset.model_assets.config_path="$model_dir" \
    --dataset.image_path="$image_path" \
    "$@"
