# GLM-5.3-Flash training on Ascend A5

This recipe starts the 371,822,658-parameter crop through the standard VLM Trainer.
The crop preserves 34 KDA blocks, 11 indexed attention blocks, the additional MTP
block, the 24-block vision tower, four mHC streams and all 288 routed experts.
The official tokenizer and processor are shared with the released model.
[Crop configuration](config_a5.json) · [Z.ai release](https://huggingface.co/zai-org/GLM-5.3-Flash)

Use Transformers 5.22.1 or a compatible release with native `glm5_next` support.
Supply a local processor directory with the tokenizer payload, and a JSONL file
of `messages` records that end in an assistant reply. Image and video blocks
may refer to local files. The preparation command resolves paths relative to
the source JSONL and uses the native HF processor for images, videos and text.

```bash
python examples/training_demo/glm5_next/prepare_replay.py \
  --input /path/to/train.jsonl \
  --processor /path/to/GLM-5.3-Flash \
  --output /path/to/replay.safetensors \
  --documents-per-record 3 \
  --repeat 24
```

For a decoder with four tokens per KPool, independent document boundaries are
kept even if a pool crosses the CP cut. To launch the fixed replay, run from the
repository root. The recipe initializes in FP32, uses BF16 FSDP parameters,
reduces gradients in FP32 and enables FP32 optimizer main parameters.
With two packed records, repeating them 24 times supplies batches for the
default 12 steps at global batch size four.

```bash
HCCL_DETERMINISTIC=true torchrun --standalone --nproc_per_node=1 \
  --module scripts.train_vl examples/training_demo/glm5_next/train_a5.yaml \
  --dataset.data_path=/path/to/replay.safetensors \
  --dataset.model_assets.pretrained_model_name_or_path=/path/to/GLM-5.3-Flash \
  --checkpoint.checkpoint_dir=/path/to/run
```

| Topology | Processes | Trainer overrides |
| --- | --- | --- |
| Single device | 1 | Recipe defaults |
| CP2 | 2 | `--accelerator.cp_size=2 --fsdp_config.dp_shard_size=2` |
| EP2 | 2 | `--accelerator.ep_size=2 --fsdp_config.dp_shard_size=2` |
| CP2 × EP2 | 4 | `--accelerator.cp_size=2 --accelerator.ep_size=2 --fsdp_config.dp_shard_size=4 --fsdp_config.edp_shard_size=2` |

To save a continuation point, set `--checkpoint.save_steps=2` and keep the same
training and scheduler limits in the prepare and restore launches. Restore from
the written step directory with `--checkpoint.restore_from=/path/to/global_step_2`.
Keep `restore_dataloader_state` enabled for a true continuation. The
[validation manifest](../../../hyper_parallel/models/glm5_next/validation.yaml)
describes the fixed four-process matrix. Copy the Trainer YAML with concrete
data and asset paths before running it.

For Online SFT, replace the three dataset targets with
`hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset`,
`hyper_parallel.data.omni.omni_transform.build_auto_processor` and
`hyper_parallel.data.omni.omni_transform.AutoProcessorTransform` respectively.
Point `dataset.data_path` to the JSONL and set
`dataset.model_assets.local_files_only: true`. The same model, media fusion and
target alignment run for both dataset sources.

## Training objectives

- Main LM predicts the next token. The separate MTP layer uses the next
  position's media-fused embedding and predicts two tokens ahead through the
  same vocabulary projection. Label shifting never crosses document boundaries.
- `model.indexer_mode=frozen` uses the published GLM selection path for SFT.
  The discrete top-k operation is also detached in the trainable variants.
  [NeMo GLM5.3-Flash KPool implementation](https://docs.nvidia.com/nemo/automodel/v0.4/nemo-automodel/nemo_automodel/components/models/glm5_next/layers)
- `dense_warmup` freezes the backbone and distills the indexer over all
  visible complete pools. `sparse` uses the selected complete pools during
  joint training. The teacher averages detached per-head attention probability
  mass and sums that mass over each four-token pool. The incomplete tail still
  participates in main attention. Auxiliary supervision starts when at least
  two pools are comparable. This is the pool-level adaptation of the published
  DSA training approach.
  [GLM-5 report](https://arxiv.org/html/2602.15763) · [DeepSeek-V3.2 report](https://arxiv.org/html/2512.02556)

Trainer reduces the three weighted loss numerators over one global valid input
token count per optimizer step. It reports LM labels, MTP labels and supervised
KPool layer-query pairs separately. This permits prompt tokens to train the
indexer and keeps gradients correct when a CP rank has no supervised LM labels.
Router correction bias is loaded as checkpoint state and held fixed in this
recipe.

The recipe selects native absorbed sparse attention for the query-length corner
case. The published weights use FP8 storage. To load a complete local checkpoint
instead of the crop, set `--model.config_path=null` and
`--model.checkpoint_path=/path/to/GLM-5.3-Flash` with the DCP pretrained loader.
The crop validates training and structure at reduced width.
