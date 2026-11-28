# Qwen3.8-Flash-Next 并行训练

[四卡训练配置](../../hyper_parallel/models/qwen4_exp/recipes/train_crop.yaml) 使用 Ascend HCCL，默认 FSDP4、CP2、EP2、TP1、PP1。全局 batch 为 4，每个数据并行副本累积两个单样本 micro-batch。使用 FP32 初始化及优化器主权重、BF16 前向参数与 FP32 梯度规约。

模型以官方 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa) 配置为基准，骨干使用 Transformers 5.24.1 的 `qwen4_exp`，额外构造训练用的单层 MTP。MTP 读取最终门控残差混合前的四条流，与下一个 token 的共享嵌入融合，使用同一 LM head 监督再下一个 token。默认 MTP loss 权重为 0.1。融合结构与发布权重的参数命名一致。 [vLLM MTP 实现](https://docs.vllm.ai/en/v0.31.0/api/vllm/models/qwen4_exp/amd/mtp/)

## 启动

在已有匹配版本 PyTorch、CANN 和 torch_npu 的环境中，从仓库根目录运行。选取四张空闲卡，并提供本地官方模型配置目录和 PNG 或 JPEG 图片。裁剪采用随机初始化。

```bash
python -m pip install -e . 'transformers==5.24.1'
GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://huggingface.co/Qwen/Qwen3.8-Flash-Next
GIT_LFS_SKIP_SMUDGE=1 git -C Qwen3.8-Flash-Next fetch --depth=1 origin \
  812d96e10a0dd82b69d749e68693698cf6c36daa
GIT_LFS_SKIP_SMUDGE=1 git -C Qwen3.8-Flash-Next checkout --detach FETCH_HEAD
git -C Qwen3.8-Flash-Next lfs pull --include=tokenizer.json --exclude=""
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash examples/qwen3_8_flash_next/launch_crop.sh \
  ./Qwen3.8-Flash-Next ./image.png
```

启动脚本读取本地 `config.json` 、`preprocessor_config.json` 和 tokenizer 文件。验证集复用一张实际经过图像处理器的图片及一段固定目标文本，样本长度为 16 且无填充，默认训练 12 个 optimizer step。可用 `--training.train_iters=40` 延长观察窗口。Trainer 以 runtime 模式记录输入所有权、视觉梯度、参数探针和 loss。此配置适用于短样本功能与过拟合验证。

## 实机验证

4 张 Ascend 950DT 使用 FSDP4、EP2、CP2 运行最终配置，12 个 optimizer step 均完成。图文样本过拟合 loss 逐步从 6.830916 降至 5.761545，下降 15.65%。各步梯度范数有限，视觉 patch embedding 和投影梯度检查通过。优化器布局证据确认 FP32 主权重和主梯度。

同权重 FP32 对照中，CP2+EP2 与单进程的 logits、loss 及抽查梯度对齐。FP32 和 BF16 的模块对照覆盖 GDN 替换、原生多模态骨干和 MTP。裁剪权重通过本地保存与重新加载，全部 1,531 个状态字典键逐元素一致。

## 裁剪和并行覆盖

| 模块 | 保留的结构 | 裁剪或并行布局 |
| --- | --- | --- |
| 文本骨干 | 48 层、原层序、36 层 GDN 和 12 层 QSA | hidden 2560 → 64。QSA head 24 → 4，KV head 保持 2。采用连续序列分片 |
| GDN | 卷积宽度 4、value/key head 比例 3 | K/V heads 16/48 → 2/6，head dim 128 → 16。用 Ulysses 交换序列与 head，卷积和递推在完整序列、局部 head 上运行 |
| QSA | 压缩率 4、4 个索引查询头、1 个索引 KV 头 | head dim 256 → 64、索引 head dim 128 → 32。验证预算 2048 → 8，使 16 token 样本执行实际稀疏筛选。保留索引历史和局部尾块 |
| 门控残差、RoPE | 4 条残差流，文本多轴 RoPE 与视觉轴向 RoPE | 门控低秩宽度 320 → 8。缩小文本 RoPE 各维分段，保留轴顺序 |
| N-gram PLE | 第 2 层注入、3-gram、每 n-gram 8 个头、卷积核 4、表对齐 128 | 嵌入宽度 2560 → 64，表大小基数 20000000 → 127。保留哈希、EOS 边界和膨胀卷积，CP 聚合历史 |
| MoE | softmax top-10、共享专家、sigmoid 门 | 512 → 16 个专家、每 EP rank 8 个。专家宽度 640 → 32。路由共享，专家 EP2，专家数据并行分片度 2 |
| 视觉塔和投影 | 27 个块、16×16 patch、2 帧 temporal patch、2×2 merge | hidden 1152 → 64、MLP 4304 → 128、heads 16 → 4。FSDP4，CP 组内复制视觉计算和融合 |
| 词表、MTP 和 loss | 共享 token embedding 与 LM head、单层 QSA+MoE MTP | 词表 248320 → 512，重映射模态和 EOS 标记。骨干与 MTP 使用相同 EP 策略，两个 loss 使用原始全序列标签 |

上下文并行在视觉融合后切分文本序列，GDN 使用仓库内的 Torch Ulysses 实现。QSA 与 PLE 聚合历史后返回本地输出。残差混合后聚合输出供原生 loss 使用。当前实现使用可微 all-gather 和原生算子处理边界，适合验证短序列的语义和组合并行。QSA 的骨干参数通过所选掩码收到梯度。离散 top-k 选择沿用 HF 行为。这一训练配置适用使用现有索引器做微调，论文中的稠密蒸馏和稀疏索引器 KL 训练属于单独阶段。 [Qwen 技术报告](https://arxiv.org/html/2608.30320v1#S2.SS1)

[模型验收 manifest](../../hyper_parallel/models/qwen4_exp/validation.yaml) 可用于本地 inventory、结构检查和模块精度对照。执行前填写模型配置与训练数据的本地路径。裁剪验收包含合成文本监督、图片分支、GDN、稀疏 QSA、PLE、门控残差、共享及路由专家和 MTP。全量权重读写、视频数据和长序列性能需分别验证。
