# FP32 epilogue 加载顺序：完整 Muon step 实验

实验分支：`experiment/muon-epilogue-prefetch`，基于 `4884e67c`。
原 `feat/muon-symm-epilogue` 分支保持不变，未引入 Gram。

## 实现

将 FP32 scratch 与 additive A 的独立加载放到同一个 `PIPE_ALL` 等待点之前。
保持乘法、加法、转换、转置写回的次序，保持 Cube tile、跨核握手和静态路由。
为满足复杂度检查，局部算术提取为内联 `combine_epilogue`；没有改变公式。
使用正式接口 `Muon(..., use_symm_gemm=True, use_symm_gemm_epilogue=True)`。
FP32 融合仍默认关闭，本轮没有增加新的公共开关。

优化依据分为两方面：

- QuACK 的 SM90 路径分别组织 additive C 加载和输出存储流水。这里借鉴独立加载提前发起的思路，
  没有复制 GPU 实现。[上游源码](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm90.py)。
- [前一轮消融](../muon_epilogue_cost/README.md)确认 Cube tile 与 MAC 时间基本不变，额外成本主要来自
  FP32 scratch、Vector 转换和写回。该改动针对加载与等待；没有消除 GM scratch，也未实现完整双缓冲。

## 对照与边界

t129，8 张 Ascend910B3，CANN 8.5.0，Torch 2.9.0+cpu，torch_npu 2.9.0。
`CLOSE_MATMUL_K_SHIFT=1`，`OMP_NUM_THREADS=1`。启动前各卡 AICore 利用率为 0；设备 0 有
其他用户持有约 119 MiB 的 Python 上下文，未干预它，因此不声称获得主机独占。

使用 Qwen3-30B-A3B、Qwen3.5-27B 语言模型主体的完整参数规格，合成 BF16 参数和梯度。
每个模型按原版 A、实验版 B、实验版 B、原版 A 顺序启动四个独立进程组。
每个进程组分别测 CANN、symm、symm+epilogue，均预热 2 步、测量 5 步；每个构建每种模式共 10 个样本。
三个模式每次都从相同初始参数、梯度和新建优化器状态开始。

统计完整 `Muon.step()` 的同步墙钟时间，每步取最慢 rank。包含 Muon 内部通信、动量、NS 和参数更新，
不包含前反向、先前梯度归约和 AdamW。不是完整训练 step。
报告本轮同时测得的数据，不把历史运行作为本轮基线。

首次 A/B 进程组另用相同变化梯度执行 5 步，在第 1、5 步记录所有 rank 的全量参数和动量 SHA256。
检查包括张量名称、形状、dtype、原始字节和元素数量，不是抽样。
哈希在计时之外计算；跨构建相同哈希表示本次输入的输出位模式一致。
这是与原 FP32 融合实现比较，不意味着与 BF16 分步计算恢复逐 bit 一致，也不验证训练收敛。

## 复现

[运行脚本](run_t129.sh)包含独立环境 editable 自动构建、单测、安装路径检查、算子冒烟和全部模型命令。
基线安装目录保留原动态库；两个构建运行同一份基准脚本，通过 JSON 中的源文件及动态库哈希区分。
复杂度整理发生在首个 A 运行期间：暂停外层调度，让 A 正常完成，再重建并验证最终 B 后恢复后续运行。
最终候选实机源码哈希须与分支文件一致。

实际重建时，在首个基线结束后执行（随后恢复本次外层调度器）：

```bash
source /home/cann_custom_path/8.5.0/cann-8.5.0/set_env.sh
cd /home/townwish/workspace/projects/hp-prefetch-step-20261116
/home/townwish/workspace/projects/hp-prefetch-step-env-20261116/bin/python -m pip install -e . --no-deps
CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1 \
  /home/townwish/workspace/projects/hp-prefetch-step-env-20261116/bin/python \
  tests/torch/functional/symmetric_matmul_smoke.py \
  --expected-root /home/townwish/workspace/projects/hp-prefetch-step-20261116 \
  --output validation/smoke.json
```

独立重现可直接打包最终实验分支源码运行脚本，无需中途暂停或重建。

## 正确性与验证范围

两个模型第 1、5 步，全部 8 个 rank 的参数和动量完整 SHA256 均与原融合版本相同。
每个检查点覆盖 30B 的 29,909,581,824 个 Muon 参数元素和等量动量元素，
27B 的 24,352,522,240 个 Muon 参数元素和等量动量元素。
两个构建使用相同的基准脚本哈希，实验版 epilogue.h 哈希与提交源码一致。

- editable 自动编译成功，最终重建后再次验证安装路径与算子冒烟通过。
- 针对性单测：25 passed、21 subtests passed；6 个已有 pytest marker 警告。
- 冒烟覆盖 BF16/FP16、非对齐输入、独立 stream、对称写回、NS 和参数更新。
  FP32 融合的小矩阵 FP64 参考检查 relative L2 为 0；这是测试输入结果，不是普遍精度保证。
- CANN 安装目录所有者警告和 HCCL barrier 当前设备提示未影响运行。
- 不包含真实训练梯度反馈、收敛验证或所有算子输入分支的穷举。

全量指纹与参数/动态库源码哈希在各模型首次 A/B 的 JSON 中；后两组只做计时。

## 完整 step 结果

单位 ms，每项合并两次独立进程组的 10 个慢 rank 同步墙钟样本后取中位数。

| 模型 | 构建 | CANN | symm | symm + FP32 epilogue |
| --- | --- | ---: | ---: | ---: |
| qwen3-30b-a3b | baseline | 989.844 | 929.730 | 911.393 |
| qwen3-30b-a3b | prefetch | 988.230 | 929.801 | 911.097 |
| qwen3.5-27b | baseline | 3198.433 | 2982.495 | 2966.699 |
| qwen3.5-27b | prefetch | 3198.253 | 2981.537 | 2968.941 |

实验版相对原融合版本：30B 减少 0.296 ms（0.033%）；27B 增加 2.242 ms（0.076%）。
30B 的两次 A/B 配对方向相反，27B 两次候选均略慢。样本范围存在重叠，不能把这种量级的差异
宣称为稳定加速。八次进程组全部成功退出；峰值 NPU 分配量未改变。

这并不说明整个 FP32 融合没有收益。本轮原融合版本相对 symm 未融合版本，30B 约快 1.97%，
27B 约快 0.53%。本次没有确认收益的是追加的加载顺序调整。
局部 kernel 改善只作用于每轮 NS 的方形多项式部分，完整 step 仍包含其余 GEMM、动量、通信和写回。
建议只保留在实验分支，不作为已经证明有效的 step 优化推广；融合默认关闭状态不变。

## 数据

- [汇总及逐次中位数](summary.json)、[最终安装与算子冒烟](smoke.json)。
- [编译、单测、八次运行日志和设备快照](evidence.tar.gz)。
- qwen3-30b-a3b：[1 baseline](qwen3-30b-a3b-1-baseline.json)、[2 prefetch](qwen3-30b-a3b-2-prefetch.json)、[3 prefetch](qwen3-30b-a3b-3-prefetch.json)、[4 baseline](qwen3-30b-a3b-4-baseline.json)。
- qwen3.5-27b：[1 baseline](qwen3.5-27b-1-baseline.json)、[2 prefetch](qwen3.5-27b-2-prefetch.json)、[3 prefetch](qwen3.5-27b-3-prefetch.json)、[4 baseline](qwen3.5-27b-4-baseline.json)。
