# 方形 GEMM 的 FP32 epilogue 成本诊断

基于 `3fde953e6b8a596f2a86c0659505c19279bdd71b`，在 t129 / Ascend910B3 /
CANN 8.5.0 / Torch 2.9.0+cpu / torch_npu 2.9.0 上执行。
本轮不修改生产 Muon 或原生算子；新增诊断脚本，并在隔离目录测试一个同步调整原型。

## 对照方式

固定同一份逐 bit 对称的 BF16 A、A 的转置副本、输出和 FP32 scratch，直接调用现有 ABI 4。
输入复制、缓冲区分配、plan 创建都在计时之外。A 的形状分别为 `[32,768,768]` 和 `[1,5120,5120]`。
所有进程都在启动前设置 `CLOSE_MATMUL_K_SHIFT=1`；随机种子为 43。

| 对照 | Cube 输出 | Vector 处理 |
| --- | --- | --- |
| direct_bf16 | 直接写 BF16 输出 | 转置镜像 |
| scratch_identity | FP32 scratch | 乘 1、转 BF16、写回并镜像 |
| scratch_scale | FP32 scratch | 乘 c、转 BF16、写回并镜像 |
| scratch_polynomial | FP32 scratch | 乘 c、读 A、转 FP32、乘 b、相加、转 BF16、写回并镜像 |

每组分别正序和逆序运行；每块预热 5 次，然后测 7 个样本，每样本包含 20 次原生调用。
另外单独采集每组 5 次 kernel 的 profiler，使用明确的 warmup/active schedule。
原始 Event 数据存在个别异常偏低样本，因此完整保留样本并使用中位数；后续反序复测额外加入
同步墙钟计时。墙钟复测出现明显停顿，未能支持稳定的吞吐收益结论；不把单个极小样本当成性能收益。

## 已确认的原因

两种形状、四个对照的 base tile 都是 `128×256×64`，任务 tile 都是 `256×256`，使用 20 个 Cube。
比较整个 128-word tiling payload，BF16 和 FP32 输出计划无任何不同。
变化发生在 plan 的 direct 标志和实际输出类型，不是 CANN 给 FP32 选了另一个差的 tile。

FP32 scratch 走一遍、再转 BF16的 identity 对照，与 direct 输出不同 bit 数为 0。
因此这里的附加成本在不改变输出值的对照中已经存在。

典型非对角元素及其镜像的逻辑输出路径流量为：

- direct：Cube 写 BF16 2 B + Vector 读 2 B + 镜像写 2 B，共 6 B。
- scratch identity：Cube 写 FP32 4 B + Vector 读 4 B + 两侧 BF16 写回 4 B，共 12 B。
- scratch polynomial：再增加 additive A 的 BF16 读取 2 B，共 14 B。

这是忽略对角 tile、公共 A/B 读取的逻辑 GM 访问模型，**不是实测 HBM 流量**；L2 命中和缓存竞争会影响实际代价。
FP32 路径还将输出转换移到 Vector，增加了下三角写回、Cast 和同步。单独将乘数从 1 改为 c，
指令结构几乎不变，实测耗时也基本不变。

硬件计数器显示 Cube 的 MAC 时间基本不变，而 Fixpipe、Vector 和相关搬运时间增加。
这些流水可以重叠，不能把各个 pipe 的 busy time 直接相加当作总耗时，也不能据此断言全程 HBM 带宽饱和。

## 优化方向

1. **Vector 预取与双缓冲。** 当前逐个 `64×64` 子块处理，存在多次 `PIPE_ALL`，并在 Cube 结果就绪后才加载 additive A。
   可尝试提前加载 A，并让下一子块的读取与当前子块的计算/写回重叠。必须保留真实数据依赖和跨核 ready/consumed 握手。
2. **紧凑的循环 scratch。** 当前申请完整 `[B,N,N]` FP32 scratch，但 Cube/Vector 只允许有限任务领先。
   可研究每个 Cube 两个 `256×256` FP32 slot；20 核约 10 MiB，而两个测试的完整 scratch 分别为 72 MiB、100 MiB。
   这能降低工作集与显存需求，但必须调整输出 stride、索引和 slot 生命周期；不能只缩小 allocation。
   此方案是否降低 HBM 访问或提高 L2 命中还需测量。
3. **评估 Matmul 到 Vector 的另一种交付方式。** CANN 的 GetTensorC 支持 VECIN 输出，但要求 NZ 布局，
   异步方式仍可能依赖 GM workspace；不能把 LocalTensor 返回值直接等同于零 GM 往返。
   需要连同布局转换和同步成本评估，不能假定 GPU 的寄存器 epilogue 可以直接移植。
   依据：[CANN 8.5 GetTensorC](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0639.html)。

这一轮不优先改 Cube tile：计划和 MAC 时间已经排除了它作为本次增量耗时的主要原因。

对照上游 QuACK 源码，它的线性 epilogue 在 accumulator dtype 上处理系数，SM90 路径为 additive C 加载
和输出存储分别组织流水。值得借鉴的是预取、分阶段交接和缓冲复用；TMA/WGMMA 等 GPU 机制不直接用于 NPU。
来源：[linear epilogue](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_default_epi.py)、
[SM90 pipeline](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm90.py)。本轮没有复制这些实现。

## 可复现证据

- [诊断脚本](../../../tests/torch/optimizer/diagnose_muon_epilogue.py)。
- [原始消融数据与完整 tiling](baseline.json)、[复测基线](baseline-repeat.json)、[原型](prefetch-probe.json)。
- [原始 profiler CSV](epilogue-cost-profile.tar.gz)、[计数器汇总](profile_summary.json)。
- [隔离原型补丁](prefetch-probe.patch)。它没有应用到生产代码；只调整独立输入加载的位置与等待次数。

## 定量结果

以下为原实现隔离 profiler 的 5 次 kernel 平均耗时，单位 µs；每项是一次方形运算，
不是五轮 NS 或完整 optimizer step。

| A 形状 | direct BF16 | FP32 identity | FP32 scale | FP32 polynomial |
| --- | ---: | ---: | ---: | ---: |
| [32,768,768] | 112.406 | 132.453 | 132.405 | 152.108 |
| [1,5120,5120] | 700.308 | 752.292 | 753.836 | 765.174 |

从 direct 到 polynomial，expert 的 Cube MAC 为 66.209 → 66.295 µs，dense 为
492.581 → 492.472 µs。相应 Vector 时间为 8.775 → 28.049 µs、10.606 → 35.626 µs；
Fixpipe 为 10.214 → 19.527 µs、10.369 → 18.468 µs。
单独引入 FP32 scratch 就已经增加约 20 / 52 µs，随后读取 A、转换并计算加法又增加约 20 / 13 µs。
这些差值只用于本组受控消融，不能跨测量批次直接外推。
融合 kernel 承担了原来多个 kernel 的工作，收益应对比完整 P 计算区间或 optimizer step。

## 加载顺序原型

只将 scratch 与 additive A 的两次独立 DataCopy 放到同一个 PIPE_ALL 等待点之前，
其余计算次序、转换、跨核握手和写回均保持不变。没有改变 Cube tile。
隔离 CMake 构建成功；两个形状的完整 polynomial 输出 SHA256 与原融合版本一致，
identity 输出仍与 direct 逐 bit 一致，有限性和对称性检查通过。
这里的对齐对象是原 FP32 融合版本，不是 BF16 分步 polynomial。

下表为两轮独立进程对照的 Event 样本中位数（µs），每轮内部各模式正序、逆序各运行一次。

| 形状 | 原实现→原型，第一轮 | 耗时下降 | 原型先跑的复测，原实现→原型 | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| expert | 154.504→148.650 | 3.79% | 153.691→150.182 | 2.28% |
| dense | 761.325→756.451 | 0.64% | 756.730→739.987 | 2.21% |

原型单独 profiler 的 polynomial 平均为 144.872 / 757.313 µs，方向与 Event 一致，
但这是跨进程采样，不能视为严格同条件的配对加速比。
同步墙钟复测反而明显变慢：expert 159.557→793.986 µs，dense 763.054→1150.722 µs。
原型进程的其他模式墙钟也出现大量停顿；具体宿主调度或同步停顿原因未定位。
因此目前只报告设备计时的改善信号，**不宣称稳定的应用吞吐收益**。

没有对该原型运行完整 NS、完整模型 optimizer step、FP16 或其他 alpha/beta 分支回归。
保留独立补丁，不应用到生产实现；FP32 融合仍默认关闭。

- [反序原型数据](probe-reverse.json)、[反序基线数据](baseline-after.json)、[比较汇总](probe_summary.json)。
- [原型 profiler CSV](epilogue-prefetch-profile.tar.gz)。
- [构建、运行与验证范围](commands.md)。
