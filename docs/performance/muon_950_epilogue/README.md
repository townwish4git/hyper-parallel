# Ascend 950：Muon FP32 epilogue 片上交付实验

## 环境与原版兼容性

2026-11-16 在 a5（rack06os7）验证，Ascend950DT_9582，CANN 9.2.0，
PyTorch/torch_npu 2.14.0。基线是 `2439d01ea80edb18bd6cb341d461efd432a3a3e0`，
即此前 GM scratch + additive prefetch 实验版本。没有接入 Gram 算法。

基线源码不做兼容性修改，`pip install -e . --no-deps --no-build-isolation` 自动编译成功，
安装后的算子、标准 NS 和五轮 Muon smoke 均通过。`--no-build-isolation` 复用已安装的构建依赖，
避免共享主机访问包索引时阻塞；本轮另以 `pip wheel . --no-deps --no-build-isolation` 自动编译 wheel，
安装到独立 venv 并在 `/tmp`（checkout 外）运行验证成功；未测试联网下载构建依赖。
SoC 自动识别为 Ascend950DT_9582。本轮时序数据均设置 `CLOSE_MATMUL_K_SHIFT=1`。

基线和优化版分别安装在独立 checkout / venv。原始兼容性 smoke 使用设备 0；
正式算子对照在同一设备 6 上先基线、后候选运行。其他用户仍在设备 4、5 上运行，
因此不是整机独占基准。四卡完整模型 step 暂未取得有效结果：非特权隔离环境在初始化 NPU 时失败，
不能把单卡算子收益解释成整模型收益。

## 第一轮：L0C → UB，消除 FP32 GM scratch

### 两个分析依据

1. [QuACK 的 Hopper GEMM](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm90.py)
   在 accumulator / register 中处理结果，并通过 epilogue 的流水线交付；
   [线性 epilogue](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_default_epi.py)
   在累加精度中计算 alpha/beta，最后转换输出。可迁移的是避免中间矩阵落到全局内存、
   将生产和消费重叠的思路；不复用 GPU 源码或调度实现。
2. 同输入的原版消融显示，950 上 FP32 融合方形 kernel 比 BF16 直接写回慢。
   设备 profiler 中 FP32 路径增加 Fixpipe 和 Vector 的 MTE2 时间。
   因而先保持 CANN Matmul 的计算与 tiling，单独替换累加结果的传递路径。

### 实现边界

- 仅 `__NPU_ARCH__ == 3510` / Ascend950 构建启用。其他架构保留原实现。
- 只处理 CANN 选择 `baseM=baseN=256`、任务块 `256×256`、padded N 为 256 倍数的 FP32 epilogue。
  K 的 tiling 仍由 CANN 决定，边界 / 不匹配 tiling 使用原 GM 路径。
- 使用 CANN Matmul 的输出 callback；矩阵乘、K 归约顺序、FP32 epilogue 算术顺序均保持原有语义。
- 通过 950 `Fixpipe` 的双目标按 M 拆分，将一个 `256×256` FP32 tile 分两次送入两个 AIV。
  每次每个 AIV 收到 `64×256`（64 KiB）。两个接收 slot 合计 128 KiB，
  再加 48.5 KiB 的向量工作区，容纳于实测 248 KiB UB。
- Cube 等待 slot 消费完成才复用；Vector 等待 FIX 完成才读取。最后一个 strip 才释放 L0C 累加缓冲。
- Python 桥接查询原生 plan 是否走片上路径，符合条件时不再分配完整 FP32 scratch。
  未改 public Muon / symmetric_matmul 参数。仍需 `use_symm_gemm=True, use_symm_gemm_epilogue=True`。
- 纯 BF16 对称 GEMM 保持原路径。FP32 epilogue 仍为默认关闭的实验选项。

### 同卡性能

下表为两轮正序 / 逆序运行，每轮 7 组、每组 20 次调用，取两轮中位数的中位数。
输入、输出和 plan 预分配；这里是 Event 测量的算子时间，不是 Python 或完整 optimizer step。

| FP32 polynomial 任务 | 原 GM 版 / μs | UB 版 / μs | 耗时下降 |
| --- | ---: | ---: | ---: |
| `[32,768,768]` | 97.658 | 81.845 | 16.2% |
| `[1,5120,5120]` | 421.379 | 401.883 | 4.6% |

Profiler 独立采集，各 5 次 kernel 平均，不与未 profile 的时序混用：

| 场景 / 指标 | 原 GM 版 / μs | UB 版 / μs |
| --- | ---: | ---: |
| expert：Cube MAC | 47.525 | 47.844 |
| expert：Fixpipe | 12.880 | 5.454 |
| expert：Vector MTE2 | 16.198 | 7.605 |
| expert：Vector 计算 | 24.642 | 25.810 |
| dense：Cube MAC | 355.737 | 357.819 |
| dense：Fixpipe | 13.122 | 5.515 |
| dense：Vector MTE2 | 24.103 | 14.026 |

各流水线时间会重叠，不能相加当作 kernel 时长。大方阵的 Cube 时间占比更高，
故同样减少约 20 μs，比例收益更小。Vector 时间略增，包含新的 UB 整理操作。
这些是 profiler 观测与结构分析；未采集 HBM 字节计数，不能声称测得了某个 HBM 带宽下降比例。

### 数值与验证

- 两个代表性形状的 FP32 polynomial 输出 SHA256 在原 GM 版与 UB 版完全一致。
- 将 FP32 epilogue 设置为恒等变换，与直接 BF16 GEMM 输出不同 bit 数均为 0。
- UB 版安装 smoke 通过，标准 NS 的五轮更新与 CANN no-k-shift 对照不同 bit 数为 0。
- FP32 epilogue 与原 BF16 逐算子 NS 仍存在舍入差异：smoke 的 NS 相对 L2 为 0.01235877，
  与原 FP32 融合版本相同。取消 scratch 不等于消除融合算法本身的舍入语义变化。
- 小矩阵独立 FP64 公式对照相对 L2 为 0。

原始数据：[基线](baseline-kernel.json)、[UB](ub-kernel.json)、
[基线 smoke](baseline-smoke.json)、[UB smoke](ub-smoke.json)、[profiler 汇总](profile-summary.json)。
同目录 CSV 保留每次 kernel 的设备指标。后续轮次应另存结果，不能覆盖本轮数据。

### 代表性参数的本地 Muon step

每个模式使用全新的参数、动量和相同输入；第 1/5 步保存完整参数与动量 SHA256，
随后测 15 次同步 wall time。不是完整模型，也没有通信。

| 参数 shape | 模式 | 原 GM 版 / ms | UB 版 / ms |
| --- | --- | ---: | ---: |
| `[32,768,2048]` | CANN NS | 4.3105 | 4.2958 |
| `[32,768,2048]` | BF16 symm NS | 4.4132 | 4.4274 |
| `[32,768,2048]` | FP32 epilogue NS | 4.2392 | 4.1652 |
| `[5120,17408]` | CANN NS | 28.0049 | 28.0006 |
| `[5120,17408]` | BF16 symm NS | 23.6128 | 23.5865 |
| `[5120,17408]` | FP32 epilogue NS | 23.2195 | 23.1644 |

FP32 路径的本地 step 分别下降 1.74% 和 0.24%。后一项很小，不能据一次对照宣称稳定收益。
相同路径下原版和 UB 版的第 1/5 步参数及动量指纹全部一致；
8 个 BF16/FP16 边界用例输出也逐 bit 一致。这里没有声称 FP32 融合与 BF16 CANN NS 逐 bit 一致，
也没有将 smoke 的单个 expert 场景对齐结论推广到所有 950 shapes。

实际峰值 allocated memory：expert 852 → 796 MiB，dense 1440 → 1356 MiB。
峰值由多种临时张量共同决定，不能简单等同于完整 scratch 字节数。
数据：[原版本地 step](baseline-local-steps.json)、[UB 本地 step](ub-local-steps.json)。

### Wheel 安装复核

提交 `0ef434ba` 的源码构建出 cp312/aarch64 wheel。安装环境复用系统 Torch，
使用 `pip install --ignore-installed --no-deps <wheel>` 确保该 venv 实际安装 wheel，
避免相同版本的外部元数据使 pip 跳过安装。没有改动系统 Python 包。
包路径经 smoke 断言位于独立 wheel venv 的 `site-packages`，不依赖源目录或 PYTHONPATH。
8 个边界用例以及两个参数场景第 1/5 步的结果，均与 editable 版逐 bit 一致。
见 [wheel smoke](ub-wheel-smoke.json) 和 [wheel 本地 step](ub-wheel-local-steps.json)。

## 第二轮：整条 additive strip 预取（未保留）

- QuACK 依据：其独立 C-load / epilogue pipeline 提前加载 additive 输入，减少消费者等待。
- 本机依据：第一轮 expert / dense 的 AIV MTE2 仍分别约 7.6 / 14.0 μs。
  尝试在等待 Cube ready 之前，一次加载 `64×256` additive strip，增加 32 KiB UB，
  再拆成 64×64 子块计算；总 UB 208.5 KiB，仍低于 248 KiB。
- 实测 FP32 polynomial：expert 81.845 → 84.215 μs（慢 2.9%），
  dense 401.883 → 404.225 μs（慢 0.6%）。
  MTE2 时间反而变成 12.101 / 19.227 μs，Cube MAC 基本不变。
- 结构上新增一次 UB 整理，而且对角任务会多加载不用的上三角 additive 数据。
  粗粒度提前加载没有抵消这些成本。本轮没有更换矩阵乘 tiling 或 FP32 算术。
- 本地 step 为 4.1625 / 23.1810 ms，相对第一轮 4.1652 / 23.1644 ms 变化很小，
  expert 的微小正向波动不足以推翻 kernel 的稳定负收益。
- 所有边界输出和第 1/5 步参数、动量指纹仍与第一轮相同。撤回预取源码，保留第一轮实现。

[可重放实验补丁](round2-strip-prefetch.patch)、[kernel 数据](round2-kernel.json)、
[本地 step 数据](round2-local-steps.json)，以及同目录 round2 CSV。
这是一轮被实测否定的候选优化，不应作为性能改进接入训练。

## 与原生 CANN 的 NS profiler 对照

另外采集了每个模式各两次五轮 NS 的完整设备 trace。
950 / CANN 9.2.0 下每次 NS：CANN 为 43 个 kernel，BF16 symm 为 48 个，
FP32 epilogue 为 33 个。UB 优化前后 kernel 数不变，仅缩短已有融合 kernel。
不能把此前其他版本的 kernel 数直接套到本次 CANN 上。

按实际 launch 顺序拆出每轮 `A²`（CANN）或 `bA+cA²`（融合）的 kernel 平均时长：

| 输入 NS shape | CANN 的 A² / μs | 原 GM 融合 / μs | UB 融合 / μs |
| --- | ---: | ---: | ---: |
| `[32,768,2048]` | 73.762 | 98.559 | 84.383 |
| `[1,5120,17408]` | 630.021 | 418.892 | 401.810 |

CANN 列不包含后续三个逐元素 kernel，不能只凭此列判断完整融合收益。
expert 的 UB kernel 仍慢于单独 CANN 方形 GEMM，但融合省去逐元素 launch；
dense 的对称计算本身已有收益，这次 UB 优化是在原收益上继续降低中转成本。
矩形 `P @ X`、norm、转置、更新等没有被本次优化消除。

未 profile 的完整 NS wall time：

| 输入 NS shape | CANN / ms | 原 GM epilogue / ms | UB epilogue / ms |
| --- | ---: | ---: | ---: |
| expert | 3.554 | 3.454 | 3.370 |
| dense | 26.562 | 21.870 | 21.762 |

它们不含 Muon 的动量和参数更新；前面的本地 step 表包含这些操作。
[原版 NS 时序](baseline-ns-timings.json)、[UB NS 时序](ub-ns-timings.json)、
[逐算子 profiler 汇总](ns-profile-summary.json)，原始 kernel CSV 同目录。
Profiler 在无 schedule 的两次调用采集结束时报告 schedule 警告；已核对导出的两次 NS
各类 kernel 数与调用结构一致。时序结论使用 profiler 外的独立测量。

## 检查与限制

a5 上相关单元测试 13 项通过：

```bash
python -m pytest -q tests/ut/components/functional/test_symmetric_matmul.py \
  tests/ut/core/optimizer/test_muon_symm_gemm.py
```

本地 lint 环境没有 Torch，UT 无法在本地启动，以上通过结果来自 a5 的 Torch 环境。
此次仅验证 950 / CANN 9.2.0；没有重新运行其他架构的设备测试。
完整 Qwen 模型、多卡通信、真实训练收敛不在已完成的证据范围中。
测试容器未能初始化 NPU；自动审批拒绝了进一步使用特权容器，等待用户单独授权。
已清理本次无法使用的非特权容器，没有改动主机 `/etc` 或其他任务。

本轮 wheel SHA256：
`ffd9b1d80f42def5d0604a6d71e0eacdbbbc9b25f2bfc48478607072885beb78`。

## 复现

源树可编辑安装后，先核对 `hyper_parallel.__file__` 指向本树。机器依赖已准备好时：

```bash
source /usr/local/Ascend/cann/set_env.sh
export CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=6
python -m pip install -e . --no-deps --no-build-isolation
python tests/torch/functional/symmetric_matmul_smoke.py \
  --expected-root "$PWD" --output /tmp/muon-950-smoke.json
python -m tests.torch.optimizer.diagnose_muon_epilogue --output /tmp/muon-950-profile
python -m tests.torch.optimizer.validate_muon_950 --output /tmp/muon-950-local-steps.json
```

`validate_muon_950` 测试 BF16/FP16、K 尾块、未对齐 out、N 尾块，
并测量两个代表性参数的完整本地 Muon step（动量和更新包括在内，无分布式通信）。
它不是完整模型 benchmark，不能替代 `benchmark_muon_models` 的多卡结果。
