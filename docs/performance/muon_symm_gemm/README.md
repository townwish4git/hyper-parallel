# Muon use_symm_gemm 验证报告

验证主机：t129，8 x Ascend 910B3，CANN 8.5.0，Torch/torch_npu 2.9.0。
分支基于 upstream master `02077404bc38c9d8f88fb7dd6423383550d1648f`。
当前功能只接入对称 GEMM 到标准 NS；不含 Gram 算法或 `use_gram` 配置。

## 完整 optimizer step

基线和候选分别为生产 `Muon(..., use_symm_gemm=False/True)`。所有 worker 在启动前设置
`CLOSE_MATMUL_K_SHIFT=1`。没有优化器 callback 或 monkeypatch。使用模型真实规格、
合成 BF16 权重和梯度，测完整 Muon step（含其通信、动量、参数更新），不测前后向、
先前梯度归约或 AdamW。27B 只包含语言主体；30B 所有专家均提供梯度。

每次 step 使用五轮 asym5；lr=0.01、momentum=0.95、Nesterov=True、weight_decay=0.1。
每配置两块测试，顺序先 False/True，再 True/False；每块预热 2 步、测量 5 步。
每个样本取同步后最慢 rank 的墙钟耗时；跨 rank 汇总通信在计时区间之外。

| 模型 | 原生 GEMM：ms/step | use_symm_gemm：ms/step | 耗时降低 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | 993.98 | 932.96 | 6.14% | 1.0654x |
| Qwen3.5-27B 语言主体 | 3196.49 | 2985.41 | 6.60% | 1.0707x |

## 全量精度

27B 初次运行在主机重启前完成全部计时和 1 步精度；5 步精度通过同一脚本的
`--cycles 0 --accuracy-steps 5` 独立补测，输入种子、生产实现和原生库哈希相同。

两模型在 1 步、5 步的全元素检查中，更新后的参数、实际更新量、动量的不同 bit 模式数均为 **0**，
相对 L2 差异均为 **0**，非有限值数均为 **0**。30B 检查 29,909,581,824 个 Muon 参数元素，
27B 检查 24,352,522,240 个。实际更新量定义为 `final - initial`，包含 BF16 写回和 weight decay；
第 5 步报告的是五步累计位移，不是第 5 步单步更新，也不是单独的 NS 输出。

精度测试在每个检查点重新初始化参数与优化器状态，各配置输入一致，梯度随步数变化。
统计使用每 rank 16 个 CPU 线程；optimizer 执行与计时保持 `OMP_NUM_THREADS=1`。

## 安装与测试

- CANN 已加载、未设置 HP_BUILD_ASCENDC 覆盖值时，默认 `pip install -e . --no-deps` 自动编译通过，PEP 517 构建隔离开启。
- 从另一份不含任何 `.so` 的源码目录执行默认 `pip wheel . --no-deps`，自动编译成功；未依赖 editable 产物。
- wheel 安装到独立环境，checkout 外导入路径确认位于该环境的 site-packages；包含全部三个原生库，无 Gram 模块。
- editable 在 t129、独立源码构建的 wheel 在 t131（同型号设备、同软件版本）的 NPU smoke 均检查实际原生算子、静态选路、当前 stream、非对齐输出、标准 NS、五次 optimizer step 及 checkpoint 恢复。
- 针对性单元测试：24 passed，17 subtests passed；包含开关默认值/参数验证、YAML 传递、合批、回调优先级、参数组 checkpoint、CANN 目标检测和回退语义。
- autogit 静态检查通过。自动测试路径推断有 advisory 提示；实际 UT/ST 路径见命令和日志。
- 为满足复杂度检查，将已有的 step 计数与无通信动量输入准备提取为私有辅助方法；顺序、调用条件与通信语义保持一致，并补充测试。

## 使用范围

上述“性能提升且逐 bit 一致”仅对应当前验证的设备、软件栈、dtype、形状和输入。
不是任意 shape 的逐 bit 保证，也不是长期训练收敛实验。其他 SoC、FP16 训练、其他 NS 系数、
full-graph optimizer 编译和 graph capture 未纳入完整模型验收。实现按静态成本规则回退小矩阵；
因此并非所有模型都能获得相同收益。

完整部署步骤：[中文指南](../../guide/muon_symm_gemm.zh.md)、[English guide](../../guide/muon_symm_gemm.md)。

## 复现与原始证据

- [完整 t129 命令](run_t129.sh)、[构建及安装命令](installation_commands.txt)。
- [30B 结果](qwen3-30b-a3b.json)、[27B 计时及 1 步结果](qwen3.5-27b.json)、
  [27B 五步补测](qwen3.5-27b-five-step.json)、[汇总数据](summary.json)。
- [editable smoke](editable-smoke-final.json)、[wheel smoke](wheel-smoke.json)、[wheel 内容](wheel-contents.json)。
- [源码哈希](source_hashes.json)、[最终审计](audit.json)、[静态检查](static-checks.txt)、[测试日志](ut-final.txt)。
- [构建验证记录](build_validation.json)记录命令、版本、退出状态及产物哈希。
