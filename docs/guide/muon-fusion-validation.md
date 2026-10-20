# Muon 融合优化实测记录

本分支将四组方案分开提交，所有开关默认关闭。基线为 `3a001291`；实现版本为
`2869edaa`。原工作区尚未提交的通信重叠改动不在本分支中，也未声称验证其组合效果。

## 环境与方法

- 主机 t129，Ascend 910B3；PyTorch 2.9.0+cpu、torch_npu 2.9.0、CANN 8.5.0。
- Triton-Ascend 3.2.0 安装在独立虚拟环境，没有修改共享训练环境。
- 单卡算子测试使用物理卡 2；FSDP 验证使用物理卡 2、3。
- 每项独立启用，对比原有标准路径。预热 10 次，测量 7 组；小形状每组 40 次，大形状每组 10 次。
- 延迟包含主机提交开销和同步后的设备执行时间；编译预热不计入。保留每组的最小、中位、最大值。
- 加速比为基线中位延迟除以候选中位延迟，小于 1 表示退化。没有把 Python 调用数当成设备 kernel 数。
- 原始结果、代码 SHA256、环境和阈值见 [JSON 数据](muon-fusion-results.json)。

## 独立方案结果

| 方案 | 精度证据 | 性能证据 | 处理结论 |
| --- | --- | --- | --- |
| 标准 NS 的 addmm/baddbmm | FP32 微测误差为 0；BF16 有舍入差异；真实训练参数容差未通过 | 小形状约 1.26–1.93x；大 BF16 矩阵约 0.59–0.89x | 保留实验开关，不默认启用 |
| 动量 Triton 融合 | FP32/FP16/BF16 的 20 步微测输出与状态误差均为 0；轨迹、分片、恢复和训练通过 | 4 个 1024×4096 FP32 张量、Nesterov 开启时约 1.27x；小模型完整 step 约 0.93–0.95x；FP16 大矩阵明显退化 | 仅供特定 FP32 大矩阵实验 |
| 缩放直接写回 | 微测误差为 0；轨迹、分片、恢复和训练通过 | 局部约 1.27–1.56x；小模型完整 step 约 1.03–1.05x | 本轮最明确的可用候选，仍需目标模型复测 |
| 衰减与参数更新 Triton 融合 | 微测误差为 0；轨迹、分片、恢复和训练通过 | 局部约 0.28–0.63x；小模型完整 step 约 0.90x | 无收益，不建议启用 |
| Gram-NS | FP64 代数测试通过；FP32 Gram 微测最大相对 L2 约 2.4e-6；BF16 输入有数值差异；默认 Gram 的真实训练参数容差未通过 | FP32 Gram 大矩阵约 0.43–0.92x；FP16 Gram 在 2048×8192 上约 1.10x，其他所测大形状退化 | 保留实验开关，不默认启用 |

Gram 的方阵路径回退到标准融合 NS，其加速不能归因于 Gram 递推。
FP16 Gram 对 FP32 输入的精度筛查有失败项；不能将降低计算精度当成等精度加速。
FP16 Gram 仅进行了算子级比较，完整优化器验证使用默认 FP32 Gram。

优化器目录 UT：211 passed，51 subtests passed；实测源码 SHA256 与实现提交一致。

## 精度验收边界

1. 单元测试以 float64 验证融合表达式、Gram 递推、两种朝向、batch、系数和 restart 的代数一致性。
2. 算子微测使用固定的相对 L2 筛查阈值。BF16 的 5% 阈值只用于筛查，不代表训练精度验收。
3. 单卡逻辑轨迹分别开启、关闭 Nesterov，使用双 momentum `(0.95, 0.85)` 和 30 步变化梯度。
4. checkpoint 恢复后继续一步，要求与未中断的同实现结果完全相同。
5. 双卡 FSDP 使用已有 attention/MoE 模型与 10 步逻辑梯度，比较完整参数和动量状态。
6. 真实训练使用两层 MLP、20 步真实 autograd。参数沿用 `rtol=1e-2, atol=1e-3`，
   状态沿用 `rtol=1e-4, atol=1e-6`。没有为 NS 或 Gram 的失败放宽容差。

NS 融合和默认 Gram 均通过了固定梯度轨迹、checkpoint、双卡分片检查，loss 也接近，
但真实训练中有参数未通过既定容差。这两项不能标记为整体精度验收通过。
复查最终两层参数：NS 融合最大相对 L2 为 3.3619%，
最大绝对差为 0.011214；默认 Gram 分别为
3.2127% 和 0.010173。
这些短测试也不构成长时间大模型收敛证明。

## 开关与使用限制

```python
from hyper_parallel.core.optimizer.muon import Muon

# 每次只启用一个选项，便于独立比较。
optimizer = Muon(params, fused_scale_writeback=True)
# Muon(params, ns_implementation="standard_fused")
# Muon(params, momentum_implementation="triton")
# Muon(params, update_implementation="triton")
# Muon(params, ns_implementation="gram", gram_compute_dtype="float32",
#      gram_reset_iterations=(2,))
```

使用 `get_hyper_optimizer` 时，将相应键加上 `muon_` 前缀放入 `muon_kwargs`。

- Triton 路径要求连续的本地 NPU 张量；参数更新路径额外要求 FP32 master 参数。
- 动量融合保留原有双 momentum 状态语义，没有改为 EMA，也不需要迁移动量 checkpoint。
- FP16/BF16 中间舍入显式保留；同尺寸张量按最多四个一组启动。FP16 精确舍入的代价很高。
- Triton 是可选依赖，仅在显式选择相关路径时导入。未支持的设备、布局会报错，不静默换算法。
- 缩放写回仅优化原生 NS 整理路径；自定义 transform/restore 路径保持原实现。
- Gram 沿用当前归一化与系数；默认以 FP32 迭代，可显式选择 FP16/BF16。
  使用独立临时缓冲区，restart 时不重复归一化。
- 没有修改 collective 顺序、owner 分配或通信规则；未进行多 stream overlap 组合验证。

## 复现命令

在 t129 的独立环境中先执行 editable 安装并确认导入路径：

```bash
cd /home/townwish/workspace/projects/hp-muon-fusion-validation
PY=/home/townwish/workspace/projects/hp-muon-fusion-env/bin/python
"$PY" -m pip install -e . --no-deps --no-build-isolation
"$PY" -c 'import hyper_parallel; print(hyper_parallel.__file__)'
"$PY" -m pytest tests/ut/core/optimizer -q
```

逐项微测，`--scheme` 可选 `ns`、`momentum`、`writeback`、`apply`、`gram`：

```bash
"$PY" -m tests.torch.optimizer.benchmark_muon_fusion \
  --scheme writeback --device 2 --repeats 40 --output /tmp/muon-writeback.json
"$PY" -m tests.torch.optimizer.benchmark_muon_fusion \
  --scheme gram --gram-dtype float16 --large --dtype bfloat16 \
  --device 2 --repeats 10 --output /tmp/muon-gram-large.json
```

完整精度检查：

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m tests.torch.optimizer.validate_muon_fusion \
  --scheme writeback --require-pass --output /tmp/muon-validate-writeback.json
```

诊断模式不传 `--require-pass`，仍会将失败字段写入 JSON。
传入该参数时，任何既定精度检查失败都会产生非零退出码。
`--training-only` 可单独复查真实训练参数差异。

实测日志出现 CANN 目录属主警告、TypedStorage 弃用提示和 Triton 编译参数提示。
这些提示没有作为精度失败的豁免；结果中的失败项保持失败。
