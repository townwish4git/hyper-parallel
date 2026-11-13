# Muon 标准 NS 的 Ascend C 对称 GEMM 加速

本特性只替换标准 Newton–Schulz（NS）中的对称矩阵乘法，默认关闭。
不引入 Gram 算法，不修改 NS 系数、动量、weight decay 或通信语义。
目标使用方式是在 Ascend NPU 训练中启用 `Muon(..., use_symm_gemm=True)`。

## 获取与安装

分支只有一个功能提交，包含算子、Python 桥接、Muon 开关、自动构建、测试与文档。
同事不需要依赖实验存档分支，也不应只复制 Python 文件。

```bash
# remote 已存在时不必重复添加。
git remote add townwish4git git@github.com:townwish4git/hyper-parallel.git
git fetch townwish4git feat/muon-symm-gemm
git cherry-pick FETCH_HEAD

# 替换为训练机器实际使用的 CANN 环境脚本。
source /usr/local/Ascend/cann/set_env.sh
python -m pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
```

请基于近期 HyperParallel master 使用本提交。更老训练分支可能需要解决 Muon
参数组接口的冲突。安装环境需要完整 CANN 编译工具链、C++ 构建工具和 CMake >= 3.16；
只有运行库的容器不能编译算子。Torch 与匹配的 torch_npu 应预先安装。

`pip install -e .` 在检测到 CANN 后自动编译，支持默认的 PEP 517 构建隔离。
无需手工执行 CMake，也没有首次 optimizer step 时的 JIT 编译。
确认 import 指向当前 checkout，并在重装后重启训练进程。

也可构建 wheel 后分发给相同软硬件环境：

```bash
wheel_dir=$(mktemp -d)
python -m pip wheel . --no-deps -w "$wheel_dir"
python -m pip install --force-reinstall --no-deps "$wheel_dir"/*.whl
```

wheel 包含编译产物，安装 wheel 不会再次编译。分发时应匹配 Python/CPU 平台标签、
CANN ABI 与 Ascend SoC。构建默认查询完整 SoC 标识，不会在查询失败时默认编译 910B。
编译机器没有设备时，显式设置 `HP_ASCEND_SOC_VERSION` 为目标设备的完整标识。
其他 SoC 构建成功，不代表已经验证该设备的精度和性能。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `HP_BUILD_ASCENDC` | `auto` | 有 CANN 时自动编译；`on` 强制要求；`off` 显式跳过 |
| `ASCEND_HOME_PATH` / `ASCEND_CANN_PACKAGE_PATH` | 自动发现 | 指定 CANN 编译工具链 |
| `HP_ASCEND_SOC_VERSION` | 查询设备 | 指定交叉编译的完整 SoC 标识 |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `4` | 本地编译并行度 |

无 CANN 的机器可以安装纯 Python 回退版本，但没有 Ascend C 性能收益。
如果已发现 CANN 而编译失败，安装会失败，不会静默伪装成加速版本。
部署 CI 可以设置 `HP_BUILD_ASCENDC=on`，要求必须构建算子。

## 训练开关与精度环境

在启动 Python、导入 Torch/CANN 之前，为所有 worker 设置：

```bash
export CLOSE_MATMUL_K_SHIFT=1
torchrun <原有训练启动参数>
```

**1 表示关闭 K-shift。** 容器、远端启动器也要传递该变量。
不要在已经执行过 matmul 的进程内临时修改它。
优化器不会替用户改写这个全局环境变量。

精度对照的基线也必须使用相同设置。CANN K-shift 会改变累加顺序，
不能将本次“关闭 K-shift”的逐 bit 验证结论套用到开启 K-shift 的基线。

```python
from hyper_parallel.core.optimizer.muon import Muon

optimizer = Muon(
    muon_parameters,
    lr=0.01,  # 这里保留你原有的超参数。
    use_symm_gemm=True,
)
```

混合 Muon/AdamW 优化器在现有 `muon_kwargs` 或 `muon_config` 中增加：

```yaml
muon_config:
  muon_use_symm_gemm: true
```

不带 `muon_` 前缀的 `use_symm_gemm: true` 也支持。
此配置不会把 embedding、norm 等原属于 AdamW 的参数转交给 Muon。
显式 `zeropower_fn` 回调仍优先于内置 NS；如需使用本实现，不能同时保留替代 NS 的回调。

## 从 checkpoint 恢复时的注意事项

开关保存在每个参数组以及 optimizer checkpoint 中。
加载旧 checkpoint 或关闭开关时保存的 checkpoint，可能覆盖构造时的参数组设置。
因此在恢复状态后，针对 **Muon 实例** 设置：

```python
optimizer.load_state_dict(checkpoint["optimizer"])
for group in optimizer.param_groups:
    group["use_symm_gemm"] = True
```

混合优化器只操作其中的 Muon 子优化器，不修改 AdamW 子优化器。
如需回退，将所有 Muon 参数组的开关设为 `False`，即可恢复原始 NS matmul 路径。

## 开训前检查

首先确认 import 位置和三个编译产物：

```python
from pathlib import Path
import hyper_parallel

root = Path(hyper_parallel.__file__).parent
folder = root / "components/functional/_ascendc/symmetric_matmul"
for name in ("libhp_symmetric.so", "libhp_epilogue_kernel.so", "libhp_fused_kernel.so"):
    assert (folder / name).is_file(), folder / name
print(root)
```

然后运行安装后的算子和 Muon smoke：

```bash
CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 python \
  tests/torch/functional/symmetric_matmul_smoke.py \
  --expected-root "$PWD" --output /tmp/symmetric-muon-smoke.json
```

该测试检查真实原生算子调用、静态选路、当前 stream、非对齐输出保护、
标准 NS、连续 5 个 optimizer step 和 checkpoint 恢复。
验证 wheel 时应从 checkout 外通过绝对路径运行脚本，并将 `--expected-root`
设置为 wheel 安装环境的 `site-packages` 路径，防止源码目录遮蔽已安装的 wheel。

完整模型规格的复现命令、数值口径和结果见
[验证报告](../performance/muon_symm_gemm/README.md)。基准脚本使用生产开关，
没有给优化器打补丁或挂测速 callback。构造这些模型规格额外需要 Transformers 5.13.0，
算子和 Muon 本身不新增该依赖。

## 实现及适用范围

每轮 NS 中 `A = X @ X.T`、`B = A @ A` 可使用 Ascend C；后续矩形乘法仍为原生 GEMM。
对称算子只计算一侧三角并转置写回另一侧。Muon 保留原来的独立 BF16 系数乘法、
加法和预分配缓冲区布局，不调用通用算子的系数融合 epilogue。

静态选择使用 reshape、合批后的实际输入。对于 `[B,N,K] @ [B,K,N]`，
当原生库已安装、输入为 NPU FP16/BF16、`N > 256`，且
`2*B*N*N*K >= 20_000_000_000` 时选择 Ascend C，其余走原生 matmul。
没有在线测速、调参或按张量数值触发的设备同步。小矩阵即使打开开关，也可能没有加速。

当前完整模型验收范围为 8 张 Ascend 910B3、CANN 8.5.0、Torch/torch_npu 2.9.0，
BF16 参数/梯度/动量、五轮 asym5、关闭 K-shift。使用真实模型规格和合成参数、梯度，
测完整 Muon step 的通信、动量与更新，不含前后向、先前的梯度归约或 AdamW。

逐 bit 一致是已测 shape、输入、dtype 和软件栈的实测结果，不是任意矩阵的数学保证：
某些原生 GEMM 的 `X @ X.T` 本身可能不逐 bit 对称。换环境或形状后应重新验证。
本次不覆盖其他 Ascend SoC、FP16 优化器训练、其他 NS 系数和长期训练收敛。

本版本定位为 eager `optimizer.step()`；将其放在模型编译区域之外。
模型前后向的编译不要求编译 optimizer step。该 ctypes 原生算子的完整图编译、
graph capture 和 replay 未验证，不应把这部分也包进 full-graph 编译。
