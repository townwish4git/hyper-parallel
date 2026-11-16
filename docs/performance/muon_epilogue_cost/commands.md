# 运行记录与复现

日期：2026-11-16。主机 t129，Ascend910B3，设备 0。
生产代码基于 `feat/muon-symm-epilogue` 的 `3fde953e6b8a596f2a86c0659505c19279bdd71b`。
诊断通过原生 ABI 调用，不经过 Muon 的静态路由。

## 实际运行命令

在以下环境执行；原型在单独目录编译，不覆盖安装目录的动态库。

```bash
source /home/cann_custom_path/8.5.0/cann-8.5.0/set_env.sh
cd /home/townwish/workspace/projects/hp-epilogue-20261116
export CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=0
probe=/home/townwish/workspace/projects/hp-epilogue-prefetch-probe-20261116
python=/home/townwish/workspace/projects/hp-epilogue-env-20261116/bin/python

"$python" -m tests.torch.optimizer.diagnose_muon_epilogue --output validation/epilogue-cost
"$python" -m tests.torch.optimizer.diagnose_muon_epilogue --output validation/epilogue-cost-repeat --no-profile
"$python" -m tests.torch.optimizer.diagnose_muon_epilogue --output validation/epilogue-prefetch-probe --native-library-dir "$probe/build/lib"
"$python" -m tests.torch.optimizer.diagnose_muon_epilogue --output validation/epilogue-probe-reverse --native-library-dir "$probe/build/lib" --no-profile
"$python" -m tests.torch.optimizer.diagnose_muon_epilogue --output validation/epilogue-baseline-after --no-profile
```

五次进程均正常退出。最终两次加入同步墙钟计时，早期数据没有此字段。
初次 baseline 也尚未加入 polynomial SHA256 字段；其余四次都记录完整输出哈希。
保留每次实际运行结果，不用后来版本补写不存在的测量字段。

## 重建隔离原型

实际构建使用复制的源码和修改后的 epilogue.h。以下用已存档的等价补丁重建该目录；
`probe` 应选一个新的空目录，避免覆盖已有试验。

```bash
mkdir -p "$probe"
cp -a hyper_parallel/components/functional/_ascendc/symmetric_matmul "$probe/"
patch -d "$probe/symmetric_matmul" -p1 < docs/performance/muon_epilogue_cost/prefetch-probe.patch
cmake -S "$probe/symmetric_matmul" -B "$probe/build" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION=Ascend910B3 -DCMAKE_BUILD_TYPE=Release
cmake --build "$probe/build" --parallel 4
```

编译和两种形状的原生调用均通过。各次 JSON 的 `native_sha256` 保存实际加载的库哈希。
输出哈希、有限性、对称性通过，不代表已覆盖所有输入类型和分支。
本轮未重新运行完整模型 step；原型的应用性能尚未确认。

## 本地检查

```bash
git add docs/performance/muon_epilogue_cost tests/torch/optimizer/diagnose_muon_epilogue.py
PATH=/tmp/hp-symm-lint/bin:$PATH /tmp/hp-symm-lint/bin/python .agent/skills/autogit/scripts/autogit.py check
git diff --check
```

检查结果见本轮提交说明；该检查会自动取消暂存，提交前需重新暂存。
