# Ascend C symmetric GEMM for Muon

This feature accelerates **standard Newton–Schulz (NS)** inside Muon. It does not
change the NS algorithm, coefficients, momentum, weight decay or communication.
The default remains `use_symm_gemm=False`.

The intended training setup is an eager `optimizer.step()` on Ascend NPU. Keep
that step outside a compiled model region; full-graph compilation and graph
capture of this ctypes-based optimizer kernel are not validated. Compiling the
model's forward/backward does not require compiling its optimizer step.

## 1. Cherry-pick the feature

The feature branch contains one commit on top of upstream master. It includes the
operator, Python dispatch, Muon switch, automatic build support, tests and docs.
There is no dependency on the archived experimental branch.

```bash
# Add this remote once; omit the command if it already exists.
git remote add townwish4git git@github.com:townwish4git/hyper-parallel.git
git fetch townwish4git feat/muon-symm-gemm
git cherry-pick FETCH_HEAD
```

Use a recent HyperParallel master. If your training branch predates the current
Muon advanced-options/parameter-group interfaces, resolve those API conflicts
before installing. Do not copy only the Python file: the native sources and
packaging changes are required.

## 2. Build and install on the training host

Use the same Python environment, CANN installation and SoC family that will run
training. The compiler toolkit (including `ascendc.cmake`), C++ build tools and
CMake >= 3.16 must be installed. A driver/runtime-only container cannot compile
this operator. Torch and the matching torch_npu should already be installed.

```bash
# Replace this with the set_env.sh belonging to your installed CANN toolkit.
source /usr/local/Ascend/cann/set_env.sh
python -m pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
```

`pip install -e .` automatically builds the Ascend C libraries when CANN is
available. PEP 517 build isolation is supported; no manual CMake command,
`PYTHONPATH`, or first-call JIT compilation is required. The import path must
point to this checkout. Restart existing training workers after reinstalling.

A wheel can instead be built from the same source and installed on equivalent
training hosts:

```bash
wheel_dir=$(mktemp -d)
python -m pip wheel . --no-deps -w "$wheel_dir"
python -m pip install --force-reinstall --no-deps "$wheel_dir"/*.whl
```

Installing a wheel loads its already compiled libraries; it does not recompile
on the destination. Match the Python/platform tag, CANN ABI and Ascend SoC of the
build environment. The native build queries the full device SoC identifier; it
does not hard-code 910B as a fallback. For a compiler-only build machine, set
`HP_ASCEND_SOC_VERSION` to the exact target identifier supported by your CANN
compiler. A successful build for another SoC is not a performance or precision
validation for that device.

Build controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HP_BUILD_ASCENDC` | `auto` | Build when CANN is found; `on` requires it; `off` explicitly skips it |
| `ASCEND_HOME_PATH` / `ASCEND_CANN_PACKAGE_PATH` | Toolkit discovery | Select the compiler toolkit |
| `HP_ASCEND_SOC_VERSION` | Query runtime | Override the full target SoC for cross-compilation |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `4` | Parallel native compilation jobs |

On a host without CANN, installation remains possible and automatic dispatch
uses native framework matmul. If CANN is detected but compilation fails, the
installation fails rather than silently reporting a successful accelerated
build. Use `HP_BUILD_ASCENDC=on` in deployment CI when native support is required.

## 3. Set the precision environment before starting Python

```bash
export CLOSE_MATMUL_K_SHIFT=1
# Export in every worker environment, including remote/container launchers.
torchrun <your existing training arguments>
```

The value **1 closes K-shift**. It must be set before Torch/CANN initialization,
not after the first matmul or only in an interactive parent shell that workers
do not inherit. HyperParallel does not mutate this process-wide environment
variable when the optimizer is constructed.

Use the same setting for both the baseline and accelerated run. CANN K-shift
changes accumulation order; comparing the accelerated operator against CANN with
K-shift enabled is not the validated bitwise comparison.

## 4. Enable the optimizer switch

```python
from hyper_parallel.core.optimizer.muon import Muon

optimizer = Muon(
    muon_parameters,
    lr=0.01,  # Keep the existing training hyperparameters.
    use_symm_gemm=True,
)
```

With the mixed Muon/AdamW builder, put `muon_use_symm_gemm: true` in the existing
`muon_kwargs` or `muon_config` mapping. The normalized spelling
`use_symm_gemm: true` is also accepted. This does not move embedding, norm or
other AdamW parameters into Muon.

```yaml
muon_config:
  muon_use_symm_gemm: true
  # Keep the rest of your existing Muon settings.
```

The option is stored per parameter group and in optimizer checkpoints. Loading
an old checkpoint, or one saved with the switch disabled, can replace constructor
group settings. To enable it while resuming, set it **after** loading optimizer
state on the Muon optimizer instance:

```python
optimizer.load_state_dict(checkpoint["optimizer"])
for group in optimizer.param_groups:
    group["use_symm_gemm"] = True
```

For a mixed/chained optimizer, apply this to its Muon leaf, not the AdamW leaf.
An explicit `zeropower_fn` callback continues to override the built-in NS path;
remove that override if you intend to use this implementation. The option also
passes through reshape/batching and reversible NS input transforms.

To disable the feature, set `use_symm_gemm=False` on every Muon group. The default
path retains the original `torch.matmul(..., out=...)` implementation.

## 5. Verify installation before a long training run

First verify that the imported package contains all three native libraries:

```python
from pathlib import Path
import hyper_parallel

root = Path(hyper_parallel.__file__).parent
folder = root / "components/functional/_ascendc/symmetric_matmul"
for name in ("libhp_symmetric.so", "libhp_epilogue_kernel.so", "libhp_fused_kernel.so"):
    assert (folder / name).is_file(), folder / name
print(root)
```

Then exercise the installed native backend and standard NS on the actual NPU:

```python
# Start this process with CLOSE_MATMUL_K_SHIFT=1 already exported.
import torch
import torch_npu
from hyper_parallel.components.functional.symmetric_matmul import symmetric_matmul
from hyper_parallel.core.optimizer.muon import zeropower_via_newtonschulz5

torch.npu.set_device(0)
torch.manual_seed(42)
with torch.no_grad():
    x = torch.randn(2048, 4096, device="npu", dtype=torch.bfloat16)
    native = symmetric_matmul(x, x.mT, backend="ascendc")
    reference = torch.matmul(x, x.mT)
    assert torch.equal(native.view(torch.int16), reference.view(torch.int16))
    base = zeropower_via_newtonschulz5(x, 5)
    fast = zeropower_via_newtonschulz5(x, 5, use_symm_gemm=True)
    assert torch.equal(base.view(torch.int16), fast.view(torch.int16))
torch.npu.synchronize()
```

For the repository smoke test, which also checks current-stream execution,
unaligned output guards, static routing, five optimizer steps and checkpoint
restoration:

```bash
CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 python \
  tests/torch/functional/symmetric_matmul_smoke.py \
  --expected-root "$PWD" --output /tmp/symmetric-muon-smoke.json
```

For a wheel validation, run that script by its absolute filename from outside the
checkout and set `--expected-root` to the installed environment's `site-packages`
directory. This prevents a checkout import from masking a broken wheel.

## 6. What changes, and what does not

Each standard NS iteration computes `A = X @ X.T`, `B = A @ A`, combines the
coefficients, then computes a rectangular update. Only the first two symmetric
products can use Ascend C. The kernel computes one triangle and mirrors it;
CANN computes both triangles independently. Muon keeps the original separate
BF16 coefficient multiplies/addition and preallocated output layouts. It does
not use the generic operator's fused coefficient epilogue for NS.

The static selector uses the actual matrix after reshape and batching. For
`[B,N,K] @ [B,K,N]`, Ascend C is selected only when native libraries are installed,
tensors are NPU FP16/BF16, `N > 256`, and `2*B*N*N*K >= 20_000_000_000`. Otherwise
it uses native matmul. There is no online timing, calibration callback or
value-dependent device synchronization in selection. Small matrices can therefore
show no speedup even when the switch is enabled.

The standalone `symmetric_matmul` interface requires a mathematically symmetric
product and is forward-only. It is not a general GEMM replacement. Its optional
`alpha`/`beta` epilogue has different rounding from separate low-precision
operations; that epilogue is not enabled by the Muon switch.

## 7. Precision and performance acceptance

The validated setup is CANN 8.5.0, Torch/torch_npu 2.9.0 and 8 x Ascend 910B3,
BF16 parameters/gradients/momentum, five-round asym5, with K-shift closed.
The full-model benchmark uses the Qwen3-30B-A3B specifications and the language
body of Qwen3.5-27B, with deterministic synthetic weights and gradients. It
measures `Muon.step` communication, momentum and parameter updates, excluding
forward/backward, previous gradient reduction and AdamW.

Run both configurations through the production flag:

```bash
CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  -m tests.torch.optimizer.benchmark_muon_models \
  --model qwen3-30b-a3b --warmup 2 --steps 5 --cycles 2 \
  --accuracy-steps 1 5 --metric-threads 16 --output /tmp/qwen3-symmetric.json

CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  -m tests.torch.optimizer.benchmark_muon_models \
  --model qwen3.5-27b --warmup 2 --steps 5 --cycles 2 \
  --accuracy-steps 1 5 --metric-threads 16 --output /tmp/qwen35-symmetric.json
```

The model-construction benchmark additionally needs Transformers 5.13.0; this is
not a new runtime dependency of Muon or the native operator. Use idle cards.
Each configuration starts from identical fresh state. Timings use the slowest
rank and two blocks in reversed order. Precision compares every parameter and
momentum element after 1 and 5 optimizer steps; effective update means final
parameter minus initial parameter, including BF16 writeback and weight decay.

Results and build evidence for this branch are recorded in the
[validation report](../performance/muon_symm_gemm/README.md).
Bitwise equality is an observed result for the tested shapes, dtype, inputs and
software stack, not a universal algebraic guarantee: even `X @ X.T` from native
GEMM can be non-bitwise-symmetric on other kernels. Validate a changed stack or
new shape distribution before relying on identical trajectories. Other Ascend
SoCs, FP16 optimizer training, alternate coefficient families, long training
convergence and graph capture are not covered by the full-model acceptance run.
