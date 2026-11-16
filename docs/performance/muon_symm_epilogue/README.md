# Muon FP32 polynomial epilogue experiment

This increment is based on `bcc231388eccdbc81e36c0bd7b22f4bfea223956` and keeps the
previous `feat/muon-symm-gemm` branch unchanged. No Gram algorithm is introduced.

## Implementation and numerical contract

Enable `Muon(..., use_symm_gemm=True, use_symm_gemm_epilogue=True)` to replace only
`A@A` and its three subsequent coefficient operations with the existing native
`alpha=c, beta=b, additive=A` symmetric epilogue. The polynomial is written into
the other preallocated square buffer; it never aliases the active input A.
The rectangular GEMM consumes that buffer. The first `X@X.T`, rectangular product,
final residual update, momentum, parameter update and communication are unchanged.

The new flag defaults to false, is forwarded through both batching paths and YAML
configuration, and is stored in optimizer parameter groups/checkpoints. It requires
`use_symm_gemm=True`. An explicit NS callback retains precedence. Existing static
Ascend C eligibility is unchanged; unsupported/small cases use native matmul and
separate input-dtype coefficient operations. No online calibration is introduced.

For eligible BF16 matrices, the polynomial changes from repeated BF16 rounding to:

```text
P = BF16(c * GEMM_FP32(A, A) + b * FP32(A))
```

This is deliberately different floating-point arithmetic. Closing K-shift does not
restore baseline bitwise equivalence. The native implementation already supports
this epilogue; no C++ kernel, tiling policy or build hook was changed this round.

## Experimental protocol

Host: t129, 8 x Ascend910B3, CANN 8.5.0, Torch 2.9.0+cpu, torch_npu 2.9.0.
All workers start with `CLOSE_MATMUL_K_SHIFT=1`, `OMP_NUM_THREADS=1`.
Use the full Qwen3-30B-A3B and Qwen3.5-27B language-model specifications with
synthetic BF16 parameters and gradients, FSDP sharding and the production Muon
constructor. There is no optimizer monkeypatch or replacement NS callback.

Compare CANN, symmetric GEMM, and symmetric GEMM plus epilogue in one run.
Each configuration has two blocks, in forward and reversed order. Each block
warms up for two steps and measures five steps. The reported median combines
all ten slowest-rank synchronized wall-time samples. Timed scope includes Muon's
communication, momentum and updates, excluding forward/backward, earlier gradient
reduction and AdamW. Hyperparameters: lr=0.01, weight_decay=0.1, momentum=0.95,
Nesterov enabled, five asym5 iterations.

Precision runs start from identical fresh states, with identical changing gradients.
Compare every Muon parameter shard and momentum element at one and five optimizer
steps against CANN. Effective update is final minus initial parameter, including
BF16 writeback and weight decay. At five steps it is cumulative displacement,
not the isolated fifth-step update. CPU statistics use 16 threads per rank;
optimizer execution retains one CPU thread. Differences from the BF16 baseline
are not errors against an exact orthogonalization oracle or a training-loss metric.
Momentum equality here uses replayed gradients; real training can feed parameter
differences back into subsequent gradients. Long-run convergence is not tested.

## Optimization reasoning

The upstream QuACK linear epilogue applies scaling and the additive input in the
accumulator dtype before output conversion. Its SM90 implementation also pipelines
loading the additive input and storing epilogue output. This supports the idea of
combining the polynomial with the second GEMM; GPU-specific TMA/WGMMA mechanisms
are not copied into the NPU implementation. Sources inspected on 2026-11-16:
[linear epilogue](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_default_epi.py),
[SM90 pipeline](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm90.py).

On NPU, the comparison must account for both saved elementwise launches/traffic
and the cost of FP32 GM scratch plus Cube/Vector coordination. This implementation
is not a register-only epilogue. The baseline direct path writes BF16; the fused
polynomial path uses FP32 scratch, so a reduction in operator count is not by
itself evidence of a speedup. Separate profiler runs compare the same five-round
NS function with all three production configurations on representative expert
and dense shapes. Profiled durations are diagnostic, not optimizer-step timings.

## Reproduction and evidence

- [Build, unit tests, smoke and full-model commands](run_t129.sh).
- Add `--include-epilogue` to `tests.torch.optimizer.benchmark_muon_models`.
- [Changed Python source hashes](source_hashes.json).
- After the full-model run, collect separate NS profiles on an idle card:

  ```bash
  CLOSE_MATMUL_K_SHIFT=1 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=0 python \
    -m tests.torch.optimizer.profile_muon_epilogue --output validation/profile
  ```

- Deployment and checkpoint details: [Chinese guide](../../guide/muon_symm_gemm.zh.md),
  [English guide](../../guide/muon_symm_gemm.md).

## Measured results

All three configurations were measured in the same run; percentages below use wall-time reductions.

| Model | CANN ms | Symmetric ms | Epilogue ms | Extra reduction vs symmetric | Total reduction vs CANN |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-30b-a3b | 994.77 | 930.70 | 911.24 | 2.09% | 8.40% |
| qwen3.5-27b | 3196.75 | 2982.47 | 2969.21 | 0.44% | 7.12% |

Relative L2 differences below compare the epilogue configuration with CANN. The symmetric-only
configuration remains bitwise identical at both checkpoints, so these are also its differences.

| Model | Steps | Parameter relative L2 | Effective-update relative L2 | Elements with different bit patterns | Max parameter abs difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-30b-a3b | 1 | 0.1846% | 1.8467% | 24.28% | 0.00048828 |
| qwen3-30b-a3b | 5 | 0.4037% | 1.1975% | 49.68% | 0.00195312 |
| qwen3.5-27b | 1 | 0.1823% | 1.8221% | 24.64% | 0.00048828 |
| qwen3.5-27b | 5 | 0.4034% | 1.1747% | 53.73% | 0.00195312 |

Momentum bit differences and nonfinite counts are zero for all configurations/checkpoints.

| Model | CANN peak GiB | Symmetric peak GiB | Epilogue peak GiB | Extra epilogue MiB vs symmetric |
| --- | ---: | ---: | ---: | ---: |
| qwen3-30b-a3b | 21.651 | 21.722 | 21.861 | 142.5 |
| qwen3.5-27b | 20.580 | 20.579 | 20.580 | 0.7 |

The epilogue is left default-off. It is a measured performance/numerical tradeoff,
not the previous bitwise-compatible optimization. The small FP64 formula check
does not establish that five-round NS or real training is more accurate.

Raw full-element results: [30B](qwen3-30b-a3b.json), [27B](qwen3.5-27b.json),
[summary](summary.json), [NPU smoke](smoke.json), [unit tests](ut.txt).

## Validation status

Automatic editable installation rebuilt the unchanged native implementation in an
isolated environment and succeeded. The installed-package smoke verified the
import path, original bitwise path, fused symmetry and input preservation. The
small FP64 polynomial reference rounded to BF16 matched the native fused output.
On `[32,768,2048]`, five-round fused NS had relative L2 difference 1.2364% from
unfused NS; this is measured before optimizer scaling and parameter writeback.
Targeted unit tests: 25 passed, 21 subtests passed; six existing marker warnings.
Both eight-rank model commands exited successfully. No nonfinite values were found.

## Kernel-level evidence

Each trace contains two complete five-round NS calls. CANN records 43 device kernels per NS,
symmetric-only records 48 (including five transpose copies), and epilogue records 33.
The epilogue removes exactly 15 standalone kernels per NS: ten scalar multiplies and five adds.
These are observed kernel counts, not estimates from Python calls. The profiler emitted a
stop-in-RECORD scheduling warning; all six traces were checked for both normalization calls,
all 30 GEMMs and the expected elementwise counts before using the data.

Independent, unprofiled NS wall times (representative shapes, not the full optimizer):

| Input shape | CANN ms | Symmetric ms | Epilogue ms |
| --- | ---: | ---: | ---: |
| [32, 768, 2048] | 6.420 | 6.376 | 6.251 |
| [1, 5120, 17408] | 50.048 | 43.234 | 42.881 |

Kernel durations below are summed over five rounds and averaged over the two NS calls.

| Shape class | CANN square GEMM ms | Symmetric square GEMM ms | Fused square polynomial ms | Standalone elementwise time saved ms |
| --- | ---: | ---: | ---: | ---: |
| expert | 0.699 | 0.565 | 0.740 | 0.559 |
| dense | 5.029 | 3.358 | 3.647 | 0.530 |

The fused square kernel itself is slower than the direct BF16 symmetric product: it now
also performs the polynomial and uses FP32 scratch. This offsets part of the saved
elementwise work. For the dense example, the removed pointwise work is only about 0.53 ms
out of a 43 ms NS call, and the square kernel grows by about 0.29 ms. This supports the
small incremental dense-model benefit. The unchanged rectangular product also varies
in the traces, so these differences are not isolated causal measurements of scratch cost.
These representative shapes do not reconstruct every actual model batching group.
Full-model wall-time measurements above remain the performance acceptance result.

Peak allocated memory is measured across the whole optimizer step. A small peak change
does not mean that the FP32 workspace is absent; another phase may set the peak.

Evidence: [raw kernel CSV archive](profile-selected.tar.gz),
[kernel summary and completeness checks](profile_summary.json), [NS wall times](ns_timings.json).
