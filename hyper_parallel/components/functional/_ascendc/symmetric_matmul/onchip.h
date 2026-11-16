// Copyright 2026 Huawei Technologies Co., Ltd
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
constexpr FixpipeConfig ONCHIP_FIXPIPE{CO2Layout::ROW_MAJOR, true};

struct OnchipOutputState {
  uint32_t sequence;
  uint32_t address;
};

__aicore__ inline void copy_onchip(const __gm__ void *, const LocalTensor<int8_t> &source, const void *parameters,
                                   uint64_t state_address, uint64_t) {
  auto &state = *reinterpret_cast<OnchipOutputState *>(state_address);
  auto &metadata = *reinterpret_cast<const matmul::DataCopyOutParams *>(parameters);
  auto input = source.ReinterpretCast<float>();
  LocalTensor<float> output;
  TBuffAddr address{};
  address.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
  address.bufferAddr = state.address;
  address.dataLen = 2 * 64 * 256 * sizeof(float);
  output.SetAddr(address);
  // A full FP32 tile is 256 KiB. Splitting it twice gives each AIV a 64 KiB
  // strip, so two receive slots and the epilogue workspace fit in its 248 KiB UB.
  for (int part = 0; part < 2; ++part, ++state.sequence) {
    int slot = state.sequence % 2;
    if (state.sequence >= 2) CrossCoreWaitFlag<0x2>(8 + slot);
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> transfer;
    transfer.nSize = 256;
    transfer.mSize = 128;
    transfer.srcStride = 256;
    transfer.dstStride = 256;
    transfer.dualDstCtl = 1;
    // Only the final strip may release the Matmul accumulator for reuse.
    transfer.unitFlag = metadata.enUnitFlag ? (part == 0 ? 2 : 3) : 0;
    Fixpipe<float, float, ONCHIP_FIXPIPE>(output[slot * 64 * 256], input[part * 128 * 16], transfer);
    CrossCoreSetFlag<0x2, PIPE_FIX>(6 + slot);
  }
}

template <class T>
__aicore__ inline void product_onchip(GM_ADDR a, GM_ADDR b, GM_ADDR out, SymmetricPlan &plan) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> received;
  // Both core types reserve this buffer first to agree on its UB address.
  pipe.InitBuffer(received, 2 * 64 * 256 * sizeof(float));
  OnchipOutputState state{0, static_cast<uint32_t>(received.Get<float>().GetPhyAddr())};
  using Left = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;         // codespell:ignore nd
  using Right = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;  // codespell:ignore nd
  using Result = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;   // codespell:ignore nd
  using Callback = matmul::MatmulCallBackFunc<copy_onchip>;
  matmul::MatmulImpl<Left, Right, Result, Result, CFG_MDL, Callback> mm;
  TCubeTiling tiling;
  auto *words = reinterpret_cast<uint32_t *>(&tiling);
  for (int i = 0; i < sizeof(tiling) / sizeof(uint32_t); ++i) words[i] = plan.tiling[i];
  mm.SetSubBlockIdx(0);
  mm.Init(&tiling, &pipe);
  mm.SetUserDefInfo(reinterpret_cast<uint64_t>(&state));
  GlobalTensor<T> left, right;
  GlobalTensor<float> unused;
  left.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a));
  right.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b));
  // The callback replaces the GM store; this address is never dereferenced.
  unused.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
  for (int task = GetBlockIdx(); task < task_count<true>(plan); task += GetBlockNum()) {
    auto region = task_region<true>(plan, task);
    produce_region(mm, left, right, unused, plan, region);
  }
  for (int slot = 0; slot < 2 && slot < state.sequence; ++slot) CrossCoreWaitFlag<0x2>(8 + slot);
}

template <class T>
class OnchipConsumer {
 public:
  __aicore__ OnchipConsumer(TPipe &pipe, GM_ADDR c, GM_ADDR out, SymmetricPlan &plan, float alpha, float beta)
      : plan_(plan), alpha_(alpha), beta_(beta) {
    pipe.InitBuffer(values, tile * tile * sizeof(float));
    pipe.InitBuffer(additive, tile * tile * sizeof(float));
    pipe.InitBuffer(rounded, tile * tile * sizeof(T));
    pipe.InitBuffer(transposed, tile * tile * sizeof(T));
    pipe.InitBuffer(selection, tile * tile / 8);
    initialize_diagonal_mask(selection);
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(c));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
  }

  __aicore__ void process(LocalTensor<float> input, const TaskRegion &region, int stage) {
    auto v = values.Get<float>();
    auto add = additive.Get<float>();
    auto r = rounded.Get<T>();
    auto tr = transposed.Get<T>();
    int row = region.row + stage * 128 + GetSubBlockIdx() * tile;
    for (int offset = 0; offset < 256; offset += tile) {
      int col = region.col + offset;
      if (row < col) continue;
      int64_t lower = static_cast<int64_t>(region.batch) * plan_.n * plan_.n + row * plan_.n + col;
      int64_t upper = static_cast<int64_t>(region.batch) * plan_.n * plan_.n + col * plan_.n + row;
      DataCopy(v, input[offset], {tile, tile * 4 / 32, (256 - tile) * 4 / 32, 0});
      if (beta_ != 0)
        DataCopy(r, cg[lower], {tile, tile * 2 / 32, static_cast<uint16_t>((plan_.n - tile) * 2 / 32), 0});
      PipeBarrier<PIPE_ALL>();
      combine_epilogue(v, add, r, plan_, alpha_, beta_);
      PipeBarrier<PIPE_ALL>();
      transpose_tile(r, tr);
      if (row == col) {
        Select(r.template ReinterpretCast<half>(), selection.Get<uint8_t>(), r.template ReinterpretCast<half>(),
               tr.template ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, tile * tile);
      }
      PipeBarrier<PIPE_ALL>();
      DataCopy(dst[lower], r, {tile, tile * 2 / 32, 0, static_cast<uint16_t>((plan_.n - tile) * 2 / 32)});
      if (row != col)
        DataCopy(dst[upper], tr, {tile, tile * 2 / 32, 0, static_cast<uint16_t>((plan_.n - tile) * 2 / 32)});
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  static constexpr int tile = 64;
  SymmetricPlan &plan_;
  float alpha_, beta_;
  TBuf<TPosition::VECCALC> values, additive, rounded, transposed, selection;
  GlobalTensor<T> cg, dst;
};

template <class T>
__aicore__ inline void epilogue_onchip(GM_ADDR c, GM_ADDR out, SymmetricPlan &plan, float alpha, float beta) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> received;
  pipe.InitBuffer(received, 2 * 64 * 256 * sizeof(float));
  OnchipConsumer<T> consumer(pipe, c, out, plan, alpha, beta);
  for (int task = GetBlockIdx() / 2; task < task_count<true>(plan); task += plan.cores) {
    auto region = task_region<true>(plan, task);
    for (int stage = 0; stage < 2; ++stage) {
      CrossCoreWaitFlag<0x2>(6 + stage);
      consumer.process(received.Get<float>()[stage * 64 * 256], region, stage);
      CrossCoreSetFlag<0x2, PIPE_MTE3>(8 + stage);
    }
  }
}
#endif
