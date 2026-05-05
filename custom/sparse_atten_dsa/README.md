# DeepSeek V4 Sparse Attention (DSA) — PTO kernel

Reference implementation of the DeepSeek V4 / V3.2-Exp **DeepSeek Sparse
Attention (DSA)** compute path, written in PTO-Auto style on top of the
PTO Tile intrinsics.

## Algorithm

DSA replaces dense softmax attention with two stages:

1. **Lightning Indexer** *(separate kernel, not implemented here)*: a cheap
   scoring pass that, for each query block, selects `TOP_K` key/value blocks
   to attend to. Output: `block_indices : [num_q_blocks, TOP_K]` of `int32`.
2. **Sparse attention compute** *(this kernel)*: standard scaled-dot-product
   attention with FlashAttention-2 streaming softmax, but the inner KV loop
   walks the selected blocks listed in `block_indices` instead of all KV
   blocks.

Per query block:

```
m = -inf;  l = 0;  O = 0
scale = 1 / sqrt(HEAD)
load Q

for t in 0..TOP_K-1:
    blk = block_indices[q_block, t]
    if blk < 0: continue                      # sentinel: unused slot
    load K[blk], V[blk]
    X = scale * (Q @ K^T)                     # [BQ, BK]  fp32 acc
    m_new = max(m, rowmax(X))
    P     = exp(X - m_new)                    # [BQ, BK]  fp32 → fp16 cast
    alpha = exp(m - m_new)                    # [BQ]
    l     = alpha * l + rowsum(P)
    O     = alpha * O + (P @ V)               # [BQ, HEAD] fp32 acc
    m     = m_new

O = O / l
store O
```

`alpha` is the running rescale factor that keeps the streaming softmax
numerically stable across blocks (FA-2). The first selected block takes the
init path so we don't multiply by an undefined `m`/`l`/`O`.

## Mapping to the PTO programming model

References below point at the docs that informed each choice.

| Concept | Mapping in this kernel | Doc |
|---|---|---|
| Execution model | SPMD: one core per query block (`get_block_idx()`); `kNumQBlocks = S_Q/BQ` cores in the launch grid | [coding/ProgrammingModel.md](../../../docs/coding/ProgrammingModel.md) |
| Style | **PTO-Auto** — direct `TLOAD → compute → TSTORE` dataflow, no `TASSIGN`, no manual flags | [auto_mode/Auto_Mode_Overview.md](../../../docs/auto_mode/Auto_Mode_Overview.md) |
| Q / K / V tiles | `Tile<TileType::Mat, __fp16, ...>` with boxed (`SLayout`) layouts so they're legal operands for `TMATMUL` | [coding/Tile.md](../../../docs/coding/Tile.md) |
| Accumulators | `TileAcc<float, BQ, BK>` for QK and `TileAcc<float, BQ, HEAD>` for PV | [coding/Tile.md](../../../docs/coding/Tile.md) |
| Per-row state (`m`, `l`, `alpha`) | `Tile<TileType::Vec, float, BQ, 1, BLayout::ColMajor, BQ, 1>` reduce tiles | [coding/Tile.md](../../../docs/coding/Tile.md) |
| GM views of Q / K / V / O | 5-D `GlobalTensor<...>` with the leading three dims set to 1 | [coding/GlobalTensor.md](../../../docs/coding/GlobalTensor.md) |
| Sparse KV indexing | Per-iteration scalar GM read of `block_indices[q_block*TOP_K + t]`, then a fresh `GlobalTensor` view at offset `kv_blk * BK * HEAD`. No `MGATHER` needed because each tile is a contiguous block. | [isa/MGATHER.md](../../../docs/isa/MGATHER.md) (only used for elementwise gather; block-level indexing is just pointer arithmetic) |
| Online softmax math | `TROWMAX`, `TROWEXPANDSUB`, `TEXP`, `TROWSUM`, `TMAX`, `TSUB`, `TMUL`, `TADD`, `TROWEXPANDMUL`, `TROWEXPANDDIV` — same instruction palette as the dense flash_atten softmax helper | [isa/README.md](../../../docs/isa/README.md), [kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp](../../manual/common/flash_atten/pto_macro_fa_softmax.hpp) |
| `__PTO_AUTO__` rule compliance | Loop-invariant guards live outside the inner loop; the per-iteration skip uses a single-expression `bool skip` evaluated once before the `if`; no `TASSIGN`, no `set_flag`/`wait_flag` | [auto_mode/Kernel_Developer_Rules_And_Limitations.md](../../../docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md) §1.1, §1.3, §3.3 |

## What's intentionally left out

- **Lightning Indexer**: out of scope for this kernel; assumed upstream.
- **Causal masking**: trivially added if needed (apply `TTRI` + `TMULS(-inf)`
  + `TADD` to `qk_acc` when `kv_blk*BK <= q_block*BQ + i`). Omitted here to
  keep the sparse path readable.
- **Pipelined Cube/Vec MPMD**: the dense flash_atten in
  [`kernels/manual/common/flash_atten`](../../manual/common/flash_atten) shows
  what a fully pipelined version looks like with explicit FIFOs, ping-pong
  L1 buffers, and `TSyncCVID`-driven cross-core handoff. The DSA hot path
  here is written for clarity in PTO-Auto; the same Cube/Vec split applies
  identically once the sparse loop body is settled.
- **Sliding-window / compressed components** of NSA-family designs: those
  are additional attention paths fused with the selected-blocks path; this
  kernel implements the selected-blocks path only.

## Files

- `sparse_atten_dsa_kernel.h` — host-facing launch wrapper signature.
- `sparse_atten_dsa_kernel.cpp` — kernel + softmax helper + explicit template
  instantiations for common DeepSeek V4 configs (`HEAD=128`, `BQ=BK=64`,
  `TOP_K=64`).

## Suggested configurations

| `S_Q`  | `S_KV`  | `HEAD` | `BQ` | `BK` | `TOP_K` | Notes |
|--------|---------|--------|------|------|---------|-------|
| 4096   | 16384   | 128    | 64   | 64   | 64      | small reference |
| 8192   | 32768   | 128    | 64   | 64   | 64      | mid-size |
| 16384  | 65536   | 128    | 64   | 64   | 64      | long-context regime where DSA wins most |

`BK = 64` matches the indexer block size used in the DeepSeek V3.2-Exp
report; `TOP_K = 64` keeps roughly `O(S_Q * TOP_K * BK) = O(S_Q * 4096)`
attention work per layer regardless of `S_KV`, which is the whole point of
DSA at long context.
