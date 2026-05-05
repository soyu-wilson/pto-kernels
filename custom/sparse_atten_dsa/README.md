# DeepSeek V4 Sparse Attention (DSA) — Ascend 910B PTO kernel

PTO implementation of the DeepSeek V4 / V3.2-Exp **DeepSeek Sparse Attention
(DSA)** compute path, mapped to the Ascend 910B AI Core split architecture
(reference: [arXiv:2505.15112](https://arxiv.org/html/2505.15112v1)).

## Algorithm

DSA replaces dense softmax attention with two stages:

1. **Lightning Indexer** *(separate kernel, not implemented here)*: a cheap
   scoring pass that, for each query block, selects `TOP_K` KV blocks to
   attend to. Output: `block_indices : [num_q_blocks, TOP_K]` of `int32`.
2. **Sparse attention compute** *(this kernel)*: standard scaled-dot-product
   attention with FlashAttention-2 streaming softmax, but the inner KV loop
   walks the selected blocks listed in `block_indices` instead of all KV
   blocks. Per query block:

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

## Mapping to the Ascend 910B AI Core

Each AI Core on 910B is a **1 Cube + 2 Vector** composite (1:2 ratio). The
two halves have disjoint local memories and can only exchange data through
GM (or, on-chip, through L2-cached fast paths exposed as the same API).

This kernel runs as MPMD on a single AI Core, with the same source compiled
twice — once into the Cube image and once into the Vec image — and the
`__DAV_CUBE__` / `__DAV_VEC__` predicates select which half runs which work
at compile time.

### Work assignment

| Stage | Hardware | PTO instructions |
|---|---|---|
| Load Q (once) and K[blk] per iteration | Cube | `TLOAD` (Mat tile in L1) |
| Q @ K^T → fp32 [BQ, BK] | **Cube** | `TMATMUL` (`TileLeft @ TileRight^T → TileAcc`) |
| FA-2 streaming softmax: rowmax, exp, rowsum, alpha rescale | **Vec** (×2) | `TROWMAX`, `TROWEXPANDSUB`, `TEXP`, `TROWSUM`, `TMAX`, `TSUB`, `TMUL`, `TADD` |
| P fp32 → fp16 cast | Vec | `TCVT` |
| Load V[blk]; P @ V → fp32 [BQ, HEAD] | **Cube** | `TLOAD`, `TMATMUL` |
| Streaming O update: O = α·O + PV; final O /= l | **Vec** (×2) | `TROWEXPANDMUL`, `TADD`, `TROWEXPANDDIV` |
| Store O | Vec | `TSTORE` |

### Vector subblock partitioning (1:2 ratio)

`BQ` query rows are split row-wise across the two Vec subblocks:
`BQ_sub = BQ / 2`, with `get_subblockid()` choosing 0 or 1. Each subblock
maintains its own `m_run`, `l_run`, `alpha`, and resident `o_run` UB tiles
covering its row range. Since softmax + GU happen on the same vec subblock,
the per-row state never crosses cores — only Cube↔Vec **tile** data does.

The `TPipe` consumer side declares `TileSplitAxis::TILE_UP_DOWN`, which is
exactly the primitive in [docs/isa/TALLOC.md](../../../docs/isa/TALLOC.md)
for "vector subblocks map to row halves."

### Cross-core data movement

All Cube↔Vec data flow goes through GM-backed ring FIFOs declared as
`TPipe`s and accessed with `TALLOC` / `TPUSH` (producer) and `TPOP` /
`TFREE` (consumer):

```
              direction        slot size              role
QKPipe        Cube → Vec       BQ * BK * f32          QK = Q @ K^T to be softmaxed
PPipe         Vec  → Cube      BQ * BK * f16          P = softmax(QK), input to PV matmul
PVPipe        Cube → Vec       BQ * HEAD * f32        PV = P @ V, to be folded into O
```

Slot size is the full BQ-row tile; the consumer picks its subblock's row
half via `TILE_UP_DOWN` so the producer doesn't need to know how many vec
subblocks consume it. Producer/consumer ready notifications are driven by
the **FFTS** sync mechanism (`set_ffts_base_addr` at kernel entry, distinct
`FlagID`s per pipe).

### Sparse loop on both cores

Both Cube and Vec read `block_indices[q_block * TOP_K + t]` each iteration
and skip in lockstep on `kv_blk < 0`. Because both halves walk the same
selection sequence, the FIFOs stay balanced — no extra control message
needs to flow when a slot is masked.

### What we did *not* do here (room for further perf tuning)

The dense [`kernels/manual/common/flash_atten`](../../manual/common/flash_atten)
reference layers several perf optimizations on top of this same Cube/Vec
split. They are intentionally omitted here so the architecture mapping
stays readable; each is a knob the kernel can adopt with minimal logic
change:

- **Pipeline preload** of `QK_PRELOAD` tiles before the steady-state loop,
  to fill the FIFOs and hide the first iteration's matmul latency.
- **Ping-pong L1 / L0 buffers** for K, P, V, and the QK/PV accumulators
  (`L0A_BUF0/1`, `L0C_BUF0/1`).
- **Sub-tile factor** (`Tile_S1 / Cube_S1`) so a single QK FIFO entry
  represents multiple cube tiles, amortising notification cost.
- **Causal masking**: trivially added by ORing a `TTRI`-generated upper-tri
  mask × `-inf` into `qk_acc` when `kv_blk * BK <= q_block * BQ + i`.
- **Lightning Indexer** itself — out of scope for this kernel.

## Files

- `sparse_atten_dsa_kernel.h` — host-facing launch wrapper signature
  (FFTS pointer + three FIFO buffers in addition to Q/K/V/indices/O).
- `sparse_atten_dsa_kernel.cpp` — the kernel: four `compute_*` helpers
  (one per pipeline stage) plus the SPMD entry point that dispatches them
  via `__DAV_CUBE__` / `__DAV_VEC__`.
- Shares the softmax helper (`pto_macro_fa_softmax.hpp`) and matmul helper
  (`pto_macro_matmul.hpp`) and GU helper (`pto_macro_fa_gu.hpp`) with the
  dense flash_atten reference.

## Suggested configurations

| `S_Q`  | `S_KV`  | `HEAD` | `BQ` | `BK` | `TOP_K` | Notes |
|--------|---------|--------|------|------|---------|-------|
| 4096   | 16384   | 128    | 64   | 64   | 64      | small reference |
| 8192   | 32768   | 128    | 64   | 64   | 64      | mid-size |
| 16384  | 65536   | 128    | 64   | 64   | 64      | long-context regime where DSA wins most |

`BK = 64` matches the indexer block size used in the DeepSeek V3.2-Exp
report; `TOP_K = 64` keeps roughly `O(S_Q · TOP_K · BK) = O(S_Q · 4096)`
attention work per layer regardless of `S_KV`, which is the whole point of
DSA at long context.
