/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// =============================================================================
// DeepSeek V4 (DSA) sparse attention — Ascend 910B Cube+Vec MPMD kernel.
//
// Hardware model (Ascend 910B / A2/A3):
//   Each AI Core is a composite of 1 Cube unit and 2 Vector subblocks (1:2).
//   - Cube: dense matmul / cube-shaped tensor ops (TMATMUL, TGEMV) on tiles
//     bound to L1 (`TileType::Mat`) and L0A/L0B/L0C (`Left`/`Right`/`Acc`).
//   - Vec : SIMD elementwise + reductions (TADD/TMUL/TROWMAX/TEXP/...) on
//     `TileType::Vec` tiles in UB. Two subblocks share the same Vec ISA;
//     `get_subblockid()` picks 0 or 1 and partitions row work between them.
//   - The two units cannot directly read each other's local memories, so
//     all Cube<->Vec data exchange goes through GM-backed ring FIFOs (TPipe)
//     written/read with TSTORE/TLOAD against TALLOC/TPOP'd slot views.
//
// Dataflow for sparse attention (per query block, per selected KV block t):
//
//        Cube                                          Vec
//        ----                                          ---
//        TLOAD K[blk]
//        TMATMUL qk_acc = Q @ K^T   (fp32 [BQ, BK])
//        TSTORE qk_acc -> qk_fifo
//        TPUSH qkPipe ----------------------> TPOP qkPipe
//                                              TLOAD QK from slot
//                                              FA-2 softmax: update m, l, alpha;
//                                              produce P = exp(scaled X - m_new)
//                                              TCVT P -> fp16
//                                              TSTORE P -> p_fifo
//                              <--------------- TPUSH pPipe
//        TPOP pPipe
//        TLOAD V[blk]
//        TLOAD P from slot
//        TMATMUL pv_acc = P @ V    (fp32 [BQ, HEAD])
//        TSTORE pv_acc -> pv_fifo
//        TPUSH pvPipe ----------------------> TPOP pvPipe
//                                              TLOAD PV from slot
//                                              O = alpha * O + PV   (rescale)
//                                              On last selected block: O = O / l
//                                              TSTORE O
//
// Each Vec subblock owns BQ/2 rows; m, l, alpha, and the resident O buffer
// live in that subblock's UB across all TOP_K iterations. The exp_max
// rescale factor is therefore *not* shipped between cores — only QK/P/PV
// tiles cross the Cube<->Vec boundary.
//
// Sparse loop: both Cube and Vec read `block_indices[q_block * TOP_K + t]`
// each iteration and skip in lockstep when the entry is < 0, so the FIFOs
// stay balanced without the producer needing to notify the consumer about
// skipped slots.
//
// Programming style: PTO-Manual (TASSIGN, TPipe, explicit pipeline).
// References:
//   docs/coding/ProgrammingModel.md  -- SPMD/MPMD, Auto vs Manual
//   docs/coding/Tile.md              -- TileType, BLayout, fractal SLayout
//   docs/coding/GlobalTensor.md      -- 5-D shape/stride model
//   docs/isa/TALLOC.md / TPUSH.md    -- TPipe FIFO primitives
//   docs/isa/TMATMUL.md              -- Cube matmul (Left x Right -> Acc)
//   docs/isa/TROWMAX.md / TEXP.md    -- Vec softmax building blocks
//   kernels/manual/common/flash_atten -- dense FA reference (this kernel
//                                        reuses its softmax/GU helpers)
// =============================================================================

#include <acl/acl.h>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

#include "sparse_atten_dsa_kernel.h"
#include "../../manual/common/flash_atten/pto_macro_matmul.hpp"
#include "../../manual/common/flash_atten/pto_macro_fa_softmax.hpp"
#include "../../manual/common/flash_atten/pto_macro_fa_gu.hpp"

using namespace pto;

// 1:2 cube:vec ratio on a 910B AI Core.
constexpr int VEC_CORES = 2;

// FFTS buffer-flag IDs for the three CV pipes. Distinct values are required
// so the runtime can route producer/consumer ready notifications correctly.
enum DsaPipeFlag : uint32_t
{
    QK_PIPE_FLAG = 0,
    P_PIPE_FLAG  = 2,
    PV_PIPE_FLAG = 4,
};

// Detect the active half of the AI Core at compile time. The same kernel
// source compiles twice (once for cube, once for vec); these flags pick
// which side runs which work without any runtime branching.
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif
#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

namespace {

constexpr AICORE inline float ConstexprSqrt(float x)
{
    if (x <= 0.0f) return 0.0f;
    float g = x;
    for (int i = 0; i < 8; ++i) g = 0.5f * (g + x / g);
    return g;
}
constexpr AICORE inline float ConstexprInvSqrt(float x) { return 1.0f / ConstexprSqrt(x); }

// ---------------------------------------------------------------------------
// CUBE: QK matmul for one selected KV block.
//
// Reads K[blk*BK : (blk+1)*BK, :HEAD] from GM, computes Q @ K^T into the
// L0C accumulator, stores the fp32 result into the QK FIFO slot, and
// publishes the slot to the Vec consumer with TPUSH.
// ---------------------------------------------------------------------------
template <int BQ, int BK, int HEAD, typename QKPipe, typename QMatT, typename KMatT, typename QKAccT,
          typename QKSlotG>
AICORE inline void compute_qk_cube(QKPipe &qkPipe, __gm__ half *k, int kv_blk, QMatT &qMat, KMatT &kMat,
                                   QKAccT &qkAcc, QKSlotG &qkSlot)
{
    using KGlobal = GlobalTensor<half, Shape<1, 1, 1, HEAD, BK>, Stride<1, 1, 1, 1, HEAD>, Layout::DN>;
    KGlobal kGlobal(k + static_cast<size_t>(kv_blk) * BK * HEAD);

    TLOAD(kMat, kGlobal);

    // The dense flash_atten's pto_macro_matmul handles L0A/L0B ping-pong and
    // the M-pipe -> MTE1-pipe handshake; reusing it keeps the Cube path
    // architecturally identical to the dense reference.
    pto_macro_matmul<BQ, HEAD, BK>(qMat, kMat, qkAcc, AccMode::Init);

    // Allocate a producer slot in the Cube->Vec QK FIFO, store the fp32
    // accumulator into it, and commit. TILE_NO_SPLIT means the Cube writes
    // the whole [BQ, BK] tile; the Vec side splits it across subblocks.
    TALLOC<QKPipe, QKSlotG, TileSplitAxis::TILE_NO_SPLIT>(qkPipe, qkSlot);
    TSTORE(qkSlot, qkAcc);
    TPUSH<QKPipe, QKSlotG, TileSplitAxis::TILE_NO_SPLIT>(qkPipe, qkSlot);
}

// ---------------------------------------------------------------------------
// CUBE: PV matmul for one selected KV block.
//
// Pops the matching P tile from the Vec->Cube FIFO, loads V[blk], runs
// P @ V into an L0C accumulator, and pushes the PV result back to Vec.
// ---------------------------------------------------------------------------
template <int BQ, int BK, int HEAD, typename PPipe, typename PVPipe, typename PMatT, typename VMatT, typename PVAccT,
          typename PSlotG, typename PVSlotG>
AICORE inline void compute_pv_cube(PPipe &pPipe, PVPipe &pvPipe, __gm__ half *v, int kv_blk, PMatT &pMat,
                                   VMatT &vMat, PVAccT &pvAcc, PSlotG &pSlot, PVSlotG &pvSlot)
{
    using VGlobal = GlobalTensor<half, Shape<1, 1, 1, BK, HEAD>, Stride<1, 1, 1, HEAD, 1>>;
    VGlobal vGlobal(v + static_cast<size_t>(kv_blk) * BK * HEAD);
    TLOAD(vMat, vGlobal);

    // P arrives via the Vec->Cube pipe.
    TPOP<PPipe, PSlotG, TileSplitAxis::TILE_NO_SPLIT>(pPipe, pSlot);
    TLOAD(pMat, pSlot);
    TFREE<PPipe, PSlotG, TileSplitAxis::TILE_NO_SPLIT>(pPipe, pSlot);

    pto_macro_matmul<BQ, BK, HEAD>(pMat, vMat, pvAcc, AccMode::Init);

    TALLOC<PVPipe, PVSlotG, TileSplitAxis::TILE_NO_SPLIT>(pvPipe, pvSlot);
    TSTORE(pvSlot, pvAcc);
    TPUSH<PVPipe, PVSlotG, TileSplitAxis::TILE_NO_SPLIT>(pvPipe, pvSlot);
}

// ---------------------------------------------------------------------------
// VEC: FA-2 streaming softmax for one [BQ_sub, BK] QK tile (BQ_sub = BQ/2).
//
// Pops the QK fp32 tile from Cube->Vec, runs the stable streaming-softmax
// update (m, l, alpha), casts P to fp16, and pushes the P slot to Cube.
// `alpha_log` (per-row) is captured in the caller's UB ring so the GU
// stage on this same Vec subblock can rescale the running O.
// ---------------------------------------------------------------------------
template <int BQ_sub, int BK, int HEAD, typename QKPipe, typename PPipe, typename QKVecSlotG, typename PVecSlotG,
          typename QKVecT, typename PFp32T, typename PFp16T, typename RowT, typename TmpT>
AICORE inline void compute_softmax_vec(QKPipe &qkPipe, PPipe &pPipe, QKVecSlotG &qkSlot, PVecSlotG &pSlot,
                                       QKVecT &qkVec, PFp32T &pFp32, PFp16T &pFp16, RowT &m_run, RowT &l_run,
                                       RowT &alpha, TmpT &tmp, float scale, bool is_first)
{
    // Pop QK from the Cube->Vec FIFO. TILE_UP_DOWN tells TPOP to expose only
    // this subblock's row half (BQ_sub = BQ/2 rows out of the BQ-row slot).
    TPOP<QKPipe, QKVecSlotG, TileSplitAxis::TILE_UP_DOWN>(qkPipe, qkSlot);
    TLOAD(qkVec, qkSlot);
    TFREE<QKPipe, QKVecSlotG, TileSplitAxis::TILE_UP_DOWN>(qkPipe, qkSlot);

    TMULS(qkVec, qkVec, scale);

    if (is_first) {
        TROWMAX(m_run, qkVec, tmp);
        TROWEXPANDSUB(pFp32, qkVec, m_run);
        TEXP(pFp32, pFp32);
        TROWSUM(l_run, pFp32, tmp);
        // alpha is undefined on the first iteration; the GU stage takes its
        // own init path, so we don't need to write it here.
    } else {
        RowT m_new;
        TROWMAX(m_new, qkVec, tmp);
        TMAX(m_new, m_new, m_run);

        // alpha = exp(m_run - m_new): the per-row factor that rescales the
        // running normalizer l and the running output O once a new max is
        // observed.
        TSUB(alpha, m_run, m_new);
        TEXP(alpha, alpha);

        TROWEXPANDSUB(pFp32, qkVec, m_new);
        TEXP(pFp32, pFp32);

        RowT l_local;
        TROWSUM(l_local, pFp32, tmp);
        TMUL(l_run, l_run, alpha);
        TADD(l_run, l_run, l_local);

        TMOV(m_run, m_new);
    }

    // Cast to fp16 and push to the Vec->Cube P FIFO so Cube can run P@V.
    using PFp32_1D = Tile<TileType::Vec, float,  1, BQ_sub * BK, BLayout::RowMajor, 1, BQ_sub * BK>;
    using PFp16_1D = Tile<TileType::Vec, half,   1, BQ_sub * BK, BLayout::RowMajor, 1, BQ_sub * BK>;
    PFp32_1D pFp32_1d; PFp16_1D pFp16_1d;
    TRESHAPE(pFp32_1d, pFp32);
    TRESHAPE(pFp16_1d, pFp16);
    TCVT(pFp16_1d, pFp32_1d, RoundMode::CAST_ROUND);

    TALLOC<PPipe, PVecSlotG, TileSplitAxis::TILE_UP_DOWN>(pPipe, pSlot);
    TSTORE(pSlot, pFp16);
    TPUSH<PPipe, PVecSlotG, TileSplitAxis::TILE_UP_DOWN>(pPipe, pSlot);
}

// ---------------------------------------------------------------------------
// VEC: streaming O update for one selected KV block.
//
// Pops PV from Cube->Vec, then either initializes (first block) or rescales
// the running O by alpha and adds the new PV. On the last *selected* block
// we divide by the global normalizer l and store O.
// ---------------------------------------------------------------------------
template <int BQ_sub, int HEAD, typename PVPipe, typename PVVecSlotG, typename OutT, typename RowT>
AICORE inline void compute_gu_vec(PVPipe &pvPipe, PVVecSlotG &pvSlot, OutT &o_run, OutT &pv_vec, RowT &alpha,
                                  RowT &l_run, bool is_first, bool is_last_selected, __gm__ float *o_block,
                                  size_t out_row_offset)
{
    TPOP<PVPipe, PVVecSlotG, TileSplitAxis::TILE_UP_DOWN>(pvPipe, pvSlot);
    if (is_first) {
        TLOAD(o_run, pvSlot);
    } else {
        TLOAD(pv_vec, pvSlot);
        // O = alpha * O + PV   (FA-2 streaming rescale; pto_macro_fa_gu helper
        // from the dense FA reference does exactly this in two ops)
        pto_macro_fa_gu(o_run, pv_vec, alpha);
    }
    TFREE<PVPipe, PVVecSlotG, TileSplitAxis::TILE_UP_DOWN>(pvPipe, pvSlot);

    if (is_last_selected) {
        // Final normalization: O /= l. Same as pto_macro_fa_gu_last but no
        // additional PV term to add (we already folded the last one above).
        TROWEXPANDDIV(o_run, o_run, l_run);

        using OutGlobal = GlobalTensor<float, Shape<1, 1, 1, BQ_sub, HEAD>, Stride<1, 1, 1, HEAD, 1>>;
        OutGlobal oGlobal(o_block + out_row_offset * HEAD);
        TSTORE(oGlobal, o_run);
    }
}

} // namespace

// ===========================================================================
// SPMD entry: one AI Core (1 Cube + 2 Vec) per query block.
// ===========================================================================
template <int S_Q, int S_KV, int HEAD, int BQ, int BK, int TOP_K, int FIFO_DEPTH>
__global__ AICORE void runSparseAttenDSA(__gm__ uint64_t *ffts_addr, __gm__ half *q, __gm__ half *k, __gm__ half *v,
                                         __gm__ int32_t *block_indices, __gm__ half *p_fifo, __gm__ float *qk_fifo,
                                         __gm__ float *pv_fifo, __gm__ float *o)
{
    static_assert(S_Q  % BQ == 0,  "S_Q must be a multiple of BQ");
    static_assert(S_KV % BK == 0,  "S_KV must be a multiple of BK");
    static_assert(BQ   % VEC_CORES == 0, "BQ must split evenly across the 2 vec subblocks");
    static_assert((HEAD * sizeof(half)) % 32 == 0, "HEAD row stride must be 32B-aligned");
    static_assert((BK   * sizeof(half)) % 32 == 0, "BK row stride must be 32B-aligned");

    set_ffts_base_addr((uint64_t)ffts_addr);

    constexpr int   BQ_sub = BQ / VEC_CORES; // rows handled by one vec subblock
    constexpr float kScale = ConstexprInvSqrt(static_cast<float>(HEAD));

    const int q_block = get_block_idx();

    // ------------------------------------------------------------------
    // Mat tiles (Cube side): bind to L1 / L0 fractal layouts as required
    // by TMATMUL (see docs/coding/Tile.md, kernels/manual/common/flash_atten).
    // ------------------------------------------------------------------
    using QMatT  = Tile<TileType::Mat, half, BQ,   HEAD, BLayout::ColMajor, BQ,   HEAD, SLayout::RowMajor, 512>;
    using KMatT  = Tile<TileType::Mat, half, HEAD, BK,   BLayout::RowMajor, HEAD, BK,   SLayout::ColMajor, 512>;
    using VMatT  = Tile<TileType::Mat, half, BK,   HEAD, BLayout::ColMajor, BK,   HEAD, SLayout::RowMajor, 512>;
    using PMatT  = Tile<TileType::Mat, half, BQ,   BK,   BLayout::ColMajor, BQ,   BK,   SLayout::RowMajor, 512>;
    using QKAccT = TileAcc<float, BQ, BK,   BQ, BK>;
    using PVAccT = TileAcc<float, BQ, HEAD, BQ, HEAD>;

    QMatT qMat;  KMatT kMat;  VMatT vMat;  PMatT pMat;
    QKAccT qkAcc; PVAccT pvAcc;

    if constexpr (DAV_CUBE) {
        // L1 layout: Q | K | P | V, with single-buffered tiles for clarity
        // (the dense flash_atten ping-pongs K/P/V to overlap stages; that is
        // a perf knob orthogonal to the architecture mapping shown here).
        uint32_t off = 0;
        TASSIGN(qMat, off); off += BQ   * HEAD * sizeof(half);
        TASSIGN(kMat, off); off += HEAD * BK   * sizeof(half);
        TASSIGN(pMat, off); off += BQ   * BK   * sizeof(half);
        TASSIGN(vMat, off);

        // L0C accumulators live at the standard A2/A3 ping-pong addresses.
        TASSIGN(qkAcc, 0x0u);
        TASSIGN(pvAcc, 0x10000u);
    }

    // ------------------------------------------------------------------
    // Vec tiles (Vec side): UB-resident; sized to one subblock's row range.
    // ------------------------------------------------------------------
    using QKVecT = Tile<TileType::Vec, float, BQ_sub, BK,   BLayout::RowMajor, BQ_sub, BK>;
    using PFp32T = Tile<TileType::Vec, float, BQ_sub, BK,   BLayout::RowMajor, BQ_sub, BK>;
    using PFp16T = Tile<TileType::Vec, half,  BQ_sub, BK,   BLayout::RowMajor, BQ_sub, BK>;
    using OutT   = Tile<TileType::Vec, float, BQ_sub, HEAD, BLayout::RowMajor, BQ_sub, HEAD>;
    using RowT   = Tile<TileType::Vec, float, BQ_sub, 1,    BLayout::ColMajor, BQ_sub, 1>;
    using TmpT   = Tile<TileType::Vec, float, BQ_sub, BK,   BLayout::RowMajor, BQ_sub, BK>;

    QKVecT qkVec;  PFp32T pFp32;  PFp16T pFp16;
    OutT   o_run;  OutT   pv_vec;
    RowT   m_run;  RowT   l_run;  RowT alpha;
    TmpT   tmp;

    if constexpr (DAV_VEC) {
        uint32_t off = 0;
        TASSIGN(o_run,  off); off += BQ_sub * HEAD * sizeof(float);
        TASSIGN(pv_vec, off); off += BQ_sub * HEAD * sizeof(float);
        TASSIGN(qkVec,  off); off += BQ_sub * BK   * sizeof(float);
        TASSIGN(pFp32,  off); off += BQ_sub * BK   * sizeof(float);
        TASSIGN(pFp16,  off); off += BQ_sub * BK   * sizeof(half);
        TASSIGN(tmp,    off); off += BQ_sub * BK   * sizeof(float);
        TASSIGN(m_run,  off); off += BQ_sub * sizeof(float);
        TASSIGN(l_run,  off); off += BQ_sub * sizeof(float);
        TASSIGN(alpha,  off);
    }

    // ------------------------------------------------------------------
    // Cube<->Vec FIFOs. SlotSize is the full BQ-row tile; TILE_UP_DOWN on
    // the Vec side splits it row-wise across the two subblocks.
    // ------------------------------------------------------------------
    using QKPipe = TPipe<QK_PIPE_FLAG, Direction::DIR_C2V, BQ * BK   * sizeof(float), FIFO_DEPTH>;
    using PPipe  = TPipe<P_PIPE_FLAG,  Direction::DIR_V2C, BQ * BK   * sizeof(half),  FIFO_DEPTH>;
    using PVPipe = TPipe<PV_PIPE_FLAG, Direction::DIR_C2V, BQ * HEAD * sizeof(float), FIFO_DEPTH>;
    QKPipe qkPipe(qk_fifo, (uint32_t)(uint64_t)qkVec.data(), 0x0);
    PPipe  pPipe (p_fifo,  0x0,                              (uint32_t)(uint64_t)pMat.data());
    PVPipe pvPipe(pv_fifo, (uint32_t)(uint64_t)pv_vec.data(), 0x0);

    using QKSlotCubeG = GlobalTensor<float, Shape<1, 1, 1, BQ,     BK>,   Stride<1, 1, 1, BK,   1>>;
    using QKSlotVecG  = GlobalTensor<float, Shape<1, 1, 1, BQ_sub, BK>,   Stride<1, 1, 1, BK,   1>>;
    using PSlotCubeG  = GlobalTensor<half,  Shape<1, 1, 1, BQ,     BK>,   Stride<1, 1, 1, BK,   1>>;
    using PSlotVecG   = GlobalTensor<half,  Shape<1, 1, 1, BQ_sub, BK>,   Stride<1, 1, 1, BK,   1>>;
    using PVSlotCubeG = GlobalTensor<float, Shape<1, 1, 1, BQ,     HEAD>, Stride<1, 1, 1, HEAD, 1>>;
    using PVSlotVecG  = GlobalTensor<float, Shape<1, 1, 1, BQ_sub, HEAD>, Stride<1, 1, 1, HEAD, 1>>;
    QKSlotCubeG qkSlotCube;  QKSlotVecG qkSlotVec;
    PSlotCubeG  pSlotCube;   PSlotVecG  pSlotVec;
    PVSlotCubeG pvSlotCube;  PVSlotVecG pvSlotVec;

    // ------------------------------------------------------------------
    // Cube path: load Q once, then walk selected KV blocks.
    // ------------------------------------------------------------------
    if constexpr (DAV_CUBE) {
        using QGlobal = GlobalTensor<half, Shape<1, 1, 1, BQ, HEAD>, Stride<1, 1, 1, HEAD, 1>>;
        QGlobal qGlobal(q + static_cast<size_t>(q_block) * BQ * HEAD);
        TLOAD(qMat, qGlobal);

        for (int t = 0; t < TOP_K; ++t) {
            const int32_t kv_blk = block_indices[static_cast<size_t>(q_block) * TOP_K + t];
            const bool skip = (kv_blk < 0);
            if (skip) continue; // both cores observe the same skip pattern

            compute_qk_cube<BQ, BK, HEAD>(qkPipe, k, kv_blk, qMat, kMat, qkAcc, qkSlotCube);
            compute_pv_cube<BQ, BK, HEAD>(pPipe, pvPipe, v, kv_blk, pMat, vMat, pvAcc, pSlotCube, pvSlotCube);
        }
    }

    // ------------------------------------------------------------------
    // Vec path: each subblock owns BQ_sub rows starting at subblock_row_off.
    // ------------------------------------------------------------------
    if constexpr (DAV_VEC) {
        const size_t subblock_row_off = static_cast<size_t>(get_subblockid()) * BQ_sub;
        __gm__ float *o_block = o + static_cast<size_t>(q_block) * BQ * HEAD;

        // Find the index of the last *selected* KV block so we know when
        // to apply the final O /= l normalization. Both cores compute it
        // the same way; cheap relative to the matmul work.
        int last_selected = -1;
        for (int t = 0; t < TOP_K; ++t) {
            if (block_indices[static_cast<size_t>(q_block) * TOP_K + t] >= 0) last_selected = t;
        }

        bool is_first = true;
        for (int t = 0; t < TOP_K; ++t) {
            const int32_t kv_blk = block_indices[static_cast<size_t>(q_block) * TOP_K + t];
            const bool skip = (kv_blk < 0);
            if (skip) continue;

            compute_softmax_vec<BQ_sub, BK, HEAD>(qkPipe, pPipe, qkSlotVec, pSlotVec, qkVec, pFp32, pFp16, m_run,
                                                  l_run, alpha, tmp, kScale, is_first);

            compute_gu_vec<BQ_sub, HEAD>(pvPipe, pvSlotVec, o_run, pv_vec, alpha, l_run, is_first,
                                         /*is_last_selected=*/(t == last_selected), o_block, subblock_row_off);

            is_first = false;
        }

        if (last_selected < 0) {
            // Pathological all-masked case: still emit zeros so the output
            // is well-defined.
            TMULS(o_run, o_run, 0.0f);
            using OutGlobal = GlobalTensor<float, Shape<1, 1, 1, BQ_sub, HEAD>, Stride<1, 1, 1, HEAD, 1>>;
            OutGlobal oGlobal(o_block + subblock_row_off * HEAD);
            TSTORE(oGlobal, o_run);
        }
    }

    pipe_barrier(PIPE_ALL);
}

__global__ AICORE __attribute__((aic)) void warmup_kernel() {}

// ===========================================================================
// Host launch wrapper.
// ===========================================================================
template <int S_Q, int S_KV, int HEAD, int BQ, int BK, int TOP_K, int FIFO_DEPTH>
void LaunchSparseAttenDSA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, int32_t *block_indices,
                          aclFloat16 *p_fifo, float *qk_fifo, float *pv_fifo, float *o, aclrtStream stream)
{
    static_assert(S_Q % BQ == 0,  "S_Q must be a multiple of BQ");
    static_assert(S_KV % BK == 0, "S_KV must be a multiple of BK");
    constexpr int kNumQBlocks = S_Q / BQ;

    warmup_kernel<<<kNumQBlocks, nullptr, stream>>>();

    runSparseAttenDSA<S_Q, S_KV, HEAD, BQ, BK, TOP_K, FIFO_DEPTH>
        <<<kNumQBlocks, nullptr, stream>>>((__gm__ uint64_t *)ffts, (__gm__ half *)q, (__gm__ half *)k,
                                           (__gm__ half *)v, (__gm__ int32_t *)block_indices,
                                           (__gm__ half *)p_fifo, (__gm__ float *)qk_fifo,
                                           (__gm__ float *)pv_fifo, (__gm__ float *)o);
}

// ---------------------------------------------------------------------------
// Explicit instantiations for common DeepSeek V4 (DSA) configurations.
//   HEAD=128, BQ=BK=64, TOP_K=64 — matches the indexer block size and
//   selection budget used in the V3.2-Exp report.
// ---------------------------------------------------------------------------
#define INSTANTIATE_SPARSE_DSA(S_Q, S_KV, HEAD, BQ, BK, TOP_K)                                                  \
    template void LaunchSparseAttenDSA<S_Q, S_KV, HEAD, BQ, BK, TOP_K, 2>(                                      \
        uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, int32_t * block_indices,               \
        aclFloat16 * p_fifo, float *qk_fifo, float *pv_fifo, float *o, aclrtStream stream);

INSTANTIATE_SPARSE_DSA(4096,  16384, 128, 64, 64, 64)
INSTANTIATE_SPARSE_DSA(8192,  32768, 128, 64, 64, 64)
INSTANTIATE_SPARSE_DSA(16384, 65536, 128, 64, 64, 64)

#undef INSTANTIATE_SPARSE_DSA
