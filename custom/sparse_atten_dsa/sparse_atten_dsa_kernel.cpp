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
// DeepSeek V4 / V3.2-Exp sparse attention (DSA) — FlashAttention-2 hot path.
//
// DSA structure:
//   1. A cheap "Lightning Indexer" (separate kernel) scores every (q_block,
//      kv_block) pair and selects the TOP_K most relevant KV blocks for each
//      query block, producing `block_indices`.
//   2. This kernel does the actual attention compute, restricted to those
//      selected KV blocks. The math is standard scaled-dot-product attention
//      with FA-2 streaming softmax; the only difference vs. dense FA is that
//      the KV loop walks `block_indices[q_block, :]` instead of all KV blocks.
//
// Programming style: PTO-Auto.
//   - No TASSIGN, no explicit flags/events. The compiler handles placement
//     and synchronization (see docs/coding/ProgrammingModel.md, docs/auto_mode/*).
//   - SPMD: one core per query block (get_block_idx selects it).
//   - All loop bounds are compile-time except the TOP_K iteration, whose body
//     skips when block_indices[t] < 0 (sentinel for "unused slot").
// =============================================================================

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "sparse_atten_dsa_kernel.h"

using namespace pto;

namespace {

// ---------------------------------------------------------------------------
// constexpr 1/sqrt(HEAD) for the attention scale. Keeping it constexpr lets
// the compiler bake it into a TMULS immediate (cheaper than a runtime divide).
// ---------------------------------------------------------------------------
constexpr AICORE inline float ConstexprSqrt(float x)
{
    if (x <= 0.0f) return 0.0f;
    float g = x;
    for (int i = 0; i < 8; ++i) {
        g = 0.5f * (g + x / g);
    }
    return g;
}

constexpr AICORE inline float ConstexprInvSqrt(float x)
{
    return 1.0f / ConstexprSqrt(x);
}

// ---------------------------------------------------------------------------
// FA-2 streaming softmax + online output update for one KV block.
//
// Maintains per-row running state:
//   m  : running max (length BQ)
//   l  : running normalizer (length BQ)
//   O  : running output accumulator [BQ, HEAD] (fp32)
//
// Given the freshly-computed QK tile X = scale * (Q @ K^T) [BQ, BK] and the
// matmul accumulator path PV = P @ V [BQ, HEAD], this routine:
//   m_new = max(m, rowmax(X))
//   alpha = exp(m - m_new)                  (scalar per row)
//   P     = exp(X - m_new)                  ([BQ, BK])
//   l     = alpha * l + rowsum(P)
//   O     = alpha * O + (P @ V)             (PV computed by caller)
//   m     = m_new
//
// `is_first` peels the first iteration so we can initialize from X directly
// (avoids a multiply-by-zero on l/O and avoids the m=-inf rescale path).
// ---------------------------------------------------------------------------
template <int BQ, int BK, int HEAD, typename QKAcc, typename PVAcc, typename PFp16Tile, typename PFp32Tile,
          typename RowTile, typename TmpTile>
AICORE inline void OnlineSoftmaxUpdate(QKAcc &qk_acc, PFp32Tile &p_fp32, PFp16Tile &p_fp16, RowTile &m_running,
                                       RowTile &l_running, RowTile &m_new, RowTile &alpha, TmpTile &reduce_tmp,
                                       float scale, bool is_first)
{
    // Apply the attention scale once on the QK accumulator before reduction.
    // Doing it in fp32 on qk_acc keeps numerics tight and lets TEXP work on
    // already-scaled inputs (cheaper than scaling inside the exp argument).
    TMULS(qk_acc, qk_acc, scale);

    if (is_first) {
        // First selected block: initialise running state from this tile only.
        TROWMAX(m_running, qk_acc, reduce_tmp);
        TROWEXPANDSUB(p_fp32, qk_acc, m_running);
        TEXP(p_fp32, p_fp32);
        TROWSUM(l_running, p_fp32, reduce_tmp);
    } else {
        // Streaming update: combine the new tile's max with the running max,
        // rescale the running normalizer, and emit the rescale factor `alpha`
        // so the caller can apply it to the running output O.
        TROWMAX(m_new, qk_acc, reduce_tmp);
        TMAX(m_new, m_new, m_running);

        // alpha = exp(m_running - m_new); use it both to rescale l_running
        // and (in the caller) to rescale O before adding the new PV.
        TSUB(alpha, m_running, m_new);
        TEXP(alpha, alpha);

        // P = exp(X - m_new), then sum into per-row local sum.
        TROWEXPANDSUB(p_fp32, qk_acc, m_new);
        TEXP(p_fp32, p_fp32);

        // l = alpha * l + rowsum(P)
        RowTile l_local;
        TROWSUM(l_local, p_fp32, reduce_tmp);
        TMUL(l_running, l_running, alpha);
        TADD(l_running, l_running, l_local);

        // commit the new running max
        TMOV(m_running, m_new);
    }

    // Cast P to fp16 for the upcoming P@V matmul (V is fp16; cube path
    // expects matching operand precision).
    TCVT(p_fp16, p_fp32, RoundMode::CAST_ROUND);
}

} // namespace

// ===========================================================================
// SPMD kernel entry point.
//
// One core per query block; the core identifies its block via get_block_idx.
// Each core walks its TOP_K selected KV blocks (looked up from block_indices)
// and accumulates the streaming-softmax output.
// ===========================================================================
template <int S_Q, int S_KV, int HEAD, int BQ, int BK, int TOP_K>
__global__ __aicore__ void SparseAttenDSAKernel(__gm__ __fp16 *q_gm, __gm__ __fp16 *k_gm, __gm__ __fp16 *v_gm,
                                                __gm__ int32_t *block_indices_gm, __gm__ float *o_gm)
{
    // Compile-time validation matching the docs' tile-shape constraints.
    static_assert(S_Q % BQ == 0,   "S_Q must be a multiple of BQ");
    static_assert(S_KV % BK == 0,  "S_KV must be a multiple of BK");
    static_assert((HEAD * sizeof(__fp16)) % 32 == 0, "HEAD row stride must be 32B-aligned (Tile.md)");
    static_assert((BK   * sizeof(__fp16)) % 32 == 0, "BK row stride must be 32B-aligned (Tile.md)");

    constexpr int kNumQBlocks = S_Q / BQ;
    constexpr float kScale    = ConstexprInvSqrt(static_cast<float>(HEAD));

    const int q_block = get_block_idx();
    if (q_block >= kNumQBlocks) {
        return;
    }

    // -----------------------------------------------------------------------
    // GlobalTensor views.
    //   - Q-block view: a [BQ, HEAD] window into q_gm at row offset q_block*BQ.
    //   - K-block view: per-iteration runtime offset, see below.
    //   - V-block view: per-iteration runtime offset, see below.
    //   - O-block view: a [BQ, HEAD] window into o_gm.
    // We keep the leading 3 shape dims = 1 per the GlobalTensor model
    // (docs/coding/GlobalTensor.md).
    // -----------------------------------------------------------------------
    using GTQ = GlobalTensor<__fp16, Shape<1, 1, 1, BQ, HEAD>, Stride<1, 1, 1, HEAD, 1>, Layout::ND>;
    using GTK = GlobalTensor<__fp16, Shape<1, 1, 1, BK, HEAD>, Stride<1, 1, 1, HEAD, 1>, Layout::ND>;
    using GTV = GlobalTensor<__fp16, Shape<1, 1, 1, BK, HEAD>, Stride<1, 1, 1, HEAD, 1>, Layout::ND>;
    using GTO = GlobalTensor<float,  Shape<1, 1, 1, BQ, HEAD>, Stride<1, 1, 1, HEAD, 1>, Layout::ND>;

    GTQ q_view(q_gm + static_cast<size_t>(q_block) * BQ * HEAD);
    GTO o_view(o_gm + static_cast<size_t>(q_block) * BQ * HEAD);

    // -----------------------------------------------------------------------
    // Tile types.
    //   Q is a Mat tile (operand for the QK^T matmul). It outlives the loop.
    //   K is loaded once per selected block; we set its SLayout to ColMajor so
    //   that Q@K^T uses the docs' "TileLeft @ TileRight^T" matmul form.
    //   V is loaded once per selected block as a TileRight for P@V.
    //   QK accumulator is fp32 [BQ, BK] (TileAcc). PV accumulator is fp32
    //   [BQ, HEAD] (TileAcc), kept resident across all TOP_K iterations.
    //   Vec tiles below carry the softmax intermediates (P_fp32, P_fp16,
    //   per-row max/sum/alpha) and stay in UB.
    // -----------------------------------------------------------------------
    using QMatTile = Tile<TileType::Mat, __fp16, BQ,   HEAD, BLayout::ColMajor, BQ,   HEAD, SLayout::RowMajor, 512>;
    using KMatTile = Tile<TileType::Mat, __fp16, HEAD, BK,   BLayout::RowMajor, HEAD, BK,   SLayout::ColMajor, 512>;
    using VMatTile = Tile<TileType::Mat, __fp16, BK,   HEAD, BLayout::ColMajor, BK,   HEAD, SLayout::RowMajor, 512>;
    using QKAcc    = TileAcc<float, BQ, BK,   BQ, BK>;
    using PVAcc    = TileAcc<float, BQ, HEAD, BQ, HEAD>;

    using PFp32Tile  = Tile<TileType::Vec, float,   BQ, BK, BLayout::RowMajor, BQ, BK>;
    using PFp16Tile  = Tile<TileType::Vec, __fp16,  BQ, BK, BLayout::RowMajor, BQ, BK>;
    using OFp32Tile  = Tile<TileType::Vec, float,   BQ, HEAD, BLayout::RowMajor, BQ, HEAD>;
    using RowTile    = Tile<TileType::Vec, float,   BQ, 1,  BLayout::ColMajor, BQ, 1>;
    using TmpTile    = Tile<TileType::Vec, float,   BQ, BK, BLayout::RowMajor, BQ, BK>;

    QMatTile q_mat;
    KMatTile k_mat;
    VMatTile v_mat;
    QKAcc    qk_acc;
    PVAcc    pv_acc;

    PFp32Tile p_fp32;
    PFp16Tile p_fp16;
    OFp32Tile o_running;            // [BQ, HEAD] fp32, vec-side mirror of pv_acc for streaming rescale
    RowTile   m_running, l_running, m_new, alpha;
    TmpTile   reduce_tmp;

    // Q is loaded once per query block; it's reused across all TOP_K iterations.
    TLOAD(q_mat, q_view);

    // -----------------------------------------------------------------------
    // Sparse KV loop. We read each selected KV block id as a scalar GM read,
    // then construct the K/V GlobalTensor views at the corresponding offset.
    // This is the same idiom the dense flash_atten kernel uses for its
    // s1_index*HEAD offset; the only change is sparsity comes from the index
    // table rather than a dense increment.
    // -----------------------------------------------------------------------
    bool is_first = true;
    for (int t = 0; t < TOP_K; ++t) {
        const int32_t kv_blk = block_indices_gm[static_cast<size_t>(q_block) * TOP_K + t];
        // Sentinel for "unused slot" — emit nothing for this t.
        // Per docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.3,
        // we evaluate the guard once into a local before the if so the auto
        // compiler sees a clean branch.
        const bool skip = (kv_blk < 0);
        if (skip) {
            continue;
        }

        const size_t kv_row_offset = static_cast<size_t>(kv_blk) * BK;
        GTK k_view(k_gm + kv_row_offset * HEAD);
        GTV v_view(v_gm + kv_row_offset * HEAD);

        // ------ Q @ K^T → QK accumulator (fp32, [BQ, BK]) -------------------
        TLOAD(k_mat, k_view);
        TMATMUL(qk_acc, q_mat, k_mat);

        // ------ Streaming softmax on the QK tile + cast P to fp16 ----------
        // Updates m_running, l_running; produces alpha = exp(m_old - m_new);
        // p_fp16 holds exp(X - m_new) ready for the P@V matmul.
        OnlineSoftmaxUpdate<BQ, BK, HEAD>(qk_acc, p_fp32, p_fp16, m_running, l_running, m_new, alpha, reduce_tmp,
                                          kScale, is_first);

        // ------ P @ V → PV accumulator (fp32, [BQ, HEAD]) -------------------
        // Move P_fp16 vec tile into a Right-operand mat tile. Per the docs
        // (TileLeft/TileRight aliases, Tile.md) the matmul engine expects
        // boxed mat operands; TMOV does the layout conversion.
        TLOAD(v_mat, v_view);
        Tile<TileType::Mat, __fp16, BQ, BK, BLayout::ColMajor, BQ, BK, SLayout::RowMajor, 512> p_mat;
        TMOV(p_mat, p_fp16);

        if (is_first) {
            // Init the running PV accumulator on the first selected block.
            TMATMUL(pv_acc, p_mat, v_mat);
            // Mirror pv_acc into a vec tile so the rescale path can act on it
            // with TROWEXPANDMUL on subsequent iterations.
            TMOV(o_running, pv_acc);
        } else {
            // Rescale the running output by alpha = exp(m_old - m_new), then
            // add the new P@V contribution. This keeps the streaming softmax
            // numerically stable across blocks.
            TROWEXPANDMUL(o_running, o_running, alpha);
            // Compute new P@V into pv_acc and fold it back into the vec mirror.
            TMATMUL(pv_acc, p_mat, v_mat);
            OFp32Tile pv_vec;
            TMOV(pv_vec, pv_acc);
            TADD(o_running, o_running, pv_vec);
        }

        is_first = false;
    }

    // -----------------------------------------------------------------------
    // Finalize: O = O / l. If every TOP_K slot was masked (degenerate input)
    // we still emit zeros to keep the output well-defined.
    // -----------------------------------------------------------------------
    if (is_first) {
        // No selected blocks: clear the output to zero. (Pathological case;
        // a sane indexer never produces all-(-1) rows.)
        TMULS(o_running, o_running, 0.0f);
    } else {
        TROWEXPANDDIV(o_running, o_running, l_running);
    }

    TSTORE(o_view, o_running);
}

// ---------------------------------------------------------------------------
// Empty warm-up kernel mirrors the convention used in flash_atten — keeps
// the host driver simple when prefetching Q/K/V into L2 before launch.
// ---------------------------------------------------------------------------
__global__ __aicore__ void SparseAttenDSAWarmup() {}

// ---------------------------------------------------------------------------
// Host launch wrapper. Grid = number of query blocks; one core per query block.
// `stream` is a runtime stream handle (aclrtStream); kept as void* in the
// header so this file does not have to pull acl/acl.h transitively when only
// being type-checked by the simulator.
// ---------------------------------------------------------------------------
template <int S_Q, int S_KV, int HEAD, int BQ, int BK, int TOP_K>
void LaunchSparseAttenDSA(__fp16 *q, __fp16 *k, __fp16 *v, int32_t *block_indices, float *o, void *stream)
{
    static_assert(S_Q % BQ == 0,  "S_Q must be a multiple of BQ");
    static_assert(S_KV % BK == 0, "S_KV must be a multiple of BK");
    constexpr int kNumQBlocks = S_Q / BQ;

    SparseAttenDSAKernel<S_Q, S_KV, HEAD, BQ, BK, TOP_K>
        <<<kNumQBlocks, nullptr, stream>>>((__gm__ __fp16 *)q, (__gm__ __fp16 *)k, (__gm__ __fp16 *)v,
                                           (__gm__ int32_t *)block_indices, (__gm__ float *)o);
}

// ---------------------------------------------------------------------------
// Explicit instantiations. Add more as needed; these match common DSA configs:
//   - HEAD=128 (DeepSeek V4 per-head dim)
//   - BQ=64, BK=64  (block size used by the lightning indexer)
//   - TOP_K=64      (default DSA selection budget)
// ---------------------------------------------------------------------------
#define INSTANTIATE_SPARSE_DSA(S_Q, S_KV, HEAD, BQ, BK, TOP_K) \
    template void LaunchSparseAttenDSA<S_Q, S_KV, HEAD, BQ, BK, TOP_K>(                                   \
        __fp16 * q, __fp16 * k, __fp16 * v, int32_t * block_indices, float *o, void *stream);

INSTANTIATE_SPARSE_DSA(4096,  16384, 128, 64, 64, 64)
INSTANTIATE_SPARSE_DSA(8192,  32768, 128, 64, 64, 64)
INSTANTIATE_SPARSE_DSA(16384, 65536, 128, 64, 64, 64)

#undef INSTANTIATE_SPARSE_DSA
