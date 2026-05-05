/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_ATTEN_DSA_KERNEL_H
#define SPARSE_ATTEN_DSA_KERNEL_H

#include <cstdint>

// DeepSeek V4 (DSA) sparse-attention compute kernel.
//
// Per-block parameters (all compile-time):
//   S_Q       : total query tokens
//   S_KV      : total key/value tokens (must satisfy S_KV % BK == 0)
//   HEAD      : head dim
//   BQ        : query block size (rows handled by one core)
//   BK        : KV block size (must match the indexer's block granularity)
//   TOP_K     : max number of KV blocks attended to per query block
//
// The kernel expects:
//   q             : [S_Q,  HEAD]              fp16, row-major
//   k             : [S_KV, HEAD]              fp16, row-major
//   v             : [S_KV, HEAD]              fp16, row-major
//   block_indices : [S_Q / BQ, TOP_K]         int32, row-major
//                   Entry < 0 (e.g. -1) marks an unused slot and is skipped.
//                   Indices are KV-block ids in [0, S_KV / BK).
//   o             : [S_Q,  HEAD]              fp32, row-major (output)
//
// SPMD: launch with grid = (S_Q / BQ); each core processes one query block.
template <int S_Q, int S_KV, int HEAD, int BQ, int BK, int TOP_K>
void LaunchSparseAttenDSA(__fp16 *q, __fp16 *k, __fp16 *v, int32_t *block_indices, float *o, void *stream);

#endif // SPARSE_ATTEN_DSA_KERNEL_H
