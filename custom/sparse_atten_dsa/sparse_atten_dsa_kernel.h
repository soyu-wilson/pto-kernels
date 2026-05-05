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

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

// DeepSeek V4 (DSA) sparse-attention compute kernel for Ascend 910B (A2/A3).
//
// Each AI Core on 910B is a 1-Cube + 2-Vector composite (1:2 cube:vec ratio).
// This kernel uses an MPMD split that maps:
//   - QK matmul and PV matmul to the Cube unit (TMATMUL on Mat/Left/Right/Acc).
//   - FA-2 streaming softmax and the running O update to the two Vec
//     subblocks; each subblock handles BQ/2 rows (selected by get_subblockid).
//   - Cross-unit data flows over three GM-backed ring FIFOs (TPipe):
//         Cube --[QK fp32]--> Vec      (QKPipe, DIR_C2V)
//         Vec  --[P  fp16]--> Cube     (PPipe,  DIR_V2C)
//         Cube --[PV fp32]--> Vec      (PVPipe, DIR_C2V)
//
// Inputs:
//   q             : [S_Q,  HEAD]              fp16, row-major
//   k, v          : [S_KV, HEAD]              fp16, row-major
//   block_indices : [S_Q / BQ, TOP_K]         int32, row-major
//                   Entry < 0 marks an unused slot (skipped on both cores in
//                   lockstep so the FIFOs stay balanced).
//   o             : [S_Q,  HEAD]              fp32, row-major (output)
//
// FIFO scratch buffers (GM, owned by the host):
//   qk_fifo : Cube->Vec, slot = BQ*BK*sizeof(float)
//   p_fifo  : Vec->Cube, slot = BQ*BK*sizeof(half)
//   pv_fifo : Cube->Vec, slot = BQ*HEAD*sizeof(float)
//   ffts    : FFTS base for cross-pipe sync (standard A2/A3 host artifact)
//
// SPMD: launch grid = (S_Q / BQ); one AI Core per query block.
template <int S_Q, int S_KV, int HEAD, int BQ, int BK, int TOP_K, int FIFO_DEPTH = 2>
void LaunchSparseAttenDSA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, int32_t *block_indices,
                          aclFloat16 *p_fifo, float *qk_fifo, float *pv_fifo, float *o, aclrtStream stream);

#endif // SPARSE_ATTEN_DSA_KERNEL_H
