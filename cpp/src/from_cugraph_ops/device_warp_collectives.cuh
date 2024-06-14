/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device_core.hpp"
#include "device_dim.cuh"
#include "macros.hpp"

#include <cstdint>

namespace cugraph::ops::utils {

/**
 * @brief warp-wide any boolean aggregator
 *
 * @param[in] in_flag flag to be checked across threads in the warp
 * @param[in] mask    set of threads to be checked in this warp
 *
 * @return true if any one of the threads have their `in_flag` set to true
 */
__device__ inline bool d_any(bool in_flag, uint32_t mask = 0xffffffffU)
{
  static_assert(CUDART_VERSION >= CUDA_VER_WARP_SHFL,
                "Expected CUDA >= 9 for warp synchronous shuffle");
  return static_cast<bool>(__any_sync(mask, static_cast<int>(in_flag)));
}

/**
 * @brief warp-wide all boolean aggregator
 *
 * @param[in] in_flag flag to be checked across threads in the warp
 * @param[in] mask    set of threads to be checked in this warp
 *
 * @return true if all of the threads have their `in_flag` set to true
 */
__device__ inline bool all(bool in_flag, uint32_t mask = 0xffffffffU)
{
  static_assert(CUDART_VERSION >= CUDA_VER_WARP_SHFL,
                "Expected CUDA >= 9 for warp synchronous shuffle");
  return static_cast<bool>(__all_sync(mask, static_cast<int>(in_flag)));
}

/**
 * @defgroup LaneMaskUtils Utility methods to obtain lane mask. Refer to the
 *           PTX ISA document to know more details on these masks.
 * @{
 */
__device__ inline unsigned lanemask_lt()
{
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}
__device__ inline unsigned lanemask_le()
{
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}
__device__ inline unsigned lanemask_gt()
{
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}
__device__ inline unsigned lanemask_ge()
{
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}
/** @} */

/**
 * @brief get a bit mask for the `n_threads` lowest threads of a warp
 *
 * @param[in] n_threads  number of threads in the mask
 *
 * @return the bit mask
 */
__host__ __device__ constexpr uint32_t low_thread_mask(int n_threads)
{
  return n_threads >= WARP_SIZE ? 0xffffffffU : (1U << n_threads) - 1U;
}

/**
 * apply a warp-wide sync (useful from Volta+ archs)
 *
 * @tparam NP number of participating threads
 *
 * @note This works on Pascal and earlier archs as well, but all threads with
 * lane id <= NP must enter this function together and in convergence.
 */
template <int NP = WARP_SIZE>
__device__ inline void warp_sync()
{
  __syncwarp(low_thread_mask(NP));
}

/**
 * @brief Shuffle the data inside a warp
 *
 * @tparam DataT the data type (currently assumed to be 4B)
 *
 * @param[in] val      value to be shuffled
 * @param[in] src_lane lane from where to shuffle
 * @param[in] width    lane width
 * @param[in] mask     mask of participating threads (Volta+)
 *
 * @return the shuffled data
 */
template <typename DataT>
__device__ inline DataT shfl(DataT val,
                             int src_lane,
                             int width     = WARP_SIZE,
                             uint32_t mask = 0xffffffffU)
{
  static_assert(CUDART_VERSION >= CUDA_VER_WARP_SHFL,
                "Expected CUDA >= 9 for warp synchronous shuffle");
  return __shfl_sync(mask, val, src_lane, width);
}

/**
 * @brief XOR-Shuffle the data inside a warp
 *
 * @tparam DataT the data type (currently assumed to be 4B)
 *
 * @param[in] val      value to be shuffled
 * @param[in] xor_mask mask to get source lane (XOR with caller's lane)
 * @param[in] width    lane width
 * @param[in] mask     mask of participating threads (Volta+)
 *
 * @return the shuffled data
 */
template <typename DataT>
__device__ inline DataT shfl_xor(DataT val,
                                 int xor_mask,
                                 int width     = WARP_SIZE,
                                 uint32_t mask = 0xffffffffU)
{
  static_assert(CUDART_VERSION >= CUDA_VER_WARP_SHFL,
                "Expected CUDA >= 9 for warp synchronous shuffle");
  return __shfl_xor_sync(mask, val, xor_mask, width);
}

/**
 * @brief Warp-level sum reduction
 *
 * @tparam DataT data type
 * @tparam NP number of participating threads.
 *         must be a power of 2 and at most warp size
 *
 * @param[in] val input value
 *
 * @return only the lane0 will contain valid reduced result
 *
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block.
 *
 * @note All threads with lane id <= NP must enter this function together
 *
 * TODO(mjoux) Expand this to support arbitrary reduction ops
 */
template <typename DataT, int NP = WARP_SIZE>
__device__ inline DataT warp_reduce(DataT val)
{
  // see https://nvbugswb.nvidia.com/NVBugs5/redir.aspx?url=/3934593
  // for now, we ensure that warp is converged before reduction wtih warp-sync
#if __CUDA_ARCH__ < 700
  warp_sync();
#endif
  static constexpr uint32_t MASK = low_thread_mask(NP);
  CUGRAPH_OPS_UNROLL
  for (int i = NP / 2; i > 0; i >>= 1) {
    DataT tmp = shfl(val, lane_id() + i, NP, MASK);
    val += tmp;
  }
  return val;
}

/**
 * @brief Warp-level sum reduction, s.t. all threads obtain the result
 *
 * @tparam DataT data type
 * @tparam NP number of participating threads.
 *         must be a power of 2 and at most warp size
 *
 * @param[in] val input value
 *
 * @return all lanes will contain valid reduced result
 *
 * @note All threads with lane id <= NP must enter this function together
 *
 * TODO(mjoux) Expand this to support arbitrary reduction ops
 */
template <typename DataT, int NP = WARP_SIZE>
__device__ inline DataT warp_all_reduce(DataT val)
{
  static constexpr uint32_t MASK = low_thread_mask(NP);
  CUGRAPH_OPS_UNROLL
  for (int i = NP / 2; i > 0; i >>= 1) {
    DataT tmp = shfl_xor(val, i, NP, MASK);
    val += tmp;
  }
  return val;
}

template <typename DataT, int NP = WARP_SIZE>
__device__ inline DataT warp_max_reduce(DataT val)
{
  static constexpr uint32_t MASK = low_thread_mask(NP);
  CUGRAPH_OPS_UNROLL
  for (int i = NP / 2; i > 0; i >>= 1) {
    DataT tmp = shfl(val, lane_id() + i, NP, MASK);
    val       = max(tmp, val);
  }
  return val;
}

/**
 * @brief Warp-level max reduction, s.t. all threads obtain the result
 *
 * @tparam DataT data type
 * @tparam NP number of participating threads.
 *         must be a power of 2 and at most warp size
 *
 * @param[in] val input value
 *
 * @return all lanes will contain valid reduced result
 *
 * @note All threads with lane id <= NP must enter this function together
 *
 * TODO(mjoux) Expand this to support arbitrary reduction ops
 */
// TODO(@stadlmax): check max w.r.t. __half2, __nv_bfloat162, we would need max
// for both halves independently
template <typename DataT, int NP = WARP_SIZE>
__device__ inline DataT warp_all_max_reduce(DataT val)
{
  static constexpr uint32_t MASK = low_thread_mask(NP);
  CUGRAPH_OPS_UNROLL
  for (int i = NP / 2; i > 0; i >>= 1) {
    DataT tmp = shfl_xor(val, i, NP, MASK);
    val       = max(tmp, val);
  }
  return val;
}

// TODO(@stadlmax): check usage w.r.t. __half2, __nv_bfloat162
// we would need two indices for that
template <typename IdxT, typename DataT, bool COMPARE_IDX = false>
__device__ inline void warp_max_idx_reduce(IdxT& idx, DataT& val)
{
  CUGRAPH_OPS_UNROLL
  for (int i = WARP_SIZE / 2; i > 0; i >>= 1) {
    DataT tmp  = shfl(val, lane_id() + i);
    IdxT i_tmp = shfl(idx, lane_id() + i);
    // we check tmp > val since we get tmp from a larger lane than ours.
    if (tmp > val) {
      idx = i_tmp;
      val = tmp;
    }
    // to guruantee deterministic results it might be preferred to
    // find the occurence of the maximum with the lowest index
    // in case the values to be reduced are not sorted within a warp
    // (e.g. if each thread uses a warp-stride loop to reduce locally)
    else if (COMPARE_IDX && tmp == val && i_tmp < idx) {
      idx = i_tmp;
    }
  }
}

}  // namespace cugraph::ops::utils
