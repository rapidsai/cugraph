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
  static constexpr uint32_t MASK = low_thread_mask(NP);
  CUGRAPH_OPS_UNROLL
  for (int i = NP / 2; i > 0; i >>= 1) {
    DataT tmp = shfl(val, lane_id() + i, NP, MASK);
    val += tmp;
  }
  return val;
}

}  // namespace cugraph::ops::utils
