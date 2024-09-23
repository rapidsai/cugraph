/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device_core.hpp"

namespace cugraph::ops::utils {

/** get the lane id of the current thread */
__device__ __forceinline__ int lane_id()
{
  int id;
  asm("mov.s32 %0, %%laneid;" : "=r"(id));
  return id;
}

/**
 * get the flat id of the current thread (within block)
 * template parameters allow to control which CTA dimensions are used
 */
template <bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ __forceinline__ int flat_id()
{
  if (!USE_X && !USE_Y && !USE_Z)
    return 0;  // weird case, but if we get here, we should have 1 thread
  if (!USE_X && !USE_Y && USE_Z) return threadIdx.z;
  if (!USE_X && USE_Y && !USE_Z) return threadIdx.y;
  if (!USE_X && USE_Y && USE_Z) return threadIdx.y + threadIdx.z * blockDim.y;
  if (USE_X && !USE_Y && !USE_Z) return threadIdx.x;
  if (USE_X && !USE_Y && USE_Z) return threadIdx.x + threadIdx.z * blockDim.x;
  if (USE_X && USE_Y && !USE_Z) return threadIdx.x + threadIdx.y * blockDim.x;
  // USE_X && USE_Y && USE_Z
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

/**
 * get the number of warps of the current block
 * template parameters allow to control which CTA dimensions are used
 */
template <bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ __forceinline__ int num_warps()
{
  if (!USE_X && !USE_Y && !USE_Z)
    return 1;  // weird case, but if we get here, we should have 1 thread
  if (!USE_X && !USE_Y && USE_Z) return ceil_div<int>(blockDim.z, WARP_SIZE);
  if (!USE_X && USE_Y && !USE_Z) return ceil_div<int>(blockDim.y, WARP_SIZE);
  if (!USE_X && USE_Y && USE_Z) return ceil_div<int>(blockDim.y * blockDim.z, WARP_SIZE);
  if (USE_X && !USE_Y && !USE_Z) return ceil_div<int>(blockDim.x, WARP_SIZE);
  if (USE_X && !USE_Y && USE_Z) return ceil_div<int>(blockDim.x * blockDim.z, WARP_SIZE);
  if (USE_X && USE_Y && !USE_Z) return ceil_div<int>(blockDim.x * blockDim.y, WARP_SIZE);
  // USE_X && USE_Y && USE_Z
  return ceil_div<int>(blockDim.x * blockDim.y * blockDim.z, WARP_SIZE);
}

/**
 * get the warp id of the current thread
 * template parameters allow to control which CTA dimensions are used
 * @note: this only makes sense if the first used dimension of the CTA size
 * is a multiple of WARP_SIZE. If this is not the case, use
 * `flat_id<...>() / WARP_SIZE` to get the warp id of the current thread
 */
template <bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ __forceinline__ int warp_id()
{
  if (!USE_X && !USE_Y && !USE_Z)
    return 0;  // weird case, but if we get here, we should have 1 thread
  if (!USE_X && !USE_Y && USE_Z) return threadIdx.z / WARP_SIZE;
  if (!USE_X && USE_Y && !USE_Z) return threadIdx.y / WARP_SIZE;
  if (!USE_X && USE_Y && USE_Z)
    return threadIdx.y / WARP_SIZE + threadIdx.z * num_warps<false, true, false>();
  if (USE_X && !USE_Y && !USE_Z) return threadIdx.x / WARP_SIZE;
  if (USE_X && !USE_Y && USE_Z)
    return threadIdx.x / WARP_SIZE + threadIdx.z * num_warps<true, false, false>();
  if (USE_X && USE_Y && !USE_Z)
    return threadIdx.x / WARP_SIZE + threadIdx.y * num_warps<true, false, false>();
  // USE_X && USE_Y && USE_Z
  return threadIdx.x / WARP_SIZE + threadIdx.y * num_warps<true, false, false>() +
         threadIdx.z * blockDim.y * num_warps<true, false, false>();
}

/**
 * get the block dimension of the current executing block
 * template parameters allow to control which CTA dimensions are used
 */
template <bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ __forceinline__ int block_dim()
{
  if (!USE_X && !USE_Y && !USE_Z)
    return 1;  // weird case, but if we get here, we should have 1 thread
  if (!USE_X && !USE_Y && USE_Z) return blockDim.z;
  if (!USE_X && USE_Y && !USE_Z) return blockDim.y;
  if (!USE_X && USE_Y && USE_Z) return blockDim.y * blockDim.z;
  if (USE_X && !USE_Y && !USE_Z) return blockDim.x;
  if (USE_X && !USE_Y && USE_Z) return blockDim.x * blockDim.z;
  if (USE_X && USE_Y && !USE_Z) return blockDim.x * blockDim.y;
  // USE_X && USE_Y && USE_Z
  return blockDim.x * blockDim.y * blockDim.z;
}

/**
 * get the flat id of the current thread (within device/grid)
 * template parameters allow to control which grid and block/CTA dimensions are used
 */
template <bool G_USE_X = true,
          bool G_USE_Y = false,
          bool G_USE_Z = false,
          bool B_USE_X = true,
          bool B_USE_Y = false,
          bool B_USE_Z = false>
__device__ __forceinline__ int flat_grid_id()
{
  auto b_id  = flat_id<B_USE_X, B_USE_Y, B_USE_Z>();
  auto b_dim = block_dim<B_USE_X, B_USE_Y, B_USE_Z>();
  if (!G_USE_X && !G_USE_Y && !G_USE_Z)
    return 0;  // weird case, but if we get here, we should have 1 thread
  if (!G_USE_X && !G_USE_Y && G_USE_Z) return blockIdx.z * b_dim + b_id;
  if (!G_USE_X && G_USE_Y && !G_USE_Z) return blockIdx.y * b_dim + b_id;
  if (!G_USE_X && G_USE_Y && G_USE_Z) return blockIdx.y * b_dim + blockIdx.z * blockDim.z + b_id;
  if (G_USE_X && !G_USE_Y && !G_USE_Z) return blockIdx.x * b_dim + b_id;
  if (G_USE_X && !G_USE_Y && G_USE_Z) return blockIdx.x * b_dim + blockIdx.z * blockDim.z + b_id;
  if (G_USE_X && G_USE_Y && !G_USE_Z) return blockIdx.x * b_dim + blockIdx.y * blockDim.y + b_id;
  // G_USE_X && G_USE_Y && G_USE_Z
  return blockIdx.x * b_dim + blockIdx.y * blockDim.y * blockDim.z + blockIdx.z * blockDim.z + b_id;
}

}  // namespace cugraph::ops::utils
