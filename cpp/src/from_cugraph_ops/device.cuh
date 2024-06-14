/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device_atomics.cuh"
#include "device_block_collectives.cuh"
#include "device_core.hpp"
#include "device_dim.cuh"
#include "device_reg.cuh"
#include "device_smem_helper.cuh"
#include "device_warp_collectives.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cugraph::ops::utils {

/**
 * @defgroup Popc Device population count operation
 *
 * @{
 */
template <typename DataT>
__device__ inline DataT popc(DataT val);
template <>
__device__ inline int32_t popc(int32_t val)
{
  auto uval = *reinterpret_cast<unsigned*>(&val);
  return __popc(uval);
}
template <>
__device__ inline uint32_t popc(uint32_t val)
{
  auto uval = *reinterpret_cast<unsigned*>(&val);
  return __popc(uval);
}
template <>
__device__ inline int64_t popc(int64_t val)
{
  auto uval = *reinterpret_cast<uint64_t*>(&val);
  return __popcll(uval);
}
template <>
__device__ inline uint64_t popc(uint64_t val)
{
  auto uval = *reinterpret_cast<uint64_t*>(&val);
  return __popcll(uval);
}
/** @} */

}  // namespace cugraph::ops::utils
