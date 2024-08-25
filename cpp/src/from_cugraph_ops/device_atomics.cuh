/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace cugraph::ops::utils {

/**
 * @defgroup AtomicMax Device atomic max operation
 *
 * @{
 */
template <typename DataT>
__device__ inline DataT atomic_max(DataT* address, DataT val)
{
  return atomicMax(address, val);
}
template <>
__device__ inline float atomic_max(float* address, float val)
{
  using u32_t          = unsigned int;
  auto* address_as_u32 = reinterpret_cast<u32_t*>(address);
  u32_t old            = *address_as_u32, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_u32, assumed, __float_as_uint(max(val, __uint_as_float(assumed))));
  } while (assumed != old);
  return __uint_as_float(old);
}
template <>
__device__ inline double atomic_max(double* address, double val)
{
  using u64_t          = unsigned long long;  // NOLINT(google-runtime-int)
  auto* address_as_ull = reinterpret_cast<u64_t*>(address);
  u64_t old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(
      address_as_ull, assumed, __double_as_longlong(max(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}
template <>
__device__ inline int64_t atomic_max(int64_t* address, int64_t val)
{
  using u64_t          = unsigned long long;  // NOLINT(google-runtime-int)
  auto* val_as_u64     = reinterpret_cast<u64_t*>(&val);
  auto* address_as_u64 = reinterpret_cast<u64_t*>(address);
  auto ret             = atomicMax(address_as_u64, *val_as_u64);
  return *reinterpret_cast<int64_t*>(&ret);
}
template <>
__device__ inline uint64_t atomic_max(uint64_t* address, uint64_t val)
{
  using u64_t          = unsigned long long;  // NOLINT(google-runtime-int)
  auto* val_as_u64     = reinterpret_cast<u64_t*>(&val);
  auto* address_as_u64 = reinterpret_cast<u64_t*>(address);
  auto ret             = atomicMax(address_as_u64, *val_as_u64);
  return *reinterpret_cast<uint64_t*>(&ret);
}
/** @} */

}  // namespace cugraph::ops::utils
