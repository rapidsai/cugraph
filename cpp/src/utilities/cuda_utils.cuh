/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <thrust/extrema.h>

namespace cugraph {
//
//  This should go into RAFT...
//
__device__ static __forceinline__ int64_t atomicMin(int64_t *addr, int64_t val)
{
  unsigned long long *addr_as_ull{reinterpret_cast<unsigned long long *>(addr)};
  unsigned long long *val_addr_as_ull{reinterpret_cast<unsigned long long *>(&val)};
  unsigned long long old        = *addr_as_ull;
  unsigned long long val_as_ull = *val_addr_as_ull;
  int64_t *p_old{reinterpret_cast<int64_t *>(&old)};
  unsigned long long expected;

  do {
    expected = old;
    old      = ::atomicCAS(addr_as_ull, expected, thrust::min(val_as_ull, expected));
  } while (expected != old);
  return *p_old;
}

__device__ static __forceinline__ int32_t atomicMin(int32_t *addr, int32_t val)
{
  return ::atomicMin(addr, val);
}

__device__ static __forceinline__ int64_t atomicAdd(int64_t *addr, int64_t val)
{
  unsigned long long *addr_as_ull{reinterpret_cast<unsigned long long *>(addr)};
  unsigned long long *val_addr_as_ull{reinterpret_cast<unsigned long long *>(&val)};
  unsigned long long old        = *addr_as_ull;
  unsigned long long val_as_ull = *val_addr_as_ull;
  int64_t *p_old{reinterpret_cast<int64_t *>(&old)};
  unsigned long long expected;

  do {
    expected = old;
    old      = ::atomicCAS(addr_as_ull, expected, (expected + val_as_ull));
  } while (expected != old);
  return *p_old;
}

__device__ static __forceinline__ int32_t atomicAdd(int32_t *addr, int32_t val)
{
  return ::atomicAdd(addr, val);
}

__device__ static __forceinline__ double atomicAdd(double volatile *addr, double val)
{
  return ::atomicAdd(const_cast<double *>(addr), val);
}

__device__ static __forceinline__ float atomicAdd(float volatile *addr, float val)
{
  return ::atomicAdd(const_cast<float *>(addr), val);
}

}  // namespace cugraph
