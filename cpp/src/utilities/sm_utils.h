/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <type_traits>

#ifdef _MSC_VER
#include <stdint.h>
#else
#include <inttypes.h>
#endif

#define DEFAULT_MASK 0xffffffff

#define USE_CG 1
//(__CUDACC_VER__ >= 80500)

namespace cugraph {
namespace detail {
namespace utils {
static __device__ __forceinline__ int lane_id()
{
  int id;
  asm("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

static __device__ __forceinline__ int lane_mask_lt()
{
  int mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

static __device__ __forceinline__ int lane_mask_le()
{
  int mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

static __device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }

static __device__ __forceinline__ unsigned int ballot(int p, int mask = DEFAULT_MASK)
{
#if __CUDA_ARCH__ >= 300
#if USE_CG
  return __ballot_sync(mask, p);
#else
  return __ballot(p);
#endif
#else
  return 0;
#endif
}

template <typename T>                                                           
static __device__ __forceinline__ T shfl(
    T r, 
    int lane, 
    int bound = 32, 
    int mask = DEFAULT_MASK
    ){
  return __shfl_sync(mask, r, lane, bound);
}

}  // namespace utils
}  // namespace detail
}  // namespace cugraph
