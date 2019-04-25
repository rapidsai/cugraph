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

#ifdef _MSC_VER
#include <stdint.h>
#else
#include <inttypes.h>
#endif

#define DEFAULT_MASK 0xffffffff

#define USE_CG 1
//(__CUDACC_VER__ >= 80500)


namespace nvgraph
{
namespace utils
{
    static __device__ __forceinline__ int lane_id()
    {
        int id;
        asm ( "mov.u32 %0, %%laneid;" : "=r"(id) );
        return id;
    }

    static __device__ __forceinline__ int lane_mask_lt()
    {
        int mask;
        asm ( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
        return mask;
    }

    static __device__ __forceinline__ int lane_mask_le()
    {
        int mask;
        asm ( "mov.u32 %0, %%lanemask_le;" : "=r"(mask) );
        return mask;
    }

    static __device__ __forceinline__ int warp_id()
    {
        return threadIdx.x >> 5;
    }

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

    static __device__ __forceinline__ int shfl(int r, int lane, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#if USE_CG
        return __shfl_sync(mask, r, lane, bound );
#else
        return __shfl(r, lane, bound );
#endif
    #else
        return 0;
    #endif
    }

    static __device__ __forceinline__ float shfl(float r, int lane, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#if USE_CG
        return __shfl_sync(mask, r, lane, bound );
#else
        return __shfl(r, lane, bound );
#endif
    #else
        return 0.0f;
    #endif
    }

    /// Warp shuffle down function
    /** Warp shuffle functions on 64-bit floating point values are not
    *  natively implemented as of Compute Capability 5.0. This
    *  implementation has been copied from
    *  (http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler).
    *  Once this is natively implemented, this function can be replaced
    *  by __shfl_down.
    *
    */
    static __device__ __forceinline__ double shfl(double r, int lane, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_sync(mask, a.x, lane, bound);
        a.y = __shfl_sync(mask, a.y, lane, bound);
        return *reinterpret_cast<double*>(&a);
#else
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl(a.x, lane, bound);
        a.y = __shfl(a.y, lane, bound);
        return *reinterpret_cast<double*>(&a);
#endif
    #else
        return 0.0;
    #endif
    }

    static __device__ __forceinline__ long long shfl(long long r, int lane, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_sync(mask, a.x, lane, bound);
        a.y = __shfl_sync(mask, a.y, lane, bound);
        return *reinterpret_cast<long long*>(&a);
#else
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl(a.x, lane, bound);
        a.y = __shfl(a.y, lane, bound);
        return *reinterpret_cast<long long*>(&a);
#endif
    #else
        return 0.0;
    #endif
    }

    static __device__ __forceinline__ int shfl_down(int r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        return __shfl_down_sync( mask, r, offset, bound );
#else
        return __shfl_down( r, offset, bound );
#endif
    #else
        return 0.0f;
    #endif
    }

    static __device__ __forceinline__ float shfl_down(float r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        return __shfl_down_sync( mask, r, offset, bound );
#else
        return __shfl_down( r, offset, bound );
#endif
    #else
        return 0.0f;
    #endif
    }

    static __device__ __forceinline__ double shfl_down(double r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_down_sync(mask, a.x, offset, bound);
        a.y = __shfl_down_sync(mask, a.y, offset, bound);
        return *reinterpret_cast<double*>(&a);
#else
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_down(a.x, offset, bound);
        a.y = __shfl_down(a.y, offset, bound);
        return *reinterpret_cast<double*>(&a);
#endif
    #else
        return 0.0;
    #endif
    }

    static __device__ __forceinline__ long long shfl_down(long long r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_down_sync(mask, a.x, offset, bound);
        a.y = __shfl_down_sync(mask, a.y, offset, bound);
        return *reinterpret_cast<long long*>(&a);
#else
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_down(a.x, offset, bound);
        a.y = __shfl_down(a.y, offset, bound);
        return *reinterpret_cast<long long*>(&a);
#endif
    #else
        return 0.0;
    #endif
    }

    // specifically for triangles counting
    static __device__ __forceinline__ uint64_t shfl_down(uint64_t r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_down_sync(mask, a.x, offset, bound);
        a.y = __shfl_down_sync(mask, a.y, offset, bound);
        return *reinterpret_cast<uint64_t*>(&a);
#else
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_down(mask, a.x, offset, bound);
        a.y = __shfl_down(mask, a.y, offset, bound);
        return *reinterpret_cast<uint64_t*>(&a);
#endif
    #else
        return 0.0;
    #endif
    }

    static __device__ __forceinline__ int shfl_up(int r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        return __shfl_up_sync( mask, r, offset, bound );
#else
        return __shfl_up( r, offset, bound );
#endif
    #else
        return 0.0f;
    #endif
    }

    static __device__ __forceinline__ float shfl_up(float r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        return __shfl_up_sync( mask, r, offset, bound );
#else
        return __shfl_up( r, offset, bound );
#endif
    #else
        return 0.0f;
    #endif
    }

    static __device__ __forceinline__ double shfl_up(double r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_up_sync(mask, a.x, offset, bound);
        a.y = __shfl_up_sync(mask, a.y, offset, bound);
        return *reinterpret_cast<double*>(&a);
#else
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_up(a.x, offset, bound);
        a.y = __shfl_up(a.y, offset, bound);
        return *reinterpret_cast<double*>(&a);
#endif
    #else
        return 0.0;
    #endif
    }

    static __device__ __forceinline__ long long shfl_up(long long r, int offset, int bound = 32, int mask = DEFAULT_MASK)
    {
    #if __CUDA_ARCH__ >= 300
#ifdef USE_CG
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_up_sync(mask, a.x, offset, bound);
        a.y = __shfl_up_sync(mask, a.y, offset, bound);
        return *reinterpret_cast<long long*>(&a);
#else
        int2 a = *reinterpret_cast<int2*>(&r);
        a.x = __shfl_up(a.x, offset, bound);
        a.y = __shfl_up(a.y, offset, bound);
        return *reinterpret_cast<long long*>(&a);
#endif
    #else
        return 0.0;
    #endif
    }
}

}
