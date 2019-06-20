/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date January, 2018
 * @version v1.4
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Host/Algorithm.hpp"                    //xlib::upper_bound_left
#include "Device/DataMovement/RegReordering.cuh" //xlib::shuffle_reordering
#include "Device/Util/Basic.cuh"                 //xlib::sync
#include "Device/Util/DeviceProperties.cuh"      //xlib::WARP_SIZE
#include "Host/Metaprogramming.hpp"              //xlib::get_arity

namespace xlib {

template<unsigned ITEMS_PER_BLOCK, typename T>
__global__
void binarySearchLBPartition(const T* __restrict__ d_prefixsum,
                             int                   prefixsum_size,
                             int*     __restrict__ d_partitions,
                             int                   num_partitions) {

    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_partitions; i += stride) {
    	T searched      = static_cast<T>(i) * ITEMS_PER_BLOCK;
		d_partitions[i] = xlib::upper_bound_left(d_prefixsum, prefixsum_size,
                                                 searched);
    }
    if (id == 0)
        d_partitions[num_partitions] = prefixsum_size - 2;
}

//==============================================================================
//==============================================================================

template<unsigned BLOCK_SIZE, bool LAST_BLOCK,
         unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void blockBinarySearchLB(const T* __restrict__ d_prefixsum,
                         int                   block_search_low,
                         T*       __restrict__ smem_prefix,
                         int                   smem_size,
                         int                 (&reg_pos)[ITEMS_PER_THREAD],
                         T                   (&reg_offset)[ITEMS_PER_THREAD]) {

    T   searched  = block_search_low +
                    static_cast<T>(threadIdx.x) * ITEMS_PER_THREAD;

    auto smem_tmp = smem_prefix + threadIdx.x;
    auto d_tmp    = d_prefixsum  + threadIdx.x;

    for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE) {
        *smem_tmp = *d_tmp;
        smem_tmp += BLOCK_SIZE;
        d_tmp    += BLOCK_SIZE;
    }

    // ALTERNATIVE 1
    //for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE)
    //    smem_prefix[i] = d_prefixsum[i];

    // ALTERNATIVE 2
    /*auto smem_tmp = smem_prefix + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int INDEX = i * BLOCK_SIZE;
        bool       pred = INDEX + threadIdx.x < smem_size;
        smem_tmp[INDEX] = d_tmp[(pred) ? INDEX : smem_size - 1];
        if (INDEX >= smem_size)
            break;
    }*/

    xlib::sync<BLOCK_SIZE>();

    int smem_pos = xlib::upper_bound_left(smem_prefix, smem_size, searched);
    T   next     = smem_prefix[smem_pos + 1];
    T   offset   = searched - smem_prefix[smem_pos];
    T   limit    = smem_prefix[smem_size - 1];

    const int LOWEST = xlib::numeric_limits<int>::lowest;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i]    = (!LAST_BLOCK || searched < limit) ? smem_pos : LOWEST;
        reg_offset[i] = offset;
        searched++;
        bool pred = (searched == next);
        offset    = (pred) ? 0 : offset + 1;
        smem_pos  = (pred) ? smem_pos + 1 : smem_pos;
        next      = smem_prefix[smem_pos + 1];
    }
    xlib::sync<BLOCK_SIZE>();
}

//==============================================================================
//==============================================================================

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, bool LAST_BLOCK,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB2(const int* __restrict__ d_partitions,
                     int                     num_partitions,
                     const T*   __restrict__ d_prefixsum,
                     int                     prefixsum_size,
                     void*      __restrict__ smem,
                     const Lambda&           lambda) {

    static_assert(xlib::get_arity<Lambda>() == 2, "binarySearchLB2 must have "
                  "lambda expression with two arguments");
    const unsigned ITEMS_PER_WARP  = xlib::WARP_SIZE * ITEMS_PER_THREAD;
    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

    int reg_pos   [ITEMS_PER_THREAD];
    T   reg_offset[ITEMS_PER_THREAD];

    int  block_start_pos  = d_partitions[ blockIdx.x ];
    int  block_end_pos    = d_partitions[ blockIdx.x + 1 ];
    int  smem_size        = block_end_pos - block_start_pos + 2;
    int  block_search_low = blockIdx.x * ITEMS_PER_BLOCK;
    auto smem_prefix      = static_cast<T*>(smem);

    blockBinarySearchLB<BLOCK_SIZE, LAST_BLOCK>
        (d_prefixsum + block_start_pos, block_search_low,
         smem_prefix, smem_size, reg_pos, reg_offset);

    xlib::smem_reordering(reg_pos, smem_prefix);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i] += block_start_pos;
        assert(reg_pos[i] < prefixsum_size);
    }

    int id    = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int index = (id / xlib::WARP_SIZE) * ITEMS_PER_WARP + xlib::lane_id();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (!LAST_BLOCK || reg_pos[i] >= 0) {
            assert(reg_pos[i] < prefixsum_size);
            lambda(reg_pos[i], index + i * xlib::WARP_SIZE);
        }
    }
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, bool LAST_BLOCK,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB3(const int* __restrict__ d_partitions,
                     int                     num_partitions,
                     const T*   __restrict__ d_prefixsum,
                     int                     prefixsum_size,
                     void*      __restrict__ smem,
                     const Lambda&           lambda) {
    static_assert(xlib::get_arity<Lambda>() == 3, "binarySearchLB3 must have "
                  "lambda expression with three arguments");
    const unsigned ITEMS_PER_WARP  = xlib::WARP_SIZE * ITEMS_PER_THREAD;
    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

    int reg_pos   [ITEMS_PER_THREAD];
    T   reg_offset[ITEMS_PER_THREAD];

    int  block_start_pos  = d_partitions[ blockIdx.x ];
    int  block_end_pos    = d_partitions[ blockIdx.x + 1 ];
    int  smem_size        = block_end_pos - block_start_pos + 2;
    int  block_search_low = blockIdx.x * ITEMS_PER_BLOCK;
    auto smem_prefix      = static_cast<T*>(smem);

    blockBinarySearchLB<BLOCK_SIZE, LAST_BLOCK>
        (d_prefixsum + block_start_pos, block_search_low,
         smem_prefix, smem_size, reg_pos, reg_offset);

    xlib::smem_reordering(reg_pos, reg_offset, smem_prefix);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i] += block_start_pos;
        assert(reg_pos[i] < prefixsum_size);
    }

    int id    = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int index = (id / xlib::WARP_SIZE) * ITEMS_PER_WARP + xlib::lane_id();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (!LAST_BLOCK || reg_pos[i] >= 0) {
            assert(reg_pos[i] < prefixsum_size);
            lambda(reg_pos[i], reg_offset[i], index + i * xlib::WARP_SIZE);
        }
    }
}

//==============================================================================
//==============================================================================
/*
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         bool LAST_BLOCK = true, typename T>
__device__ __forceinline__
void blockBinarySearchLB3(const T* __restrict__ d_prefixsum,
                          int                   smem_size,
                          int                   block_search_low,
                          T*       __restrict__ smem_prefix,
                          int                 (&reg_pos)[ITEMS_PER_THREAD]) {

    T   searched  = block_search_low +
                    static_cast<T>(threadIdx.x) * ITEMS_PER_THREAD;
    auto smem_tmp = smem_prefix + threadIdx.x;
    auto d_tmp    = d_prefixsum + threadIdx.x;

    for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE) {
        *smem_tmp = *d_tmp;
        smem_tmp += BLOCK_SIZE;
        d_tmp    += BLOCK_SIZE;
    }
    xlib::sync<BLOCK_SIZE>();

    int smem_pos = xlib::upper_bound_left(smem_prefix, smem_size, searched);
    T   next     = smem_prefix[smem_pos + 1];
    T   limit    = smem_prefix[smem_size - 1];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i] = (!LAST_BLOCK || searched < limit) ? smem_pos : smem_size;
        searched++;
        bool pred = (searched == next);
        smem_pos  = (pred) ? smem_pos + 1 : smem_pos;
        next      = smem_prefix[smem_pos + 1];
    }
    xlib::sync<BLOCK_SIZE>();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        smem_prefix[threadIdx.x * ITEMS_PER_THREAD + i] = reg_pos[i];

    xlib::sync<BLOCK_SIZE>();
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         bool LAST_BLOCK = true, typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB3(const int* __restrict__ d_partitions,
                     int                     num_partitions,
                     const T*   __restrict__ d_prefixsum,
                     int                     prefixsum_size,
                     void*      __restrict__ smem,
                     const Lambda&           lambda) {

    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

    int reg_pos    [ITEMS_PER_THREAD];
    int reg_indices[ITEMS_PER_THREAD];
    T   reg_offset [ITEMS_PER_THREAD];

    int  block_start_pos  = d_partitions[ blockIdx.x ];
    int  block_end_pos    = d_partitions[ blockIdx.x + 1 ];
    int  smem_size        = block_end_pos - block_start_pos + 2;
    int  block_search_low = blockIdx.x * ITEMS_PER_BLOCK;
    auto smem_buffer      = static_cast<T*>(smem);// + ITEMS_PER_BLOCK;

    blockBinarySearchLB3<BLOCK_SIZE, ITEMS_PER_THREAD, LAST_BLOCK>
        (d_prefixsum + block_start_pos, smem_size, block_search_low,
         smem_buffer, reg_pos);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int index      = threadIdx.x + i * BLOCK_SIZE;
        reg_pos[i]     = smem_buffer[index];
        reg_indices[i] = block_search_low + index;
    }

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        reg_pos[i] += block_start_pos;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (!LAST_BLOCK || reg_pos[i] < block_start_pos + smem_size)
            lambda(reg_pos[i], reg_offset[i], reg_indices[i]);
    }
}*/

} // namespace xlib
