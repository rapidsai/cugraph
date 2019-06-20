/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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

#include "Device/DataMovement/Indexing.cuh"
#include "Device/Util/DeviceProperties.cuh"
#include "Device/Util/Basic.cuh"
#include "Host/Algorithm.hpp"

namespace xlib {
namespace detail {

template<unsigned PARTITION_SIZE, typename T>
__global__ void blockPartition(const T* __restrict__ d_prefixsum,
                               int                   prefixsum_size,
                               int*     __restrict__ d_partition,
                               int                   num_partitions) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_partitions; i += stride) {
    	T     searched = static_cast<T>(i) * PARTITION_SIZE;
		d_partition[i] = xlib::upper_bound_left(d_prefixsum, prefixsum_size,
                                                searched);
    }
    if (id == 0)
        d_partition[num_partitions] = prefixsum_size - 2;
}

template<unsigned BLOCK_SIZE, typename T, unsigned ITEMS_PER_THREAD>
__device__ __forceinline__
void threadPartitionAuxLoop(const T* __restrict__ ptr,
                            int                   block_start_pos,
                            int                   chunk_size,
                            T                     searched,
                            T* __restrict__       smem_prefix,
                            int               (&reg_pos)[ITEMS_PER_THREAD],
                            T                 (&reg_offset)[ITEMS_PER_THREAD]) {

    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;
    T low_limit = 0;
    while (chunk_size > 0) {
        int smem_size = ::min(chunk_size, ITEMS_PER_BLOCK);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int index = i * BLOCK_SIZE + threadIdx.x;
            if (index < smem_size)
                smem_prefix[index] = ptr[index];
        }
        xlib::sync<BLOCK_SIZE>();

        int   ubound = xlib::upper_bound_left(smem_prefix, smem_size, searched);
        int smem_pos = ::min(::max(0, ubound), ITEMS_PER_BLOCK - 2);
        assert(smem_pos >= 0 && smem_pos + 1 < ITEMS_PER_BLOCK);
        T     offset = ::max(searched - smem_prefix[smem_pos], 0);
        T       next = smem_prefix[smem_pos + 1];
        T high_limit = smem_prefix[smem_size - 1];

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            T loc_search = searched + i;
            if (loc_search < low_limit || loc_search >= high_limit)
                continue;
            if (loc_search == next) {
                do {
                    smem_pos++;
                    assert(smem_pos >= 0 && smem_pos + 1 < smem_size);
                    next = smem_prefix[smem_pos + 1];
                } while (loc_search == next);
                offset = 0;
            }
            reg_pos[i]    = block_start_pos + smem_pos;
            reg_offset[i] = offset;
            offset++;
        }
        xlib::sync<BLOCK_SIZE>();
        low_limit        = high_limit;
        chunk_size      -= ITEMS_PER_BLOCK - 1;
        ptr             += ITEMS_PER_BLOCK - 1;
        block_start_pos += ITEMS_PER_BLOCK - 1;
    }
}

/**
 * @brief
 * @details
 * @verbatim
 *    d_prefixsum input: 0, 3, 7, 10, 13
 *     ITEMS_PER_THREAD: 5
 *    reg_pos  output: t1(0, 0, 0, 1, 1) t2(1, 1, 2, 2, 2) t3(3, 3, 3, *, *)
 * reg_offset  output: t1(0, 1, 2, 0, 1) t2(2, 3, 0, 1, 2) t3(0, 1, 2, *, *)
 *                    *: undefined
 * @endverbatim
 *
 * @tparam BLOCK_SIZE
 * @tparam T
 * @tparam ITEMS_PER_THREAD
 * @param[in] d_partition
 * @param[in] d_prefixsum
 * @param[in] reg_pos
 * @param[in] reg_offset
 * @param[in] smem
 *
 * @warning |smem| must be equal to BLOCK_SIZE * ITEMS_PER_THREAD
 * @remark The best way to detect unused registers in the last thread block
 *            is to fill the `reg_offset` array with a special value
 * @warning requires `__syncthreads()` at the end if the shared memory is used
 * @remark the function uses static indexing for `reg_pos` and `reg_offset`
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void threadPartition(const T* __restrict__ d_prefixsum, int prefixsum_size,
                     void*    __restrict__ smem,
                     int (&reg_pos)[ITEMS_PER_THREAD],
                     T   (&reg_offset)[ITEMS_PER_THREAD],
                     int block_start_pos,
                     int block_end_pos) {

    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;
    T  block_search_low = static_cast<T>(blockIdx.x) * ITEMS_PER_BLOCK;
    T          searched = block_search_low +
                          static_cast<T>(threadIdx.x) * ITEMS_PER_THREAD;

    int      chunk_size = block_end_pos - block_start_pos + 2;
    const T*        ptr = d_prefixsum + block_start_pos;

    if (blockIdx.x == gridDim.x - 1)
        xlib::reg_fill(reg_pos, -1);

    auto smem_prefix = static_cast<T*>(smem);

    detail::threadPartitionAuxLoop<BLOCK_SIZE>
        (ptr, block_start_pos, chunk_size, searched,
         smem_prefix, reg_pos, reg_offset);
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void threadPartitionNoDup(const T* __restrict__ d_prefixsum,
                          int                   prefixsum_size,
                          void*    __restrict__ smem,
                          int                 (&reg_pos)[ITEMS_PER_THREAD],
                          T                   (&reg_offset)[ITEMS_PER_THREAD],
                          int                   block_start_pos,
                          int                   block_end_pos) {

    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;
    T block_search_low = static_cast<T>(blockIdx.x) * ITEMS_PER_BLOCK;
    T searched         = block_search_low +
                         static_cast<T>(threadIdx.x) * ITEMS_PER_THREAD;

    const T* ptr         = d_prefixsum + block_start_pos;
    int      smem_size   = block_end_pos - block_start_pos + 2;
    auto     smem_prefix = static_cast<T*>(smem);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int index = i * BLOCK_SIZE + threadIdx.x;
        if (index < smem_size)
            smem_prefix[index] = ptr[index];
    }
    xlib::sync<BLOCK_SIZE>();

    int smem_pos = xlib::upper_bound_left(smem_prefix, smem_size, searched);
    T   next     = smem_prefix[smem_pos + 1];
    T   offset   = searched - smem_prefix[smem_pos];
    T   limit    = smem_prefix[smem_size - 1];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i]    = searched < limit ? block_start_pos + smem_pos : -1;
        reg_offset[i] = offset;
        searched++;
        if (searched == next) {
            smem_pos++;
            next   = smem_prefix[smem_pos + 1];
            offset = 0;
        } else
            offset++;
    }
    xlib::sync<BLOCK_SIZE>();
}
//==============================================================================
//==============================================================================

//==============================================================================
//==============================================================================

template<bool BLOCK_PARTITION_FROM_GLOBAL, bool NO_DUPLICATE,
         bool LAST_BLOCK_CHECK, unsigned BLOCK_SIZE,
         unsigned ITEMS_PER_THREAD = 0, typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLBGen(const T* __restrict__ d_prefixsum,
                       int                   prefixsum_size,
                       int*     __restrict__ d_partitions,
                       void*    __restrict__ smem,
                       const Lambda&         lambda) {

    const unsigned _ITEMS_PER_THREAD = ITEMS_PER_THREAD == 0 ?
                                       smem_per_thread<T, BLOCK_SIZE>() :
                                       ITEMS_PER_THREAD;
    int reg_pos[_ITEMS_PER_THREAD];
    T   reg_offset[_ITEMS_PER_THREAD];

    static_assert(!BLOCK_PARTITION_FROM_GLOBAL, "Deprecated");
    int block_start_pos, block_end_pos;
    if (BLOCK_PARTITION_FROM_GLOBAL) {
        block_start_pos = d_partitions[ blockIdx.x ];
        block_end_pos   = d_partitions[ blockIdx.x + 1 ];
    }
    else {
        const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * _ITEMS_PER_THREAD;
        const unsigned            IDX1 = BLOCK_SIZE >= 64 ? xlib::WARP_SIZE : 1;

        auto smem_prefix = static_cast<T*>(smem);
        if (threadIdx.x == 0) {
            T block_search = static_cast<T>(blockIdx.x) * ITEMS_PER_BLOCK;
            smem_prefix[0] = xlib::upper_bound_left(d_prefixsum, prefixsum_size,
                                                    block_search);
        }
        else if (threadIdx.x == IDX1) {
            T block_search = static_cast<T>(blockIdx.x + 1) * ITEMS_PER_BLOCK;
            smem_prefix[1] = blockIdx.x == gridDim.x - 1 ? prefixsum_size - 2 :
                             xlib::upper_bound_left(d_prefixsum, prefixsum_size,
                                                    block_search);
        }
        xlib::sync<BLOCK_SIZE>();
        block_start_pos = smem_prefix[0];
        block_end_pos   = smem_prefix[1];
        xlib::sync<BLOCK_SIZE>();
    }
    if (NO_DUPLICATE) {
        detail::threadPartitionNoDup<BLOCK_SIZE>
            (d_prefixsum, prefixsum_size, smem, reg_pos, reg_offset,
             block_start_pos, block_end_pos);
    }
    else {
        detail::threadPartition<BLOCK_SIZE>
            (d_prefixsum, prefixsum_size, smem, reg_pos, reg_offset,
             block_start_pos, block_end_pos);
    }

    threadToWarpIndexing<_ITEMS_PER_THREAD>(reg_pos, reg_offset, smem);

    //if (LAST_BLOCK_CHECK && blockIdx.x == gridDim.x - 1) {
        #pragma unroll
        for (int i = 0; i < _ITEMS_PER_THREAD; i++) {
            if (reg_pos[i] != -1) {
                assert(reg_pos[i] < prefixsum_size);
                lambda(reg_pos[i], reg_offset[i]);
            }
        }
    //}
    /*else {
        #pragma unroll
        for (int i = 0; i < _ITEMS_PER_THREAD; i++) {
            assert(reg_pos[i] < prefixsum_size);
            lambda(reg_pos[i], reg_offset[i]);
        }
    }*/
}


template<unsigned BLOCK_SIZE, typename T, typename Lambda>
__device__ __forceinline__
void simpleBinarySearchLBGen(const T* __restrict__ d_prefixsum,
                       int                   prefixsum_size,
                       void*    __restrict__ smem,
                       const Lambda&         lambda) {
    const unsigned ITEMS_PER_BLOCK = blockDim.x;
    T work_index = blockIdx.x * ITEMS_PER_BLOCK + threadIdx.x;
    int pos = xlib::upper_bound_left(d_prefixsum, prefixsum_size,
                                            work_index);
    if ((pos >= 0) && (pos < prefixsum_size - 1) && (work_index < d_prefixsum[prefixsum_size - 1])) {
        T offset = work_index - d_prefixsum[pos];
        lambda(pos, offset);
    }
}

} // namespace detail

//==============================================================================
//==============================================================================
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB(const T* __restrict__ d_prefixsum,
                    int                   prefixsum_size,
                    void*    __restrict__ smem,
                    const Lambda&         lambda) {

    detail::binarySearchLBGen<false, false, true, BLOCK_SIZE, ITEMS_PER_THREAD>
        (d_prefixsum, prefixsum_size, nullptr, smem, lambda);
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         typename T, typename Lambda>
__device__ __forceinline__
void simpleBinarySearchLB(const T* __restrict__ d_prefixsum,
                    int                   prefixsum_size,
                    void*    __restrict__ smem,
                    const Lambda&         lambda) {

    detail::simpleBinarySearchLBGen<BLOCK_SIZE>
        (d_prefixsum, prefixsum_size, smem, lambda);
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLBAllPos(const T* __restrict__ d_prefixsum,
                          int                   prefixsum_size,
                          void*    __restrict__ smem,
                          const Lambda&         lambda) {

    detail::binarySearchLBGen<false, false, false, BLOCK_SIZE, ITEMS_PER_THREAD>
        (d_prefixsum, prefixsum_size, nullptr, smem, lambda);
}

//==============================================================================

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB(const T* __restrict__ d_prefixsum,
                    int                   prefixsum_size,
                    int*     __restrict__ d_partitions,
                    void*    __restrict__ smem,
                    const Lambda&         lambda) {

    detail::binarySearchLBGen<true, false, true, BLOCK_SIZE, ITEMS_PER_THREAD>
        (d_prefixsum, prefixsum_size, d_partitions, smem, lambda);
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLBAllPos(const T* __restrict__ d_prefixsum,
                          int                   prefixsum_size,
                          int*     __restrict__ d_partitions,
                          void*    __restrict__ smem,
                          const Lambda&         lambda) {

    detail::binarySearchLBGen<true, false, false, BLOCK_SIZE, ITEMS_PER_THREAD>
        (d_prefixsum, prefixsum_size, d_partitions, smem, lambda);
}

//------------------------------------------------------------------------------

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLBAllPosNoDup(const T* __restrict__ d_prefixsum,
                               int                   prefixsum_size,
                               void*    __restrict__ smem,
                               const Lambda&         lambda) {

    detail::binarySearchLBGen<false, true, false, BLOCK_SIZE, ITEMS_PER_THREAD>
        (d_prefixsum, prefixsum_size, nullptr, smem, lambda);
}

} // namespace xlib
