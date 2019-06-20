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

namespace xlib {

/**
 * @brief
 * @remark \f$ num_partitions = \lceil d_prefixsum[size] / PARTITION_SIZE \rceil
 *         \f$
 * @remark \f$|d_partition| == num_partitions + 1\f$
 * @remark size < \f$2^{31}\f$; num_partitions < \f$2^{31}\f$
 */
template<unsigned PARTITION_SIZE = 0, typename T>
__global__ void blockPartition(const T* __restrict__ d_prefixsum,
                               int                   prefixsum_size,
                               int*     __restrict__ d_partitions,
                               int                   num_partitions);

//------------------------------------------------------------------------------
/**
 * @brief
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD = 0,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB(const T* __restrict__ d_prefixsum,
                    int                   prefixsum_size,
                    void*    __restrict__ smem,
                    const Lambda&         lambda);

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD = 0,
         typename T, typename Lambda>
__device__ __forceinline__
void simpleBinarySearchLB(const T* __restrict__ d_prefixsum,
                    int                   prefixsum_size,
                    void*    __restrict__ smem,
                    const Lambda&         lambda);

/**
 * @brief
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD = 0,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLBAllPos(const T* __restrict__ d_prefixsum,
                          int                   prefixsum_size,
                          void*    __restrict__ smem,
                          const Lambda&         lambda);

//------------------------------------------------------------------------------
/**
 * @brief
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD = 0,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB(const T* __restrict__ d_prefixsum,
                    int                   prefixsum_size,
                    int*     __restrict__ d_partitions,
                    void*    __restrict__ smem,
                    const Lambda&         lambda);

/**
 * @brief
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD = 0,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLBAllPos(const T* __restrict__ d_prefixsum,
                          int                   prefixsum_size,
                          int*     __restrict__ d_partitions,
                          void*    __restrict__ smem,
                          const Lambda&         lambda);

//==============================================================================

/**
 * @brief
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD = 0,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLBAllPosNoDup(const T* __restrict__ d_prefixsum,
                               int                   prefixsum_size,
                               void*    __restrict__ smem,
                               const Lambda&         lambda);

} // namespace xlib

#include "impl/BinarySearchLB.i.cuh"
