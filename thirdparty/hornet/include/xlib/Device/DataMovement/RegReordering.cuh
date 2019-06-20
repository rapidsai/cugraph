/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
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

namespace xlib {

/**
 * ITEMS_PER_THREAD == |reg1| == |reg2|     in the example: 3
 * SMEM_ITEMS == shared memory items        in the example: 2
 *
 *  before:
 *  thread0: reg1 = {  1,  2,  3,  4,  5 }
 *  thread1: reg1 = {  6,  7,  8,  9, 10 }
 *  thread2: reg1 = { 11, 12, 13, 14, 15 }
 *  thread3: reg1 = { 16, 17, 18, 19, 20 }
 *
 *  after:
 *  thread0: reg1 = { 1, 5,  9, 13, 17 }
 *  thread1: reg1 = { 2, 6, 10, 14, 18 }
 *  thread2: reg1 = { 3, 7, 11, 15, 19 }
 *  thread3: reg1 = { 4, 8, 12, 16, 20 }
 *
 */
template<unsigned ITEMS_PER_THREAD, typename T, typename R>
 __device__ __forceinline__
void smem_reordering(T   (&reg1)[ITEMS_PER_THREAD],
                     R   (&reg2)[ITEMS_PER_THREAD],
                     void* smem);

template<unsigned SMEM_PER_WARP = 0, unsigned ITEMS_PER_THREAD, typename T>
 __device__ __forceinline__
void smem_reordering(T (&reg)[ITEMS_PER_THREAD], void* smem);

//------------------------------------------------------------------------------

template<typename T, int SIZE>
__device__ __forceinline__
void shuffle_reordering(T (&A)[SIZE]);

template<typename T>
__device__ __forceinline__
void shuffle_reordering_v4(T (&A)[8]);

template<typename T, int SIZE>
__device__ __forceinline__
void shuffle_reordering_inv(T (&A)[SIZE]);

//------------------------------------------------------------------------------

template<unsigned ITEMS_PER_THREAD, typename T>
 __device__ __forceinline__
typename std::enable_if<xlib::mcd(ITEMS_PER_THREAD, xlib::WARP_SIZE) == 1 ||
                        xlib::is_power2(ITEMS_PER_THREAD)>::type
reordering_dispatch(T (&reg)[ITEMS_PER_THREAD], void* smem);


template<unsigned ITEMS_PER_THREAD, typename T>
 __device__ __forceinline__
typename std::enable_if<xlib::mcd(ITEMS_PER_THREAD, xlib::WARP_SIZE) != 1 &&
                        !xlib::is_power2(ITEMS_PER_THREAD)>::type
reordering_dispatch(T (&reg)[ITEMS_PER_THREAD], void* smem);

} // namespace xlib

#include "impl/RegReordering.i.cuh"
