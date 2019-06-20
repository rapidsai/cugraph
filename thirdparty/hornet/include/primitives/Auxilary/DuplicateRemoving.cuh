/**
 * @brief
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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

#include <Device/Util/DeviceProperties.cuh> //xlib::smem_per_block

namespace hornets_nest {

template<unsigned HASHTABLE_SIZE, int ITERS = 1>
__device__ __forceinline__
bool duplicate_removing_aux(vid_t vertex, void* smem) {
    static_assert(xlib::is_power2(HASHTABLE_SIZE),
                 "HASHTABLE_SIZE must be a power of two");
    const vid_t PRIME_ARRAY[] = { 1803059, 397379, 433781, 590041, 593689,
                                  931517, 1049897, 1285607, 1636007, 1803059 };

    //auto hash_table = static_cast<vid2_t*>(smem);
    auto hash_table = static_cast<volatile int64_t*>(smem);

    #pragma unroll
	for (int i = 0; i < ITERS; i++) {
        const unsigned hash = static_cast<unsigned>(vertex * PRIME_ARRAY[i]) %
                              HASHTABLE_SIZE;
    	vid2_t to_write = xlib::make2(static_cast<vid_t>(threadIdx.x), vertex);

    	hash_table[hash] = reinterpret_cast<volatile int64_t&>
                                (const_cast<volatile vid2_t&>(to_write));
        xlib::sync<xlib::WARP_SIZE>();

    	vid2_t recover = const_cast<vid2_t&>(
                         reinterpret_cast<volatile vid2_t&>(hash_table[hash]));
        if (recover.x != threadIdx.x && recover.y == vertex)
            return true;

        xlib::sync<xlib::WARP_SIZE>();
    }
    return false;
}

template<int ITERS = 1>
__device__ __forceinline__
bool is_duplicate(vid_t vertex) {
    const unsigned      SMEM_SIZE = xlib::smem_per_block<vid2_t, 128>();
    const unsigned HASHTABLE_SIZE = xlib::rounddown_pow2(SMEM_SIZE);

    return duplicate_removing_aux<HASHTABLE_SIZE, 1>(vertex, xlib::dyn_smem);

    /*const unsigned      WARP_SMEM = xlib::SMemPerWarp<vid2_t, 128>::value;
    const unsigned HASHTABLE_SIZE = xlib::rounddown_pow2(WARP_SMEM);
    void* warp_shmem = reinterpret_cast<vid2_t*>(xlib::dyn_smem) +
                        (WARP_SMEM * xlib::warp_id());

    return duplicate_removing_aux<HASHTABLE_SIZE, ITERS>(vertex, warp_shmem);*/
}

} // namespace hornets_nest
