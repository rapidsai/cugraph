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

#include "Device/Util/Basic.cuh"
#include "Device/Util/DeviceProperties.cuh"
#include <type_traits>

// N === 2
/*
if (lane_id() % N == 0)
    swap
    col = lane_id() / (WARP_SIZE / N)
source = (lane_id() * N + col) % WARP_SIZE;
if (lane_id() % N == 0)
    swap
*/
namespace xlib {

namespace detail {

template<int SIZE = 1, int OFFSET = 0, int LEFT_BOUND = 0>
struct ThreadToWarpIndexing {

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__
     static void run(T   (&reg)[ITEMS_PER_THREAD],
                     void* smem_thread,
                     void* smem_warp) {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            static_cast<T*>(smem_thread)[i] = reg[i];
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            reg[i] = static_cast<T*>(smem_warp)[i * xlib::WARP_SIZE];
    }

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__
    static void run(const T (&reg_in)[ITEMS_PER_THREAD],
                    T       (&reg_out)[ITEMS_PER_THREAD],
                    T*      smem_thread,
                    T*      smem_warp) {

        const int      TH_NUM = xlib::WARP_SIZE / (ITEMS_PER_THREAD / SIZE);
        const int RIGHT_BOUND = LEFT_BOUND + TH_NUM;

        if (static_cast<int>(lane_id()) >= LEFT_BOUND &&
                lane_id() < RIGHT_BOUND) {
            #pragma unroll
             for (int i = 0; i < ITEMS_PER_THREAD; i++)
                smem_thread[i] = reg_in[i];
        }
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            reg_out[OFFSET + i] = smem_warp[i * xlib::WARP_SIZE];

        const bool END_COND = LEFT_BOUND + TH_NUM >= xlib::WARP_SIZE;
        ThreadToWarpIndexing<END_COND ? 0 : SIZE,
                             OFFSET + SIZE, LEFT_BOUND + TH_NUM>
            ::run(reg_in, reg_out, smem_thread, smem_warp);
    }

    static const int TOTAL_SIZE = WARP_SIZE * SIZE;

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__ static
    void run(const T (&reg_in)[ITEMS_PER_THREAD],
             T (&reg_out)[ITEMS_PER_THREAD],
             T* smem, T* smem_warp, int& offset, int& index) {

        while (index < ITEMS_PER_THREAD && offset < TOTAL_SIZE)
            smem[offset++] = reg_in[index++];
        offset -= TOTAL_SIZE;

        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            reg_out[OFFSET + j] = smem_warp[j * WARP_SIZE];

        const bool END_COND = OFFSET + SIZE > ITEMS_PER_THREAD;
        ThreadToWarpIndexing<END_COND ? 0 : SIZE, SIZE, OFFSET + SIZE>
            ::run(reg_in, reg_out, smem, smem_warp, offset, index);
    }
};

template<int OFFSET, int LEFT_BOUND>
struct ThreadToWarpIndexing<0, OFFSET, LEFT_BOUND> {

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__ static
    void run(T (&)[ITEMS_PER_THREAD], T*, T*) {}

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__ static
    void run(const T (&)[ITEMS_PER_THREAD], T (&)[ITEMS_PER_THREAD], T*, T*) {}

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__ static
    void run(const T (&)[ITEMS_PER_THREAD], T (&)[ITEMS_PER_THREAD],
             T*, T*, int&, int&) {}
};

} // namespace detail

/**
 * ITEMS_PER_THREAD == |reg1| == |reg2|     in the example: 3
 * SMEM_ITEMS == shared memory items        in the example: 2
 *
 *  before:
 *  thread0: reg1 = { 1,  2,  3,  4, 5 }
 *  thread1: reg1 = { 6,  7,  8, 9, 10 }
 *  thread2: reg1 = { 11, 12, 13, 14, 15}
 *  thread3: reg1 = { 16, 17, 18, 19, 20}
 *
 *  after:
 *  thread0: reg1 = { 1, 5,  9, 13, 17 }
 *  thread1: reg1 = { 2, 6, 10, 14, 18 }
 *  thread2: reg1 = { 3, 7, 11, 15, 19 }
 *  thread3: reg1 = { 4, 8, 12, 16, 20 }
 *
 */
template<unsigned SMEM_ITEMS_ = 0, unsigned ITEMS_PER_THREAD,
         typename T, typename R>
 __device__ __forceinline__
typename std::enable_if<sizeof(T) == sizeof(R)>::type
threadToWarpIndexing(T (&reg1)[ITEMS_PER_THREAD],
                     R (&reg2)[ITEMS_PER_THREAD],
                     void* smem) {

    const int SMEM_ITEMS_TMP = SMEM_ITEMS_ ? SMEM_ITEMS_ :
                               xlib::smem_per_thread<T>();
    static_assert(ITEMS_PER_THREAD <= SMEM_ITEMS_TMP,
                 "n. register > shared memory : to do");

    const unsigned    SMEM_ITEMS = ITEMS_PER_THREAD;
    const unsigned SIZE_PER_WARP = xlib::WARP_SIZE * SMEM_ITEMS;
    T*       smemT = static_cast<T*>(smem) + xlib::warp_id() * SIZE_PER_WARP;
    T* smem_thread = smemT + xlib::lane_id() * SMEM_ITEMS;
    T*   smem_warp = smemT + xlib::lane_id();

    detail::ThreadToWarpIndexing<>::run(reg1, smem_thread, smem_warp);
    detail::ThreadToWarpIndexing<>::run(reg2, smem_thread, smem_warp);
}

template<unsigned ITEMS_PER_THREAD, unsigned SMEM_ITEMS_ = 0, typename T>
 __device__ __forceinline__
void threadToWarpIndexing(T (&reg)[ITEMS_PER_THREAD], T* smem) {
    using namespace detail;
    const unsigned SMEM_ITEMS = SMEM_ITEMS_ ? SMEM_ITEMS_ :
                                xlib::smem_per_thread<T>();

    smem        += xlib::warp_id() * xlib::WARP_SIZE * SMEM_ITEMS;
    T* smem_warp = smem + xlib::lane_id();

    if (ITEMS_PER_THREAD <= SMEM_ITEMS || ITEMS_PER_THREAD % SMEM_ITEMS == 0) {
        const unsigned SIZE = xlib::min(SMEM_ITEMS, ITEMS_PER_THREAD);
        T*      smem_thread = smem + xlib::lane_id() * SIZE;

        if (ITEMS_PER_THREAD <= SMEM_ITEMS)
            detail::ThreadToWarpIndexing<>::run(reg, smem_thread, smem_warp);
        else {
            T tmp[ITEMS_PER_THREAD];
            detail::ThreadToWarpIndexing<SIZE>::run(reg, tmp, smem_thread,
                                                    smem_warp);
            reg_copy(tmp, reg);
        }
    }
    else {
        T tmp[ITEMS_PER_THREAD];
        int offset = lane_id() * ITEMS_PER_THREAD;
        int  index = 0;
        ThreadToWarpIndexing<SMEM_ITEMS>::run(reg, tmp, smem, smem_warp,
                                              offset, index);
        reg_copy(tmp, reg);
    }
}

} // namespace xlib
