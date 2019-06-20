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

#include "Device/Util/Basic.cuh"            //xlib::shfl
#include "Device/Util/DeviceProperties.cuh" //xlib::WARP_SIZE

namespace xlib {
namespace detail {

template<int SIZE, int INDEX = 0>
struct Unroll {
    template<typename Lambda>
    __device__ __forceinline__
    static void apply(const Lambda& lambda) {
        lambda(INDEX);
        Unroll<SIZE, INDEX + 1>::apply(lambda);
    }
};

template<int SIZE>
struct Unroll<SIZE, SIZE> {
    template<typename Lambda>
    __device__ __forceinline__
    static void apply(const Lambda&) {}
};

template<int SIZE = 1, int OFFSET = 0, int LEFT_BOUND = 0>
struct SMemReordering {

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

    //--------------------------------------------------------------------------

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__
    static void run(const T (&reg_in)[ITEMS_PER_THREAD],
                    T       (&reg_out)[ITEMS_PER_THREAD],
                    T*      smem_thread,
                    T*      smem_warp) {

        const int TH_NUM      = xlib::WARP_SIZE / (ITEMS_PER_THREAD / SIZE);
        const int RIGHT_BOUND = LEFT_BOUND + TH_NUM;

        if (static_cast<int>(xlib::lane_id()) >= LEFT_BOUND &&
            xlib::lane_id() < RIGHT_BOUND) {

            #pragma unroll
             for (int i = 0; i < ITEMS_PER_THREAD; i++)
                smem_thread[i] = reg_in[i];
        }
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            reg_out[OFFSET + i] = smem_warp[i * xlib::WARP_SIZE];

        const bool END_COND = (LEFT_BOUND + TH_NUM >= xlib::WARP_SIZE);
        SMemReordering<END_COND ? 0 : SIZE,
                       OFFSET + SIZE, LEFT_BOUND + TH_NUM>
            ::run(reg_in, reg_out, smem_thread, smem_warp);
    }

    //--------------------------------------------------------------------------

    template<typename T, unsigned ITEMS_PER_THREAD>
    __device__ __forceinline__
    static void run(const T (&reg_in)[ITEMS_PER_THREAD],
                   T        (&reg_out)[ITEMS_PER_THREAD],
                   T*        smem,
                   T*        smem_warp,
                   int&      offset,
                   int&      index) {

        const int WARP_ITEMS = xlib::WARP_SIZE * SIZE;

        while (index < ITEMS_PER_THREAD && offset < WARP_ITEMS)
            smem[offset++] = reg_in[index++];
        offset -= WARP_ITEMS;

        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            reg_out[OFFSET + j] = smem_warp[j * xlib::WARP_SIZE];

        const bool END_COND = (OFFSET + SIZE > ITEMS_PER_THREAD);
        SMemReordering<END_COND ? 0 : SIZE, SIZE, OFFSET + SIZE>
            ::run(reg_in, reg_out, smem, smem_warp, offset, index);
    }
};

template<int OFFSET, int LEFT_BOUND>
struct SMemReordering<0, OFFSET, LEFT_BOUND> {

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

//==============================================================================
//==============================================================================
//////////////////////////////
// SHARED MEMORY REORDERING //
//////////////////////////////

template<unsigned ITEMS_PER_THREAD, typename T, typename R>
 __device__ __forceinline__
void smem_reordering(T   (&reg1)[ITEMS_PER_THREAD],
                     R   (&reg2)[ITEMS_PER_THREAD],
                     void* smem) {
    if (ITEMS_PER_THREAD == 1)
        return;

    const unsigned WARP_ITEMS = xlib::WARP_SIZE * ITEMS_PER_THREAD;
    T* smem_tmp    = static_cast<T*>(smem) + xlib::warp_id() * WARP_ITEMS;
    T* smem_thread = smem_tmp + xlib::lane_id() * ITEMS_PER_THREAD;
    T* smem_warp   = smem_tmp + xlib::lane_id();

    detail::SMemReordering<>::run(reg1, smem_thread, smem_warp);
    detail::SMemReordering<>::run(reg2, smem_thread, smem_warp);
}

//==============================================================================

template<unsigned SMEM_PER_WARP, unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void smem_reordering(T (&reg)[ITEMS_PER_THREAD], void* smem) {
    if (ITEMS_PER_THREAD == 1)
        return;

    const unsigned SMEM_THREAD = (SMEM_PER_WARP == 0) ? ITEMS_PER_THREAD :
                                 SMEM_PER_WARP / xlib::WARP_SIZE;

    T* smem_T    = static_cast<T*>(smem) +
                   xlib::warp_id() * xlib::WARP_SIZE * SMEM_THREAD;
    T* smem_warp = smem_T + xlib::lane_id();

    if (ITEMS_PER_THREAD <= SMEM_THREAD ||
        ITEMS_PER_THREAD % SMEM_THREAD == 0) {

        const unsigned MIN_ITEMS = xlib::min(SMEM_THREAD, ITEMS_PER_THREAD);
        T*  smem_thread = smem_T + xlib::lane_id() * MIN_ITEMS;

        if (ITEMS_PER_THREAD <= SMEM_THREAD)
            detail::SMemReordering<>::run(reg, smem_thread, smem_warp);
        else {
            T tmp[ITEMS_PER_THREAD];
            detail::SMemReordering<MIN_ITEMS>::run(reg, tmp, smem_thread,
                                                    smem_warp);
            xlib::reg_copy(tmp, reg);
        }
    }
    else {
        T tmp[ITEMS_PER_THREAD];
        int offset = xlib::lane_id() * ITEMS_PER_THREAD;
        int  index = 0;
        detail::SMemReordering<SMEM_THREAD>::run(reg, tmp, smem_T, smem_warp,
                                                 offset, index);
        xlib::reg_copy(tmp, reg);
    }
}

//==============================================================================
//==============================================================================
//==============================================================================
////////////////////////
// SHUFFLE REORDERING //
////////////////////////

template<typename T, int SIZE>
__device__ __forceinline__
void shuffle_reordering(T (&A)[SIZE]) {
    static_assert(xlib::mcd(SIZE, xlib::WARP_SIZE) == 1 ||
                  xlib::is_power2(SIZE),
                  "Does not work if mcd(SIZE, WARP_SIZE) != 1 && SIZE is not "
                  "a power of 2");
    using namespace xlib::detail;

    if (SIZE == 1)
        return;

    T B[SIZE];
    int laneid = xlib::lane_id();

    if (xlib::mcd(SIZE, xlib::WARP_SIZE) == 1) {
        /*
        //                     !!!  Enable in CUDA Toolkit >= 9.2  !!!
        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            int index = (i * xlib::WARP_SIZE + laneid) % SIZE;
            #pragma unroll
            for (int j = 0; j < SIZE; j++)
                B[j] = (j == index) ? A[i] : B[j];
        }
        */
        Unroll<SIZE>::apply([&](int I) {
                        int index = (I * xlib::WARP_SIZE + laneid) % SIZE;
                        Unroll<SIZE>::apply([&](int J) {
                                        B[J] = (J == index) ? A[I] : B[J];
                                    });
                    });

        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            A[i] = xlib::shfl(B[i], (laneid * SIZE + i) % xlib::WARP_SIZE);
    }
    else if (xlib::is_power2(SIZE)) {
        const unsigned NUM_GROUPS = xlib::WARP_SIZE / SIZE;

        /*
        //                     !!!  Enable in CUDA Toolkit >= 9.2  !!!
        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            int index = (SIZE - i + laneid) % SIZE; //also: i * (SIZE - 1)
            #pragma unroll
            for (int j = 0; j < SIZE; j++)
                B[i] = (j == index) ? A[j] : B[i];
        }
        */
        Unroll<SIZE>::apply([&](int I) {
                        int index = (SIZE - I + laneid) % SIZE;
                        Unroll<SIZE>::apply([&](int J) {
                                        B[I] = (J == index) ? A[J] : B[I];
                                    });
                    });

        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            //also (laneid % NUM_GROUPS) * SIZE;
            int offset = (laneid * SIZE) % xlib::WARP_SIZE;
            B[i] = xlib::shfl(B[i], offset + (laneid / NUM_GROUPS + i) % SIZE);
        }

        /*
        //                     !!!  Enable in CUDA Toolkit >= 9.2  !!!
        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            int index = (i + laneid / NUM_GROUPS) % SIZE;
            #pragma unroll
            for (int j = 0; j < SIZE; j++)
                A[j] = (j == index) ? B[i] : A[j];
        }
        */
        Unroll<SIZE>::apply([&](int I) {
                        int index = (I + laneid / NUM_GROUPS) % SIZE;
                        Unroll<SIZE>::apply([&](int J) {
                                        A[J] = (J == index) ? B[I] : A[J];
                                    });
                    });
    }
}

//==============================================================================
//==============================================================================

template<typename T>
__device__ __forceinline__
void shuffle_reordering_v4(T (&A)[8]) {
    using namespace detail;
    const unsigned SIZE       = 8;
    const unsigned VECT       = 4;
    const unsigned NUM_GROUPS = SIZE / VECT;
    const unsigned GROUP_SIZE = xlib::WARP_SIZE / NUM_GROUPS;
    T B[SIZE];
    int laneid = xlib::lane_id();

    /*
    //                     !!!  Enable in CUDA Toolkit >= 9.2  !!!
    #pragma unroll
    for (int i = 0; i < SIZE; i++) {
        int index = ((laneid % NUM_GROUPS) * VECT + i) % SIZE;
        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            B[i] = (j == index) ? A[j] : B[i];
    }
    */
    Unroll<SIZE>::apply([&](int I) {
                    int index = ((laneid % NUM_GROUPS) * VECT + I) % SIZE;
                    Unroll<SIZE>::apply([&](int J) {
                                    B[I] = (J == index) ? A[J] : B[I];
                                });
                });

    #pragma unroll
    for (int i = 0; i < SIZE; i++) {
        int offset = (laneid / GROUP_SIZE + i / VECT) % NUM_GROUPS;
        int index  = (offset + laneid * NUM_GROUPS) % xlib::WARP_SIZE;
        B[i] = xlib::shfl(B[i], index);
    }

    /*
    //                     !!!  Enable in CUDA Toolkit >= 9.2  !!!
    #pragma unroll
    for (int i = 0; i < SIZE; i++) {
        int index = ((laneid / GROUP_SIZE) * VECT + i) % SIZE;
        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            A[i] = (j == index) ? B[j] : A[i];
    }
    */
    Unroll<SIZE>::apply([&](int I) {
                    int index = ((laneid / GROUP_SIZE) * VECT + I) % SIZE;
                    Unroll<SIZE>::apply([&](int J) {
                                    A[I] = (J == index) ? B[J] : A[I];
                                });
                });
}

//==============================================================================

template<typename T, int SIZE>
__device__ __forceinline__
void shuffle_reordering_inv(T (&A)[SIZE]) {
    static_assert(xlib::WARP_SIZE % SIZE == 0,
                  "WARP_SIZE and SIZE must be divisible");
    using namespace xlib::detail;
    if (SIZE == 1)
        return;

    T B[SIZE];
    int laneid = xlib::lane_id();

    const unsigned NUM_GROUPS = xlib::WARP_SIZE / SIZE;

    //                     !!!  Enable in CUDA Toolkit >= 9.2  !!!
    /*
    #pragma unroll            //index = (SIZE - i + laneid / NUM_GROUPS) % SIZE;
    for (int i = 0; i < SIZE; i++) {
        int index = (SIZE - i + laneid) % SIZE;
        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            B[i] = (j == index) ? A[j] : B[i];
    }
    */

    Unroll<SIZE>::apply([&](int I) {
                    int index = (SIZE - I + laneid / NUM_GROUPS) % SIZE;
                    Unroll<SIZE>::apply([&](int J) {
                                    B[I] = (J == index) ? A[J] : B[I];
                                });
                });

    #pragma unroll
    for (int i = 0; i < SIZE; i++) {
        int base  = (laneid % SIZE) * NUM_GROUPS;
        int index = (base + laneid / SIZE + i * NUM_GROUPS) % xlib::WARP_SIZE;
        B[i] = xlib::shfl(B[i], index);
    }

    //                     !!!  Enable in CUDA Toolkit >= 9.2  !!!
    /*
    #pragma unroll
    for (int i = 0; i < SIZE; i++) {
        int index = (i + laneid) % SIZE; //<-- (i + laneid % SIZE) % SIZE
        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            A[j] = (j == index) ? B[i] : A[j];
    }
    */

    Unroll<SIZE>::apply([&](int I) {
                    int index = (I + laneid) % SIZE;
                    Unroll<SIZE>::apply([&](int J) {
                                    A[J] = (J == index) ? B[I] : A[J];
                                });
                });
}

} // namespace xlib
