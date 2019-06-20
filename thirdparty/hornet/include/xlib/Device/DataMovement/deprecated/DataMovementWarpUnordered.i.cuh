/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
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
#include "DataMovementWarp.i.cuh"
#include "../../../Util/Util.cuh"

namespace data_movement {
namespace warp {
namespace unordered {

namespace {

/**
* SMem must be in the correct position for each lane
*/
template<int SIZE, typename T>
void __device__ __forceinline__ RegToSharedSupport(T (&Queue)[SIZE],
                                                   T* __restrict__ SMem) {

    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(SMem)[i * WARP_SIZE] =
                                            reinterpret_cast<int2*>(Queue)[i];
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(SMem)[i * WARP_SIZE] =
                                            reinterpret_cast<int*>(Queue)[i];
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
           reinterpret_cast<short*>(SMem)[i * WARP_SIZE] =
                                            reinterpret_cast<short*>(Queue)[i];
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            SMem[i * WARP_SIZE] = Queue[i];
    }
}

/**
* SMem must be in the correct position for each lane
*/
template<int SIZE, typename T>
void __device__ __forceinline__ SharedToRegSupport(T* __restrict__ SMem,
                        T (&Queue)[SIZE]) {

    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Queue)[i] =
                                reinterpret_cast<int2*>(SMem)[i * WARP_SIZE];
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(Queue)[i] =
                                reinterpret_cast<int*>(SMem)[i * WARP_SIZE];
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
           reinterpret_cast<short*>(Queue)[i] =
                                reinterpret_cast<short*>(SMem)[i * WARP_SIZE];
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Queue[i] = SMem[i * WARP_SIZE];
    }
}

} // namespace

//==============================================================================

template<cub::CacheStoreModifier M, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ Pointer) {

    computeOffset<GLOBAL, SIZE>(Pointer);
    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i++)
            cub::ThreadStore<M>(reinterpret_cast<int4*>(Pointer) + i * WARP_SIZE,
                                              reinterpret_cast<int4*>(Queue)[i]);
    }
    if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
        cub::ThreadStore<M>(reinterpret_cast<int2*>(Pointer) + i * WARP_SIZE,
                                          reinterpret_cast<int2*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
        cub::ThreadStore<M>(reinterpret_cast<int*>(Pointer) + i * WARP_SIZE,
                                          reinterpret_cast<int*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
       cub::ThreadStore<M>(reinterpret_cast<short*>(Pointer) + i * WARP_SIZE,
                                         reinterpret_cast<short*>(Queue)[i]);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            cub::ThreadStore<M>(Pointer + i * WARP_SIZE, Queue[i]);
    }
}

//------------------------------------------------------------------------------

template<cub::CacheLoadModifier M, typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T (&Queue)[SIZE]) {

    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i++)
            reinterpret_cast<int4*>(Queue)[i] = cub::ThreadLoad<M>(
                            reinterpret_cast<int4*>(Pointer) + i * WARP_SIZE);
    }
    if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Queue)[i] = cub::ThreadLoad<M>(
                            reinterpret_cast<int2*>(Pointer) + i * WARP_SIZE);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(Queue)[i] = cub::ThreadLoad<M>(
                            reinterpret_cast<int*>(Pointer) + i * WARP_SIZE);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
            reinterpret_cast<short*>(Queue)[i] = cub::ThreadLoad<M>(
                            reinterpret_cast<short*>(Pointer) + i * WARP_SIZE);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Queue[i] = cub::ThreadLoad<M>(Pointer + i * WARP_SIZE);
    }
}

//------------------------------------------------------------------------------

template<int SIZE, typename T>
void __device__ __forceinline__ RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    computeOffset<SHARED, SIZE>(SMem);
    RegToSharedSupport(Queue, SMem);
}

template<int SIZE, typename T>
void __device__ __forceinline__ SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    computeOffset<SHARED, SIZE>(SMem);
    SharedToRegSupport(SMem, Queue);
}

} //@unordered
} //@warp
} //@data_movement
