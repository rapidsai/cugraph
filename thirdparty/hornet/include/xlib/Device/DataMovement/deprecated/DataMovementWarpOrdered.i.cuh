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
#include "../../../Util/Util.cuh"
#include "../../../../Host/BaseHost.hpp"

namespace data_movement {

using namespace PTX;
using namespace numeric;

namespace {

/**
* SMem must be in the correct position for each lane
*/
template<int SIZE, typename T>
void __device__ __forceinline__ SharedRegSupport(T* __restrict__ Source,
                                                 T* __restrict__ Dest) {

    const int SIZE_CHAR = SIZE * sizeof(T);
    if (SIZE_CHAR % 8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR / 8; i++)
            reinterpret_cast<int2*>(Dest)[i] = reinterpret_cast<int2*>(Source)[i];
    }
    else if (SIZE_CHAR % 4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR / 4; i++)
            reinterpret_cast<int*>(Dest)[i] = reinterpret_cast<int*>(Source)[i];
    }
    else if (SIZE_CHAR % 2 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR / 2; i++)
            reinterpret_cast<short*>(Dest)[i] = reinterpret_cast<short*>(Source)[i];
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE_CHAR; i++)
            Dest[i] = Source[i];
    }
}
} // namespace

//==============================================================================

namespace warp_ordered {

template<cub::CacheLoadModifier M, typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    T* SMemThread = SMem + LaneID() * SIZE;

    warp::computeOffset<GLOBAL, SIZE>(SMem);
    warp::computeOffset<GLOBAL, SIZE>(Pointer);
    warp::GlobalToSharedSupport<M, SIZE * WARP_SIZE>(Pointer, SMem);

    SharedRegSupport<SIZE>(SMemThread, Queue);
}

template<cub::CacheStoreModifier M, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ Pointer) {

    T* SMemThread = SMem + LaneID() * SIZE;
    SharedRegSupport<SIZE>(Queue, SMemThread);

    warp::computeOffset<GLOBAL, SIZE>(SMem);
    warp::computeOffset<GLOBAL, SIZE>(Pointer);
    warp::SharedToGlobalSupport<M, SIZE * WARP_SIZE>(SMem, Pointer);
}

template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    SMem += LaneID() * SIZE;
    SharedRegSupport<SIZE>(Queue, SMem);
}

template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    SMem += LaneID() * SIZE;
    SharedRegSupport<SIZE>(SMem, Queue);
}

} //@warp_ordered

//------------------------------------------------------------------------------

namespace warp_ordered_adv {

template<cub::CacheLoadModifier M, typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T* __restrict__ SMemThread,
                                            T (&Queue)[SIZE]) {

    warp::GlobalToSharedSupport<M, SIZE * WARP_SIZE>(Pointer, SMem);
    SharedRegSupport<SIZE>(SMemThread, Queue);
}

template<cub::CacheStoreModifier M, typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ SMemThread,
                                            T* __restrict__ Pointer) {

    SharedRegSupport<SIZE>(Queue, SMemThread);
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("%d %d %d\n", SMemThread[0], SMemThread[1], SMemThread[2]);
    warp::SharedToGlobalSupport<M, SIZE * WARP_SIZE>(SMem, Pointer);
}

template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    SharedRegSupport<SIZE>(Queue, SMem);
}

template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    SharedRegSupport<SIZE>(SMem, Queue);
}

} //@warp_ordered_advanced
} //@data_movement
