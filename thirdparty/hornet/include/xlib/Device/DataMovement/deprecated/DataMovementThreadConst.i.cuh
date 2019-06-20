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
namespace data_movement {
namespace thread {

namespace {
/**
* CONSTANT SIZE
*/
template<int SIZE, typename T>
__device__ __forceinline__ void SharedRegFormat(T* __restrict__ Source, T* __restrict__ Dest) {
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Dest)[i] = reinterpret_cast<int2*>(Source)[i];
    } else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(Dest)[i] = reinterpret_cast<int*>(Source)[i];
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
       #pragma unroll
       for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
           reinterpret_cast<short*>(Dest)[i] = reinterpret_cast<short*>(Source)[i];
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Dest[i] = Source[i];
    }
}
} // namespace

//------------------------------------------------------------------------------

/**
* CONSTANT SIZE
*/
template<int SIZE, typename T>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem) {
    SharedRegFormat<SIZE>(Queue, SMem);
}

/**
* CONSTANT SIZE
*/
template<int SIZE, typename T>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]) {
    SharedRegFormat<SIZE>(SMem, Queue);
}

/**
* CONSTANT SIZE
*/
template<cub::CacheStoreModifier M, int SIZE, typename T>
void __device__ __forceinline__ RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ Pointer) {

    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i++)
            cub::ThreadStore<M>(reinterpret_cast<int4*>(Pointer) + i,
                                reinterpret_cast<int4*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            cub::ThreadStore<M>(reinterpret_cast<long long int*>(Pointer) + i,
                                reinterpret_cast<long long int*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            cub::ThreadStore<M>(reinterpret_cast<int*>(Pointer) + i,
                                reinterpret_cast<int*>(Queue)[i]);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
            cub::ThreadStore<M>(reinterpret_cast<short*>(Pointer) + i,
                                reinterpret_cast<short*>(Queue)[i]);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            cub::ThreadStore<M>(Pointer + i, Queue[i]);
    }
}

/**
* CONSTANT SIZE
*/
template<cub::CacheLoadModifier M, int SIZE, typename T>
void __device__ __forceinline__ GlobalToReg(T* __restrict__ Pointer,
                                            T (&Queue)[SIZE]) {

    const int AGGR_SIZE_16 = sizeof(T) < 16 ? 16 / sizeof(T) : 1;
    const int AGGR_SIZE_8 = sizeof(T) < 8 ? 8 / sizeof(T) : 1;
    const int AGGR_SIZE_4 = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const int AGGR_SIZE_2 = sizeof(T) < 2 ? 2 / sizeof(T) : 1;

    if (SIZE % AGGR_SIZE_16 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_16; i++)
            reinterpret_cast<int4*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<int4*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_8 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_8; i++)
            reinterpret_cast<int2*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<int2*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_4 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_4; i++)
            reinterpret_cast<int*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<int*>(Pointer) + i);
    }
    else if (SIZE % AGGR_SIZE_2 == 0) {
        #pragma unroll
        for (int i = 0; i < SIZE / AGGR_SIZE_2; i++)
            reinterpret_cast<short*>(Queue)[i] = cub::ThreadLoad<M>(reinterpret_cast<short*>(Pointer) + i);
    }
    else {
        #pragma unroll
        for (int i = 0; i < SIZE; i++)
            Queue[i] = cub::ThreadLoad<M>(Pointer + i);
    }
}

} //@thread
} //@data_movement
