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
#include "Base/Device/Primitives/CudaFunctional.cuh"
#include "Base/Device/Primitives/ThreadReduce.cuh"

namespace xlib {
namespace ThreadInclusiveScanILP {
namespace detail {

template<int SIZE, typename T, cuda_functional::binary_op<T> BinaryOP,
         int STRIDE = SIZE / 4>
struct ThreadInclusiveScanSupport {
    static_assert(IsPower2<SIZE>::value,
                  "ThreadReduce : SIZE must be a power of 2");

    __device__ __forceinline__ static void DownSweepRight(T (&Array)[SIZE]) {
        #pragma unroll
        for (int INDEX = STRIDE * 2; INDEX < SIZE; INDEX += STRIDE * 2)
            Array[INDEX - 1 + STRIDE] = BinaryOP(Array[INDEX - 1],
                                                 Array[INDEX - 1 + STRIDE]);
        ThreadInclusiveScanSupport<SIZE, T, BinaryOP, STRIDE / 2>
            ::DownSweepRight(Array);
    }
};

template<int SIZE, typename T, cuda_functional::binary_op<T> BinaryOP>
struct ThreadInclusiveScanSupport<SIZE, T, BinaryOP, 0> {
    __device__ __forceinline__ static void DownSweepRight(T (&Array)[SIZE]) {}
};

} // namespace detail

//==========================================================================

template<typename T, int SIZE>
__device__ __forceinline__ static void Add(T (&Array)[SIZE]) {
    /*using namespace cuda_functional;
    using namespace detail;
    ThreadReduce::detail::
        ThreadReduceSupport<SIZE, T, plus<T>>::UpSweepRight(Array);
    ThreadInclusiveScanSupport<SIZE, T, plus<T>>::DownSweepRight(Array);*/
}

} // namespace ThreadInclusiveScanILP

namespace ThreadInclusiveScan {

template<typename T, int SIZE>
__device__ __forceinline__ static void Add(T (&Array)[SIZE]) {
    #pragma unroll
    for (int i = 1; i < SIZE; i++)
        Array[i] += Array[i - 1];
}

template<typename T>
__device__ __forceinline__ static void Add(T* Array, const int size) {
    for (int i = 1; i < size; i++)
        Array[i] += Array[i - 1];
}

} // namespace ThreadInclusiveScan

namespace ThreadExclusiveScan {
    template<typename T, int SIZE>
    __device__ __forceinline__ static void Add(T (&Array)[SIZE]) {
        T tmp = Array[0], tmp2;
        Array[0] = 0;
        #pragma unroll
        for (int i = 1; i < SIZE; i++) {
            tmp2 = Array[i];
            Array[i] = tmp;
            tmp += tmp2;
        }
    }

    template<typename T>
    __device__ __forceinline__ static void Add(T* Array, const int size) {
        T tmp = Array[0], tmp2;
        Array[0] = 0;
        for (int i = 1; i < size; i++) {
            tmp2 = Array[i];
            Array[i] = tmp;
            tmp += tmp2;
        }
    }

} // namespace ThreadExclusiveScan
} // namespace xlib
