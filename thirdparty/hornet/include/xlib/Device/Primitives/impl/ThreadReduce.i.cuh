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
#include "Base/Host/Numeric.hpp"    //xlib::is_power2

namespace xlib {
namespace thread_reduce {
namespace detail {

template<typename T, typename Lambda>
__device__ __forceinline__
static void upSweepLeft(T (&array)[SIZE], const Lambda& lambda) {
    static_assert(xlib::is_power2(SIZE),
                  "ThreadReduce : SIZE must be a power of 2");

    #pragma unroll
    for (int STRIDE = 1; STRIDE < SIZE; STRIDE *= 2) {
        #pragma unroll
        for (int INDEX = 0; INDEX < SIZE; INDEX += STRIDE * 2)
            array[INDEX] = lambda(array[INDEX], array[INDEX + STRIDE]);
    }
}

    /*template<typename T, typename Lambda>
    __device__ __forceinline__
    static void upSweepLeft(T (&array)[SIZE], const Lambda& lambda) {
        #pragma unroll
        for (int INDEX = 0; INDEX < SIZE; INDEX += STRIDE * 2)
            array[INDEX] = lambda(array[INDEX], array[INDEX + STRIDE]);
        ThreadReduceSupport<SIZE, STRIDE * 2>::upSweepLeft(array, lambda);
    }*/

    /*__device__ __forceinline__
    static void UpSweepRight(T (&array)[SIZE]) {
        #pragma unroll
        for (int INDEX = STRIDE - 1; INDEX < SIZE; INDEX += STRIDE * 2) {
            array[INDEX + STRIDE] = BinaryOP(array[INDEX],
                                             array[INDEX + STRIDE]);
        }
        ThreadReduceSupport<SIZE, T, BinaryOP, STRIDE * 2>::UpSweepRight(array);
    }*/
};
/*
template<int SIZE>
struct ThreadReduceSupport<SIZE, SIZE> {
    template<typename T, typename Lambda>
    __device__ __forceinline__
    static void upSweepLeft(T (&)[SIZE], const Lambda&) {}
    //__device__ __forceinline__ static void UpSweepRight(T (&array)[SIZE]) {}
};*/

} // namespace detail

//==============================================================================

template<typename T, int SIZE>
__device__ __forceinline__
static void add(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return a + b; };
    detail::upSweepLeft(array, lambda);
}

template<typename T, int SIZE>
__device__ __forceinline__
static void min(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return min(a, b); };
    detail::upSweepLeft(array, lambda);
}

template<typename T, int SIZE>
__device__ __forceinline__
static void max(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return max(a, b); };
    detail::upSweepLeft(array, lambda);
}

template<typename T, int SIZE>
__device__ __forceinline__
static void logicAnd(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return a && b; };
    detail::upSweepLeft(array, lambda);
}

template<typename T, int SIZE>
__device__ __forceinline__
static void logicOr(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return a || b; };
    detail::upSweepLeft(array, lambda);
}

template<typename T, int SIZE, typename Lambda>
__device__ __forceinline__
static void custom(T (&array)[SIZE], const Lambda& lambda) {
    detail::upSweepLeft(array, lambda);
}

} // namespace thread_reduce
} // namespace xlib
