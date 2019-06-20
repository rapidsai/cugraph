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
#pragma once

#include "Base/Device/Util/Basic.cuh"
#include <type_traits>

namespace xlib {

template<typename T, int SIZE>
__device__ __forceinline__
void vector_copy(T (&queue)[SIZE], T* ptr) {
    const int CAST_SIZE = (SIZE * sizeof(T)) % 16 == 0 ? 16 :
                          (SIZE * sizeof(T)) % 8 == 0 ? 8 :
                          (SIZE * sizeof(T)) % 4 == 0 ? 4 :
                          (SIZE * sizeof(T)) % 2 == 0 ? 2 : 1;
    using R = xlib::Pad<CAST_SIZE>;
    const int LOOPS = (SIZE * sizeof(T)) / CAST_SIZE;

    #pragma unroll
    for (int i = 0; i < LOOPS; i++)
        reinterpret_cast<R*>(ptr)[i] = reinterpret_cast<R*>(queue)[i];
}

template<typename T, typename Lambda>
__device__ __forceinline__
void vector_op(const T* ptr, int size, const Lambda& lambda) {
    using R1 = typename std::conditional<
                sizeof(T) <= 16 && xlib::IsPower2<sizeof(T)>::value, int4, T>
              ::type;
    using R2 = int2;
    using R3 = int;
    using R4 = short;

    const unsigned SIZE = sizeof(R1) / sizeof(T);
    T queue[SIZE];
    int size_loop = size / SIZE;

    for (int i = 0; i < size_loop; i++) {
        reinterpret_cast<R1*>(queue)[0] = reinterpret_cast<const R1*>(ptr)[i];
        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            lambda(queue[j]);
    }
    if (sizeof(R1) == sizeof(T))
        return;
    int remain = size - size_loop * SIZE;
    ptr       += size_loop * SIZE;
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R2)) {
        reinterpret_cast<R2*>(queue)[0] = reinterpret_cast<const R2*>(ptr)[0];
        const int L_SIZE = sizeof(R2) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++)
            lambda(queue[j]);
        if (sizeof(R2) == sizeof(T))
            return;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R3)) {
        reinterpret_cast<R3*>(queue)[0] = reinterpret_cast<const R3*>(ptr)[0];
        const int L_SIZE = sizeof(R3) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++)
            lambda(queue[j]);
        if (sizeof(R3) == sizeof(T))
            return;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R4)) {
        reinterpret_cast<R4*>(queue)[0] = reinterpret_cast<const R4*>(ptr)[0];
        const int L_SIZE = sizeof(R4) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++)
            lambda(queue[j]);
        if (sizeof(R4) == sizeof(T))
            return;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain != 0)
        lambda(*ptr);
}

template<typename T, typename Lambda>
__device__ __forceinline__
bool vector_first_of(const T* ptr, int size, const Lambda& lambda) {
    using R1 = typename std::conditional<
                sizeof(T) <= 16 && xlib::IsPower2<sizeof(T)>::value, int4, T>
              ::type;
    using R2 = int2;
    using R3 = int;
    using R4 = short;

    const unsigned SIZE = sizeof(R1) / sizeof(T);
    T queue[SIZE];
    int size_loop = size / SIZE;

    for (int i = 0; i < size_loop; i++) {
        reinterpret_cast<R1*>(queue)[0] = reinterpret_cast<const R1*>(ptr)[i];
        #pragma unroll
        for (int j = 0; j < SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
    }
    if (sizeof(R1) == sizeof(T))
        return false;
    int remain = size - size_loop * SIZE;
    ptr       += size_loop * SIZE;
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R2)) {
        reinterpret_cast<R2*>(queue)[0] = reinterpret_cast<const R2*>(ptr)[0];
        const int L_SIZE = sizeof(R2) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
        if (sizeof(R2) == sizeof(T))
            return false;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R3)) {
        reinterpret_cast<R3*>(queue)[0] = reinterpret_cast<const R3*>(ptr)[0];
        const int L_SIZE = sizeof(R3) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
        if (sizeof(R3) == sizeof(T))
            return false;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R4)) {
        reinterpret_cast<R4*>(queue)[0] = reinterpret_cast<const R4*>(ptr)[0];
        const int L_SIZE = sizeof(R4) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
        if (sizeof(R4) == sizeof(T))
            return false;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain != 0)
        return lambda(*ptr);
    return false;
}

} // namespace xlib
