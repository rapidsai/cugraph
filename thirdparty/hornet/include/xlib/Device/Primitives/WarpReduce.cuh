/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
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

#include "Device/Util/DeviceProperties.cuh" //xlib::WARP_SZ

namespace xlib {

template<int WARP_SZ = 32>  //VW_SIZE == 1  --> SPECIALIZATION
struct WarpReduce {
    static_assert(xlib::is_power2(WARP_SZ) && WARP_SZ >= 1 &&
                  WARP_SZ <= WARP_SIZE, "WarpReduce : WARP_SZ must be a power"
                  " of 2 and 2 <= WARP_SZ <= WARP_SIZE");

    template<typename T>
    __device__ __forceinline__
    static void add(T& value);

    template<typename T>
    __device__ __forceinline__
    static void min(T& value);

    template<typename T>
    __device__ __forceinline__
    static void max(T& value);

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void add(T (&value)[SIZE]);

    //--------------------------------------------------------------------------

    template<typename T>
    __device__ __forceinline__
    static void addAll(T& value);

    template<typename T>
    __device__ __forceinline__
    static void minAll(T& value);

    template<typename T>
    __device__ __forceinline__
    static void maxAll(T& value);

    //--------------------------------------------------------------------------

    template<typename T, typename R>
    __device__ __forceinline__
    static void add(T& value, R* pointer);

    template<typename T, typename R>
    __device__ __forceinline__
    static void min(T& value, R* pointer);

    template<typename T, typename R>
    __device__ __forceinline__
    static void max(T& value, R* pointer);

    //--------------------------------------------------------------------------

    template<typename T, typename R>
    __device__ __forceinline__
    static T atomicAdd(const T& value, R* pointer);

    template<typename T, typename R>
    __device__ __forceinline__
    static void atomicMin(const T& value, R* pointer);

    template<typename T, typename R>
    __device__ __forceinline__
    static void atomicMax(const T& value, R* pointer);
};

} // namespace xlib

#include "impl/WarpReduce.i.cuh"
