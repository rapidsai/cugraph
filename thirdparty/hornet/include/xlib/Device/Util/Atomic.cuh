/**
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

namespace xlib {
/**
 * @brief provides overloading atomic operations for common types. In particular
 * it allows atomic operations for char, int8_t, unsigned char, int, unsigned,
 * half, float, double. However, the pointer type must be of size supported by
 * CUDA APIs
 */
namespace atomic {

/**
 * @brief overloading of CUDA atomicAdd
 * @param[in] value value to sum
 * @param[in] ptr pointer where store the sum
 * @return previous value of ptr
 */
template<typename T, typename R>
__device__ __forceinline__
T add(const T& value, R* ptr);

/**
 * @brief overloading of CUDA atomicMax
 * @param[in] value value to compare
 * @param[in] ptr pointer where store the minimum between the provided value and
 *                the stored value
 * @return previous value of ptr
 */
template<typename T, typename R>
__device__ __forceinline__
T max(const T& value, R* address);

/**
 * @brief overloading of CUDA atomicMin
 * @param[in] value value to compare
 * @param[in] ptr pointer where store the maximum between the provided value and
 *                the stored value
 * @return previous value of ptr
 */
template<typename T, typename R>
__device__ __forceinline__
T min(const T& value, R* address);

} // namespace atomic
} // namespace xlib

#include "impl/Atomic.i.cuh"
