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
 */
#pragma once

#include <cuda_fp16.h>

namespace xlib {
namespace atomic {

template<typename T, typename R>
__device__ __forceinline__
T add(const T& value, R* ptr) {
    static_assert(std::is_same<T, R>::value, "T and R must be the same type");
    return atomicAdd(ptr, value);
}

template<>
__device__ __forceinline__
double add<double, double>(const double& value, double* double_ptr) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(double_ptr, value);
#else
    auto     ull_ptr = reinterpret_cast<unsigned long long*>(double_ptr);
    uint64_t old_ull = *ull_ptr;
    double assumed_double, old_double;
    do {
        auto assumed_ull = old_ull;
        auto     sum = value + assumed_ull;
        //auto sum_ull = reinterpret_cast<unsigned long long&>(sum);
        old_ull      = atomicCAS(ull_ptr, assumed_ull, sum);

        assumed_double = reinterpret_cast<double&>(assumed_ull);
        old_double     = reinterpret_cast<double&>(old_ull);
    } while (assumed_double != old_double);
    return old_double;
#endif
}

template<>
__device__ __forceinline__
char add<char, int>(const char& value, int* address) {
    return static_cast<char>(atomicAdd(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
int8_t add<int8_t, int>(const int8_t& value, int* address) {
    return static_cast<int8_t>(atomicAdd(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
unsigned char add<unsigned char, unsigned>(const unsigned char& value,
                                           unsigned* address) {
    return static_cast<unsigned char>(
                atomicAdd(address, static_cast<unsigned>(value)));
}

template<>
__device__ __forceinline__
short add<short, int>(const short& value, int* address) {
    return static_cast<short>(atomicAdd(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
unsigned short add<unsigned short, unsigned>(const unsigned short& value,
                                             unsigned* address) {
    return static_cast<unsigned char>(
                atomicAdd(address, static_cast<unsigned>(value)));
}

template<>
__device__ __forceinline__
half add<half, float>(const half& value, float* address) {
    return __float2half(xlib::atomic::add(__half2float(value), address));
}

//==============================================================================

template<typename T, typename R>
__device__ __forceinline__
T max(const T& value, R* address) {
    static_assert(std::is_same<T, R>::value, "T and R must be the same type");
    return atomicMax(address, value);
}

template<>
__device__ __forceinline__
char max<char, int>(const char& value, int* address) {
    return static_cast<char>(atomicMax(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
unsigned char max<unsigned char, unsigned>(const unsigned char& value,
                                           unsigned* address) {
    return static_cast<unsigned char>(
                atomicMax(address, static_cast<unsigned>(value)));
}

template<>
__device__ __forceinline__
int8_t max<int8_t, int>(const int8_t& value, int* address) {
    return static_cast<int8_t>(atomicMax(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
short max<short, int>(const short& value, int* address) {
    return static_cast<short>(atomicMax(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
unsigned short max<unsigned short, unsigned>(const unsigned short& value,
                                             unsigned* address) {
    return static_cast<unsigned short>(
                atomicMax(address, static_cast<unsigned>(value)));
}

//address must be initialized with std::numeric_limits<int>::min()
template<>
__device__ __forceinline__
float max<float, float>(const float& value, float* address) {
    int value_int = reinterpret_cast<const int&>(value);
    if (value_int < 0)
        value_int = 0x80000000 - value_int;
    auto ret = atomicMax(reinterpret_cast<int*>(address), value_int);
    return reinterpret_cast<float&>(ret);
}

//address must be initialized with std::numeric_limits<long long int>::min()
template<>
__device__ __forceinline__
double max<double, double>(const double& value, double* address) {
    auto value_ll = reinterpret_cast<const long long int&>(value);
    if (value_ll < 0)
        value_ll = 0x8000000000000000 - value_ll;
    auto ret = atomicMax(reinterpret_cast<long long int*>(address), value_ll);
    return reinterpret_cast<double&>(ret);
}

template<>
__device__ __forceinline__
half max<half, float>(const half& value, float* address) {
    return __float2half(xlib::atomic::max(__half2float(value), address));
}

//==============================================================================

template<typename T, typename R>
__device__ __forceinline__
T min(const T& value, R* address) {
    static_assert(std::is_same<T, R>::value, "T and R must be the same type");
    return atomicMin(address, value);
}

template<>
__device__ __forceinline__
char min<char, int>(const char& value, int* address) {
    return static_cast<char>(atomicMin(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
int8_t min<int8_t, int>(const int8_t& value, int* address) {
    return static_cast<int8_t>(atomicMin(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
short min<short, int>(const short& value, int* address) {
    return static_cast<short>(atomicMin(address, static_cast<int>(value)));
}

template<>
__device__ __forceinline__
unsigned char min<unsigned char, unsigned>(const unsigned char& value,
                                           unsigned* address) {
    return static_cast<unsigned char>(
                atomicMin(address, static_cast<unsigned>(value)));
}

template<>
__device__ __forceinline__
unsigned short min<unsigned short, unsigned>(const unsigned short& value,
                                             unsigned* address) {
    return static_cast<unsigned short>(
                atomicMin(address, static_cast<unsigned short>(value)));
}

template<>
__device__ __forceinline__
float min<float, float>(const float& value, float* address) {
    int value_int = reinterpret_cast<const int&>(value);
    if (value_int < 0)
        value_int = 0x80000000 - value_int;
    auto ret = atomicMin(reinterpret_cast<int*>(address), value_int);
    return reinterpret_cast<float&>(ret);
}

template<>
__device__ __forceinline__
double min<double, double>(const double& value, double* address) {
    long long int value_ll = reinterpret_cast<const long long int&>(value);
    if (value_ll < 0)
        value_ll = 0x8000000000000000 - value_ll;
    auto ret = atomicMin(reinterpret_cast<long long int*>(address), value_ll);
    return reinterpret_cast<double&>(ret);
}

template<>
__device__ __forceinline__
half min<half, float>(const half& value, float* address) {
    return __float2half(xlib::atomic::min(__half2float(value), address));
}

} // namespace atomic
} // namespace xlib
