/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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

#include "HostDevice.hpp"
#include <algorithm>    //std::min
#include <ratio>        //std::ratio

namespace xlib {

// nearest multiple of 4 : ((n-1)|3) + 1

// ==================== CONST EXPR TIME numeric methods ========================

template<typename T>
constexpr T max(const T& a, const T& b) noexcept;

template<typename T>
constexpr T min(const T& a, const T& b) noexcept;

template<typename T, typename... TArgs>
constexpr T min(const T& a, const T& b, const TArgs&... args) noexcept;

template<typename T, typename... TArgs>
constexpr T max(const T& a, const T& b, const TArgs&... args) noexcept;

template<typename T>
HOST_DEVICE
constexpr bool addition_is_safe(T a, T b) noexcept;

template<typename T>
HOST_DEVICE
constexpr bool mul_is_safe(T a, T b) noexcept;

template<typename R, typename T>
void check_overflow(T value);

template<typename T, typename R>
HOST_DEVICE
constexpr T ceil_div(T value, R div) noexcept;

template<typename T, typename R>
HOST_DEVICE
constexpr T uceil_div(T value, R div) noexcept;

/*
template<uint64_t DIV, typename T>
HOST_DEVICE CONST_EXPR
T ceil_div(T value) noexcept;*/

/**
 *  15 / 4 = 3.75 --> 4
 *  13 / 4 = 3.25 --> 3
 * .0 to .0.4 round down, .5 to .9 round up
 */
template<typename T>
HOST_DEVICE
constexpr T round_div(T value, T div) noexcept;

template<typename T>
HOST_DEVICE
constexpr T uround_div(T value, T div) noexcept;

/*
template<uint64_t DIV, typename T>
HOST_DEVICE CONST_EXPR
T round_div(T value) noexcept;
*/

template<typename T>
HOST_DEVICE
constexpr T upper_approx(T value, T mul) noexcept;

template<uint64_t MUL, typename T>
HOST_DEVICE
constexpr T upper_approx(T value) noexcept;

template<typename T>
HOST_DEVICE
constexpr T lower_approx(T value, T mul) noexcept;

template<int64_t MUL, typename T>
HOST_DEVICE
constexpr T lower_approx(T value) noexcept;

template<typename T>
HOST_DEVICE
constexpr bool is_power2(T value) noexcept;

template<typename T>
HOST_DEVICE
constexpr T factorial(T value) noexcept;

template<typename T, typename R>
HOST_DEVICE
bool read_bit(const T* array, R pos) noexcept;

template<typename T, typename R>
HOST_DEVICE
void write_bit(T* array, R pos) noexcept;

template<typename T, typename R>
HOST_DEVICE
void write_bits(T* array, R start, R end) noexcept;

template<typename T, typename R>
HOST_DEVICE
void delete_bit(T* array, R pos) noexcept;

template<typename T, typename R>
HOST_DEVICE
void delete_bits(T* array, R start, R end) noexcept;

// ========================== RUN TIME numeric methods =========================

template<typename T>
HOST_DEVICE
constexpr T roundup_pow2(T value) noexcept;

template<typename T>
HOST_DEVICE
constexpr T rounddown_pow2(T value) noexcept;

/** @fn T log2(const T value)
 *  @brief calculate the integer logarithm of 'value'
 *  @return &lfloor; log2 ( value ) &rfloor;
 */
template<typename T>
HOST_DEVICE
int log2(T value) noexcept;

template<typename T>
HOST_DEVICE
int ceil_log2(T value) noexcept;

template<unsigned BASE, typename T>
HOST_DEVICE
int log(T value) noexcept;

template<unsigned BASE, typename T>
HOST_DEVICE
int ceil_log(T value) noexcept;

/*template<typename T>
T pow2(unsigned exp, T value) {
    assert(xlib::is_power2(value));
    static const int multiply_deBruijn_bitposition[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };
    auto  pos = static_cast<uint32_t>(value * 0x077CB531U) >> 27;
    auto log2 = multiply_deBruijn_bitposition[pos];
    return value << xlib::log2(value);
}*/

template<typename T>
constexpr T pow(unsigned exp, T value) noexcept;

template<unsigned N>
constexpr unsigned geometric_serie(unsigned repetition) noexcept;

template<typename T, typename R>
float per_cent(T part, R max) noexcept;

template<typename T, typename R>
constexpr typename std::common_type<T, R>::type mcd(T a, R b) noexcept;

template<typename R>
struct CompareFloatABS;

template<std::intmax_t Num, std::intmax_t Den>
struct CompareFloatABS<std::ratio<Num, Den>> {
    template<typename T>
    bool operator() (T a, T b) noexcept;
};

template<typename R>
struct CompareFloatRelativeErr;

template<std::intmax_t Num, std::intmax_t Den>
struct CompareFloatRelativeErr<std::ratio<Num, Den>> {
    template<typename T>
    bool operator() (T a, T b) noexcept;
};

template <typename T, typename = void>
class WeightedRandomGenerator {
    template<typename R>
    WeightedRandomGenerator(const R* input, size_t size);

    size_t get() const noexcept;
};

template<unsigned A, unsigned B, unsigned BINS>
struct MultiplyShiftHash32;

template<uint64_t A, uint64_t B, unsigned BINS>
struct MultiplyShiftHash64;

} // namespace xlib

#include "impl/Numeric.i.hpp"
