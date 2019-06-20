/**
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
 */
#include "Host/Basic.hpp"
#include "Host/Metaprogramming.hpp"
#include <cassert>      //assert
#include <chrono>       //std::chrono
#include <cmath>        //std::abs
#include <limits>       //std::numeric_limits
#include <random>       //std::mt19937_64
#include <type_traits>  //std::is_integral

namespace xlib {

#if ((__clang_major__ >= 3 && __clang_minor__ >= 4) || __GNUC__ > 5)

template<typename T>
HOST_DEVICE
constexpr bool addition_is_safe(T a, T b) noexcept {
    return true;
    //return std::is_integral<T>::value ? __builtin_add_overflow(a, b, &a)
    //                                  : true;
    //return __builtin_add_overflow(a, b, &a);
}

template<typename T>
HOST_DEVICE
constexpr bool mul_is_safe(T a, T b) noexcept {
    return true;
    //return std::is_integral<T>::value ? __builtin_mul_overflow(a, b, &a)
    //                                  : true;
}

#else

template<typename T>
HOST_DEVICE
bool constexpr addition_is_safe(T a, T b) noexcept {
    return std::is_integral<T>::value && std::is_unsigned<T>::value ?
           (a + b) < a : true;
}

template<typename T>
HOST_DEVICE
bool constexpr mul_is_safe(T, T) noexcept { return true; }

#endif

template<typename R, typename T>
void check_overflow(T value) {
    if (value > std::numeric_limits<R>::max())
        ERROR("value overflow")
}

//------------------------------------------------------------------------------
template<typename T>
constexpr T min(const T& a) noexcept {
    return a;
}

template<typename T>
constexpr T min(const T& a, const T& b) noexcept {
    return a < b ? a : b;
}

template<typename T, typename... TArgs>
constexpr T min(const T& a, const T& b, const TArgs&... args) noexcept {
    const auto& min_args = xlib::min(args...);
    const auto& min_value = xlib::min(a, b);
    return min_value < min_args ? min_value : min_args;
}

template<typename T>
constexpr T max(const T& a) noexcept {
    return a;
}

template<typename T>
constexpr T max(const T& a, const T& b) noexcept {
    return a > b ? a : b;
}

template<typename T, typename... TArgs>
constexpr T max(const T& a, const T& b, const TArgs&... args) noexcept {
    const auto&  max_args = xlib::max(args...);
    const auto& max_value = xlib::max(a, b);
    return max_value > max_args ? max_value : max_args;
}

//------------------------------------------------------------------------------

namespace detail {

template<typename T>
HOST_DEVICE
constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type
ceil_div_aux(T value, T div) noexcept {
    //return value == 0 ? 0 : 1u + ((value - 1u) / div);       // not overflow
    return (value + div - 1) / div;       // may overflow
    //return (value / div) + ((value % div) > 0)
    //  --> remainer = zero op, but not GPU devices
}

template<typename T>
HOST_DEVICE
constexpr typename std::enable_if<std::is_signed<T>::value, T>::type
ceil_div_aux(T value, T div) noexcept {
    using U = typename std::make_unsigned<T>::type;
    return (value > 0) ^ (div > 0) ? value / div :
       static_cast<T>(ceil_div_aux(static_cast<U>(value), static_cast<U>(div)));
}

/**
 *
 * @warning division by zero
 * @warning division may overflow if (value + div / 2) > numeric_limits<T>::max()
 */
template<typename T>
HOST_DEVICE
constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type
round_div_aux(T value, T div) noexcept {
    assert(addition_is_safe(value, div / 2u) && "division overflow");
    return (value + (div / 2u)) / div;
}

/**
 *
 * @warning division by zero
 * @warning division may overflow/underflow. If value > 0 && div > 0 -> assert
 *         value / div > 0 --> (value - div / 2) may underflow
 *         value / div < 0 --> (value + div / 2) may overflow
 */
template<typename T>
HOST_DEVICE
constexpr typename std::enable_if<std::is_signed<T>::value, T>::type
round_div_aux(T value, T div) noexcept {
    assert(addition_is_safe(value, div / 2) && "division overflow");
    assert(value > 0 && div > 0 && "value, div > 0");
    return (value < 0) ^ (div < 0) ? (value - div / 2) / div
                                   : (value + div / 2) / div;
}

} // namespace detail

//------------------------------------------------------------------------------

template<typename T, typename R>
HOST_DEVICE
constexpr T ceil_div(T value, R div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    static_assert(std::is_integral<R>::value, "R must be integral");
    assert(div != 0 && "division by zero in integer arithmetic");

    return detail::ceil_div_aux(value, static_cast<T>(div));
}

template<typename T, typename R>
HOST_DEVICE
constexpr T uceil_div(T value, R div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    assert(div != 0 && "division by zero in integer arithmetic");

    using U = typename std::make_unsigned<T>::type;
    return detail::ceil_div_aux(static_cast<U>(value), static_cast<U>(div));
}

template<uint64_t DIV, typename T>
HOST_DEVICE
constexpr T ceil_div(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    static_assert(DIV != 0, "division by zero in integer arithmetic");
#if !defined(__NVCC__)
    assert(std::is_unsigned<T>::value || value >= 0);
#endif

    const auto DIV_ = static_cast<T>(DIV);
    //return value == 0 ? 0 : 1 + ((value - 1) / DIV_);
    return (value + DIV_ - 1) / DIV_;
}

//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE
constexpr T round_div(T value, T div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    assert(div != 0 && "division by zero in integer arithmetic");

    return detail::round_div_aux(value, div);
}

template<typename T>
HOST_DEVICE
constexpr T uround_div(T value, T div) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    assert(div != 0 && "division by zero in integer arithmetic");

    using R = typename std::make_unsigned<T>::type;
    return detail::round_div_aux(static_cast<R>(value), static_cast<R>(div));
}

template<uint64_t DIV, typename T>
HOST_DEVICE
constexpr T round_div(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    static_assert(DIV > 0, "division by zero");
#if !defined(__NVCC__)
    assert(std::is_unsigned<T>::value || value >= 0);
#endif
    assert(addition_is_safe(value, static_cast<T>(DIV / 2u)));

    const auto DIV_ = static_cast<T>(DIV);
    return (value + (DIV_ / 2u)) / DIV_;
}

//------------------------------------------------------------------------------

/**
 * @pre T must be integral
 * @warning division by zero
 * @warning division may overflow if (value + (mul / 2)) > numeric_limits<T>::max()
 */
template<typename T>
HOST_DEVICE
constexpr T upper_approx(T value, T mul) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return ceil_div(value, mul) * mul;
}

template<uint64_t MUL, typename T>
HOST_DEVICE
constexpr T upper_approx(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
#if !defined(__NVCC__)
    assert(std::is_unsigned<T>::value || value >= 0);
#endif

    const auto MUL_ = static_cast<T>(MUL);
    return MUL == 1 ? value :
            !xlib::is_power2(MUL) ? ceil_div(value, MUL) * MUL_
                                  : (value + MUL_ - 1) & ~(MUL_ - 1);
}

template<typename T>
HOST_DEVICE
constexpr T lower_approx(T value, T mul) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return (value / mul) * mul;
}

template<int64_t MUL, typename T>
HOST_DEVICE
constexpr T lower_approx(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
#if !defined(__NVCC__)
    assert(std::is_unsigned<T>::value || value >= 0);
#endif

    const auto MUL_ = static_cast<T>(MUL);
    return MUL == 1 ? value :
           !xlib::is_power2(MUL) ? (value / MUL_) * MUL_ : value & ~(MUL_ - 1);
}

//------------------------------------------------------------------------------

template<typename T>
HOST_DEVICE
constexpr bool is_power2(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    using R = typename std::conditional<std::is_integral<T>::value,
                                        T, uint64_t>::type;
    auto value_ = static_cast<R>(value);
#if !defined(__NVCC__)
    assert(std::is_unsigned<R>::value || value_ >= 0);
#endif
    return (value_ != 0) && !(value_ & (value_ - 1));
}

template<typename T>
HOST_DEVICE
constexpr T factorial(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return value <= 1 ? 1 : value * factorial(value - 1);
}

template<typename T, typename R>
HOST_DEVICE bool read_bit(const T* array, R pos) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto           upos = static_cast<unsigned>(pos);
    return array[upos / SIZE] & (static_cast<T>(1) << (upos % SIZE));
}

template<typename T, typename R>
HOST_DEVICE void write_bit(T* array, R pos) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto           upos = static_cast<unsigned>(pos);
    array[upos / SIZE] |= static_cast<T>(1) << (upos % SIZE);
}

template<typename T, typename R>
HOST_DEVICE void write_bits(T* array, R start, R end) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto         ustart = static_cast<unsigned>(start);
    auto           uend = static_cast<unsigned>(end);
    auto     start_word = ustart / SIZE;
    auto       end_word = uend / SIZE;
    array[start_word]  |= ~((static_cast<T>(1) << (uend % SIZE)) - 1);
    array[end_word]    |= (static_cast<T>(1) << (ustart % SIZE)) - 1;
    std::fill(array + start_word + 1, array + end_word - 1, static_cast<T>(-1));
}

template<typename T, typename R>
HOST_DEVICE void delete_bit(T* array, R pos) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto           upos = static_cast<unsigned>(pos);
    array[upos / SIZE] &= ~(static_cast<T>(1) << (upos % SIZE));
}

template<typename T, typename R>
HOST_DEVICE void delete_bits(T* array, R start, R end) noexcept {
    static_assert(std::is_integral<T>::value && std::is_integral<R>::value,
                  "T/R must be integral");

    const unsigned SIZE = sizeof(T) * 8u;
    auto         ustart = static_cast<unsigned>(start);
    auto           uend = static_cast<unsigned>(end);
    auto     start_word = ustart / SIZE;
    auto       end_word = uend / SIZE;
    array[start_word]  &= (static_cast<T>(1) << (ustart % SIZE)) - 1;
    array[end_word]    &= ~((static_cast<T>(1) << (uend % SIZE)) - 1);
    for (int i = start_word + 1; i < end_word - 1; i++)
        array[i] = 0;
}

// =============================================================================

template<typename T>
HOST_DEVICE
constexpr T roundup_pow2(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    //const bool is_integral = std::is_integral<T>::value;
    //using R = typename std::conditional<is_integral, T, uint64_t>::type;
    //auto  v = static_cast<R>(value);
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;
    return value;
}

template<typename T>
HOST_DEVICE
constexpr T rounddown_pow2(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    value |= (value >> 1);
    value |= (value >> 2);
    value |= (value >> 4);
    value |= (value >> 8);
    value |= (value >> 16);
    return (value & ~(value >> 1));
}
/*
template<typename T>
HOST_DEVICE
int log2(T value) noexcept {
    const bool is_integral = std::is_integral<T>::value;
    using R = typename std::conditional<is_integral && sizeof(T) <= 4, unsigned,
                                        uint64_t>::type;
    const auto value_unsigned = static_cast<R>(value);
    assert(value > 0);

    #if defined(__CUDA_ARCH__)
        return __msb(value_unsigned);
    #else
        return sizeof(T) < 8 ? 31 - __builtin_clz(value_unsigned) :
                               63 - __builtin_clzll(value_unsigned);
    #endif
}*/

template<typename T>
HOST_DEVICE
typename std::enable_if<sizeof(T) <= 4, int>::type
log2_aux(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be intergral");
    assert(value > 0);

    #if defined(__CUDA_ARCH__)
        unsigned ret;
        asm ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(value));
        return ret;
    #else
        return 31 - __builtin_clz(value);
    #endif
}

template<typename T>
HOST_DEVICE
typename std::enable_if<sizeof(T) == 8, int>::type
log2_aux(T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be intergral");
    assert(value > 0);

    #if defined(__CUDA_ARCH__)
        unsigned ret;
        asm ("bfind.u64 %0, %1;" : "=r"(ret) : "l"(value));
        return ret;
    #else
        return 63 - __builtin_clzll(value);
    #endif
}

template<typename T>
HOST_DEVICE
int log2(T value) noexcept {
    return log2_aux(value);
}

template<typename T>
HOST_DEVICE
int ceil_log2(T value) noexcept {
    return is_power2(value) ? log2(value) : log2(value) + 1;
}

namespace detail {

template<unsigned BASE, typename T>
HOST_DEVICE
typename std::enable_if<!xlib::is_power2(BASE), int>::type
log_aux(T value) noexcept {
    int count;
    for (count = 0; value; count++)
        value /= BASE;
    return count;
}

template<unsigned BASE, typename T>
HOST_DEVICE
typename std::enable_if<xlib::is_power2(BASE), int>::type
log_aux(T value) noexcept {
    return xlib::log2(value) / xlib::Log2<BASE>::value;
}

/*template<unsigned BASE, typename T>
HOST_DEVICE
typename std::enable_if<!xlib::IsPower2<BASE>::value, int>::type
ceil_log(T value);*/

template<unsigned BASE, typename T>
HOST_DEVICE
typename std::enable_if<xlib::is_power2(BASE), int>::type
ceil_log_aux(T value) noexcept {
    auto ret = xlib::ceil_div<xlib::Log2<BASE>::value>(xlib::ceil_log2(value));
    return ret;
}

} // namespace detail

template<unsigned BASE, typename T>
HOST_DEVICE
int log(T value) noexcept {
    return detail::log_aux<BASE>(value);
}

template<unsigned BASE, typename T>
HOST_DEVICE
int ceil_log(T value) noexcept {
    return detail::ceil_log_aux<BASE>(value);
}

template<unsigned BASE, typename T>
HOST_DEVICE
typename std::enable_if<xlib::is_power2(BASE), T>::type
pow(T exp) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return 1 << (xlib::Log2<BASE>::value * exp);
}

template<unsigned BASE, typename T>
HOST_DEVICE
typename std::enable_if<!xlib::is_power2(BASE), T>::type
pow(T exp) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    return pow(exp, BASE);
}

template<typename T>
constexpr T pow(unsigned exp, T value) noexcept {
    static_assert(std::is_integral<T>::value, "T must be integral");
    if (xlib::is_power2(exp))
        return 1 << (xlib::log2(exp) * value);
    T result = 1;
    while (exp) {
        if (exp & 1) {
            result *= value;
            exp--;
        }
        result *= result;
        exp /= 2u;
    }
    return result;
}

template<unsigned N>
constexpr unsigned geometric_serie(unsigned repetition) noexcept {
    return (xlib::pow<N>(repetition + 1) - 1) / (N - 1);
};

template<typename T, typename R>
float per_cent(T part, R max) noexcept {
    return (static_cast<float>(part) / static_cast<float>(max)) * 100.0f;
}

template<typename T, typename R>
constexpr typename std::common_type<T, R>::type mcd(T a, R b) noexcept {
    using CT = typename std::common_type<T, R>::type;
    static_assert(std::is_integral<T>::value && std::is_integral<T>::value,
                  "T and R must be integral");
    assert((std::is_unsigned<T>::value || a >= 0) &&
           (std::is_unsigned<R>::value || b >= 0) &&
            "a, b must be greater than zero");
    return static_cast<CT>((b == 0) ? a : xlib::mcd(b, a % b));
}

/*
HOST_DEVICE
constexpr unsigned multiplyShiftHash32(unsigned A, unsigned B,
                                       unsigned log_bins, unsigned value)
                                       noexcept {
    return static_cast<unsigned>(A * value + B) >> (32 - log_bins);
}

HOST_DEVICE
constexpr uint64_t multiplyShiftHash64(uint64_t A, uint64_t B,
                                       unsigned log_bins, uint64_t value)
                                       noexcept {
    return static_cast<uint64_t>(A * value + B) >> (64 - log_bins);
}*/

template<unsigned A, unsigned B, unsigned BINS>
struct MultiplyShiftHash32 {
    static_assert(xlib::is_power2(BINS), "BINS must be a power of 2");

    HOST_DEVICE
    static unsigned op(unsigned value) {
        return static_cast<unsigned>(A * value + B) >> (32 - Log2<BINS>::value);
    }
};

template<uint64_t A, uint64_t B, unsigned BINS>
struct MultiplyShiftHash64 {
    static_assert(xlib::is_power2(BINS), "BINS must be a power of 2");

    HOST_DEVICE
    static uint64_t op(uint64_t value) {
        return static_cast<uint64_t>(A * value + B) >> (64 - Log2<BINS>::value);
    }
};

//------------------------------------------------------------------------------

template<std::intmax_t Num, std::intmax_t Den>
template<typename T>
inline bool CompareFloatABS<std::ratio<Num, Den>>
::operator() (T a, T b) noexcept {
    const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
    return std::abs(a - b) < epsilon;
}

template<std::intmax_t Num, std::intmax_t Den>
template<typename T>
inline bool CompareFloatRelativeErr<std::ratio<Num, Den>>
::operator() (T a, T b) noexcept {
    const T epsilon = static_cast<T>(Num) / static_cast<T>(Den);
    const T    diff = std::abs(a - b);
    //return (diff < epsilon) ||
    //       (diff / std::max(std::abs(a), std::abs(b)) < epsilon);
    return (diff < epsilon) ||
           (diff / std::min(std::abs(a) + std::abs(b),
                            std::numeric_limits<float>::max()) < epsilon);
}

//------------------------------------------------------------------------------

namespace detail {

template <typename T>
class WeightedRandomGeneratorAux {
public:
    template<typename R>
    WeightedRandomGeneratorAux(const R* weights, size_t size) :
            _gen(static_cast<uint64_t>(
                 std::chrono::system_clock::now().time_since_epoch().count())),
            _size(size) {

        _cumulative    = new T[size + 1];
        _cumulative[0] = 0;
        for (size_t i = 1; i <= size; i++)
            _cumulative[i] = _cumulative[i - 1]+ static_cast<T>(weights[i - 1]);
    }
    ~WeightedRandomGeneratorAux() {
        delete[] _cumulative;
    }
protected:
    std::mt19937_64 _gen;
    T*              _cumulative { nullptr };
    size_t          _size;
};

} //namespace detail

template <typename T>
class WeightedRandomGenerator<T, typename std::enable_if<
                                    std::is_integral<T>::value>::type> :
                                public detail::WeightedRandomGeneratorAux<T> {
public:
    using detail::WeightedRandomGeneratorAux<T>::_cumulative;
    using detail::WeightedRandomGeneratorAux<T>::_gen;
    using detail::WeightedRandomGeneratorAux<T>::_size;

    template<typename R>
    WeightedRandomGenerator(const R* weights, size_t size) :
            detail::WeightedRandomGeneratorAux<T>(weights, size), _int_distr() {

        using param_t = typename std::uniform_int_distribution<T>::param_type;
        _int_distr.param(param_t(0, _cumulative[size] - 1));
    }

    inline size_t get() noexcept {
        T value = _int_distr(_gen);
        auto it = std::upper_bound(_cumulative, _cumulative + _size, value);
        return std::distance(_cumulative, it) - 1;
    }
private:
    std::uniform_int_distribution<T> _int_distr;
};

template <typename T>
class WeightedRandomGenerator<T, typename std::enable_if<
                                    std::is_floating_point<T>::value>::type> :
                                public detail::WeightedRandomGeneratorAux<T> {
public:
    using detail::WeightedRandomGeneratorAux<T>::_cumulative;
    using detail::WeightedRandomGeneratorAux<T>::_gen;
    using detail::WeightedRandomGeneratorAux<T>::_size;

    template<typename R>
    WeightedRandomGenerator(const R* weights, size_t size) :
           detail::WeightedRandomGeneratorAux<T>(weights, size), _real_distr() {

        using param_t = typename std::uniform_real_distribution<T>::param_type;
        _real_distr.param(param_t(0, _cumulative[size - 1]));
    }

    inline size_t get() noexcept {
        T value = _real_distr(_gen);
        auto it = std::upper_bound(_cumulative, _cumulative + _size, value);
        return std::distance(_cumulative, it);
    }
private:
    std::uniform_real_distribution<T> _real_distr;
};

} // namespace xlib
