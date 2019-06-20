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
#include "Host/Numeric.hpp"     //xlib::delete_bits
#include <random>

namespace xlib {

inline Bitmask::Bitmask(size_t size) noexcept :
                                _num_word(xlib::ceil_div<32>(size)),
                                _size(size) {
    try {
        _array = new unsigned[_num_word]();
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

inline Bitmask::~Bitmask() noexcept {
    delete[] _array;
}

inline void Bitmask::init(size_t size) noexcept {
    _size = size;
    _num_word = xlib::ceil_div<32>(size);
    try {
        _array = new unsigned[_num_word]();
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

inline void Bitmask::free() noexcept {
    delete[] _array;
    _array = nullptr;
}

inline BitRef Bitmask::operator[](size_t index) noexcept {
    assert(index < _size);
    return BitRef(_array[index >> 5u], 1u << (index % 32u));
}

inline bool Bitmask::operator[](size_t index) const noexcept {
    assert(index < _size);
    return static_cast<bool>(_array[index >> 5u] & (1u << (index % 32u)));
}

inline void Bitmask::randomize() noexcept {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                .count();
    randomize(seed);
}

inline void Bitmask::randomize(uint64_t seed) noexcept {
    auto    generator = std::mt19937_64(seed);
    auto distribution = std::uniform_int_distribution<unsigned>(
                            std::numeric_limits<unsigned>::lowest(),
                            std::numeric_limits<unsigned>::max());
    const auto lambda = [&]{ return distribution(generator); };
    std::generate(_array, _array + _num_word, lambda);
}

inline void Bitmask::clear() noexcept {
    std::fill(_array, _array + _num_word, 0);
}

inline size_t Bitmask::size() const noexcept {
    size_t count = 0;
    for (size_t i = 0; i < _num_word; i++)
        count += static_cast<size_t>(__builtin_popcount(_array[i]));
    return count;
}

//==============================================================================

template<unsigned SIZE>
inline BitRef BitmaskStack<SIZE>::operator[](unsigned index) noexcept {
    assert(index < SIZE);
    return BitRef(_array[index >> 5u], 1u << (index % 32u));
}

template<unsigned SIZE>
inline bool BitmaskStack<SIZE>::operator[](unsigned index) const noexcept {
    assert(index < SIZE);
    return static_cast<bool>(_array[index >> 5u] & (1u << (index % 32u)));
}

template<unsigned SIZE>
inline void BitmaskStack<SIZE>::clear() noexcept {
    std::fill(_array, _array + xlib::CeilDiv<SIZE, 32>::value, 0);
}

template<unsigned SIZE>
inline unsigned BitmaskStack<SIZE>::get_count() const noexcept {
    unsigned count = 0;
    for (unsigned i = 0; i < xlib::CeilDiv<SIZE, 32>::value; i++)
        count += static_cast<unsigned>(__builtin_popcount(_array[i]));
    return count;
}

} // namespace xlib
