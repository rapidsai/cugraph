/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
#include "StandardAPI.hpp"
#include <Device/Util/SafeCudaAPI.cuh>
#include <Device/Util/SafeCudaAPISync.cuh>
#include <cstring>

namespace hornets_nest {
////////////////
// AoSData<T> //
////////////////

template<typename T>
HOST_DEVICE
AoSData<T>::AoSData(const T& value) noexcept : _value(value) {}

template<typename T>
template<typename R>
HOST_DEVICE
AoSData<T>::AoSData(const R ptr, size_t,
                    typename std::enable_if<
                            std::is_pointer<R>::value>::type*) noexcept :
                    _value(*reinterpret_cast<const T*>(ptr)) {}

template<typename T>
template<typename R, int SIZE>
AoSData<T>::AoSData(const R* const (&array)[SIZE], size_t index) noexcept :
            _value(reinterpret_cast<const T*>(array[0])[index]) {}

template<typename T>
template<int INDEX>
HOST_DEVICE
const T& AoSData<T>::get() const noexcept {
    static_assert(INDEX == 0, "error");
    return _value;
}

template<typename T>
template<int INDEX>
HOST_DEVICE
T& AoSData<T>::get() noexcept {
    static_assert(INDEX == 0, "error");
    return _value;
}

template<typename T>
template<int INDEX, typename R, int SIZE>
void AoSData<T>::assign(const R* const (&array)[SIZE], int index) {
    _value = reinterpret_cast<const T*>(array[INDEX])[index];
}

//==============================================================================
////////////////////
// AoSData<TArgs> //
////////////////////

template<typename T, typename... TArgs>
HOST_DEVICE
AoSData<T, TArgs...>::AoSData(const T& value, const TArgs&... args) noexcept :
                                        _value(value),
                                        _tail(args...) {}

template<typename T, typename... TArgs>
template<typename R>
HOST_DEVICE
AoSData<T, TArgs...>::AoSData(const R ptr, size_t pitch,
                              typename std::enable_if<
                                   !std::is_array<R>::value>::type*) noexcept :
            _value(*reinterpret_cast<const T*>(ptr)),
            _tail(reinterpret_cast<const xlib::byte_t*>(ptr) + pitch, pitch) {}

template<typename T, typename... TArgs>
template<typename R, int SIZE>
AoSData<T, TArgs...>::AoSData(const R* const (&array)[SIZE],
                              size_t index) noexcept:
                           _value(reinterpret_cast<const T*>(array[0])[index]) {
    _tail.template assign<1>(array, index);
}

template<typename T, typename... TArgs>
template<int INDEX, typename R, int SIZE>
void AoSData<T, TArgs...>::assign(const R* const (&array)[SIZE],
                                  int index) noexcept {
    _value = reinterpret_cast<const T*>(array[INDEX])[index];
    _tail.template assign<INDEX + 1>(array, index);
}

template<typename T, typename... TArgs>
template<int INDEX>
HOST_DEVICE
typename std::enable_if<INDEX == 0, const T&>::type
AoSData<T, TArgs...>::get() const noexcept {
    return _value;
}

template<typename T, typename... TArgs>
template<int INDEX>
HOST_DEVICE
typename std::enable_if<INDEX == 0, T&>::type
AoSData<T, TArgs...>::get() noexcept {
    return _value;
}

template<typename T, typename... TArgs>
template<int INDEX>
HOST_DEVICE
typename std::enable_if<INDEX != 0,
                      typename xlib::SelectType<INDEX, const T&, TArgs...>::type
                        >::type
AoSData<T, TArgs...>::get() const noexcept {
    return _tail.template get<INDEX - 1>();
}

template<typename T, typename... TArgs>
template<int INDEX>
HOST_DEVICE
typename std::enable_if<INDEX != 0,
                        typename xlib::SelectType<INDEX, T&, TArgs...>::type
                        >::type
AoSData<T, TArgs...>::get() noexcept {
    return _tail.template get<INDEX - 1>();
}

//==============================================================================
/////////
// AoS //
/////////

template<typename... TArgs>
AoS<TArgs...>::AoS(const void* (&array)[sizeof...(TArgs)],
                   int num_items) noexcept :
                                    _num_items(num_items) {
    _h_ptr = new AoSData<TArgs...>[num_items];
    for (auto i = 0; i < num_items; i++)
        _h_ptr[i] = AoSData<TArgs...>(array, i);
    gpu::allocate(_d_ptr, num_items);
    host::copyToDevice(_h_ptr, num_items, _d_ptr);
}

template<typename... TArgs>
AoS<TArgs...>::~AoS() noexcept {
    delete[] _h_ptr;
    gpu::free(_d_ptr);
}

template<typename... TArgs>
void AoS<TArgs...>::initialize(const void* (&array)[sizeof...(TArgs)],
                               int num_items) noexcept {
    _num_items = num_items;
    _h_ptr     = new AoSData<TArgs...>[num_items];
    for (auto i = 0; i < num_items; i++)
        _h_ptr[i] = AoSData<TArgs...>(array, i);
    gpu::allocate(_d_ptr, num_items);
    host::copyToDevice(_h_ptr, num_items, _d_ptr);
}

template<typename... TArgs>
void AoS<TArgs...>::update() noexcept {
    gpu::copyToHost(_d_ptr, _num_items, _h_ptr);
}

template<typename... TArgs>
HOST_DEVICE
AoSData<TArgs...>& AoS<TArgs...>::operator[](int index) noexcept {
    assert(index < _num_items);
#if defined(__CUDA_ARCH__)
    return _d_ptr[index];
#else
    return _h_ptr[index];
#endif
}

template<typename... TArgs>
HOST_DEVICE
const AoSData<TArgs...>& AoS<TArgs...>::operator[](int index) const noexcept {
    assert(index < _num_items);
#if defined(__CUDA_ARCH__)
    return _d_ptr[index];
#else
    return _h_ptr[index];
#endif
}

template<typename... TArgs>
void* AoS<TArgs...>::device_ptr() const noexcept {
    return _d_ptr;
}

template<typename... TArgs>
size_t AoS<TArgs...>::pitch() const noexcept {
    return 0;
}

//==============================================================================
/////////
// SoA //
/////////

template<typename... TArgs>
const SoA<TArgs...>::TypeSizes SoA<TArgs...>::TYPE_SIZES;

template<typename... TArgs>
SoA<TArgs...>::SoA(void* (&array)[sizeof...(TArgs)],
                   int num_items) noexcept :
                        _num_items(num_items),
                        _pitch(xlib::upper_approx<512>(num_items) * MAX_SIZE) {
    auto allocated_items = xlib::upper_approx<512>(num_items);
    _h_ptr = new AoSData<TArgs...>[allocated_items];
    gpu::allocate(_d_ptr, allocated_items);

    for (int i = 0; i < NUM_ARGS; i++) {
        auto d_ptr = reinterpret_cast<byte_t*>(_d_ptr) +  _pitch * i;
        host::copyToDevice(static_cast<byte_t*>(array[i]),
                         allocated_items * TYPE_SIZES[i], d_ptr);
    }
}

template<typename... TArgs>
SoA<TArgs...>::~SoA() noexcept {
    delete[] _h_ptr;
    gpu::free(_d_ptr);
}

template<typename... TArgs>
void SoA<TArgs...>::initialize(const void* (&array)[sizeof...(TArgs)],
                               int num_items) noexcept {
    _num_items = num_items;
    _pitch     = xlib::upper_approx<512>(num_items) * MAX_SIZE;
    auto allocated_items = xlib::upper_approx<512>(num_items);
    gpu::allocate(_d_ptr, allocated_items);

    for (int i = 0; i < NUM_ARGS; i++) {
        auto d_ptr = reinterpret_cast<byte_t*>(_d_ptr) +  _pitch * i;
        host::copyToDevice(static_cast<const byte_t*>(array[i]),
                         allocated_items * TYPE_SIZES[i], d_ptr);
    }
}

template<typename... TArgs>
void SoA<TArgs...>::update() noexcept {
    auto allocated_items = xlib::upper_approx<512>(_num_items);
    if (_h_ptr == nullptr)
        _h_ptr = new AoSData<TArgs...>[allocated_items];
    gpu::copyToHost(_d_ptr, allocated_items, _h_ptr);
}

template<typename... TArgs>
HOST_DEVICE
AoSData<TArgs...> SoA<TArgs...>::operator[](int index) const noexcept {
    assert(index < _num_items);
    using T = typename xlib::SelectType<0, TArgs...>::type;
#if defined(__CUDA_ARCH__)
    return AoSData<TArgs...>(reinterpret_cast<T*>(_d_ptr) + index, _pitch);
#else
    if (_h_ptr == nullptr)
        update();
    return AoSData<TArgs...>(reinterpret_cast<T*>(_h_ptr) + index, _pitch);
#endif
}

template<typename... TArgs>
HOST_DEVICE
SoARef<TArgs...> SoA<TArgs...>::operator[](int index) noexcept {
    assert(index < _num_items);
    using T = typename xlib::SelectType<0, TArgs...>::type;
#if defined(__CUDA_ARCH__)
    return SoARef<TArgs...>(reinterpret_cast<T*>(_d_ptr) + index, _pitch);
#else
    return SoARef<TArgs...>(reinterpret_cast<T*>(_h_ptr) + index, _pitch);
#endif
}

template<typename... TArgs>
void* SoA<TArgs...>::device_ptr() const noexcept {
    return _d_ptr;
}

template<typename... TArgs>
size_t SoA<TArgs...>::pitch() const noexcept {
    return _pitch;
}

//==============================================================================
////////////
// SoARef //
////////////

template<int INDEX, int MAX>
struct RecursiveAssign {
    template<typename... TArgs>
    HOST_DEVICE
    static void apply(const AoSData<TArgs...>& obj, void* ptr, size_t pitch) {
        using T = typename xlib::SelectType<INDEX, TArgs...>::type;
        *static_cast<T*>(ptr) = obj.template get<INDEX>();
        RecursiveAssign<INDEX + 1, MAX>::apply
            (obj, static_cast<xlib::byte_t*>(ptr) + pitch, pitch);
    }
};

template<int MAX>
struct RecursiveAssign<MAX, MAX> {
    template<typename... TArgs>
    HOST_DEVICE
    static void apply(const AoSData<TArgs...>& obj, void* ptr, size_t) {
        using T = typename xlib::SelectType<MAX, TArgs...>::type;
        *static_cast<T*>(ptr) = obj.template get<MAX>();
    }
};

template<typename... TArgs>
HOST_DEVICE
SoARef<TArgs...>::SoARef(void* ptr, size_t pitch) noexcept :
                                            _ptr(ptr),
                                            _pitch(pitch) {}

template<typename... TArgs>
HOST_DEVICE
const SoARef<TArgs...>&
SoARef<TArgs...>::operator=(const AoSData<TArgs...>& obj) noexcept {
    RecursiveAssign<0, sizeof...(TArgs) - 1>::apply(obj, _ptr, _pitch);
    return *this;
}

template<typename... TArgs>
HOST_DEVICE
SoARef<TArgs...>::operator AoSData<TArgs...>() const noexcept {
    return AoSData<TArgs...>(_ptr, _pitch);
}

} // namespace hornets_nest
