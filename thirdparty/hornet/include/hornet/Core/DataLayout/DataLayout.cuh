/**
 * @brief Hornet
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
 *
 * @file
 */
#pragma once

#include "BasicTypes.hpp"   //xlib::byte_t

namespace hornets_nest {

template<typename... TArgs>
struct AoSData;

template<typename T>
class AoSData<T> {
    template<typename...> friend class AoSData;
public:
    HOST_DEVICE
    explicit AoSData() noexcept {}

    HOST_DEVICE
    explicit AoSData(const T& value) noexcept;

    template<typename R>
    HOST_DEVICE
    explicit AoSData(const R ptr, size_t,
                     typename std::enable_if<
                        std::is_pointer<R>::value>::type* = 0) noexcept;

    template<typename R, int SIZE>
    explicit AoSData(const R* const (&array)[SIZE], size_t index) noexcept;

    template<int INDEX = 0>
    HOST_DEVICE
    const T& get() const noexcept;

    template<int INDEX = 0>
    HOST_DEVICE
    T& get() noexcept;

    friend inline std::ostream& operator<<(std::ostream& os,
                                           const AoSData& obj) {
        os << obj._value;
        return os;
    }

private:
    T _value;

    template<int INDEX, typename R, int SIZE>
    void assign(const R* const (&array)[SIZE], int index);
};

//==============================================================================

template<unsigned SIZE>
struct __align__(SIZE) ForceAlign {};

template<typename T, typename... TArgs>
class
AoSData<T, TArgs...> :
             ForceAlign<xlib::roundup_pow2(xlib::SizeSum<T, TArgs...>::value)> {
public:
    HOST_DEVICE
    explicit AoSData() noexcept {}

    HOST_DEVICE
    explicit AoSData(const T& value, const TArgs&... args) noexcept;

    template<typename R, int SIZE>
    explicit AoSData(const R* const (&array)[SIZE], size_t index) noexcept;

    template<typename R>
    HOST_DEVICE
    explicit AoSData(const R ptr, size_t pitch,
                     typename std::enable_if<
                            !std::is_array<R>::value>::type* = 0) noexcept;

    template<int INDEX>
    HOST_DEVICE
    typename std::enable_if<INDEX == 0, const T&>::type
    get() const noexcept;

    template<int INDEX>
    HOST_DEVICE
    typename std::enable_if<INDEX == 0, T&>::type
    get() noexcept;


    template<int INDEX>
    HOST_DEVICE
    typename std::enable_if<INDEX != 0,
                      typename xlib::SelectType<INDEX, const T&, TArgs...>::type
                           >::type
    get() const noexcept;

    template<int INDEX>
    HOST_DEVICE
    typename std::enable_if<INDEX != 0,
                      typename xlib::SelectType<INDEX, T&, TArgs...>::type
                           >::type
    get() noexcept;

    template<typename...>
    friend inline std::ostream& operator<<(std::ostream& os,
                                           const AoSData& obj) {
        os << obj._value << ", " << obj._tail;
        return os;
    }

private:
    T                 _value;
    AoSData<TArgs...> _tail;

    template<int INDEX, typename R, int SIZE>
    void assign(const R* const (&array)[SIZE], int index) noexcept;
};

//==============================================================================

template<typename... TArgs>
class AoSdev;

template<typename... TArgs>
class AoS {
public:
    explicit AoS() noexcept {}

    explicit AoS(const AoSData<TArgs...>* array, int num_items) noexcept;

    explicit AoS(const void* (&array)[sizeof...(TArgs)], int num_items)
                 noexcept;

    ~AoS() noexcept;

    void initialize(const void* (&array)[sizeof...(TArgs)], int num_items)
                    noexcept;

    HOST_DEVICE
    AoSData<TArgs...>& operator[](int index) noexcept;

    HOST_DEVICE
    const AoSData<TArgs...>& operator[](int index) const noexcept;

    void update() noexcept;

    void* device_ptr() const noexcept;

    size_t pitch() const noexcept;
private:
    AoSData<TArgs...>* _d_ptr     { nullptr };
    AoSData<TArgs...>* _h_ptr     { nullptr };
    int                _num_items { 0 };
};

//==============================================================================

template<typename... TArgs>
class SoA;

template<typename... TArgs>
class SoARef {
    template<typename...>         friend class SoA;
    template<typename...>         friend class SoAdev;
    template<size_t, typename...> friend class SoAdevPitch;
public:
    HOST_DEVICE
    const SoARef& operator=(const AoSData<TArgs...>& obj) noexcept;

    HOST_DEVICE
    operator AoSData<TArgs...>() const noexcept;

    template<typename... RArgs>
    friend inline std::ostream& operator<<(std::ostream& os,
                                           const SoARef<RArgs...>& obj) {
        os << AoSData<RArgs...>(obj);
        return os;
    }
private:
    void*        _ptr;
    const size_t _pitch;

    HOST_DEVICE
    explicit SoARef(void* ptr, size_t pitch) noexcept;
};

template<typename... TArgs>
class SoA {
public:
    explicit SoA() noexcept {}

    explicit SoA(const AoSData<TArgs...>* array, int num_items) noexcept;

    explicit SoA(void* (&array)[sizeof...(TArgs)],
                 int num_items) noexcept;

    ~SoA() noexcept;

    void initialize(const void* (&array)[sizeof...(TArgs)], int num_items)
                    noexcept;

    HOST_DEVICE
    AoSData<TArgs...> operator[](int index) const noexcept;

    HOST_DEVICE
    SoARef<TArgs...> operator[](int index) noexcept;

    void update() noexcept;

    void* device_ptr() const noexcept;

    size_t pitch() const noexcept;
private:
    using TypeSizes = xlib::Seq<sizeof(TArgs)...>;

    static const unsigned NUM_ARGS = sizeof...(TArgs);
    static const unsigned MAX_SIZE = xlib::MaxSize<TArgs...>::value;
    static const TypeSizes TYPE_SIZES;

    size_t _pitch     { 0 };
    int    _num_items { 0 };

    AoSData<TArgs...>* _h_ptr { nullptr };
    AoSData<TArgs...>* _d_ptr { nullptr };
};

//==============================================================================

template<typename, typename = void>
struct AoSDataHelperAux;

template<typename... TArgs>
struct AoSDataHelperAux<std::tuple<TArgs...>,
                  typename std::enable_if<xlib::IsVectorizable<TArgs...>::value>
                  ::type> : AoSData<TArgs...> {

    HOST_DEVICE
    explicit AoSDataHelperAux(void* ptr, size_t) noexcept :
                AoSData<TArgs...>(*reinterpret_cast<AoSData<TArgs...>*>(ptr)) {}
};

template<typename... TArgs>
struct AoSDataHelperAux<std::tuple<TArgs...>,
                 typename std::enable_if<!xlib::IsVectorizable<TArgs...>::value>
                 ::type> : AoSData<TArgs...> {

    HOST_DEVICE
    explicit AoSDataHelperAux(void* ptr, size_t pitch) noexcept :
                AoSData<TArgs...>(ptr, pitch) {}
};

template<typename... TArgs>
struct AoSDataHelper : public AoSDataHelperAux<std::tuple<TArgs...>> {
    __device__ __forceinline__
    explicit AoSDataHelper(void* ptr, size_t pitch) noexcept :
                           AoSDataHelperAux<std::tuple<TArgs...>>(ptr, pitch) {}
};

//==============================================================================

template<typename, bool = false, typename = void>
struct BestLayoutAux;

template<typename... TArgs, bool FORCE_SOA>
struct BestLayoutAux<std::tuple<TArgs...>, FORCE_SOA,
                  typename std::enable_if<!FORCE_SOA &&
                                          xlib::IsVectorizable<TArgs...>::value>
                  ::type> : AoS<TArgs...> {

    explicit BestLayoutAux() noexcept : AoS<TArgs...>() {}

    explicit BestLayoutAux(void* (&array)[sizeof...(TArgs)], int num_items)
                           noexcept :
                                AoS<TArgs...>(array, num_items) {}
};

template<typename... TArgs, bool FORCE_SOA>
struct BestLayoutAux<std::tuple<TArgs...>, FORCE_SOA,
                 typename std::enable_if<FORCE_SOA ||
                                         !xlib::IsVectorizable<TArgs...>::value>
                 ::type> : SoA<TArgs...> {

    static_assert(sizeof...(TArgs) == sizeof...(TArgs), "not vectorizable");

    explicit BestLayoutAux() noexcept : SoA<TArgs...>() {}

    explicit BestLayoutAux(void* (&array)[sizeof...(TArgs)], int num_items)
                           noexcept :
                                SoA<TArgs...>(array, num_items) {}
};

} // namespace hornets_nest

#include "DataLayout.i.cuh"
