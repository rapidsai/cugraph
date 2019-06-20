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

#include "Core/DataLayout/DataLayout.cuh"

namespace hornets_nest {

template<typename... TArgs>
class AoSdev {
public:
    __device__ __forceinline__
    explicit AoSdev(void* d_ptr);

    __device__ __forceinline__
    AoSData<TArgs...>& operator[](int index);

    __device__ __forceinline__
    const AoSData<TArgs...>& operator[](int index) const;

    template<int INDEX>
    __device__ __forceinline__
    typename xlib::SelectType<INDEX, TArgs...>::type*
    ptr(int index) const;
private:
    using   TypeSizes = xlib::Seq<sizeof(TArgs)...>;
    using TypeSizesPS = typename xlib::ExcPrefixSum<TypeSizes>::type;
    static const TypeSizesPS TYPE_SIZES_PS;

    AoSData<TArgs...>* _d_ptr;
};

//==============================================================================

template<size_t PITCH, typename... TArgs>
class SoAdevPitch {
    using byte_t = xlib::byte_t;
public:
    __device__ __forceinline__
    explicit SoAdevPitch(void* ptr);

    __device__ __forceinline__
    AoSData<TArgs...> operator[](int index) const;

    __device__ __forceinline__
    SoARef<TArgs...> operator[](int index);

    template<int INDEX>
    __device__ __forceinline__
    typename xlib::SelectType<INDEX, TArgs...>::type*
    ptr(int index) const;
private:
    void* _d_ptr;
};

template<typename... TArgs>
class SoAdev {
    using byte_t = xlib::byte_t;
public:
    __device__ __forceinline__
    explicit SoAdev(void* ptr, size_t pitch);

    __device__ __forceinline__
    AoSData<TArgs...> operator[](int index) const;

    __device__ __forceinline__
    SoARef<TArgs...> operator[](int index);

    template<int INDEX>
    __device__ __forceinline__
    typename xlib::SelectType<INDEX, TArgs...>::type*
    ptr(int index) const;
private:
    void*        _d_ptr;
    const size_t _pitch;
};

//==============================================================================

template<size_t PITCH, typename, bool = false, typename = void>
struct BestLayoutDevPitchAux;

template<size_t PITCH, typename... TArgs, bool FORCE_SOA>
struct BestLayoutDevPitchAux<PITCH, std::tuple<TArgs...>, FORCE_SOA,
                  typename std::enable_if<!FORCE_SOA &&
                                          xlib::IsVectorizable<TArgs...>::value>
                  ::type> : AoSdev<TArgs...> {

    __device__ __forceinline__
    explicit BestLayoutDevPitchAux(void* ptr) noexcept :
                                AoSdev<TArgs...>(ptr) {}
};

template<size_t PITCH, typename... TArgs, bool FORCE_SOA>
struct BestLayoutDevPitchAux<PITCH, std::tuple<TArgs...>, FORCE_SOA,
                 typename std::enable_if<FORCE_SOA ||
                                         !xlib::IsVectorizable<TArgs...>::value>
                 ::type> : SoAdevPitch<PITCH, TArgs...> {

    static_assert(sizeof...(TArgs) == sizeof...(TArgs), "not vectorizable");

    __device__ __forceinline__
    explicit BestLayoutDevPitchAux(void* ptr) noexcept :
                                SoAdevPitch<PITCH, TArgs...>(ptr) {}
};

//------------------------------------------------------------------------------


template<typename, bool = false, typename = void>
struct BestLayoutDevAux;

template<typename... TArgs, bool FORCE_SOA>
struct BestLayoutDevAux<std::tuple<TArgs...>, FORCE_SOA,
                typename std::enable_if<!FORCE_SOA &&
                                        xlib::IsVectorizable<TArgs...>::value>
                ::type> : AoSdev<TArgs...> {

    __device__ __forceinline__
    explicit BestLayoutDevAux(void* ptr, size_t) noexcept :
                                AoSdev<TArgs...>(ptr) {}
};

template<typename... TArgs, bool FORCE_SOA>
struct BestLayoutDevAux<std::tuple<TArgs...>, FORCE_SOA,
                 typename std::enable_if<FORCE_SOA ||
                                         !xlib::IsVectorizable<TArgs...>::value>
                 ::type> : SoAdev<TArgs...> {

    static_assert(sizeof...(TArgs) == sizeof...(TArgs), "not vectorizable");

    __device__ __forceinline__
    explicit BestLayoutDevAux(void* ptr, size_t pitch) noexcept :
                                SoAdev<TArgs...>(ptr, pitch) {}
};

//------------------------------------------------------------------------------
/*
template<size_t PITCH, typename... TArgs>
struct BestLayoutDevPitch :
                     public BestLayoutDevPitchAux<PITCH, std::tuple<TArgs...>> {

    __device__ __forceinline__
    explicit BestLayoutDevPitch(void* ptr) noexcept :
                      BestLayoutDevPitchAux<PITCH, std::tuple<TArgs...>>(ptr) {}
};*/
/*
template<typename... TArgs>
struct BestLayoutDev : public BestLayoutDevAux<std::tuple<TArgs...>> {

    __device__ __forceinline__
    explicit BestLayoutDev(void* ptr, size_t pitch) noexcept :
                           BestLayoutDevAux<std::tuple<TArgs...>>(ptr, pitch) {}
};*/

} // namespace hornets_nest

#include "DataLayoutDev.i.cuh"
