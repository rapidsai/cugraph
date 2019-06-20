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
namespace hornets_nest {

template<typename... TArgs>
const AoSdev<TArgs...>::TypeSizesPS AoSdev<TArgs...>::TYPE_SIZES_PS;

template<typename... TArgs>
__device__ __forceinline__
AoSdev<TArgs...>::AoSdev<TArgs...>(void* d_ptr) :
                          _d_ptr(reinterpret_cast<AoSData<TArgs...>*>(d_ptr)) {}

template<typename... TArgs>
__device__ __forceinline__
AoSData<TArgs...>& AoSdev<TArgs...>::operator[](int index) {
    return _d_ptr[index];
}

template<typename... TArgs>
__device__ __forceinline__
const AoSData<TArgs...>& AoSdev<TArgs...>::operator[](int index) const {
    return _d_ptr[index];
}

template<typename... TArgs>
template<int INDEX>
__device__ __forceinline__
typename xlib::SelectType<INDEX, TArgs...>::type*
AoSdev<TArgs...>::ptr<INDEX>(int index) const {
    using T = typename xlib::SelectType<INDEX, TArgs...>::type;
    return reinterpret_cast<T*>
       (reinterpret_cast<xlib::byte_t*>(_d_ptr + index) + TYPE_SIZES_PS[INDEX]);
}

//==============================================================================

template<size_t PITCH, typename... TArgs>
__device__ __forceinline__
SoAdevPitch<PITCH, TArgs...>::SoAdevPitch(void* d_ptr) : _d_ptr(d_ptr) {}

template<size_t PITCH, typename... TArgs>
__device__ __forceinline__
SoARef<TArgs...> SoAdevPitch<PITCH, TArgs...>::operator[](int index) {
    using T = typename xlib::SelectType<0, TArgs...>::type;
    return SoARef<TArgs...>(reinterpret_cast<T*>(_d_ptr) + index, PITCH);
}

template<size_t PITCH, typename... TArgs>
template<int INDEX>
__device__ __forceinline__
typename xlib::SelectType<INDEX, TArgs...>::type*
SoAdevPitch<PITCH, TArgs...>::ptr(int index) const {
    using T = typename xlib::SelectType<0, TArgs...>::type;
    return reinterpret_cast<T*>
                   (reinterpret_cast<byte_t*>(_d_ptr) + PITCH * INDEX) + index;
}

//==============================================================================

template<typename... TArgs>
__device__ __forceinline__
SoAdev<TArgs...>::SoAdev(void* d_ptr, size_t pitch) : _d_ptr(d_ptr),
                                                      _pitch(pitch) {}

template<typename... TArgs>
__device__ __forceinline__
SoARef<TArgs...> SoAdev<TArgs...>::operator[](int index) {
    using T = typename xlib::SelectType<0, TArgs...>::type;
    return SoARef<TArgs...>(reinterpret_cast<T*>(_d_ptr) + index, _pitch);
}

template<typename... TArgs>
template<int INDEX>
__device__ __forceinline__
typename xlib::SelectType<INDEX, TArgs...>::type*
SoAdev<TArgs...>::ptr(int index) const {
    using T = typename xlib::SelectType<0, TArgs...>::type;
    return reinterpret_cast<T*>
                   (reinterpret_cast<byte_t*>(_d_ptr) + _pitch * INDEX) + index;
}

} // namespace hornets_nest
