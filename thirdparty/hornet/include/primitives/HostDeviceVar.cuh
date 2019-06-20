/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
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

#include <iostream>

namespace hornets_nest {

template<typename T>
class HostDeviceVar {
public:
    explicit HostDeviceVar() noexcept;

    explicit HostDeviceVar(const T& value) noexcept;

    HostDeviceVar(const HostDeviceVar& obj) noexcept;

    ~HostDeviceVar() noexcept;

    __host__ __device__ __forceinline__
    void sync() noexcept;

    __device__ __forceinline__
    T* ptr() noexcept;

    __host__ __device__ __forceinline__
    operator T() noexcept;

    __host__ __device__ __forceinline__
    const T& operator=(const T& value) noexcept;

    __host__ __device__ __forceinline__
    const T& operator()() const noexcept;

    __host__ __device__ __forceinline__
    T& operator()() noexcept;

    /*template<typename R>
    friend inline
    typename std::enable_if<xlib::is_stream_insertable<R>::value,
                            std::ostream&>::type
    operator<<(std::ostream& os, const HostDeviceVar<R>& obj) {
        os << obj;
        return os;
    }*/

private:
    T    _value;
    T*   _d_value_ptr { nullptr };

    int  _copy_count  { 0 };
    mutable bool _enable_sync { false };
};

} // namespace hornets_nest

#include "HostDeviceVar.i.cuh"
