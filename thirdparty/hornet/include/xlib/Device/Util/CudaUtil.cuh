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

#include <array>    //std::array
#include <limits>   //std::numeric_limits

namespace xlib {

enum THREAD_GROUP { VOID = 0, WARP, BLOCK }; //deprecated

template<typename T>
struct numeric_limits {         // available in CUDA kernels
    static const T min    = std::numeric_limits<T>::min();
    static const T max    = std::numeric_limits<T>::max();
    static const T lowest = std::numeric_limits<T>::lowest();
};

//------------------------------------------------------------------------------

template<int SIZE>
struct CuFreeAtExit {
    template<typename... TArgs>
    explicit CuFreeAtExit(TArgs... args) noexcept;

    ~CuFreeAtExit() noexcept;
private:
    const std::array<void*, SIZE> _tmp;
};

void device_info(int device_id = 0);

} // namespace xlib

//------------------------------------------------------------------------------

namespace nvtx {

enum class NvColor : int
            { GREEN  = 0x0000FF00, BLUE = 0x000000FF, YELLOW = 0x00FFFF00,
              PURPLE = 0x00FF00FF, CYAN = 0x0000FFFF, RED    = 0x00FF0000,
              WHITE  = 0x00FFFFFF };

void push_range(const std::string& event_name, NvColor color) noexcept;
void pop_range() noexcept;

} // namespace nvtx

#include "impl/CudaUtil.i.cuh"
