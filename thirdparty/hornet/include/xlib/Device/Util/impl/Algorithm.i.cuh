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
#include "Device/Util/DeviceProperties.cuh"  //xlib::WARP_SIZE
#include "Device/Util/SafeCudaAPI.cuh"       //cuMemcpyToHost
#include "Device/Util/SafeCudaAPISync.cuh"       //cuMemcpyToHost
#include "Host/Algorithm.hpp"                //xlib::equal_sorted
#include "Host/Numeric.hpp"                  //xlib::is_power2

namespace xlib {
namespace gpu {

template<typename HostIterator, typename DeviceIterator>
bool equal(HostIterator host_start, HostIterator host_end,
           DeviceIterator device_start) noexcept {
    using R = typename std::iterator_traits<DeviceIterator>::value_type;
    auto size = std::distance(host_start, host_end);
    R* array = new R[size];
    cuMemcpyToHost(&(*device_start), size, array);

    bool flag = std::equal(host_start, host_end, array);
    if (!flag) {
        for (int i = 0; i < size; i++) {
            if (host_start[i] != array[i]) {
                std::cout << "\nhost:   " << host_start[i]
                          << "\ndevice: " << array[i]
                          << "\nat:     " << i << std::endl;
                break;
            }
        }
    }
    delete[] array;
    return flag;
}

template<typename HostIterator, typename DeviceIterator>
bool equal_sorted(HostIterator host_start, HostIterator host_end,
                  DeviceIterator device_start) noexcept {
    using R = typename std::iterator_traits<DeviceIterator>::value_type;
    auto size = std::distance(host_start, host_end);
    R* array = new R[size];
    cuMemcpyToHost(&(*device_start), size, array);

    bool flag = xlib::equal_sorted(host_start, host_end, array, array + size);
    delete[] array;
    return flag;
}

} // namespace gpu

//------------------------------------------------------------------------------

template<unsigned SIZE, typename T>
__device__ __forceinline__
int binary_search_pow2(const T* shared_mem, T searched) {
    static_assert(xlib::is_power2(SIZE), "SIZE must be a power of 2");
    int low = 0;
    #pragma unroll
    for (int i = 1; i <= xlib::Log2<SIZE>::value; i++) {
        int pos = low + ((SIZE) >> i);
        if (searched >= shared_mem[pos])
            low = pos;
    }
    return low;
}

template<typename T>
__device__ __forceinline__
int binary_search_warp(T reg_value, T searched) {
    int low = 0;
    #pragma unroll
    for (int i = 1; i <= xlib::Log2<WARP_SIZE>::value; i++) {
        int pos = low + ((xlib::WARP_SIZE) >> i);
        if (searched >= __shfl_sync(0xFFFFFFFF, reg_value, pos))
            low = pos;
    }
    return low;
}

} // namespace xlib
