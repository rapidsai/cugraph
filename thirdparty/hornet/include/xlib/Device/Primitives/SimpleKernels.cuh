/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date July, 2017
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
#include "Device/Util/DeviceProperties.cuh"
#include "Host/Numeric.hpp"

namespace xlib {
namespace gpu {
namespace detail {

template<unsigned UNROLL_STEPS = 1, typename SizeT, typename T>
__global__
void memsetKernel(T* d_out, SizeT num_items, T init_value) {
    static_assert(sizeof(T) <= sizeof(int4), "sizeof(T) <= sizeof(int4)");

    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO;

    SizeT           idx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT        stride = gridDim.x * blockDim.x;
    SizeT stride_unroll = stride * UNROLL_STEPS;
    SizeT         limit = num_items / THREAD_ITEMS;

    T storage[RATIO];
    #pragma unroll
    for (SizeT K = 0; K < RATIO; K++)
        storage[K] = init_value;
    const auto& to_write = reinterpret_cast<int4&>(storage);

    auto d_out4 = reinterpret_cast<int4*>(d_out);
    for (SizeT i = idx; i < limit; i += stride_unroll) {
        #pragma unroll
        for (int K = 0; K < UNROLL_STEPS; K++) {
            SizeT index = i + stride * K;
            if (index < limit)
                d_out4[index] = to_write;
        }
    }
    for (SizeT i = limit * THREAD_ITEMS + idx; i < num_items; i += stride)
        d_out[i] = init_value;
}

template<unsigned UNROLL_STEPS = 1, typename SizeT, typename T>
__global__
void memcpyKernel(const T* __restrict__ d_in, SizeT num_items,
                  T* __restrict__ d_out) {
    static_assert(sizeof(T) <= sizeof(int4), "sizeof(T) <= sizeof(int4)");

    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO;

    SizeT           idx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeT        stride = gridDim.x * blockDim.x;
    SizeT stride_unroll = stride * UNROLL_STEPS;
    SizeT         limit = num_items / THREAD_ITEMS;

    auto  d_in4 = reinterpret_cast<const int4*>(d_in);
    auto d_out4 = reinterpret_cast<int4*>(d_out);

    for (SizeT i = idx; i < limit; i += stride_unroll) {
        #pragma unroll
        for (int K = 0; K < UNROLL_STEPS; K++) {
            SizeT index = i + stride * K;
            if (index < limit)
                d_out4[index] = d_in4[index];
        }
    }
    for (SizeT i = limit * THREAD_ITEMS + idx; i < num_items; i += stride)
        d_out[i] = d_in[i];
}

} // namespace detail

const unsigned BLOCK_SIZE = 256;

template<typename T>
void memset(T* d_out, size_t num_items, const T& init_value) {
    static_assert(sizeof(int4) % sizeof(T) == 0 && sizeof(T) <= sizeof(int4),
                  "T not aligned");

    const unsigned UNROLL_STEPS = 16;
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    unsigned         num_blocks = xlib::ceil_div<RATIO * BLOCK_SIZE>(num_items);

    if ((num_items * UNROLL_STEPS) / RATIO < std::numeric_limits<int>::max()) {
        detail::memsetKernel<UNROLL_STEPS, int>
            <<< num_blocks, BLOCK_SIZE >>> (d_out, num_items, init_value);
    }
    else {
        detail::memsetKernel<UNROLL_STEPS, int64_t>
            <<< num_blocks, BLOCK_SIZE >>>  (d_out, num_items, init_value);
    }
}

template<typename T>
void memcpy(const T* d_in, size_t num_items, T* d_out) {
    static_assert(sizeof(int4) % sizeof(T) == 0 && sizeof(T) <= sizeof(int4),
                  "T not aligned");

    const unsigned UNROLL_STEPS = 16;
    const unsigned        RATIO = sizeof(int4) / sizeof(T);
    unsigned         num_blocks = xlib::ceil_div<RATIO * BLOCK_SIZE>(num_items);

    if ((num_items * UNROLL_STEPS) / RATIO < std::numeric_limits<int>::max()) {
        detail::memcpyKernel<UNROLL_STEPS, int>
            <<< num_blocks, BLOCK_SIZE >>> (d_in, num_items, d_out);
    }
    else {
        detail::memcpyKernel<UNROLL_STEPS, int64_t>
            <<< num_blocks, BLOCK_SIZE >>> (d_in, num_items, d_out);
    }
}

} // namespace gpu
} // namespace xlib
