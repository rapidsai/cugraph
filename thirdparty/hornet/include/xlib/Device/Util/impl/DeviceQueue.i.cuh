/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
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
#include "Device/Util/PTX.cuh"
#include "Device/Primitives/WarpScan.cuh"
#include <cassert>

namespace xlib {

template<typename T, int SIZE>
__device__ __forceinline__
DeviceQueue<T, SIZE>::DeviceQueue(T    (&queue)[SIZE],
                                  T*   __restrict__ queue_ptr,
                                  int* __restrict__ size_ptr) :
                                       _queue(queue),
                                       _queue_ptr(queue_ptr),
                                       _size_ptr(size_ptr) {}

template<typename T, int SIZE>
__device__ __forceinline__
DeviceQueue<T, SIZE>::~DeviceQueue() {
    if (SIZE > 1)
        store_localqueue_aux();
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::insert(const T& item) {
    if (SIZE == 1) {
        _queue[0] = item;
        _size = 1;
    }
    else
        _queue[_size++] = item;
}

template<typename T, int SIZE>
__device__ __forceinline__
int DeviceQueue<T, SIZE>::size() const {
    return _size;
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::store() {
    if (SIZE == 1) store_ballot();
    else store_localqueue();
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::store_localqueue() {
    assert(__activemask() == static_cast<unsigned>(-1));
    if (__any_sync(0xFFFFFFFF, _size >= SIZE))
        store_localqueue_aux();
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::store_localqueue_aux() {
    int thread_offset = _size;
    int   warp_offset = xlib::WarpExclusiveScan<>::atomicAdd(thread_offset,
                                                             _size_ptr);
    T* ptr = _queue_ptr + warp_offset + thread_offset;
    for (int i = 0; i < _size; i++)
        ptr[i] = _queue[i];
    _size = 0;
}

template<typename T, int SIZE>
__device__ __forceinline__
void DeviceQueue<T, SIZE>::store_ballot() {
    unsigned       ballot = __ballot_sync(0xFFFFFFFF, _size);
    unsigned elected_lane = xlib::__msb(ballot);
    int warp_offset;
    if (xlib::lane_id() == elected_lane)
        warp_offset = atomicAdd(_size_ptr, __popc(ballot));
    int offset = __popc(ballot & xlib::lanemask_lt()) +
                 __shfl_sync(0xFFFFFFFF, warp_offset, elected_lane);
    if (_size) {
        _queue_ptr[offset] = _queue[0];
        _size = 0;
    }
}

//------------------------------------------------------------------------------

__device__ __forceinline__
DeviceQueueOffset::DeviceQueueOffset(int* __restrict__ size_ptr) :
                                        _size_ptr(size_ptr) {}

__device__ __forceinline__
int DeviceQueueOffset::offset() {
    unsigned       ballot = __activemask();
    unsigned elected_lane = xlib::__msb(ballot);
    int warp_offset;
    if (xlib::lane_id() == elected_lane)
        warp_offset = atomicAdd(_size_ptr, __popc(ballot));
    return  __popc(ballot & xlib::lanemask_lt()) +
            __shfl_sync(0xFFFFFFFF, warp_offset, elected_lane);
}

} // namespace xlib
