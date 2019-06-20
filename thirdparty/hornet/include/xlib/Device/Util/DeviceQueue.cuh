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
 *
 * @file
 */
#pragma once

namespace xlib {

/**
 * @brief The class implements a general type *device* queue
 * @tparam T type of objects stored in the queue
 * @tparam SIZE number of thread registers used to temporaly store the queue
 *         objects. If `SIZE == 1` the store operation in global memory is
 *         immediately executed by using an optimized binary prefix-sum to
 *         ensure memory coalecing. If `SIZE > 1` the object are temporaly
 *         stored in the registers and then stored in global memory by using an
 *         optimized warp-shuffle prefix-sum when the capacity is equal to
 *         `SIZE`
 * @remark the number of global atomic operations is equal to
 *         \f$\lceil \frac{total\_enqueue_items}{SIZE} \rceil$\f
 * @remark only `SIZE == 1` ensures full-coalescing. It is progressibely lost
 *         for `SIZE > 1`
 * @remark more big is `SIZE` more the register pressure increase
 */
template<typename T, int SIZE = 16>
class DeviceQueue {
public:
    /**
     * @brief Default costructor
     * @param[in] queue_ptr **initial** pointer to the global memory queue
     * @param[in] size_ptr pointer to global counter of the number of queue
     *            items
     * @pre `size_ptr` must be zero before used the class
     */
    __device__ __forceinline__
    DeviceQueue(T (&queue)[SIZE], T* __restrict__ queue_ptr,
                int* __restrict__ size_ptr);

    __device__ __forceinline__
    ~DeviceQueue();

    /**
     * @brief insert an item in the queue
     * @param item item to insert
     * @remark after this method call the item is **not** stored in global
     *         memory
     */
    __device__ __forceinline__
    void insert(const T& item);

    /**
     * @brief actual number of items temporary stored in local queue (register)
     * @return number of items in the local queue
     */
    __device__ __forceinline__
    int size() const;

    /**
     * @brief store the actual items of the queue in global memory
     */
    __device__ __forceinline__
    void store();

private:
    T*   _queue_ptr;
    int* _size_ptr;
    T    (&_queue)[SIZE];
    int  _size { 0 };

    __device__ __forceinline__
    void store_localqueue();

    __device__ __forceinline__
    void store_localqueue_aux();

    __device__ __forceinline__
    void store_ballot();
};

class DeviceQueueOffset {
public:
    __device__ __forceinline__
    DeviceQueueOffset(int* __restrict__ size_ptr);

    __device__ __forceinline__
    int offset();
private:
    int* _size_ptr;
};

} // namespace xlib

#include "impl/DeviceQueue.i.cuh"
