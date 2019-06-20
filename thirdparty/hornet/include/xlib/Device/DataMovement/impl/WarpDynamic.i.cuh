/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
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
#include "Base/Device/DataMovement/RegInsert.cuh"
#include "Base/Device/Primitives/WarpReduce.cuh"
#include "Base/Device/Primitives/WarpScan.cuh"
#include "Base/Device/Util/Basic.cuh"
#include "Base/Device/Util/Definition.cuh"
#include "Base/Device/Util/PTX.cuh"

namespace xlib {
namespace detail {

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
WarpQueueBase<T, SIZE, CM>
::WarpQueueBase(T (&queue)[SIZE],
                T*   __restrict__ queue_ptr,
                int* __restrict__ size_ptr) : _queue(queue),
                                              _queue_ptr(queue_ptr),
                                              _size_ptr(size_ptr),
                                              _size(0) {}

} // namespace detail
//------------------------------------------------------------------------------

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
WarpQueueSimple<T, SIZE, CM>::WarpQueueSimple(T (&queue)[SIZE],
                                              T*   __restrict__ queue_ptr,
                                              int* __restrict__ size_ptr) :
             detail::WarpQueueBase<T, SIZE, CM>(queue, queue_ptr, size_ptr) {}

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
WarpQueueSimple<T, SIZE, CM>::~WarpQueueSimple() {
    _store();
}

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
void WarpQueueSimple<T, SIZE, CM>::insert(T item) {
    _queue[_size++] = item;
}

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
void WarpQueueSimple<T, SIZE, CM>::store() {
    assert(__ballot(true) == static_cast<unsigned>(-1));
    if (__any(_size >= SIZE)) {
        _store();
        _size = 0;
    }
}

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
void WarpQueueSimple<T, SIZE, CM>::_store() {
    int thread_offset = _size, total;
    int   warp_offset = WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                       _size_ptr, total);
    T* ptr = _queue_ptr + warp_offset + thread_offset;
    for (int i = 0; i < _size; i++)
        Store<CM>(ptr + i, _queue[i]);
}
//------------------------------------------------------------------------------

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
WarpQueueUnroll<T, SIZE, CM>::~WarpQueueUnroll() {
    int thread_offset = _size, total;
    int   warp_offset = WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                       _size_ptr, total);
    T* ptr = _queue_ptr + warp_offset + thread_offset;
    for (int i = 0; i < _size; i++)
        Store<CM>(ptr + i, _queue[i]);
}

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
void WarpQueueUnroll<T, SIZE, CM>::insert(T item) {
    Reg<>::insert(_queue, _size, item);
}

template<typename T, int SIZE, CacheModifier CM>
__device__ __forceinline__
void WarpQueueUnroll<T, SIZE, CM>::store() {
    assert(__ballot(true) == static_cast<unsigned>(-1));
    if (__any(_size >= SIZE)) {
        int thread_offset = _size, total;
        int   warp_offset = WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                           _size_ptr, total);
        T* ptr = _queue_ptr + warp_offset + thread_offset;
        #pragma unroll
        for (int i = 0; i < _size; i++) {
            if (i < _size)
                Store<CM>(ptr + i, _queue[i]);
        }
        _size = 0;
    }
}
//------------------------------------------------------------------------------

template<typename T, CacheModifier CM>
__device__ __forceinline__
WarpQueueBallot<T, CM>
::WarpQueueBallot(T*   __restrict__ queue_ptr,
                  int* __restrict__ size_ptr) : _queue_ptr(queue_ptr),
                                                _size_ptr(size_ptr) {}

template<typename T, CacheModifier CM>
__device__ __forceinline__
void WarpQueueBallot<T, CM>::store(T item, int predicate) {
    unsigned       ballot = __ballot(predicate);
    unsigned elected_lane = __msb(ballot);
    int warp_offset;
    if (lane_id() == elected_lane)
        warp_offset = atomicAdd(_size_ptr, __popc(ballot));
    int offset = __popc(ballot & LaneMaskLT()) +
                 __shfl(warp_offset, elected_lane);
    if (predicate)
        Store<CM>(_queue_ptr + offset) = item;
}
//------------------------------------------------------------------------------

template<typename T, int SIZE, unsigned ITEMS_PER_WARP, CacheModifier CM>
__device__ __forceinline__
WarpQueueSharedMem<T, SIZE, ITEMS_PER_WARP, CM>
::WarpQueueSharedMem(T (&queue)[SIZE],
                     T*   __restrict__ queue_ptr,
                     int* __restrict__ size_ptr,
                     T* shared_mem) :
                detail::WarpQueueBase<T, SIZE, CM>(queue, queue_ptr, size_ptr),
                _shared_mem(shared_mem) {
    _lane_shared_mem = shared_mem + lane_id();
}

template<typename T, int SIZE, unsigned ITEMS_PER_WARP, CacheModifier CM>
__device__ __forceinline__
WarpQueueSharedMem<T, SIZE, ITEMS_PER_WARP, CM>::~WarpQueueSharedMem() {
    store();
}

template<typename T, int SIZE, unsigned ITEMS_PER_WARP, CacheModifier CM>
__device__ __forceinline__
void WarpQueueSharedMem<T, SIZE, ITEMS_PER_WARP, CM>::store() {
    assert(__ballot(true) == static_cast<unsigned>(-1));
    if (__any(_size >= SIZE)) {
        int thread_offset = _size, total;
        int   warp_offset = WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                           _size_ptr, total);
        T* ptr = _queue_ptr + warp_offset + lane_id();

        int j = 0;
        int loop_limit = total / ITEMS_PER_WARP;
        for (int loop = 0; loop < loop_limit; loop++) {
            int pos = thread_offset;
            while (j < _size && pos < ITEMS_PER_WARP)
                 _shared_mem[pos++] = _queue[j++];

            #pragma unroll
            for (int i = 0; i < ITEMS_PER_WARP; i += WARP_SIZE)
                Store<CM>(ptr + i, _lane_shared_mem[i]);

            total         -= ITEMS_PER_WARP;
            thread_offset -= ITEMS_PER_WARP;
            ptr           += ITEMS_PER_WARP;
        }
        int pos = thread_offset;
        while (j < _size && pos < ITEMS_PER_WARP)
             _shared_mem[pos++] = _queue[j++];
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_WARP; i += WARP_SIZE) {
            if (lane_id() + i < total)
                Store<CM>(ptr + i, _lane_shared_mem[i]);
        }
        _size = 0;
    }
}

//==============================================================================
//==============================================================================

template<cuQUEUE_MODE mode, CacheModifier CM, int Items_per_warp>
template<typename T, typename R, int SIZE>
__device__ __forceinline__
void QueueWarp<mode, CM, Items_per_warp>
::store(T (&Queue)[SIZE],
        int size,
        T* __restrict__ queue_ptr,
        R* __restrict__ queue_size_ptr,
        T* __restrict__ SMem) {

    int thread_offset = size, total;
    int warp_offset = WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                     queue_size_ptr,
                                                     total);
    warp_dyn<mode, CM, Items_per_warp>
        ::regToGlobal(Queue, size, thread_offset, queue_ptr + warp_offset,
                      total, SMem + warp_id() * Items_per_warp);
}

template<cuQUEUE_MODE mode, CacheModifier CM, int Items_per_warp>
template<typename T, typename R, int SIZE>
__device__ __forceinline__
void QueueWarp<mode, CM, Items_per_warp>
::store(T (&Queue)[SIZE],
        int size,
        T* __restrict__ queue_ptr,
        R* __restrict__ queue_size_ptr) {

    static_assert(mode != cuQUEUE_MODE::SHAREDMEM &&
                  mode != cuQUEUE_MODE::SHAREDMEM_UNROLL,
                  "SMem == nullptr not allowed with shared memory");
    QueueWarp<mode, CM, Items_per_warp>
        ::store(Queue, size, queue_ptr, queue_size_ptr,
                static_cast<T*>(nullptr));
}
/*
template<cuQUEUE_MODE mode, CacheModifier CM, int Items_per_warp>
template<typename T, typename R, int SIZE>
__device__ __forceinline__
void QueueWarp<mode, CM, Items_per_warp>
::store2(T (&Queue)[SIZE],
        const int size,
        T* __restrict__ queue_ptr,
        R* __restrict__ queue_size_ptr,
        T* __restrict__ SMem,
        int& warp_offset,
        int& total) {

    int thread_offset = size;
    warp_offset = WarpExclusiveScan<>::AtomicAdd(thread_offset,
                                                     queue_size_ptr,
                                                     total);
    warp_dyn<mode, CM, Items_per_warp>
        ::regToGlobal(Queue, size, thread_offset, queue_ptr + warp_offset,
                      total, SMem + warp_id() * Items_per_warp);
}*/

//==============================================================================
//==============================================================================

template<CacheModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::SIMPLE, CM, Items_per_warp> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            int size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            const int total,          //optional
                            T* __restrict__ SMem) {   //optional

        devPointer += thread_offset;
        for (int i = 0; i < size; i++)
            Store<CM>(devPointer + i, Queue[i]);
    }
};


template<CacheModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::UNROLL, CM, Items_per_warp> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                     int size,
                     int thread_offset,
                     T* __restrict__ devPointer,
                     const int total,          //optional
                     T* __restrict__ SMem) {   //optional

        devPointer += thread_offset;
        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            if (i < size)
                Store<CM>(devPointer + i, Queue[i]);
        }
    }
};

template<CacheModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::SHAREDMEM, CM, Items_per_warp> {
    //static_assert(Items_per_warp != 0, "Items_per_warp == 0");

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            int size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            int total,
                            T* __restrict__ SMem) {

        T* SMemTMP = SMem;
        devPointer += lane_id();
        SMem += lane_id();
        int j = 0;
        while (true) {
             while (j < size && thread_offset < Items_per_warp) {
                 SMemTMP[thread_offset] = Queue[j];
                 j++;
                 thread_offset++;
             }
            if (total < Items_per_warp) {
                #pragma unroll
                for (int i = 0; i < Items_per_warp; i += WARP_SIZE) {
                    if (lane_id() + i < total)
                        Store<CM>(devPointer + i, SMem[i]);
                }
                break;
            }
            else {
                #pragma unroll
                for (int i = 0; i < Items_per_warp; i += WARP_SIZE)
                    Store<CM>(devPointer + i, SMem[i]);
            }
            total -= Items_per_warp;
            thread_offset -= Items_per_warp;
            devPointer += Items_per_warp;
        }
    }
};

template<CacheModifier CM, int Items_per_warp>
struct warp_dyn<cuQUEUE_MODE::Min, CM, Items_per_warp> {
    static_assert(Items_per_warp != 0, "Items_per_warp == 0");

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            int size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            const int total,          //optional
                            T* __restrict__ SMem) {   //optional

        int minValue = size;
        WarpReduce<>::minAll(minValue);

        T* devPointerTMP = devPointer + lane_id();
        for (int i = 0; i < minValue; i++)
            Store<CM>(devPointerTMP + i * WARP_SIZE, Queue[i]);

        size -= minValue;
        thread_offset -= lane_id() * minValue;
        total -= minValue * WARP_SIZE;
        devPointer += minValue * WARP_SIZE;

        RegToGlobal(Queue + minValue, size, thread_offset, total,
                    SMem, devPointer);
    }
};


template<CacheModifier CM, int Items_per_warp>
struct QueueWarp<cuQUEUE_MODE::BALLOT, CM, Items_per_warp> {
    template<typename T, typename R>
    __device__ __forceinline__
    static void store(T value,
                      bool predicate,
                      T* __restrict__ queue_ptr,
                      R* __restrict__ queue_size_ptr) {

        unsigned ballot = __ballot(predicate);
        unsigned electedLane = __msb(ballot);

        //if (ballot) {
            int warp_offset;
            if (lane_id() == electedLane)
                warp_offset = atomicAdd(queue_size_ptr, __popc(ballot));
            int th_offset = __popc(ballot & LaneMaskLT()) +
                            __shfl(warp_offset, electedLane);
            if (predicate)
                queue_ptr[th_offset] = value;
        //}
    }
};

} // namespace xlib
