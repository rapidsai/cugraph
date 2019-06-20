/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date ?, 2017
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

#include "Base/Device/Util/CacheModifier.cuh"

namespace xlib {

enum class cuQUEUE_MODE { SIMPLE, UNROLL, Min, SHAREDMEM,
                          SHAREDMEM_UNROLL, BALLOT};

template<cuQUEUE_MODE mode = cuQUEUE_MODE::SIMPLE,
         CacheModifier CM = DF, int Items_per_warp = 0>
struct warp_dyn {
    template<typename T, int SIZE>
     __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            int& size,
                            int thread_offset,
                            T* __restrict__ devPointer,
                            const int total,          //optional
                            T* __restrict__ SMem);    //optional
};

template<cuQUEUE_MODE mode = cuQUEUE_MODE::SIMPLE,
         CacheModifier CM = DF, int Items_per_warp = 0>
struct QueueWarp {
     template<typename T, typename R, int SIZE>
     __device__ __forceinline__
    static void store(T (&Queue)[SIZE],
                      int size,
                      T* __restrict__ queue_ptr,
                      R* __restrict__ queue_size_ptr,
                      T* __restrict__ SMem);

    template<typename T, typename R, int SIZE>
    __device__ __forceinline__
    static void store(T (&Queue)[SIZE],
                      int size,
                      T* __restrict__ queue_ptr,
                      R* __restrict__ queue_size_ptr);

    /*template<typename T, typename R, int SIZE>
    __device__ __forceinline__
    static void store2(T (&Queue)[SIZE],
                       const int size,
                       T* __restrict__ queue_ptr,
                       R* __restrict__ queue_size_ptr,
                       T* __restrict__ SMem,
                       int& warp_offset,
                       int& total);*/
};

//==============================================================================
namespace detail {

template<typename T, int SIZE, CacheModifier CM = DF>
class WarpQueueBase {
public:
    __device__ __forceinline__
    WarpQueueBase(T (&queue)[SIZE],
                  T*   __restrict__ queue_ptr,
                  int* __restrict__ size_ptr);
protected:
    T (&_queue)[SIZE];
    T*   _queue_ptr;
    int* _size_ptr;
    int  _size;
};

} // namespace detail

template<typename T, int SIZE, CacheModifier CM = DF>
class WarpQueueSimple : public detail::WarpQueueBase<T, SIZE, CM> {
public:
    __device__ __forceinline__
    WarpQueueSimple(T (&queue)[SIZE],
                   T*   __restrict__ queue_ptr,
                   int* __restrict__ size_ptr);

    __device__ __forceinline__
    ~WarpQueueSimple();

    __device__ __forceinline__
    void insert(T item);

    __device__ __forceinline__
    void store();
private:
    using detail::WarpQueueBase<T, SIZE, CM>::_queue;
    using detail::WarpQueueBase<T, SIZE, CM>::_queue_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size;
    __device__ __forceinline__
    void _store();
};

template<typename T, int SIZE, CacheModifier CM = DF>
class WarpQueueUnroll : public detail::WarpQueueBase<T, SIZE, CM> {
public:
    __device__ __forceinline__
    WarpQueueUnroll(T (&queue)[SIZE],
                    T*   __restrict__ queue_ptr,
                    int* __restrict__ size_ptr);

    __device__ __forceinline__
    ~WarpQueueUnroll();

    __device__ __forceinline__
    void insert(T item);

    __device__ __forceinline__
    void store();
private:
    using detail::WarpQueueBase<T, SIZE, CM>::_queue;
    using detail::WarpQueueBase<T, SIZE, CM>::_queue_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size;
    __device__ __forceinline__
    void _store();
};


template<typename T, CacheModifier CM = DF>
class WarpQueueBallot {
public:
    __device__ __forceinline__
    WarpQueueBallot(T*   __restrict__ queue_ptr,
                    int* __restrict__ size_ptr);

    __device__ __forceinline__
    void store(T item, int predicate);
private:
    T*   _queue_ptr;
    int* _size_ptr;
};

template<typename T, int SIZE, unsigned ITEMS_PER_WARP, CacheModifier CM = DF>
class WarpQueueSharedMem : public detail::WarpQueueBase<T, SIZE, CM> {
public:
    __device__ __forceinline__
    WarpQueueSharedMem(T (&queue)[SIZE],
                       T*   __restrict__ queue_ptr,
                       int* __restrict__ size_ptr,
                       T* shared_mem);

    __device__ __forceinline__
    ~WarpQueueSharedMem();

    __device__ __forceinline__
    void insert(T item);

    __device__ __forceinline__
    void store();
private:
    using detail::WarpQueueBase<T, SIZE, CM>::_queue;
    using detail::WarpQueueBase<T, SIZE, CM>::_queue_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size_ptr;
    using detail::WarpQueueBase<T, SIZE, CM>::_size;
    T* _shared_mem;
    T* _lane_shared_mem;
};

} // namespace xlib

#include "impl/WarpDynamic.i.cuh"
