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
namespace xlib {
namespace detail {

template<int BLOCKSIZE, BLOCK_MODE mode,
         cub::CacheStoreModifier CM, int Items_per_warp>
struct block_dyn_common {

    template<typename T, typename R, int SIZE>
    __device__ __forceinline__
    static void queueStore(T (&Queue)[SIZE],
                           const int size,
                           T* __restrict__ queue_ptr,
                           R* __restrict__ queue_size_ptr) {

        int thread_offset = size;
        const int warp_offset = BlockExclusiveScan<BLOCKSIZE>
                                           ::Add(thread_offset, queue_size_ptr);
        block_dyn<BLOCKSIZE, mode, CM, Items_per_warp>
            ::regToGlobal(Queue, size, queue_ptr + warp_offset + thread_offset);
    }
};

} // namespace detail

//------------------------------------------------------------------------------

template<int BLOCKSIZE, cub::CacheStoreModifier CM, int Items_per_block>
struct block_dyn<BLOCKSIZE, BLOCK_MODE::SIMPLE, CM, Items_per_block> :
          block_dyn_common<BLOCKSIZE, BLOCK_MODE::SIMPLE, CM, Items_per_block> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            int thread_offset,
                            const int total,
                            T* __restrict__ shared_mem,
                            T* __restrict__ devPointer) {

        devPointer += thread_offset;
        for (int i = 0; i < size; i++)
            cub::ThreadStore<CM>(devPointer + i, Queue[i]);
    }
};

template<int BLOCKSIZE, cub::CacheStoreModifier CM, int Items_per_block>
struct block_dyn<BLOCKSIZE, BLOCK_MODE::UNROLL, CM, Items_per_block> :
          block_dyn_common<BLOCKSIZE, BLOCK_MODE::UNROLL, CM, Items_per_block> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            int thread_offset,
                            const int total,
                            T* __restrict__ shared_mem,
                            T* __restrict__ devPointer) {

        devPointer += thread_offset;
        #pragma unroll
        for (int i = 0; i < SIZE; i++) {
            if (i < size)
                cub::ThreadStore<CM>(devPointer + i, Queue[i]);
        }
    }
};

template<int BLOCKSIZE, cub::CacheStoreModifier CM, int Items_per_block>
struct block_dyn<BLOCKSIZE, BLOCK_MODE::SHAREDMEM, CM, Items_per_block> :
       block_dyn_common<BLOCKSIZE, BLOCK_MODE::SHAREDMEM, CM, Items_per_block> {

    template<typename T, int SIZE>
    __device__ __forceinline__
    static void regToGlobal(T (&Queue)[SIZE],
                            const int size,
                            int thread_offset,
                            const int total,
                            T* __restrict__ shared_mem,
                            T* __restrict__ devPointer) {
        int j = 0;
         while (true) {
             while (j < size && thread_offset < Items_per_block) {
                 shared_mem[thread_offset] = Queue[j];
                 j++;
                thread_offset++;
             }
            __syncthreads();
             #pragma unroll
             for (int i = 0; i < Items_per_block; i += BLOCKSIZE) {
                 const int index = threadIdx.x + i;
                 if (index < total)
                     cub::ThreadStore<CM>(devPointer + index, shared_mem[index]);
             }
            total -= Items_per_block;
            if (total <= 0)
                 break;
             thread_offset -= Items_per_block;
             devPointer += Items_per_block;
            __syncthreads();
         }
    }
};

} // namespace xlib
