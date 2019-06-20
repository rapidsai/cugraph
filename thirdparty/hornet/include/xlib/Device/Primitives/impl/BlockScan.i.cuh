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
#include "Base/Device/Primitives/WarpScan.cuh"
#include "Base/Device/Util/Basic.cuh"
#include "Base/Device/Util/Definition.cuh"

namespace xlib {

template<>
struct BlockExclusiveScan<32, SCAN_MODE::REDUCE> {
    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value, T& total) {
        const unsigned ballot = __ballot(predicate);
        value = __popc(ballot & LaneMaskLT());
        total = __popc(ballot);
    }

    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T*) {
        WarpExclusiveScan<>::AddBcast(value, total);
    }

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T*) {
        return WarpExclusiveScan<>::AtomicAdd(value, ptr, total);
    }
};

template<>
struct BlockExclusiveScan<32, SCAN_MODE::ATOMIC> :
    BlockExclusiveScan<32, SCAN_MODE::REDUCE> {};

template<unsigned BLOCK_SIZE>
struct BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::ATOMIC> {
    // smem_total is not SAFE!!!! require synchronization
    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value, T* smem_total) {
        if (threadIdx.x == 0)
            *smem_total = 0;
        __syncthreads();

        const unsigned ballot = __ballot(predicate);
        int warp_offset;
        if (lane_id() == 0)
            warp_offset = atomicAdd(smem_total, __popc(ballot));
        value = __popc(ballot & LaneMaskLT()) + __shfl(warp_offset, 0);
        // smem_total is not SAFE!!!! require synchronization
    }

    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T* shared_mem) {
        if (threadIdx.x < WARP_SIZE)
            *shared_mem = 0;
        __syncthreads();

        const T warp_offset = WarpExclusiveScan<>::AtomicAdd(value, shared_mem);
        value += warp_offset;
        __syncthreads();
        total = *shared_mem;
    }

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T* shared_mem) {
        BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::ATOMIC>
            ::AddBcast(value, total, shared_mem);

        if (threadIdx.x == 0)
            shared_mem[1] = atomicAdd(ptr, total);
        __syncthreads();
        return shared_mem[1];
    }
};

template<unsigned BLOCK_SIZE>
struct BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::REDUCE> {
    static_assert(BLOCK_SIZE != 0, "Missing template paramenter");

    template<typename T>
    __device__ __forceinline__
    static void BinaryAdd(int predicate, int& value,
                          T* smem_total, T* shared_mem) {

        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        const unsigned     ballot = __ballot(predicate);
        const unsigned   _warp_id = warp_id();

        value = __popc(ballot & LaneMaskLT());
        shared_mem[_warp_id] = __popc(ballot);
        __syncthreads();

        if (threadIdx.x < N_OF_WARPS) {
            T tmp = shared_mem[threadIdx.x];
            WarpExclusiveScan<N_OF_WARPS>::Add(tmp, smem_total);
            shared_mem[threadIdx.x] = tmp;
        }
        __syncthreads();
        value += shared_mem[_warp_id];
    }

    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T* shared_mem) {
        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        const unsigned   _warp_id = warp_id();

        WarpExclusiveScan<>::Add(value, shared_mem + _warp_id);
        __syncthreads();
        if (threadIdx.x < N_OF_WARPS) {
            T tmp = shared_mem[threadIdx.x];
            WarpExclusiveScan<N_OF_WARPS>::Add(tmp, shared_mem + N_OF_WARPS);
            shared_mem[threadIdx.x] = tmp;
        }
        __syncthreads();
        value += shared_mem[_warp_id];
        total  = shared_mem[N_OF_WARPS];
    }

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T* shared_mem) {
        const unsigned N_OF_WARPS = BLOCK_SIZE / WARP_SIZE;
        BlockExclusiveScan<BLOCK_SIZE, SCAN_MODE::REDUCE>
            ::AddBcast(value, total, shared_mem);

        if (threadIdx.x == 0)
            shared_mem[N_OF_WARPS + 1] = atomicAdd(ptr, total);
        __syncthreads();
        return shared_mem[N_OF_WARPS + 1];
    }
};

} // namespace xlib
