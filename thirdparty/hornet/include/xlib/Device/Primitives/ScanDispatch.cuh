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
#pragma once

#include "Base/Device/Primitives/WarpScan.cuh"
#include "Base/Device/Primitives/BlockScan.cuh"

namespace xlib {

template<xlib::THREAD_GROUP GRP, unsigned BlockSize,
         SCAN_MODE mode = SCAN_MODE::ATOMIC>
struct ExclusiveScan;

template<unsigned BlockSize, SCAN_MODE mode>
struct ExclusiveScan<WARP, BlockSize, mode> {
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

template<unsigned BlockSize, SCAN_MODE mode>
struct ExclusiveScan<BLOCK, BlockSize, mode> {
    template<typename T>
    __device__ __forceinline__
    static void AddBcast(T& value, T& total, T* SMemLocal) {
        BlockExclusiveScan<BlockSize, mode>::AddBcast(value, total, SMemLocal);
    }

    template<typename T>
    __device__ __forceinline__
    static T AtomicAdd(T& value, T* ptr, T& total, T* SMemLocal) {
        return BlockExclusiveScan<BlockSize, mode>
                    ::AtomicAdd(value, ptr, total, SMemLocal);
    }
};

} // namespace xlib
