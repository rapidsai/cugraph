/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
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

#include "Host/Numeric.hpp" //xlib::min

namespace xlib {

/*
 * @brief Compute capability-dependent device properties
 * @details Available properties at compile time:
 * - shared memory per Streaming Multiprocessor: @p SMEM_PER_SM
 * - maximum number of resident blocks per Streaming Multiprocessor:
 *   @p RBLOCKS_PER_SM
 * Supported architecture: *300, 320, 350, 370, 500, 520, 530, 600, 610, 620,
 *                          700*
 * @tparam CUDA_ARCH identifier of the GPU architecture (3 digits)
 */
template<int CUDA_ARCH>
struct DeviceProp {
    static_assert(CUDA_ARCH != CUDA_ARCH, "Unsupported Compute Cabalitity");
};

template<>
struct DeviceProp<300> {
    static const unsigned SMEM_PER_SM    = 49152;
    static const unsigned RBLOCKS_PER_SM = 16;
};

template<>
struct DeviceProp<320> {
    static const unsigned SMEM_PER_SM    = 49152;
    static const unsigned RBLOCKS_PER_SM = 16;
};

template<>
struct DeviceProp<350> {
    static const unsigned SMEM_PER_SM    = 49152;
    static const unsigned RBLOCKS_PER_SM = 16;
};

template<>
struct DeviceProp<370> {
    static const unsigned SMEM_PER_SM    = 114688;
    static const unsigned RBLOCKS_PER_SM = 16;
};

template<>
struct DeviceProp<500> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<520> {
    static const unsigned SMEM_PER_SM    = 98304;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<530> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<600> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<610> {
    static const unsigned SMEM_PER_SM    = 98304;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<620> {
    static const unsigned SMEM_PER_SM    = 65536;
    static const unsigned RBLOCKS_PER_SM = 32;
};

template<>
struct DeviceProp<700> {
    static const unsigned SMEM_PER_SM    = 98304;
    static const unsigned RBLOCKS_PER_SM = 32;
};

//==============================================================================

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
constexpr unsigned blocks_per_SM() {
    return THREADS_PER_SM / BLOCK_SIZE;
}

template<typename T, unsigned BLOCK_SIZE, int K>
__device__ __forceinline__
constexpr unsigned smem_per_thread() {
#if defined(__CUDA_ARCH__)
    unsigned SMEM_PER_SM    = DeviceProp<__CUDA_ARCH__>::SMEM_PER_SM;
    unsigned RBLOCKS_PER_SM = DeviceProp<__CUDA_ARCH__>::RBLOCKS_PER_SM;

    unsigned _BLOCK_SIZE    = (BLOCK_SIZE == 0) ?
                              THREADS_PER_SM / RBLOCKS_PER_SM : BLOCK_SIZE;
    unsigned NUM_BLOCKS     = THREADS_PER_SM / _BLOCK_SIZE;
    unsigned ACTUAL_BLOCKS  = RBLOCKS_PER_SM < NUM_BLOCKS ?
                                  RBLOCKS_PER_SM : NUM_BLOCKS;
    unsigned SMEM_PER_BLOCK = SMEM_PER_SM / ACTUAL_BLOCKS < MAX_BLOCK_SMEM ?
                                  SMEM_PER_SM / ACTUAL_BLOCKS : MAX_BLOCK_SMEM;
    return (SMEM_PER_BLOCK / _BLOCK_SIZE) / sizeof(T);
#else
    static_assert(BLOCK_SIZE != BLOCK_SIZE, "not defined");
    return 0;
#endif
};

template<unsigned BLOCK_SIZE, int K>
__device__ __forceinline__
constexpr unsigned smem_per_thread() {
    return smem_per_thread<int8_t, BLOCK_SIZE>();
}

template<typename T, unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_thread_occ() {
#if defined(__CUDA_ARCH__)
    unsigned SMEM_PER_SM    = DeviceProp<__CUDA_ARCH__>::SMEM_PER_SM;
    unsigned RBLOCKS_PER_SM = DeviceProp<__CUDA_ARCH__>::RBLOCKS_PER_SM;

    unsigned BLOCK_SIZE     = THREADS_PER_SM / NUM_BLOCKS;
    unsigned ACTUAL_BLOCKS  = xlib::min(RBLOCKS_PER_SM, NUM_BLOCKS);
    unsigned SMEM_PER_BLOCK = xlib::min(SMEM_PER_SM / ACTUAL_BLOCKS,
                                        MAX_BLOCK_SMEM);
    return (SMEM_PER_BLOCK / BLOCK_SIZE) / sizeof(T);
#else
    static_assert(NUM_BLOCKS != NUM_BLOCKS, "not defined");
    return 0;
#endif
};

template<unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_thread_occ() {
    return smem_per_thread_occ<int8_t, NUM_BLOCKS>();
}

//------------------------------------------------------------------------------

template<typename T, unsigned BLOCK_SIZE, int K>
__device__ __forceinline__
constexpr unsigned smem_per_warp() {
    return xlib::smem_per_thread<T, BLOCK_SIZE>() * xlib::WARP_SIZE;
};

template<unsigned BLOCK_SIZE, int K>
__device__ __forceinline__
constexpr unsigned smem_per_warp() {
    return xlib::smem_per_thread<BLOCK_SIZE>() * xlib::WARP_SIZE;
};

template<typename T, unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_warp_occ() {
    return xlib::smem_per_thread_occ<T, NUM_BLOCKS>() * xlib::WARP_SIZE;
};

template<unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_warp_occ() {
    return xlib::smem_per_thread_occ<NUM_BLOCKS>() * xlib::WARP_SIZE;
};

//------------------------------------------------------------------------------

template<typename T, unsigned BLOCK_SIZE, int K>
__device__ __forceinline__
constexpr unsigned smem_per_block() {
    return xlib::smem_per_thread<T, BLOCK_SIZE>() * BLOCK_SIZE;
};

template<unsigned BLOCK_SIZE, int K>
__device__ __forceinline__
constexpr unsigned smem_per_block() {
    return xlib::smem_per_thread<BLOCK_SIZE>() * BLOCK_SIZE;
};

template<typename T, unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_block_occ() {
    unsigned BLOCK_SIZE = THREADS_PER_SM / NUM_BLOCKS;
    return xlib::smem_per_thread_occ<T, NUM_BLOCKS>() * BLOCK_SIZE;
};

template<unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_block_occ() {
    unsigned BLOCK_SIZE = THREADS_PER_SM / NUM_BLOCKS;
    return xlib::smem_per_thread_occ<NUM_BLOCKS>() * BLOCK_SIZE;
};


//==============================================================================
//==============================================================================

template<typename T>
int DeviceProperty::smem_per_thread(int block_size) noexcept {
    return DeviceProperty::smem_per_block(block_size) / block_size;
}

template<typename T>
int DeviceProperty::smem_per_warp(int block_size) noexcept {
    return DeviceProperty::smem_per_block(block_size) /
           (block_size / xlib::WARP_SIZE);
}

template<typename T>
int DeviceProperty::smem_per_block(int block_size) noexcept {
    assert(block_size >= 0 && block_size < MAX_BLOCK_SIZE &&
           "BLOCK_SIZE range");

    int max_blocks     = DeviceProperty::resident_blocks_per_SM();
    int SM_smem        = DeviceProperty::smem_per_SM();
    int num_blocks     = xlib::THREADS_PER_SM / block_size;
    int actual_blocks  = std::min(max_blocks, num_blocks);
    int smem_per_block = std::min(SM_smem / actual_blocks,
                                   static_cast<int>(MAX_BLOCK_SMEM));
    return smem_per_block / sizeof(T);
}

//------------------------------------------------------------------------------

template<typename T>
int DeviceProperty::smem_per_thread_occ(int num_blocks_per_SM,
                                        int block_size) noexcept {
    return DeviceProperty::smem_per_block_occ<T>(num_blocks_per_SM) /
           block_size;
}

template<typename T>
int DeviceProperty::smem_per_warp_occ(int num_blocks_per_SM,
                                      int block_size) noexcept {
    return DeviceProperty::smem_per_block_occ<T>(num_blocks_per_SM) /
           (block_size / xlib::WARP_SIZE);
}

template<typename T>
int DeviceProperty::smem_per_block_occ(int num_blocks_per_SM) noexcept {
    int max_blocks = DeviceProperty::resident_blocks_per_SM();
    int SM_smem    = DeviceProperty::smem_per_SM();

    assert(num_blocks_per_SM >= 0 && num_blocks_per_SM < max_blocks &&
           "num_blocks_per_SM range");

    int actual_blocks  = std::min(max_blocks, num_blocks_per_SM);
    int smem_per_block = std::min(SM_smem / actual_blocks,
                                  static_cast<int>(MAX_BLOCK_SMEM));
    return smem_per_block / sizeof(T);
}

//==============================================================================
//==============================================================================

template<typename Kernel>
int kernel_occupancy(const Kernel& kernel, int block_size, int dyn_smem) {
    int num_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel,
                                                  block_size, dyn_smem);
    return num_blocks;
}

} // namespace xlib
