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

#include <cstdint>  //int8_t

namespace xlib {

///@brief Total number of threads for Streaming Multiprocessor (SM)
const unsigned THREADS_PER_SM = 2048;

///@brief Maximum number of threads per block
const unsigned MAX_BLOCK_SIZE = 1024;

///@brief Maximum allocable shared memory per block (bytes)
const unsigned MAX_BLOCK_SMEM = 49152;

///@brief Total GPU constant memory (bytes)
const unsigned CONSTANT_MEM   = 65536;

///@brief Number of shared memory banks
const unsigned MEMORY_BANKS   = 32;

///@brief Number of threads in a warp
const unsigned WARP_SIZE      = 32;

///@brief Maximum number of Streaming Multiprocessors (TitanV, CC 7.0)
const unsigned MAX_SM         = 80;

///@brief Maximum number of Resident Thread Blocks
const unsigned GPU_MAX_BLOCKS = (xlib::THREADS_PER_SM / xlib::WARP_SIZE) *
                                 xlib::MAX_SM;

//==============================================================================
#if defined(__NVCC__)

/**
 * @brief Number of thread blocks of a given size in a Streaming Multiprocessor
 * @tparam BLOCK_SIZE number of threads in a block
 * @return Thread blocks per SM
 */
template<unsigned BLOCK_SIZE>
__device__ __forceinline__
constexpr unsigned blocks_per_SM();

/**
 * @brief Available shared memory per thread by considering the maximum
 *        occupancy for a give block size
 * @tparam T type to allocate in the shared memory
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and T, BLOCK_SIZE are statically known
 * @return Shared memory available for thread (number of items of size @p T)
 */
template<typename T = int8_t, unsigned BLOCK_SIZE = 0, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_thread();

/**
 * @brief Available shared memory per thread by considering the maximum
 *        occupancy for a give block size
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and BLOCK_SIZE is statically known
 * @return Shared memory available for thread (number of bytes)
 */
template<unsigned BLOCK_SIZE, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_thread();

/**
 * @brief Available shared memory per thread by considering the occupancy
 *        provided in number of thread blocks
 * @tparam T type to allocate in the shared memory
 * @tparam NUM_BLOCKS number of threads blocks
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and T, NUM_BLOCKS are statically known
 * @return Shared memory available for thread (number of items of size @p T)
 */
template<typename T, unsigned NUM_BLOCKS, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_thread_occ();

/**
 * @brief Available shared memory per thread by considering the occupancy
 *        provided in number of thread blocks
 * @tparam NUM_BLOCKS number of threads blocks
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and BLOCK_SIZE is statically known
 * @return Shared memory available for thread (number of bytes)
 */
template<unsigned NUM_BLOCKS, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_thread_occ();

//------------------------------------------------------------------------------

/**
 * @brief Available shared memory per warp by considering the maximum
 *        occupancy for a give block size
 * @tparam T type to allocate in the shared memory
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and T, BLOCK_SIZE are statically known
 * @return Shared memory available for warp (number of items of size @p T)
 */
template<typename T = int8_t, unsigned BLOCK_SIZE = 0, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_warp();

/**
 * @brief Available shared memory per warp by considering the maximum
 *        occupancy for a give block size
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and BLOCK_SIZE is statically known
 * @return Shared memory available for warp (number of bytes)
 */
template<unsigned BLOCK_SIZE, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_warp();

/**
 * @brief Available shared memory per warp by considering the occupancy
 *        provided in number of thread blocks
 * @tparam T type to allocate in the shared memory
 * @tparam NUM_BLOCKS number of threads blocks
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and T, BLOCK_SIZE are statically known
 * @return Shared memory available for warp (number of items of size @p T)
 */
template<typename T, unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_warp_occ();

/**
 * @brief Available shared memory per warp by considering the occupancy
 *        provided in number of thread blocks
 * @tparam NUM_BLOCKS number of threads blocks
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and BLOCK_SIZE is statically known
 * @return Shared memory available for warp (number of bytes)
 */
template<unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_warp_occ();

//------------------------------------------------------------------------------

/**
 * @brief Available shared memory per block by considering the maximum
 *        occupancy for a give blocks size
 * @tparam T type to allocate in the shared memory
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and BLOCK_SIZE or T are statically known
 * @return Shared memory available for block (number of items of size @p T)
 */
template<typename T, unsigned BLOCK_SIZE, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_block();

/**
 * @brief Available shared memory per block by considering the maximum
 *        occupancy for a give blocks size
 * @tparam BLOCK_SIZE number of threads in a block. If zero, the method
 *                    considers the maximum block occupancy
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and BLOCK_SIZE is statically known
 * @return Shared memory available for block (number of bytes)
 */
template<unsigned BLOCK_SIZE, int K = 0>
__device__ __forceinline__
constexpr unsigned smem_per_block();

/**
 * @brief Available shared memory per block by considering the occupancy
 *        provided in number of thread blocks
 * @tparam T type to allocate in the shared memory
 * @tparam NUM_BLOCKS number of threads blocks
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and T, BLOCK_SIZE are statically known
 * @return Shared memory available for block (number of items of size @p T)
 */
template<typename T, unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_block_occ();

/**
 * @brief Available shared memory per block by considering the occupancy
 *        provided in number of thread blocks
 * @tparam NUM_BLOCKS number of threads blocks
 * @tparam K any value not known at compile time.
 *         It must be speficied if the kernel which use this method
 *         is not compiled and BLOCK_SIZE is statically known
 * @return Shared memory available for warp (number of bytes)
 */
template<unsigned NUM_BLOCKS, int K>
__device__ __forceinline__
constexpr unsigned smem_per_block_occ();

#endif
//==============================================================================
//==============================================================================

/**
 * @brief Run-time device properties
 * @details The class provides basic static methods to retrieve CUDA device
 *          properties for all GPU devices available. The first method invoked
 *          retrieves the information for all devices once and for all.
 *          Successive calls to any methods get the values previous retrieved.
 *          The information are related to the current GPU device
 *          (to change with `cudaSetDevice`)
 */
class DeviceProperty {
public:
    /**
     * @brief Return the total number if Streaming Multiprocessors for the
     *        current GPU device
     * @return Total number of SMs
     */
    static int num_SM() noexcept;

    /**
     * @brief Return the total shared memory for Streaming Multiprocessor for
     *        the current GPU device
     * @return Shared memory for SM
     */
    static int smem_per_SM() noexcept;

    /**
     * @brief Return the maximum number of resident blocks for Streaming
     *        Multiprocessors for the current GPU device
     * @return Resident blocks per SM
     */
    static int resident_blocks_per_SM() noexcept;

    /**
     * @brief Return the total of threads by considering all Streaming
     *        Multiprocessors for the current GPU device
     * @return Total number of GPU threads
     */
    static int resident_threads() noexcept;

    /**
     * @brief Return the total of warps by considering all Streaming
     *        Multiprocessors for the current GPU device
     * @return Total number of GPU warps
     */
    static int resident_warps() noexcept;

    /**
     * @brief Return the total of blocks of a given size by considering all
     *        Streaming Multiprocessors for the current GPU device
     * @param[in] block_size Number of threads in a block
     * @return Total number of GPU blocks
     */
    static int resident_blocks(int block_size) noexcept;

    /**
     * @brief Return the available shared memory for thread by considering
     *        data size and block size
     * @tparam T data type
     * @param block_size Number of threads in a block
     * @return Shared memory available for thread
     */
    template<typename T = int8_t>
    static int smem_per_thread(int block_size) noexcept;

    /**
     * @brief Return the available shared memory for warp by considering
     *        data size and block size
     * @tparam T data type
     * @param block_size Number of threads in a block
     * @return Shared memory available for warp
     */
    template<typename T = int8_t>
    static int smem_per_warp(int block_size) noexcept;

    /**
     * @brief Return the available shared memory for block by considering
     *        data size and block size
     * @tparam T data type
     * @param block_size Number of threads in a block
     * @return Shared memory available for block
     */
    template<typename T = int8_t>
    static int smem_per_block(int block_size) noexcept;

    /**
     * @brief Return the available shared memory for thread by considering
     *        data size and block occupancy (in terms of number of blocks
     *        for Streaming Multiprocessor)
     * @tparam T Data type
     * @param[in] num_blocks_per_SM Number of blocks per SM
     * @return Shared memory available for thread
     */
    template<typename T = int8_t>
    static int smem_per_thread_occ(int num_blocks_per_SM, int block_size)
                                   noexcept;

    /**
     * @brief Return the available shared memory for warp by considering
     *        data size and block occupancy (in terms of number of blocks
     *        for Streaming Multiprocessor)
     * @tparam T Data type
     * @param[in] num_blocks_per_SM Number of blocks per SM
     * @return Shared memory available for warp
     */
    template<typename T = int8_t>
    static int smem_per_warp_occ(int num_blocks_per_SM, int block_size)
                                 noexcept;

    /**
     * @brief Return the available shared memory for block by considering
     *        data size and block occupancy (in terms of number of blocks
     *        for Streaming Multiprocessor)
     * @tparam T Data type
     * @param[in] num_blocks_per_SM Number of blocks per SM
     * @return Shared memory available for block
     */
    template<typename T = int8_t>
    static int smem_per_block_occ(int num_blocks_per_SM) noexcept;

private:
    static constexpr int MAX_GPUS = 8;

    static int  _num_sm[MAX_GPUS];
    static int  _smem_per_SM[MAX_GPUS];
    static int  _rblock_per_SM[MAX_GPUS];
    static int  _num_gpus;
    static bool _init_flag;

    static void _init() noexcept;
};

//------------------------------------------------------------------------------

/**
 * @brief Return the number of blocks for Streaming Multiprocessor for a given
 *        kernel function (`__global__`) by considering by default no shared
 *        memory
 * @param[in] kernel Kernel function
 * @param block_size Number of threads in a block
 * @param[in] dyn_smem Dynamic shared memory allocated for @p kernel (optional)
 * @return Number of blocks per SM (occupancy)
 */
template<typename Kernel>
int kernel_occupancy(const Kernel& kernel, int block_size, int dyn_smem = 0);

} // namespace xlib

#if defined(__NVCC__)
    #include "impl/DeviceProperties.i.cuh"
#endif
