/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <utilities/error.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace cugraph {
namespace experimental {
namespace detail {

// FIXME: Better move this to RAFT

template <typename T>
struct is_atomically_addable {
  static constexpr bool value = std::is_floating_point<T>::value ||
                         (std::is_integral<T>::value && ((sizeof(T) == 4) || (sizeof(T) == 8)));
};

template <typename T>
__device__ std::enable_if_t<std::is_floating_point<T>::value, T> atomic_add(T* ptr, T val)
{
  return atomicAdd(ptr, val);
}

template <typename T>
__device__ std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4), T> atomic_add(T* ptr,
                                                                                         T val)
{
  static_assert(sizeof(unsigned int) == 4);
  return T{atomicAdd(reinterpret_cast<unsigned int*>(ptr), static_cast<unsigned int>(val))};
}

template <typename T>
__device__ std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 8), T> atomic_add(T* ptr,
                                                                                          T val)
{
  static_assert(sizeof(unsigned long long) == 8);
  return T{atomicAdd(reinterpret_cast<unsigned long long int*>(ptr),
                     static_cast<unsigned long long int>(val))};
}

/**
 * @brief Size of a warp in a CUDA kernel.
 */
static constexpr size_t warp_size{32};

/**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to threads.
 */
class grid_1d_thread_t {
 public:
  int const block_size{0};
  int const num_blocks{0};

  /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   * @param elements_per_thread Typically, a single kernel thread processes more than a single
   * element; this affects the number of threads the grid must contain
   */
  grid_1d_thread_t(size_t overall_num_elements,
                   size_t num_threads_per_block,
                   size_t max_num_blocks_1d,
                   size_t elements_per_thread = 1)
    : block_size(num_threads_per_block),
      num_blocks(
        std::min((overall_num_elements + (elements_per_thread * num_threads_per_block) - 1) /
                   (elements_per_thread * num_threads_per_block),
                 max_num_blocks_1d))
  {
    CUGRAPH_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
    CUGRAPH_EXPECTS(num_threads_per_block / warp_size > 0,
                    "num_threads_per_block / warp_size must be > 0");
    CUGRAPH_EXPECTS(elements_per_thread > 0, "elements_per_thread must be > 0");
  }
};

/**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to warps.
 */
class grid_1d_warp_t {
 public:
  int const block_size{0};
  int const num_blocks{0};

  /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   */
  grid_1d_warp_t(size_t overall_num_elements,
                 size_t num_threads_per_block,
                 size_t max_num_blocks_1d)
    : block_size(num_threads_per_block),
      num_blocks(std::min((overall_num_elements + (num_threads_per_block / warp_size) - 1) /
                            (num_threads_per_block / warp_size),
                          max_num_blocks_1d))
  {
    CUGRAPH_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
    CUGRAPH_EXPECTS(num_threads_per_block / warp_size > 0,
                    "num_threads_per_block / warp_size must be > 0");
  }
};

/**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to blocks.
 */
class grid_1d_block_t {
 public:
  int const block_size{0};
  int const num_blocks{0};

  /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   */
  grid_1d_block_t(size_t overall_num_elements,
                  size_t num_threads_per_block,
                  size_t max_num_blocks_1d)
    : block_size(num_threads_per_block),
      num_blocks(std::min(overall_num_elements, max_num_blocks_1d))
  {
    CUGRAPH_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
    CUGRAPH_EXPECTS(num_threads_per_block / warp_size > 0,
                    "num_threads_per_block / warp_size must be > 0");
  }
};

// FIXME: a temporary to test 1D pattern accelerator, should be added to RAFT::handle_t
constexpr size_t get_max_num_blocks_1D() { return static_cast<size_t>(65535); }

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph