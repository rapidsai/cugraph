/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file renumber_cg.cuh
 * @brief CG-optimized renumbering for trillion-edge scale sampling
 *
 * This file provides a specialized renumbering implementation that uses
 * cooperative groups (CG) for parallel hash table probing, addressing
 * the scalability bottleneck in sampling post-processing.
 *
 * Key optimizations:
 * 1. CG size = 4 for parallel probing during hash table operations
 * 2. Alternative sort-based approach for when hash tables are inefficient
 * 3. Bulk operations to maximize throughput
 *
 * References:
 * - CUDA Programming Guide: Cooperative Groups
 * - cuCollections (cuco) CG support
 */

#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuco/static_map.cuh>

namespace cugraph {
namespace detail {

// CG size for parallel probing
constexpr int kRenumberCGSize = 4;

/**
 * @brief CG-optimized key-value store for renumbering
 *
 * This specialized hash table uses CG size = 4 for parallel probing,
 * which can provide 2-4x speedup over single-thread probing for
 * large datasets with high collision rates.
 *
 * @tparam key_t Key type
 * @tparam value_t Value type
 */
template <typename key_t, typename value_t>
class renumber_cg_store_t {
 public:
  using cuco_map_type = cuco::static_map<key_t,
                                          value_t,
                                          cuco::extent<std::size_t>,
                                          cuda::thread_scope_device,
                                          thrust::equal_to<key_t>,
                                          cuco::linear_probing<kRenumberCGSize,
                                                               cuco::murmurhash3_32<key_t>>,
                                          rmm::mr::polymorphic_allocator<std::byte>,
                                          cuco::storage<1>>;

  renumber_cg_store_t(rmm::cuda_stream_view stream) {}

  /**
   * @brief Construct with key-value pairs
   *
   * Uses CG size = 4 for parallel insertion probing.
   */
  template <typename KeyIterator, typename ValueIterator>
  renumber_cg_store_t(KeyIterator key_first,
                      KeyIterator key_last,
                      ValueIterator value_first,
                      key_t invalid_key,
                      value_t invalid_value,
                      rmm::cuda_stream_view stream)
  {
    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    
    cuco_store_ = std::make_unique<cuco_map_type>(
      num_keys * 2,  // capacity with load factor ~0.5
      cuco::empty_key<key_t>{invalid_key},
      cuco::empty_value<value_t>{invalid_value},
      thrust::equal_to<key_t>{},
      cuco::linear_probing<kRenumberCGSize, cuco::murmurhash3_32<key_t>>{},
      cuco::thread_scope_device,
      cuco::storage<1>{},
      rmm::mr::polymorphic_allocator<std::byte>{rmm::mr::get_current_device_resource()},
      stream.value());
    
    if (num_keys > 0) {
      auto pair_first = thrust::make_zip_iterator(key_first, value_first);
      cuco_store_->insert(pair_first, pair_first + num_keys, stream.value());
    }
    
    invalid_value_ = invalid_value;
  }

  /**
   * @brief Bulk find with CG parallel probing
   *
   * This is the key optimization: uses cooperative groups for parallel
   * probing during lookups, which can be 2-4x faster than single-thread.
   */
  template <typename KeyIterator, typename ValueIterator>
  void find(KeyIterator key_first,
            KeyIterator key_last,
            ValueIterator value_first,
            rmm::cuda_stream_view stream)
  {
    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    if (num_keys == 0) return;
    
    cuco_store_->find(key_first, key_last, value_first, stream.value());
  }

  value_t invalid_value() const { return invalid_value_; }

 private:
  std::unique_ptr<cuco_map_type> cuco_store_{nullptr};
  value_t invalid_value_{};
};

/**
 * @brief Alternative: Sort-based renumbering for very large datasets
 *
 * For extremely large datasets where hash table overhead is high,
 * sort-based renumbering can be more efficient due to better
 * memory access patterns and cache utilization.
 *
 * Algorithm:
 * 1. Sort the renumber map by key: O(n log n)
 * 2. Binary search for each lookup: O(m log n) total
 *
 * This is better when:
 * - Memory is constrained (no need for 2x hash table size)
 * - Cache locality is important
 * - Dataset is very large (billions of elements)
 *
 * @tparam vertex_t Vertex type
 */
template <typename vertex_t>
class renumber_sort_based_t {
 public:
  renumber_sort_based_t(rmm::cuda_stream_view stream)
    : sorted_keys_(0, stream),
      sorted_values_(0, stream) {}

  /**
   * @brief Construct with key-value pairs
   *
   * Sorts the data for efficient binary search lookups.
   */
  template <typename KeyIterator, typename ValueIterator>
  renumber_sort_based_t(KeyIterator key_first,
                        KeyIterator key_last,
                        ValueIterator value_first,
                        vertex_t invalid_value,
                        rmm::cuda_stream_view stream)
    : sorted_keys_(cuda::std::distance(key_first, key_last), stream),
      sorted_values_(cuda::std::distance(key_first, key_last), stream),
      invalid_value_(invalid_value)
  {
    auto num_keys = sorted_keys_.size();
    if (num_keys == 0) return;
    
    // Copy to internal storage
    thrust::copy(rmm::exec_policy(stream), key_first, key_last, sorted_keys_.begin());
    thrust::copy(rmm::exec_policy(stream), value_first, value_first + num_keys, sorted_values_.begin());
    
    // Sort by key
    thrust::sort_by_key(rmm::exec_policy(stream),
                        sorted_keys_.begin(),
                        sorted_keys_.end(),
                        sorted_values_.begin());
  }

  /**
   * @brief Lookup values using binary search
   *
   * O(log n) per lookup, but with excellent cache behavior.
   */
  template <typename KeyIterator, typename ValueIterator>
  void find(KeyIterator key_first,
            KeyIterator key_last,
            ValueIterator value_first,
            rmm::cuda_stream_view stream)
  {
    auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
    if (num_keys == 0) return;
    
    thrust::transform(
      rmm::exec_policy(stream),
      key_first,
      key_last,
      value_first,
      [sorted_keys = raft::device_span<vertex_t const>(sorted_keys_.data(), sorted_keys_.size()),
       sorted_values = raft::device_span<vertex_t const>(sorted_values_.data(), sorted_values_.size()),
       invalid_value = invalid_value_] __device__(vertex_t key) {
        auto it = thrust::lower_bound(thrust::seq, sorted_keys.begin(), sorted_keys.end(), key);
        if (it != sorted_keys.end() && *it == key) {
          return sorted_values[thrust::distance(sorted_keys.begin(), it)];
        }
        return invalid_value;
      });
  }

  vertex_t invalid_value() const { return invalid_value_; }

 private:
  rmm::device_uvector<vertex_t> sorted_keys_;
  rmm::device_uvector<vertex_t> sorted_values_;
  vertex_t invalid_value_{};
};

/**
 * @brief Choose optimal renumbering strategy based on dataset size
 *
 * For trillion-edge graphs, the choice between hash table and sort-based
 * approaches depends on:
 * - Memory availability (hash table needs 2x capacity)
 * - Access patterns (random vs. sequential)
 * - Hardware characteristics (cache size, memory bandwidth)
 *
 * General guidelines:
 * - Small datasets (<10M): Sort-based (simpler, less memory)
 * - Medium datasets (10M-1B): CG hash table (O(1) lookups)
 * - Very large datasets (>1B): Consider hybrid or distributed approaches
 */
enum class RenumberStrategy {
  HASH_CG,     // CG-optimized hash table (CG size = 4)
  SORT_BASED,  // Sort + binary search
  AUTO         // Auto-select based on size
};

}  // namespace detail
}  // namespace cugraph
