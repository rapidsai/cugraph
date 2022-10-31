/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cinttypes>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {

// key type is either vertex_t (tag_t == void) or thrust::tuple<vertex_t, tag_t> (tag_t != void)
// if sorted_unique is true, stores unique key objects in the sorted (non-descending) order.
// if false, there can be duplicates and the elements may not be sorted.
template <typename vertex_t,
          typename tag_t     = void,
          bool multi_gpu     = false,
          bool sorted_unique = false>
class key_bucket_t {
 public:
  using key_type =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  static bool constexpr is_sorted_unique = sorted_unique;

  static_assert(std::is_same_v<tag_t, void> || std::is_arithmetic_v<tag_t>);

  using optional_buffer_type = std::
    conditional_t<std::is_same_v<tag_t, void>, std::byte /* dummy */, rmm::device_uvector<tag_t>>;

  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle)
    : handle_ptr_(&handle), vertices_(0, handle.get_stream()), tags_(std::byte{0})
  {
  }

  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  key_bucket_t(raft::handle_t const& handle)
    : handle_ptr_(&handle), vertices_(0, handle.get_stream()), tags_(0, handle.get_stream())
  {
  }

  /**
   * @ brief insert a vertex to the bucket
   *
   * @param vertex vertex to insert
   */
  template <typename tag_type = tag_t, std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  void insert(vertex_t vertex)
  {
    if (vertices_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp(vertex, handle_ptr_->get_stream());
      insert(tmp.data(), tmp.data() + 1);
    } else {
      vertices_.resize(1, handle_ptr_->get_stream());
      raft::update_device(vertices_.data(), &vertex, size_t{1}, handle_ptr_->get_stream());
    }
  }

  /**
   * @ brief insert a (vertex, tag) pair to the bucket
   *
   * @param vertex vertex of the (vertex, tag) pair to insert
   * @param tag tag of the (vertex, tag) pair to insert
   */
  template <typename tag_type = tag_t, std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  void insert(thrust::tuple<vertex_t, tag_type> key)
  {
    if (vertices_.size() > 0) {
      rmm::device_scalar<vertex_t> tmp_vertex(thrust::get<0>(key), handle_ptr_->get_stream());
      rmm::device_scalar<tag_t> tmp_tag(thrust::get<1>(key), handle_ptr_->get_stream());
      auto pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(tmp_vertex.data(), tmp_tag.data()));
      insert(pair_first, pair_first + 1);
    } else {
      vertices_.resize(1, handle_ptr_->get_stream());
      tags_.resize(1, handle_ptr_->get_stream());
      auto pair_first =
        thrust::make_tuple(thrust::make_zip_iterator(vertices_.begin(), tags_.begin()));
      thrust::fill(handle_ptr_->get_thrust_policy(), pair_first, pair_first + 1, key);
    }
  }

  /**
   * @ brief insert a list of vertices to the bucket
   *
   * @param vertex_first Iterator pointing to the first (inclusive) element of the vertices stored
   * in device memory.
   * @param vertex_last Iterator pointing to the last (exclusive) element of the vertices stored in
   * device memory.
   */
  template <typename VertexIterator,
            typename tag_type                                 = tag_t,
            std::enable_if_t<std::is_same_v<tag_type, void>>* = nullptr>
  void insert(VertexIterator vertex_first, VertexIterator vertex_last)
  {
    static_assert(
      std::is_same_v<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>);

    if (vertices_.size() > 0) {
      if constexpr (sorted_unique) {
        rmm::device_uvector<vertex_t> merged_vertices(
          vertices_.size() + thrust::distance(vertex_first, vertex_last),
          handle_ptr_->get_stream());
        thrust::merge(handle_ptr_->get_thrust_policy(),
                      vertices_.begin(),
                      vertices_.end(),
                      vertex_first,
                      vertex_last,
                      merged_vertices.begin());
        merged_vertices.resize(thrust::distance(merged_vertices.begin(),
                                                thrust::unique(handle_ptr_->get_thrust_policy(),
                                                               merged_vertices.begin(),
                                                               merged_vertices.end())),
                               handle_ptr_->get_stream());
        merged_vertices.shrink_to_fit(handle_ptr_->get_stream());
        vertices_ = std::move(merged_vertices);
      } else {
        auto cur_size = vertices_.size();
        vertices_.resize(cur_size + thrust::distance(vertex_first, vertex_last),
                         handle_ptr_->get_stream());
        thrust::copy(handle_ptr_->get_thrust_policy(),
                     vertex_first,
                     vertex_last,
                     vertices_.begin() + cur_size);
      }
    } else {
      vertices_.resize(thrust::distance(vertex_first, vertex_last), handle_ptr_->get_stream());
      thrust::copy(handle_ptr_->get_thrust_policy(), vertex_first, vertex_last, vertices_.begin());
    }
  }

  /**
   * @ brief insert a list of (vertex, tag) pairs to the bucket
   *
   * @param key_first Iterator pointing to the first (inclusive) element of the (vertex,tag) pairs
   * stored in device memory.
   * @param key_last Iterator pointing to the last (exclusive) element of the  (vertex,tag) pairs
   * stored in device memory.
   */
  template <typename KeyIterator,
            typename tag_type                                  = tag_t,
            std::enable_if_t<!std::is_same_v<tag_type, void>>* = nullptr>
  void insert(KeyIterator key_first, KeyIterator key_last)
  {
    static_assert(std::is_same_v<typename std::iterator_traits<KeyIterator>::value_type,
                                 thrust::tuple<vertex_t, tag_t>>);

    if (vertices_.size() > 0) {
      if constexpr (sorted_unique) {
        rmm::device_uvector<vertex_t> merged_vertices(
          vertices_.size() + thrust::distance(key_first, key_last), handle_ptr_->get_stream());
        rmm::device_uvector<tag_t> merged_tags(merged_vertices.size(), handle_ptr_->get_stream());
        auto old_pair_first =
          thrust::make_zip_iterator(thrust::make_tuple(vertices_.begin(), tags_.begin()));
        auto merged_pair_first = thrust::make_zip_iterator(
          thrust::make_tuple(merged_vertices.begin(), merged_tags.begin()));
        thrust::merge(handle_ptr_->get_thrust_policy(),
                      old_pair_first,
                      old_pair_first + vertices_.size(),
                      key_first,
                      key_last,
                      merged_pair_first);
        merged_vertices.resize(
          thrust::distance(merged_pair_first,
                           thrust::unique(handle_ptr_->get_thrust_policy(),
                                          merged_pair_first,
                                          merged_pair_first + merged_vertices.size())),
          handle_ptr_->get_stream());
        merged_tags.resize(merged_vertices.size(), handle_ptr_->get_stream());
        merged_vertices.shrink_to_fit(handle_ptr_->get_stream());
        merged_tags.shrink_to_fit(handle_ptr_->get_stream());
        vertices_ = std::move(merged_vertices);
        tags_     = std::move(merged_tags);
      } else {
        auto cur_size = vertices_.size();
        vertices_.resize(cur_size + thrust::distance(key_first, key_last),
                         handle_ptr_->get_stream());
        tags_.resize(vertices_.size(), handle_ptr_->get_stream());
        thrust::copy(
          handle_ptr_->get_thrust_policy(),
          key_first,
          key_last,
          thrust::make_zip_iterator(thrust::make_tuple(vertices_.begin(), tags_.begin())) +
            cur_size);
      }
    } else {
      vertices_.resize(thrust::distance(key_first, key_last), handle_ptr_->get_stream());
      tags_.resize(thrust::distance(key_first, key_last), handle_ptr_->get_stream());
      thrust::copy(handle_ptr_->get_thrust_policy(),
                   key_first,
                   key_last,
                   thrust::make_zip_iterator(thrust::make_tuple(vertices_.begin(), tags_.begin())));
    }
  }

  size_t size() const { return vertices_.size(); }

  template <bool do_aggregate = multi_gpu>
  std::enable_if_t<do_aggregate, size_t> aggregate_size() const
  {
    return host_scalar_allreduce(handle_ptr_->get_comms(),
                                 vertices_.size(),
                                 raft::comms::op_t::SUM,
                                 handle_ptr_->get_stream());
  }

  template <bool do_aggregate = multi_gpu>
  std::enable_if_t<!do_aggregate, size_t> aggregate_size() const
  {
    return vertices_.size();
  }

  void resize(size_t size)
  {
    vertices_.resize(size, handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) { tags_.resize(size, handle_ptr_->get_stream()); }
  }

  void clear() { resize(0); }

  void shrink_to_fit()
  {
    vertices_.shrink_to_fit(handle_ptr_->get_stream());
    if constexpr (!std::is_same_v<tag_t, void>) { tags_.shrink_to_fit(handle_ptr_->get_stream()); }
  }

  auto const begin() const
  {
    if constexpr (std::is_same_v<tag_t, void>) {
      return vertices_.begin();
    } else {
      return thrust::make_zip_iterator(thrust::make_tuple(vertices_.begin(), tags_.begin()));
    }
  }

  auto begin()
  {
    if constexpr (std::is_same_v<tag_t, void>) {
      return vertices_.begin();
    } else {
      return thrust::make_zip_iterator(thrust::make_tuple(vertices_.begin(), tags_.begin()));
    }
  }

  auto const end() const { return begin() + vertices_.size(); }

  auto end() { return begin() + vertices_.size(); }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::device_uvector<vertex_t> vertices_;
  optional_buffer_type tags_;
};

template <typename vertex_t,
          typename tag_t                = void,
          bool multi_gpu                = false,
          bool sorted_unique_key_bucket = false>
class vertex_frontier_t {
  static_assert(std::is_same_v<tag_t, void> || std::is_arithmetic_v<tag_t>);

 public:
  using key_type =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  static size_t constexpr kInvalidBucketIdx{std::numeric_limits<size_t>::max()};

  vertex_frontier_t(raft::handle_t const& handle, size_t num_buckets) : handle_ptr_(&handle)
  {
    buckets_.reserve(num_buckets);
    for (size_t i = 0; i < num_buckets; ++i) {
      buckets_.emplace_back(handle);
    }
  }

  size_t num_buckets() const { return buckets_.size(); }

  key_bucket_t<vertex_t, tag_t, multi_gpu, sorted_unique_key_bucket>& bucket(size_t bucket_idx)
  {
    return buckets_[bucket_idx];
  }

  key_bucket_t<vertex_t, tag_t, multi_gpu, sorted_unique_key_bucket> const& bucket(
    size_t bucket_idx) const
  {
    return buckets_[bucket_idx];
  }

  void swap_buckets(size_t bucket_idx0, size_t bucket_idx1)
  {
    std::swap(buckets_[bucket_idx0], buckets_[bucket_idx1]);
  }

  template <typename SplitOp>
  void split_bucket(size_t this_bucket_idx,
                    std::vector<size_t> const& move_to_bucket_indices,
                    SplitOp split_op)
  {
    auto& this_bucket = bucket(this_bucket_idx);
    if (this_bucket.size() == 0) { return; }

    // 1. apply split_op to each bucket element

    assert(buckets_.size() <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(this_bucket.size(), handle_ptr_->get_stream());
    thrust::transform(
      handle_ptr_->get_thrust_policy(),
      this_bucket.begin(),
      this_bucket.end(),
      bucket_indices.begin(),
      [split_op] __device__(auto key) {
        auto split_op_result = split_op(key);
        return static_cast<uint8_t>(split_op_result ? *split_op_result : kInvalidBucketIdx);
      });

    // 2. remove elements with the invalid bucket indices

    auto pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), this_bucket.begin()));
    bucket_indices.resize(
      thrust::distance(pair_first,
                       thrust::remove_if(handle_ptr_->get_thrust_policy(),
                                         pair_first,
                                         pair_first + bucket_indices.size(),
                                         [] __device__(auto pair) {
                                           return thrust::get<0>(pair) ==
                                                  static_cast<uint8_t>(kInvalidBucketIdx);
                                         })),
      handle_ptr_->get_stream());
    this_bucket.resize(bucket_indices.size());
    bucket_indices.shrink_to_fit(handle_ptr_->get_stream());
    this_bucket.shrink_to_fit();

    // 3. separte the elements to stay in this bucket from the elements to be moved to other buckets

    pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), this_bucket.begin()));
    auto new_this_bucket_size = static_cast<size_t>(thrust::distance(
      pair_first,
      thrust::stable_partition(  // stable_partition to maintain sorted order within each bucket
        handle_ptr_->get_thrust_policy(),
        pair_first,
        pair_first + bucket_indices.size(),
        [this_bucket_idx = static_cast<uint8_t>(this_bucket_idx)] __device__(auto pair) {
          return thrust::get<0>(pair) == this_bucket_idx;
        })));

    // 4. insert to target buckets and resize this bucket

    insert_to_buckets(bucket_indices.begin() + new_this_bucket_size,
                      bucket_indices.end(),
                      this_bucket.begin() + new_this_bucket_size,
                      move_to_bucket_indices);

    this_bucket.resize(new_this_bucket_size);
    this_bucket.shrink_to_fit();
  }

  template <typename KeyIterator>
  void insert_to_buckets(uint8_t* bucket_idx_first /* [INOUT] */,
                         uint8_t* bucket_idx_last /* [INOUT] */,
                         KeyIterator key_first /* [INOUT] */,
                         std::vector<size_t> const& to_bucket_indices)
  {
    // 1. group the elements by their target bucket indices

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(bucket_idx_first, key_first));
    auto pair_last  = pair_first + thrust::distance(bucket_idx_first, bucket_idx_last);

    std::vector<size_t> insert_bucket_indices{};
    std::vector<size_t> insert_offsets{};
    std::vector<size_t> insert_sizes{};
    if (to_bucket_indices.size() == 1) {
      insert_bucket_indices = to_bucket_indices;
      insert_offsets        = {0};
      insert_sizes          = {static_cast<size_t>(thrust::distance(pair_first, pair_last))};
    } else if (to_bucket_indices.size() == 2) {
      auto next_bucket_size = static_cast<size_t>(thrust::distance(
        pair_first,
        thrust::stable_partition(  // stable_partition to maintain sorted order within each bucket
          handle_ptr_->get_thrust_policy(),
          pair_first,
          pair_last,
          [next_bucket_idx = static_cast<uint8_t>(to_bucket_indices[0])] __device__(auto pair) {
            return thrust::get<0>(pair) == next_bucket_idx;
          })));
      insert_bucket_indices = to_bucket_indices;
      insert_offsets        = {0, next_bucket_size};
      insert_sizes          = {
        next_bucket_size,
        static_cast<size_t>(thrust::distance(pair_first + next_bucket_size, pair_last))};
    } else {
      thrust::stable_sort(  // stable_sort to maintain sorted order within each bucket
        handle_ptr_->get_thrust_policy(),
        pair_first,
        pair_last,
        [] __device__(auto lhs, auto rhs) { return thrust::get<0>(lhs) < thrust::get<0>(rhs); });
      rmm::device_uvector<uint8_t> d_indices(to_bucket_indices.size(), handle_ptr_->get_stream());
      rmm::device_uvector<size_t> d_counts(d_indices.size(), handle_ptr_->get_stream());
      auto it = thrust::reduce_by_key(handle_ptr_->get_thrust_policy(),
                                      bucket_idx_first,
                                      bucket_idx_last,
                                      thrust::make_constant_iterator(size_t{1}),
                                      d_indices.begin(),
                                      d_counts.begin());
      d_indices.resize(thrust::distance(d_indices.begin(), thrust::get<0>(it)),
                       handle_ptr_->get_stream());
      d_counts.resize(d_indices.size(), handle_ptr_->get_stream());
      std::vector<uint8_t> h_indices(d_indices.size());
      std::vector<size_t> h_counts(h_indices.size());
      raft::update_host(
        h_indices.data(), d_indices.data(), d_indices.size(), handle_ptr_->get_stream());
      raft::update_host(
        h_counts.data(), d_counts.data(), d_counts.size(), handle_ptr_->get_stream());
      handle_ptr_->sync_stream();

      size_t offset{0};
      for (size_t i = 0; i < h_indices.size(); ++i) {
        insert_bucket_indices[i] = static_cast<size_t>(h_indices[i]);
        insert_offsets[i]        = offset;
        insert_sizes[i]          = h_counts[i];
        offset += insert_sizes[i];
      }
    }

    // 2. insert to the target buckets

    for (size_t i = 0; i < insert_offsets.size(); ++i) {
      bucket(insert_bucket_indices[i])
        .insert(key_first + insert_offsets[i], key_first + (insert_offsets[i] + insert_sizes[i]));
    }
  }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  std::vector<key_bucket_t<vertex_t, tag_t, multi_gpu, sorted_unique_key_bucket>> buckets_{};
};

}  // namespace cugraph
