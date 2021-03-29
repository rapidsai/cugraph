/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <utilities/host_scalar_comm.cuh>
#include <utilities/thrust_tuple_utils.cuh>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cinttypes>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace experimental {

template <typename vertex_t, bool is_multi_gpu = false>
class SortedUniqueElementBucket {
 public:
  SortedUniqueElementBucket(raft::handle_t const& handle)
    : handle_ptr_(&handle), elements_(0, handle.get_stream())
  {
  }

  void insert(vertex_t v)
  {
    if (elements_.size() > 0) {
      rmm::device_scalar<vertex_t> vertex(v, handle_ptr_->get_stream());
      insert(vertex.data(), vertex_t{1});
    } else {
      elements_.resize(1, handle_ptr_->get_stream());
      raft::update_device(elements_.data(), &v, size_t{1}, handle_ptr_->get_stream());
    }
  }

  void insert(vertex_t const* sorted_unique_vertices, vertex_t num_sorted_unique_vertices)
  {
    if (elements_.size() > 0) {
      rmm::device_uvector<vertex_t> merged_vertices(elements_.size() + num_sorted_unique_vertices,
                                                    handle_ptr_->get_stream());
      thrust::merge(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                    elements_.begin(),
                    elements_.end(),
                    sorted_unique_vertices,
                    sorted_unique_vertices + num_sorted_unique_vertices,
                    merged_vertices.begin());
      merged_vertices.resize(
        thrust::distance(
          merged_vertices.begin(),
          thrust::unique(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                         merged_vertices.begin(),
                         merged_vertices.end())),
        handle_ptr_->get_stream());
      merged_vertices.shrink_to_fit(handle_ptr_->get_stream());
      elements_ = std::move(merged_vertices);
    } else {
      elements_.resize(num_sorted_unique_vertices, handle_ptr_->get_stream());
      thrust::copy(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                   sorted_unique_vertices,
                   sorted_unique_vertices + num_sorted_unique_vertices,
                   elements_.begin());
    }
  }

  size_t size() const { return elements_.size(); }

  template <bool do_aggregate = is_multi_gpu>
  std::enable_if_t<do_aggregate, size_t> aggregate_size() const
  {
    return host_scalar_allreduce(
      handle_ptr_->get_comms(), elements_.size(), handle_ptr_->get_stream());
  }

  template <bool do_aggregate = is_multi_gpu>
  std::enable_if_t<!do_aggregate, size_t> aggregate_size() const
  {
    return elements_.size();
  }

  void resize(size_t size) { elements_.resize(size, handle_ptr_->get_stream()); }

  void clear() { elements_.resize(0, handle_ptr_->get_stream()); }

  void shrink_to_fit() { elements_.shrink_to_fit(handle_ptr_->get_stream()); }

  auto const data() const { return elements_.data(); }

  auto data() { return elements_.data(); }

  auto const begin() const { return elements_.begin(); }

  auto begin() { return elements_.begin(); }

  auto const end() const { return elements_.end(); }

  auto end() { return elements_.end(); }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::device_uvector<vertex_t> elements_;
};

template <typename vertex_t, bool is_multi_gpu = false, size_t num_buckets = 1>
class VertexFrontier {
 public:
  static size_t constexpr kNumBuckets = num_buckets;
  static size_t constexpr kInvalidBucketIdx{std::numeric_limits<size_t>::max()};

  VertexFrontier(raft::handle_t const& handle) : handle_ptr_(&handle)
  {
    for (size_t i = 0; i < num_buckets; ++i) { buckets_.emplace_back(handle); }
  }

  SortedUniqueElementBucket<vertex_t, is_multi_gpu>& get_bucket(size_t bucket_idx)
  {
    return buckets_[bucket_idx];
  }

  SortedUniqueElementBucket<vertex_t, is_multi_gpu> const& get_bucket(size_t bucket_idx) const
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
    auto& this_bucket = get_bucket(this_bucket_idx);
    if (this_bucket.size() > 0) {
      static_assert(kNumBuckets <= std::numeric_limits<uint8_t>::max());
      rmm::device_uvector<uint8_t> bucket_indices(this_bucket.size(), handle_ptr_->get_stream());
      thrust::transform(
        rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
        this_bucket.begin(),
        this_bucket.end(),
        bucket_indices.begin(),
        [split_op] __device__(auto v) { return static_cast<uint8_t>(split_op(v)); });

      auto pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), this_bucket.begin()));
      this_bucket.resize(thrust::distance(
        pair_first,
        thrust::remove_if(
          rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
          pair_first,
          pair_first + bucket_indices.size(),
          [invalid_bucket_idx = static_cast<uint8_t>(kInvalidBucketIdx)] __device__(auto pair) {
            return thrust::get<0>(pair) == invalid_bucket_idx;
          })));
      bucket_indices.resize(this_bucket.size(), handle_ptr_->get_stream());
      this_bucket.shrink_to_fit();
      bucket_indices.shrink_to_fit(handle_ptr_->get_stream());

      pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), this_bucket.begin()));
      auto new_this_bucket_size = thrust::distance(
        pair_first,
        thrust::stable_partition(  // stalbe_partition to maintain sorted order within each bucket
          rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
          pair_first,
          pair_first + bucket_indices.size(),
          [this_bucket_idx = static_cast<uint8_t>(this_bucket_idx)] __device__(auto pair) {
            return thrust::get<0>(pair) == this_bucket_idx;
          }));

      if (move_to_bucket_indices.size() == 1) {
        get_bucket(move_to_bucket_indices[0])
          .insert(this_bucket.begin() + new_this_bucket_size,
                  thrust::distance(this_bucket.begin() + new_this_bucket_size, this_bucket.end()));
      } else if (move_to_bucket_indices.size() == 2) {
        auto next_bucket_size = thrust::distance(
          pair_first + new_this_bucket_size,
          thrust::stable_partition(  // stalbe_partition to maintain sorted order within each bucket
            rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
            pair_first + new_this_bucket_size,
            pair_first + bucket_indices.size(),
            [next_bucket_idx = static_cast<uint8_t>(move_to_bucket_indices[0])] __device__(
              auto pair) { return thrust::get<0>(pair) == next_bucket_idx; }));
        get_bucket(move_to_bucket_indices[0])
          .insert(this_bucket.begin() + new_this_bucket_size, next_bucket_size);
        get_bucket(move_to_bucket_indices[1])
          .insert(this_bucket.begin() + new_this_bucket_size + next_bucket_size,
                  thrust::distance(this_bucket.begin() + new_this_bucket_size + next_bucket_size,
                                   this_bucket.end()));
      } else {
        thrust::sort(rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
                     pair_first + new_this_bucket_size,
                     pair_first + bucket_indices.size());
        rmm::device_uvector<uint8_t> d_indices(move_to_bucket_indices.size(),
                                               handle_ptr_->get_stream());
        rmm::device_uvector<size_t> d_counts(d_indices.size(), handle_ptr_->get_stream());
        auto it = thrust::reduce_by_key(
          rmm::exec_policy(handle_ptr_->get_stream())->on(handle_ptr_->get_stream()),
          bucket_indices.begin() + new_this_bucket_size,
          bucket_indices.end(),
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
        handle_ptr_->get_stream_view().synchronize();
        std::vector<size_t> h_offsets(h_indices.size(), 0);
        std::partial_sum(h_counts.begin(), h_counts.end() - 1, h_offsets.begin() + 1);
        for (size_t i = 0; i < h_indices.size(); ++i) {
          if (h_counts[i] > 0) {
            get_bucket(h_indices[i])
              .insert(this_bucket.begin() + new_this_bucket_size + h_offsets[i], h_counts[i]);
          }
        }
      }

      this_bucket.resize(new_this_bucket_size);
      this_bucket.shrink_to_fit();
    }

    return;
  }

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  std::vector<SortedUniqueElementBucket<vertex_t, is_multi_gpu>> buckets_{};
};

}  // namespace experimental
}  // namespace cugraph
