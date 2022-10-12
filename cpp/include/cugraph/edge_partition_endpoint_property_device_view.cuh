/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/device_functors.cuh>

#include <raft/core/device_span.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename ValueIterator>
class edge_partition_endpoint_property_device_view_t {
 public:
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  edge_partition_endpoint_property_device_view_t() = default;

  edge_partition_endpoint_property_device_view_t(
    edge_major_property_view_t<vertex_t, ValueIterator> const& view, size_t partition_idx)
    : value_first_(view.value_firsts()[partition_idx]),
      range_first_(view.major_range_firsts()[partition_idx])
  {
    if (view.keys()) {
      keys_                    = (*(view.keys()))[partition_idx];
      key_chunk_start_offsets_ = (*(view.key_chunk_start_offsets()))[partition_idx];
      key_chunk_size_          = *(view.key_chunk_size());
    }
    value_first_ = view.value_firsts()[partition_idx];
    range_first_ = view.major_range_firsts()[partition_idx];
  }

  edge_partition_endpoint_property_device_view_t(
    edge_minor_property_view_t<vertex_t, ValueIterator> const& view)
  {
    if (view.keys()) {
      keys_                    = *(view.keys());
      key_chunk_start_offsets_ = *(view.key_chunk_start_offsets());
      key_chunk_size_          = *(view.key_chunk_size());
    }
    value_first_ = view.value_first();
    range_first_ = view.minor_range_first();
  }

  __device__ ValueIterator get_iter(vertex_t offset) const
  {
    auto value_offset = offset;
    if (keys_) {
      auto chunk_idx = static_cast<size_t>(offset) / (*key_chunk_size_);
      auto it        = thrust::lower_bound(thrust::seq,
                                    (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx],
                                    (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1],
                                    range_first_ + offset);
      assert((it != (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx + 1]) &&
             (*it == (range_first_ + offset)));
      value_offset = (*key_chunk_start_offsets_)[chunk_idx] +
                     static_cast<vertex_t>(thrust::distance(
                       (*keys_).begin() + (*key_chunk_start_offsets_)[chunk_idx], it));
    }
    return value_first_ + value_offset;
  }

  __device__ value_type get(vertex_t offset) const { return *get_iter(offset); }

 private:
  thrust::optional<raft::device_span<vertex_t const>> keys_{thrust::nullopt};
  thrust::optional<raft::device_span<vertex_t const>> key_chunk_start_offsets_{thrust::nullopt};
  thrust::optional<size_t> key_chunk_size_{thrust::nullopt};

  ValueIterator value_first_{};
  vertex_t range_first_{};
};

template <typename vertex_t>
class edge_partition_endpoint_dummy_property_device_view_t {
 public:
  using value_type = thrust::nullopt_t;

  edge_partition_endpoint_dummy_property_device_view_t() = default;

  edge_partition_endpoint_dummy_property_device_view_t(
    edge_endpoint_dummy_property_view_t const& view, size_t partition_idx)
  {
  }

  edge_partition_endpoint_dummy_property_device_view_t(
    edge_endpoint_dummy_property_view_t const& view)
  {
  }

  __device__ auto get(vertex_t offset) const { return thrust::nullopt; }
};

}  // namespace detail

}  // namespace cugraph
