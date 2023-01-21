/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/edge_property.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>

namespace cugraph {

namespace detail {

template <typename edge_t, typename ValueIterator>
class edge_partition_edge_property_device_view_t {
 public:
  using value_type = typename thrust::iterator_traits<ValueIterator>::value_type;

  edge_partition_edge_property_device_view_t() = default;

  edge_partition_edge_property_device_view_t(
    edge_property_view_t<edge_t, ValueIterator> const& view, size_t partition_idx)
    : value_first_(view.value_firsts()[partition_idx])
  {
    value_first_ = view.value_firsts()[partition_idx];
  }

  __host__ __device__ ValueIterator value_first() { return value_first_; }

  __device__ ValueIterator get_iter(edge_t offset) const { return value_first_ + offset; }

  __device__ value_type get(edge_t offset) const { return *get_iter(offset); }

 private:
  ValueIterator value_first_{};
};

template <typename edge_t>
class edge_partition_edge_dummy_property_device_view_t {
 public:
  using value_type = thrust::nullopt_t;

  edge_partition_edge_dummy_property_device_view_t() = default;

  edge_partition_edge_dummy_property_device_view_t(edge_dummy_property_view_t const& view,
                                                   size_t partition_idx)
  {
  }

  __device__ auto get(edge_t offset) const { return thrust::nullopt; }
};

}  // namespace detail

}  // namespace cugraph
