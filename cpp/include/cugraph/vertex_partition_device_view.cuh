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

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <type_traits>

namespace cugraph {
namespace experimental {

template <typename vertex_t>
class vertex_partition_device_view_base_t {
 public:
  vertex_partition_device_view_base_t(vertex_t number_of_vertices)
    : number_of_vertices_(number_of_vertices)
  {
  }

  template <typename vertex_type = vertex_t>
  __host__ __device__ std::enable_if_t<std::is_signed<vertex_type>::value, bool> is_valid_vertex(
    vertex_type v) const noexcept
  {
    return ((v >= 0) && (v < number_of_vertices_));
  }

  template <typename vertex_type = vertex_t>
  __host__ __device__ std::enable_if_t<std::is_unsigned<vertex_type>::value, bool> is_valid_vertex(
    vertex_type v) const noexcept
  {
    return (v < number_of_vertices_);
  }

 private:
  // should be trivially copyable to device
  vertex_t number_of_vertices_{0};
};

template <typename GraphViewType, typename Enable = void>
class vertex_partition_device_view_t;

// multi-GPU version
template <typename GraphViewType>
class vertex_partition_device_view_t<GraphViewType, std::enable_if_t<GraphViewType::is_multi_gpu>>
  : public vertex_partition_device_view_base_t<typename GraphViewType::vertex_type> {
 public:
  vertex_partition_device_view_t(GraphViewType const& graph_view)
    : vertex_partition_device_view_base_t<typename GraphViewType::vertex_type>(
        graph_view.get_number_of_vertices()),
      first_(graph_view.get_local_vertex_first()),
      last_(graph_view.get_local_vertex_last())
  {
  }

  __host__ __device__ bool is_local_vertex_nocheck(typename GraphViewType::vertex_type v) const
    noexcept
  {
    return (v >= first_) && (v < last_);
  }

  __host__ __device__ typename GraphViewType::vertex_type
  get_local_vertex_offset_from_vertex_nocheck(typename GraphViewType::vertex_type v) const noexcept
  {
    return v - first_;
  }

 private:
  // should be trivially copyable to device
  typename GraphViewType::vertex_type first_{0};
  typename GraphViewType::vertex_type last_{0};
};

// single-GPU version
template <typename GraphViewType>
class vertex_partition_device_view_t<GraphViewType, std::enable_if_t<!GraphViewType::is_multi_gpu>>
  : public vertex_partition_device_view_base_t<typename GraphViewType::vertex_type> {
 public:
  vertex_partition_device_view_t(GraphViewType const& graph_view)
    : vertex_partition_device_view_base_t<typename GraphViewType::vertex_type>(
        graph_view.get_number_of_vertices())
  {
  }

  __host__ __device__ constexpr bool is_local_vertex_nocheck(
    typename GraphViewType::vertex_type v) const noexcept
  {
    return true;
  }

  __host__ __device__ constexpr typename GraphViewType::vertex_type
  get_local_vertex_offset_from_vertex_nocheck(typename GraphViewType::vertex_type v) const noexcept
  {
    return v;
  }
};

}  // namespace experimental
}  // namespace cugraph
