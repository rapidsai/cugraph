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

#include <type_traits>

namespace cugraph {

namespace detail {

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

}  // namespace detail

template <typename GraphViewType, bool multi_gpu, typename Enable = void>
class vertex_partition_device_view_t;

// multi-GPU version
template <typename vertex_t, bool multi_gpu>
class vertex_partition_device_view_t<vertex_t, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::vertex_partition_device_view_base_t<vertex_t> {
 public:
  vertex_partition_device_view_t(vertex_partition_view_t<vertex_t, multi_gpu> view)
    : detail::vertex_partition_device_view_base_t<vertex_t>(view.number_of_vertices()),
      local_vertex_partition_range_first_(view.local_vertex_partition_range_first()),
      local_vertex_partition_range_last_(view.local_vertex_partition_range_last())
  {
  }

  __host__ __device__ bool in_local_vertex_partition_range_nocheck(vertex_t v) const noexcept
  {
    return (v >= local_vertex_partition_range_first_) && (v < local_vertex_partition_range_last_);
  }

  __host__ __device__ vertex_t
  local_vertex_partition_offset_from_vertex_nocheck(vertex_t v) const noexcept
  {
    return v - local_vertex_partition_range_first_;
  }

 private:
  // should be trivially copyable to device
  vertex_t local_vertex_partition_range_first_{0};
  vertex_t local_vertex_partition_range_last_{0};
};

// single-GPU version
template <typename vertex_t, bool multi_gpu>
class vertex_partition_device_view_t<vertex_t, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::vertex_partition_device_view_base_t<vertex_t> {
 public:
  vertex_partition_device_view_t(vertex_partition_view_t<vertex_t, multi_gpu> view)
    : detail::vertex_partition_device_view_base_t<vertex_t>(view.number_of_vertices())
  {
  }

  __host__ __device__ constexpr bool in_local_vertex_partition_range_nocheck(
    vertex_t v) const noexcept
  {
    return true;
  }

  __host__ __device__ constexpr vertex_t local_vertex_partition_offset_from_vertex_nocheck(
    vertex_t v) const noexcept
  {
    return v;
  }
};

}  // namespace cugraph
