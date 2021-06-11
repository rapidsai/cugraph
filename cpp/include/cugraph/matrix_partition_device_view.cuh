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

#include <cugraph/utilities/error.hpp>

#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {
namespace experimental {

template <typename vertex_t, typename edge_t, typename weight_t>
class matrix_partition_device_view_base_t {
 public:
  matrix_partition_device_view_base_t(edge_t const* offsets,
                                      vertex_t const* indices,
                                      weight_t const* weights,
                                      edge_t number_of_edges)
    : offsets_(offsets), indices_(indices), weights_(weights), number_of_edges_(number_of_edges)
  {
  }

  __host__ __device__ edge_t get_number_of_edges() const { return number_of_edges_; }

  __host__ __device__ edge_t const* get_offsets() const { return offsets_; }
  __host__ __device__ vertex_t const* get_indices() const { return indices_; }
  __host__ __device__ weight_t const* get_weights() const { return weights_; }

  __device__ thrust::tuple<vertex_t const*, weight_t const*, edge_t> get_local_edges(
    vertex_t major_offset) const noexcept
  {
    auto edge_offset  = *(offsets_ + major_offset);
    auto local_degree = *(offsets_ + (major_offset + 1)) - edge_offset;
    auto indices      = indices_ + edge_offset;
    auto weights      = weights_ != nullptr ? weights_ + edge_offset : nullptr;
    return thrust::make_tuple(indices, weights, local_degree);
  }

  __device__ edge_t get_local_degree(vertex_t major_offset) const noexcept
  {
    return *(offsets_ + (major_offset + 1)) - *(offsets_ + major_offset);
  }

  __device__ edge_t get_local_offset(vertex_t major_offset) const noexcept
  {
    return *(offsets_ + major_offset);
  }

 private:
  // should be trivially copyable to device
  edge_t const* offsets_{nullptr};
  vertex_t const* indices_{nullptr};
  weight_t const* weights_{nullptr};
  edge_t number_of_edges_{0};
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename Enable = void>
class matrix_partition_device_view_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
class matrix_partition_device_view_t<vertex_t,
                                     edge_t,
                                     weight_t,
                                     multi_gpu,
                                     std::enable_if_t<multi_gpu>>
  : public matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t> {
 public:
  matrix_partition_device_view_t(edge_t const* offsets,
                                 vertex_t const* indices,
                                 weight_t const* weights,
                                 edge_t number_of_matrix_partition_edges,
                                 vertex_t major_first,
                                 vertex_t major_last,
                                 vertex_t minor_first,
                                 vertex_t minor_last,
                                 vertex_t major_value_start_offset)
    : matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t>(
        offsets, indices, weights, number_of_matrix_partition_edges),
      major_first_(major_first),
      major_last_(major_last),
      minor_first_(minor_first),
      minor_last_(minor_last),
      major_value_start_offset_(major_value_start_offset)
  {
  }

  __host__ __device__ vertex_t get_major_value_start_offset() const
  {
    return major_value_start_offset_;
  }

  __host__ __device__ vertex_t get_major_first() const noexcept { return major_first_; }

  __host__ __device__ vertex_t get_major_last() const noexcept { return major_last_; }

  __host__ __device__ vertex_t get_major_size() const noexcept
  {
    return major_last_ - major_first_;
  }

  __host__ __device__ vertex_t get_minor_first() const noexcept { return minor_first_; }

  __host__ __device__ vertex_t get_minor_last() const noexcept { return minor_last_; }

  __host__ __device__ vertex_t get_minor_size() const noexcept
  {
    return minor_last_ - minor_first_;
  }

  __host__ __device__ vertex_t get_major_offset_from_major_nocheck(vertex_t major) const noexcept
  {
    return major - major_first_;
  }

  __host__ __device__ vertex_t get_minor_offset_from_minor_nocheck(vertex_t minor) const noexcept
  {
    return minor - minor_first_;
  }

  __host__ __device__ vertex_t get_major_from_major_offset_nocheck(vertex_t major_offset) const
    noexcept
  {
    return major_first_ + major_offset;
  }

  __host__ __device__ vertex_t get_minor_from_minor_offset_nocheck(vertex_t minor_offset) const
    noexcept
  {
    return minor_first_ + minor_offset;
  }

 private:
  // should be trivially copyable to device
  vertex_t major_first_{0};
  vertex_t major_last_{0};
  vertex_t minor_first_{0};
  vertex_t minor_last_{0};

  vertex_t major_value_start_offset_{0};
};

// single-GPU version
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
class matrix_partition_device_view_t<vertex_t,
                                     edge_t,
                                     weight_t,
                                     multi_gpu,
                                     std::enable_if_t<!multi_gpu>>
  : public matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t> {
 public:
  matrix_partition_device_view_t(edge_t const* offsets,
                                 vertex_t const* indices,
                                 weight_t const* weights,
                                 vertex_t number_of_vertices,
                                 edge_t number_of_edges)
    : matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t>(
        offsets, indices, weights, number_of_edges),
      number_of_vertices_(number_of_vertices)
  {
  }

  __host__ __device__ vertex_t get_major_value_start_offset() const { return vertex_t{0}; }

  __host__ __device__ constexpr vertex_t get_major_first() const noexcept { return vertex_t{0}; }

  __host__ __device__ vertex_t get_major_last() const noexcept { return number_of_vertices_; }

  __host__ __device__ vertex_t get_major_size() const noexcept { return number_of_vertices_; }

  __host__ __device__ constexpr vertex_t get_minor_first() const noexcept { return vertex_t{0}; }

  __host__ __device__ vertex_t get_minor_last() const noexcept { return number_of_vertices_; }

  __host__ __device__ vertex_t get_minor_size() const noexcept { return number_of_vertices_; }

  __host__ __device__ vertex_t get_major_offset_from_major_nocheck(vertex_t major) const noexcept
  {
    return major;
  }

  __host__ __device__ vertex_t get_minor_offset_from_minor_nocheck(vertex_t minor) const noexcept
  {
    return minor;
  }

  __host__ __device__ vertex_t get_major_from_major_offset_nocheck(vertex_t major_offset) const
    noexcept
  {
    return major_offset;
  }

  __host__ __device__ vertex_t get_minor_from_minor_offset_nocheck(vertex_t minor_offset) const
    noexcept
  {
    return minor_offset;
  }

 private:
  vertex_t number_of_vertices_;
};

}  // namespace experimental
}  // namespace cugraph
