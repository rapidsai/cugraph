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

#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <cassert>
#include <optional>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
class matrix_partition_device_view_base_t {
 public:
  matrix_partition_device_view_base_t(edge_t const* offsets,
                                      vertex_t const* indices,
                                      std::optional<weight_t const*> weights,
                                      edge_t number_of_edges)
    : offsets_(offsets),
      indices_(indices),
      weights_(weights ? thrust::optional<weight_t const*>(*weights) : thrust::nullopt),
      number_of_edges_(number_of_edges)
  {
  }

  __host__ __device__ edge_t get_number_of_edges() const { return number_of_edges_; }

  __host__ __device__ edge_t const* get_offsets() const { return offsets_; }
  __host__ __device__ vertex_t const* get_indices() const { return indices_; }
  __host__ __device__ thrust::optional<weight_t const*> get_weights() const { return weights_; }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
  __device__ thrust::tuple<vertex_t const*, thrust::optional<weight_t const*>, edge_t>
  get_local_edges(vertex_t major_idx) const noexcept
  {
    auto edge_offset  = *(offsets_ + major_idx);
    auto local_degree = *(offsets_ + (major_idx + 1)) - edge_offset;
    auto indices      = indices_ + edge_offset;
    auto weights =
      weights_ ? thrust::optional<weight_t const*>{*weights_ + edge_offset} : thrust::nullopt;
    return thrust::make_tuple(indices, weights, local_degree);
  }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
  __device__ edge_t get_local_degree(vertex_t major_idx) const noexcept
  {
    return *(offsets_ + (major_idx + 1)) - *(offsets_ + major_idx);
  }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
  __device__ edge_t get_local_offset(vertex_t major_idx) const noexcept
  {
    return *(offsets_ + major_idx);
  }

 private:
  // should be trivially copyable to device
  edge_t const* offsets_{nullptr};
  vertex_t const* indices_{nullptr};
  thrust::optional<weight_t const*> weights_{thrust::nullopt};
  edge_t number_of_edges_{0};
};

}  // namespace detail

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
  : public detail::matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t> {
 public:
  matrix_partition_device_view_t(
    matrix_partition_view_t<vertex_t, edge_t, weight_t, multi_gpu> view)
    : detail::matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t>(
        view.get_offsets(), view.get_indices(), view.get_weights(), view.get_number_of_edges()),
      dcs_nzd_vertices_(view.get_dcs_nzd_vertices()
                          ? thrust::optional<vertex_t const*>{*(view.get_dcs_nzd_vertices())}
                          : thrust::nullopt),
      dcs_nzd_vertex_count_(view.get_dcs_nzd_vertex_count()
                              ? thrust::optional<vertex_t>{*(view.get_dcs_nzd_vertex_count())}
                              : thrust::nullopt),
      major_first_(view.get_major_first()),
      major_last_(view.get_major_last()),
      minor_first_(view.get_minor_first()),
      minor_last_(view.get_minor_last()),
      major_value_start_offset_(view.get_major_value_start_offset())
  {
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

  __host__ __device__ vertex_t
  get_major_from_major_offset_nocheck(vertex_t major_offset) const noexcept
  {
    return major_first_ + major_offset;
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> get_major_hypersparse_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    if (dcs_nzd_vertices_) {
      // we can avoid binary search (and potentially improve performance) if we add an auxiliary
      // array or cuco::static_map (at the expense of additional memory)
      auto it = thrust::lower_bound(
        thrust::seq, *dcs_nzd_vertices_, *dcs_nzd_vertices_ + *dcs_nzd_vertex_count_, major);
      return it != *dcs_nzd_vertices_ + *dcs_nzd_vertex_count_
               ? (*it == major ? thrust::optional<vertex_t>{static_cast<vertex_t>(
                                   thrust::distance(*dcs_nzd_vertices_, it))}
                               : thrust::nullopt)
               : thrust::nullopt;
    } else {
      return thrust::nullopt;
    }
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> get_major_from_major_hypersparse_idx_nocheck(
    vertex_t major_hypersparse_idx) const noexcept
  {
    return dcs_nzd_vertices_
             ? thrust::optional<vertex_t>{(*dcs_nzd_vertices_)[major_hypersparse_idx]}
             : thrust::nullopt;
  }

  __host__ __device__ vertex_t
  get_minor_from_minor_offset_nocheck(vertex_t minor_offset) const noexcept
  {
    return minor_first_ + minor_offset;
  }

  __host__ __device__ vertex_t get_major_value_start_offset() const
  {
    return major_value_start_offset_;
  }

  __host__ __device__ thrust::optional<vertex_t const*> get_dcs_nzd_vertices() const
  {
    return dcs_nzd_vertices_;
  }
  __host__ __device__ thrust::optional<vertex_t> get_dcs_nzd_vertex_count() const
  {
    return dcs_nzd_vertex_count_;
  }

 private:
  // should be trivially copyable to device

  thrust::optional<vertex_t const*> dcs_nzd_vertices_{nullptr};
  thrust::optional<vertex_t> dcs_nzd_vertex_count_{0};

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
  : public detail::matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t> {
 public:
  matrix_partition_device_view_t(
    matrix_partition_view_t<vertex_t, edge_t, weight_t, multi_gpu> view)
    : detail::matrix_partition_device_view_base_t<vertex_t, edge_t, weight_t>(
        view.get_offsets(), view.get_indices(), view.get_weights(), view.get_number_of_edges()),
      number_of_vertices_(view.get_major_last())
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

  __host__ __device__ vertex_t
  get_major_from_major_offset_nocheck(vertex_t major_offset) const noexcept
  {
    return major_offset;
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> get_major_hypersparse_idx_from_major_nocheck(
    vertex_t major) const noexcept
  {
    assert(false);
    return thrust::nullopt;
  }

  // major_hypersparse_idx: index within the hypersparse segment
  __host__ __device__ thrust::optional<vertex_t> get_major_from_major_hypersparse_idx_nocheck(
    vertex_t major_hypersparse_idx) const noexcept
  {
    assert(false);
    return thrust::nullopt;
  }

  __host__ __device__ vertex_t
  get_minor_from_minor_offset_nocheck(vertex_t minor_offset) const noexcept
  {
    return minor_offset;
  }

  __host__ __device__ thrust::optional<vertex_t const*> get_dcs_nzd_vertices() const
  {
    assert(false);
    return thrust::nullopt;
  }
  __host__ __device__ thrust::optional<vertex_t> get_dcs_nzd_vertex_count() const
  {
    assert(false);
    return thrust::nullopt;
  }

 private:
  vertex_t number_of_vertices_;
};

}  // namespace experimental
}  // namespace cugraph
