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

#include <cugraph/experimental/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {
namespace experimental {

template <typename vertex_t, typename edge_t, typename weight_t>
class matrix_partition_device_base_t {
 public:
  matrix_partition_device_base_t(edge_t const* offsets,
                                 vertex_t const* indices,
                                 weight_t const* weights,
                                 edge_t number_of_edges)
    : offsets_(offsets), indices_(indices), weights_(weights), number_of_edges_(number_of_edges)
  {
  }

  __host__ __device__ edge_t get_number_of_edges() const { return number_of_edges_; }

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

template <typename GraphViewType, typename Enable = void>
class matrix_partition_device_t;

// multi-GPU version
template <typename GraphViewType>
class matrix_partition_device_t<GraphViewType, std::enable_if_t<GraphViewType::is_multi_gpu>>
  : public matrix_partition_device_base_t<typename GraphViewType::vertex_type,
                                          typename GraphViewType::edge_type,
                                          typename GraphViewType::weight_type> {
 public:
  matrix_partition_device_t(GraphViewType const& graph_view, size_t partition_idx)
    : matrix_partition_device_base_t<typename GraphViewType::vertex_type,
                                     typename GraphViewType::edge_type,
                                     typename GraphViewType::weight_type>(
        graph_view.offsets(partition_idx),
        graph_view.indices(partition_idx),
        graph_view.weights(partition_idx),
        graph_view.get_number_of_local_adj_matrix_partition_edges(partition_idx)),
      major_first_(GraphViewType::is_adj_matrix_transposed
                     ? graph_view.get_local_adj_matrix_partition_col_first(partition_idx)
                     : graph_view.get_local_adj_matrix_partition_row_first(partition_idx)),
      major_last_(GraphViewType::is_adj_matrix_transposed
                    ? graph_view.get_local_adj_matrix_partition_col_last(partition_idx)
                    : graph_view.get_local_adj_matrix_partition_row_last(partition_idx)),
      minor_first_(GraphViewType::is_adj_matrix_transposed
                     ? graph_view.get_local_adj_matrix_partition_row_first(partition_idx)
                     : graph_view.get_local_adj_matrix_partition_col_first(partition_idx)),
      minor_last_(GraphViewType::is_adj_matrix_transposed
                    ? graph_view.get_local_adj_matrix_partition_row_last(partition_idx)
                    : graph_view.get_local_adj_matrix_partition_col_last(partition_idx)),
      major_value_start_offset_(
        GraphViewType::is_adj_matrix_transposed
          ? graph_view.get_local_adj_matrix_partition_col_value_start_offset(partition_idx)
          : graph_view.get_local_adj_matrix_partition_row_value_start_offset(partition_idx))
  {
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_value_start_offset() const
  {
    return major_value_start_offset_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_first() const noexcept
  {
    return major_first_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_last() const noexcept
  {
    return major_last_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_size() const noexcept
  {
    return major_last_ - major_first_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_first() const noexcept
  {
    return minor_first_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_last() const noexcept
  {
    return minor_last_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_size() const noexcept
  {
    return minor_last_ - minor_first_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_offset_from_major_nocheck(
    typename GraphViewType::vertex_type major) const noexcept
  {
    return major - major_first_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_offset_from_minor_nocheck(
    typename GraphViewType::vertex_type minor) const noexcept
  {
    return minor - minor_first_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_from_major_offset_nocheck(
    typename GraphViewType::vertex_type major_offset) const noexcept
  {
    return major_first_ + major_offset;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_from_minor_offset_nocheck(
    typename GraphViewType::vertex_type minor_offset) const noexcept
  {
    return minor_first_ + minor_offset;
  }

 private:
  // should be trivially copyable to device
  typename GraphViewType::vertex_type major_first_{0};
  typename GraphViewType::vertex_type major_last_{0};
  typename GraphViewType::vertex_type minor_first_{0};
  typename GraphViewType::vertex_type minor_last_{0};

  typename GraphViewType::vertex_type major_value_start_offset_{0};
};

// single-GPU version
template <typename GraphViewType>
class matrix_partition_device_t<GraphViewType, std::enable_if_t<!GraphViewType::is_multi_gpu>>
  : public matrix_partition_device_base_t<typename GraphViewType::vertex_type,
                                          typename GraphViewType::edge_type,
                                          typename GraphViewType::weight_type> {
 public:
  matrix_partition_device_t(GraphViewType const& graph_view, size_t partition_idx)
    : matrix_partition_device_base_t<typename GraphViewType::vertex_type,
                                     typename GraphViewType::edge_type,
                                     typename GraphViewType::weight_type>(
        graph_view.offsets(),
        graph_view.indices(),
        graph_view.weights(),
        graph_view.get_number_of_edges()),
      number_of_vertices_(graph_view.get_number_of_vertices())
  {
    assert(partition_idx == 0);
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_value_start_offset() const
  {
    return typename GraphViewType::vertex_type{0};
  }

  __host__ __device__ constexpr typename GraphViewType::vertex_type get_major_first() const noexcept
  {
    return typename GraphViewType::vertex_type{0};
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_last() const noexcept
  {
    return number_of_vertices_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_size() const noexcept
  {
    return number_of_vertices_;
  }

  __host__ __device__ constexpr typename GraphViewType::vertex_type get_minor_first() const noexcept
  {
    return typename GraphViewType::vertex_type{0};
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_last() const noexcept
  {
    return number_of_vertices_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_size() const noexcept
  {
    return number_of_vertices_;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_offset_from_major_nocheck(
    typename GraphViewType::vertex_type major) const noexcept
  {
    return major;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_offset_from_minor_nocheck(
    typename GraphViewType::vertex_type minor) const noexcept
  {
    return minor;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_major_from_major_offset_nocheck(
    typename GraphViewType::vertex_type major_offset) const noexcept
  {
    return major_offset;
  }

  __host__ __device__ typename GraphViewType::vertex_type get_minor_from_minor_offset_nocheck(
    typename GraphViewType::vertex_type minor_offset) const noexcept
  {
    return minor_offset;
  }

 private:
  typename GraphViewType::vertex_type number_of_vertices_;
};

}  // namespace experimental
}  // namespace cugraph
