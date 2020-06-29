/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <graph.hpp>
#include <utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <functional>
#include <memory>
#include <type_traits>

namespace cugraph {
namespace experimental {

// Common for both OPG and single-GPU versions
template <typename GraphType>
class graph_base_device_view_t {
 public:
  using vertex_type                              = typename GraphType::vertex_type;
  using edge_type                                = typename GraphType::edge_type;
  using weight_type                              = typename GraphType::weight_type;
  static constexpr bool is_adj_matrix_transposed = GraphType::is_adj_matrix_transposed;
  static constexpr bool is_opg                   = GraphType::is_opg;

  graph_base_device_view_t()                                = delete;
  ~graph_base_device_view_t()                               = default;
  graph_base_device_view_t(graph_base_device_view_t const&) = default;
  graph_base_device_view_t(graph_base_device_view_t&&)      = default;
  graph_base_device_view_t& operator=(graph_base_device_view_t const&) = default;
  graph_base_device_view_t& operator=(graph_base_device_view_t&&) = default;

  __host__ __device__ bool is_symmetric() const noexcept { return is_symmetric_; }

  __host__ __device__ bool is_weighted() const noexcept { return weights_ptr_ != nullptr; }

  __host__ __device__ vertex_type get_number_of_vertices() const noexcept
  {
    return number_of_vertices_;
  }

  __host__ __device__ vertex_type get_number_of_edges() const noexcept { return number_of_edges_; }

  template <typename vertex_t = vertex_type>
  __host__ __device__ std::enable_if_t<std::is_signed<vertex_t>::value, bool> is_valid_vertex(
    vertex_type v) const noexcept
  {
    return ((v >= 0) && (v < number_of_vertices_));
  }

  template <typename vertex_t = vertex_type>
  __host__ __device__ std::enable_if_t<std::is_unsigned<vertex_t>::value, bool> is_valid_vertex(
    vertex_type v) const noexcept
  {
    return (v < number_of_vertices_);
  }

 protected:
  bool is_symmetric_{false};
  vertex_type number_of_vertices_{0};
  edge_type number_of_edges_{0};

  edge_type const* offsets_ptr_{nullptr};
  vertex_type const* indices_ptr_{nullptr};
  weight_type const* weights_ptr_{nullptr};

  graph_base_device_view_t(GraphType const& graph)
  {
    // FIXME: better not directly access graph member variables, and directed is a misnomer.
    is_symmetric_       = !graph.prop.directed;
    number_of_vertices_ = graph.number_of_vertices;
    number_of_edges_    = graph.number_of_edges;
    // FIXME: better not directly access graph member variables
    offsets_ptr_ = graph.offsets;
    indices_ptr_ = graph.indices;
    weights_ptr_ = graph.edge_data;
  }
};

template <typename GraphType, typename Enable = void>
class graph_device_view_t;

// OPG version
template <typename GraphType>
class graph_device_view_t<GraphType, std::enable_if_t<GraphType::is_opg>>
  : public graph_base_device_view_t<GraphType> {
 public:
  using vertex_type                              = typename GraphType::vertex_type;
  using edge_type                                = typename GraphType::edge_type;
  using weight_type                              = typename GraphType::weight_type;
  static constexpr bool is_adj_matrix_transposed = GraphType::is_adj_matrix_transposed;
  static constexpr bool is_opg                   = true;

  graph_device_view_t()                           = delete;
  ~graph_device_view_t()                          = default;
  graph_device_view_t(graph_device_view_t const&) = default;
  graph_device_view_t(graph_device_view_t&&)      = default;
  graph_device_view_t& operator=(graph_device_view_t const&) = default;
  graph_device_view_t& operator=(graph_device_view_t&&) = default;

  graph_device_view_t(GraphType const& graph, void* d_ptr)
    : graph_base_device_view_t<GraphType>(graph)
  {
    CUGRAPH_FAIL("unimplemented.");
  }

  // only for the create() function
  void destroy() { delete this; }

  static std::unique_ptr<graph_device_view_t, std::function<void(graph_device_view_t*)>> create(
    GraphType const& graph)
  {
    // FIXME: If we partition a graph with a graph partitioning algorithm, block-diagonal parts
    // of an adjacency matrix have more non-zeros. For load balancing, we need to evenly distribute
    // block-diagonal parts to GPUs. Over-partitioning adjacency matrix rows is necessary for this
    // purpose.
    // See E. Boman, K. Devine, and S. Rajamanickam, "Scalable matrix computation on large
    // scale-free graphs using 2D graph partitioning," 2013.
    auto num_local_adj_matrix_partitions = 1;
    rmm::device_buffer* buffer_ptr =
      new rmm::device_buffer(num_local_adj_matrix_partitions * sizeof(vertex_type) * 2);
    auto deleter = [buffer_ptr](graph_device_view_t* graph_device_view) {
      graph_device_view->destroy();
      delete buffer_ptr;
    };
    std::unique_ptr<graph_device_view_t, decltype(deleter)> graph_device_view_ptr(
      new graph_device_view_t(graph, buffer_ptr->data()), deleter);

    return graph_device_view_ptr;
  }

 private:
  vertex_type const* local_adj_matrix_partition_firsts_{nullptr};
  vertex_type const* local_adj_matrix_partition_lasts_{nullptr};
  size_t num_local_adj_matrix_partitions_{0};
};

// single GPU version
template <typename GraphType>
class graph_device_view_t<GraphType, std::enable_if_t<!GraphType::is_opg>>
  : public graph_base_device_view_t<GraphType> {
 public:
  using vertex_type                              = typename GraphType::vertex_type;
  using edge_type                                = typename GraphType::edge_type;
  using weight_type                              = typename GraphType::weight_type;
  static constexpr bool is_adj_matrix_transposed = GraphType::is_adj_matrix_transposed;
  static constexpr bool is_opg                   = false;

  graph_device_view_t()                           = delete;
  ~graph_device_view_t()                          = default;
  graph_device_view_t(graph_device_view_t const&) = default;
  graph_device_view_t(graph_device_view_t&&)      = default;
  graph_device_view_t& operator=(graph_device_view_t const&) = default;
  graph_device_view_t& operator=(graph_device_view_t&&) = default;

  graph_device_view_t(GraphType const& graph) : graph_base_device_view_t<GraphType>(graph) {}

  static std::unique_ptr<graph_device_view_t, std::function<void(graph_device_view_t*)>> create(
    GraphType const& graph)
  {
    return std::make_unique<graph_device_view_t>(graph);
  }

  __host__ __device__ vertex_type get_number_of_local_vertices() const noexcept
  {
    return this->number_of_vertices_;
  }

  __host__ __device__ vertex_type get_number_of_adj_matrix_local_rows() const noexcept
  {
    return this->number_of_vertices_;
  }

  __host__ __device__ vertex_type get_number_of_adj_matrix_local_cols() const noexcept
  {
    return this->number_of_vertices_;
  }

  __host__ __device__ constexpr bool is_local_vertex_nocheck(vertex_type v) const noexcept
  {
    return true;
  }

  __host__ __device__ constexpr bool is_adj_matrix_local_row_nocheck(vertex_type row) const noexcept
  {
    return true;
  }

  __host__ __device__ constexpr bool is_adj_matrix_local_col_nocheck(vertex_type col) const noexcept
  {
    return true;
  }

  __host__ __device__ vertex_type
  get_vertex_from_local_vertex_offset_nocheck(vertex_type offset) const noexcept
  {
    return offset;
  }

  __host__ __device__ vertex_type get_local_vertex_offset_from_vertex_nocheck(vertex_type v) const
    noexcept
  {
    return v;
  }

  __host__ __device__ vertex_type
  get_adj_matrix_local_row_offset_from_row_nocheck(vertex_type row) const noexcept
  {
    return row;
  }

  __host__ __device__ vertex_type
  get_adj_matrix_local_col_offset_from_col_nocheck(vertex_type col) const noexcept
  {
    return col;
  }

  auto local_vertex_begin() const
  {
    return thrust::make_counting_iterator(static_cast<vertex_type>(0));
  }

  auto local_vertex_end() const
  {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }

  // FIXME: this API does not work if a single process holds more than one rectangular partitions
  // of the adjacency matrix.
  auto adj_matrix_local_row_begin() const
  {
    return thrust::make_counting_iterator(static_cast<vertex_type>(0));
  }

  auto adj_matrix_local_row_end() const
  {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }

  auto adj_matrix_local_col_begin() const
  {
    return thrust::make_counting_iterator(static_cast<vertex_type>(0));
  }

  auto adj_matrix_local_col_end() const
  {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }

  template <bool transposed = is_adj_matrix_transposed>
  __device__ std::enable_if_t<transposed, edge_type> get_local_in_degree_nocheck(
    vertex_type row) const noexcept
  {
    auto row_offset = get_adj_matrix_local_row_offset_from_row_nocheck(row);
    return *(this->offsets_ptr_ + row_offset + 1) - *(this->offsets_ptr_ + row_offset);
  }

  template <bool transposed = is_adj_matrix_transposed>
  __device__ std::enable_if_t<!transposed, edge_type> get_local_out_degree_nocheck(
    vertex_type row) const noexcept
  {
    auto row_offset = get_adj_matrix_local_row_offset_from_row_nocheck(row);
    return *(this->offsets_ptr_ + row_offset + 1) - *(this->offsets_ptr_ + row_offset);
  }

  __device__ thrust::tuple<vertex_type const*, weight_type const*, vertex_type> get_local_edges(
    vertex_type row) const noexcept
  {
    auto row_offset   = get_adj_matrix_local_row_offset_from_row_nocheck(row);
    auto edge_offset  = *(this->offsets_ptr_ + row_offset);
    auto local_degree = *(this->offsets_ptr_ + row_offset + 1) - edge_offset;
    auto indices      = this->indices_ptr_ + edge_offset;
    auto weights      = this->weights_ptr_ != nullptr ? this->weights_ptr_ + edge_offset : nullptr;
    return thrust::make_tuple(indices, weights, local_degree);
  }
};

}  // namespace experimental
}  // namespace cugraph
