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

// FIXME: better move this file to include/utilities (following cuDF) and rename to error.hpp
#include <utilities/error_utils.h>

#include <utilities/traits.hpp>
#include <graph.hpp>

#include <rmm/rmm.h>

#include <thrust/iterator/counting_iterator.h>

#include <functional>
#include <memory>
#include <type_traits>


namespace cugraph {
namespace experimental {

template<typename GraphType, typename Enable = void>
class graph_compressed_sparse_base_device_view_t;

// Common for both OPG and single-GPU versions
template<typename GraphType>
class graph_compressed_sparse_base_device_view_t<
  GraphType, std::enable_if_t<is_csr<GraphType>::value || is_csc<GraphType>::value>
> {
 public:
  using vertex_type = typename GraphType::vertex_type;
  using edge_type = typename GraphType::edge_type;
  using weight_type = typename GraphType::weight_type;
  static constexpr bool is_csr_type = is_csr<GraphType>::value;
  static constexpr bool is_csc_type = is_csc<GraphType>::value;

  graph_compressed_sparse_base_device_view_t() = delete;
  ~graph_compressed_sparse_base_device_view_t() = default;
  graph_compressed_sparse_base_device_view_t(
    graph_compressed_sparse_base_device_view_t const&) = default;
  graph_compressed_sparse_base_device_view_t(
    graph_compressed_sparse_base_device_view_t&&) = default;
  graph_compressed_sparse_base_device_view_t& operator=(
    graph_compressed_sparse_base_device_view_t const&) = default;
  graph_compressed_sparse_base_device_view_t& operator=(
    graph_compressed_sparse_base_device_view_t&&) = default;

  __host__ __device__
  bool is_symmetric() const noexcept {
    return is_symmetric_;
  }

  __host__ __device__
  vertex_type get_number_of_vertices() const noexcept {
    return number_of_vertices_;
  }

  __host__ __device__
  vertex_type get_number_of_edges() const noexcept {
    return number_of_edges_;
  }

  __host__ __device__
  constexpr bool in_vertex_range(vertex_type v) const noexcept {
    // FIXME: need to check
    return true;
  }

 protected:
  bool is_symmetric_{false};
  vertex_type number_of_vertices_{0};
  edge_type number_of_edges_{0};

  edge_type const* offsets_ptr_{nullptr};
  vertex_type const* indices_ptr_{nullptr};
  weight_type const* weights_ptr_{nullptr};

  graph_compressed_sparse_base_device_view_t(GraphType const& graph) {
    // FIXME: better not directly access graph member variables, and directed is a misnomer.
    is_symmetric_ = !graph.prop.directed;
    number_of_vertices_ = graph.number_of_vertices;
    // FIXME: better not directly access graph member variables
    offsets_ptr_ = graph.offsets;
    indices_ptr_ = graph.indices;
    weights_ptr_ = graph.edge_data;
  }
};

template<typename GraphType, typename Enable = void>
class graph_compressed_sparse_device_view_t;

// OPG version
template<typename GraphType>
class graph_compressed_sparse_device_view_t<
  GraphType,
  std::enable_if_t<GraphType::is_opg && (is_csr<GraphType>::value || is_csc<GraphType>::value)>
> : public graph_compressed_sparse_base_device_view_t<GraphType> {
 public:
  using vertex_type = typename GraphType::vertex_type;
  using edge_type = typename GraphType::edge_type;
  using weight_type = typename GraphType::weight_type;
  static constexpr bool is_csr_type = is_csr<GraphType>::value;
  static constexpr bool is_csc_type = is_csc<GraphType>::value;

  graph_compressed_sparse_device_view_t() = delete;
  ~graph_compressed_sparse_device_view_t() = default;
  graph_compressed_sparse_device_view_t(
    graph_compressed_sparse_device_view_t const&) = default;
  graph_compressed_sparse_device_view_t(
    graph_compressed_sparse_device_view_t&&) = default;
  graph_compressed_sparse_device_view_t& operator=(
    graph_compressed_sparse_device_view_t const&) = default;
  graph_compressed_sparse_device_view_t& operator=(
    graph_compressed_sparse_device_view_t&&) = default;

  graph_compressed_sparse_device_view_t(GraphType const& graph, void* d_ptr)
    : graph_compressed_sparse_base_device_view_t<GraphType>(graph) {
    CUGRAPH_FAIL("unimplemented.");
  }

  void destroy() {  // only for the create() function
    delete this;
  }

  static std::unique_ptr<
    graph_compressed_sparse_device_view_t,
    std::function<void(graph_compressed_sparse_device_view_t*)>
  > create(GraphType const& graph) {
    // FIXME: If we partition a graph with a graph partitioning algorithm, block-diagonal parts
    // of an adjacency matrix have more non-zeros. For load balancing, we need to evenly distribute
    // block-diagonal parts to GPUs. Over-partitioning adjacency matrix rows is necessary for this
    // purpose.
    // See E. Boman, K. Devine, and S. Rajamanickam, "Scalable matrix computation on large
    // scale-free graphs using 2D graph partitioning," 2013.
    auto num_this_partition_adj_matrix_row_ranges = 1;
    rmm::device_buffer* buffer_ptr =
      new rmm::device_buffer(num_this_partition_adj_matrix_row_ranges * sizeof(vertex_type) * 2);
    auto deleter =
      [buffer_ptr] (graph_compressed_sparse_device_view_t* graph_device_view) {
        graph_device_view->destroy();
        delete buffer_ptr;
      };
    std::unique_ptr<graph_compressed_sparse_device_view_t, decltype(deleter)>
    graph_device_view_ptr(
      new graph_compressed_sparse_device_view_t(graph, buffer_ptr->data()), deleter);

    return graph_device_view_ptr;
  }

private:
  vertex_type const* this_partition_adj_matrix_row_firsts_ptr_{nullptr};
  vertex_type const* this_partition_adj_matrix_row_lasts_ptr_{nullptr};
  size_t num_this_partition_adj_matrix_row_ranges_{0};
};

// single GPU version
template<typename GraphType>
class graph_compressed_sparse_device_view_t<
  GraphType,
  std::enable_if_t<!GraphType::is_opg && (is_csr<GraphType>::value || is_csc<GraphType>::value)>
> : public graph_compressed_sparse_base_device_view_t<GraphType> {
public:
  using vertex_type = typename GraphType::vertex_type;
  using edge_type = typename GraphType::edge_type;
  using weight_type = typename GraphType::weight_type;
  static constexpr bool is_csr_type = is_csr<GraphType>::value;
  static constexpr bool is_csc_type = is_csc<GraphType>::value;

  graph_compressed_sparse_device_view_t() = delete;
  ~graph_compressed_sparse_device_view_t() = default;
  graph_compressed_sparse_device_view_t(
    graph_compressed_sparse_device_view_t const&) = default;
  graph_compressed_sparse_device_view_t(
    graph_compressed_sparse_device_view_t&&) = default;
  graph_compressed_sparse_device_view_t& operator=(
    graph_compressed_sparse_device_view_t const&) = default;
  graph_compressed_sparse_device_view_t& operator=(
    graph_compressed_sparse_device_view_t&&) = default;

  graph_compressed_sparse_device_view_t(GraphType const& graph)
    : graph_compressed_sparse_base_device_view_t<GraphType>(graph) {}

  static std::unique_ptr<
    graph_compressed_sparse_device_view_t,
    std::function<void(graph_compressed_sparse_device_view_t*)>
  > create(GraphType const& graph) {
    return std::make_unique<graph_compressed_sparse_device_view_t>(graph);
  }

  // FIXME: better replace offset_data(), index_data(), and weight_data() with functions returning
  // a single value. This will abstract out graph data structure internals if adopt more complex
  // data structure than CSR/CSC (e.g. for 2D partitioning with a very large number of processes
  // CSR/CSC representations can become hyper-sparse for low degree vertices; in this case,
  // DCSR/DCSC will save memory, also we can skip storing offsets for 0 degree vertices)
  __host__ __device__
  edge_type const* offset_data() const noexcept {
    return this->offsets_ptr_;
  }

  __host__ __device__
  vertex_type const* index_data() const noexcept {
    return this->indices_ptr_;
  }

  __host__ __device__
  weight_type const* weight_data() const noexcept {
    return this->weights_ptr_;
  }

  __host__ __device__
  vertex_type get_number_of_this_partition_vertices() const noexcept {
    return this->number_of_vertices_;
  }

  __host__ __device__
  vertex_type get_number_of_this_partition_adj_matrix_rows() const noexcept {
    return this->number_of_vertices_;
  }

  __host__ __device__
  vertex_type get_number_of_this_partition_adj_matrix_cols() const noexcept {
    return this->number_of_vertices_;
  }

  __host__ __device__
  constexpr bool in_this_partition_vertex_range_nocheck(vertex_type v) const noexcept {
    return true;
  }

  __host__ __device__
  constexpr bool in_this_partition_adj_matrix_row_range_nocheck(vertex_type v) const noexcept {
    return true;
  }

  __host__ __device__
  constexpr bool in_this_partition_adj_matrxi_col_range_nocheck(vertex_type v) const noexcept {
    return true;
  }

  __host__ __device__
  vertex_type get_vertex_from_this_partition_vertex_offset_nocheck(vertex_type offset) const noexcept {
    return offset;
  }

  __host__ __device__
  vertex_type get_this_partition_vertex_offset_from_vertex_nocheck(vertex_type v) const noexcept {
    return v;
  }

  __host__ __device__
  vertex_type get_this_partition_row_offset_from_row_nocheck(vertex_type row) const noexcept {
    return row;
  }

  __host__ __device__
  vertex_type get_this_partition_col_offset_from_col_nocheck(vertex_type col) const noexcept {
    return col;
  }

  auto this_partition_vertex_begin() const {
    return thrust::make_counting_iterator(static_cast<vertex_type>(0));
  }

  auto this_partition_vertex_end() const {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }
  auto this_partition_adj_matrix_row_begin() const {
    return thrust::make_counting_iterator(static_cast<vertex_type>(0));
  }

  auto this_partition_adj_matrix_row_end() const {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }

  auto this_partition_adj_matrix_col_begin() const {
    return thrust::make_counting_iterator(static_cast<vertex_type>(0));
  }

  auto this_partition_adj_matrix_col_end() const {
    return thrust::make_counting_iterator(this->number_of_vertices_);
  }
};

}  // namespace experimental
}  // namespace cugraph