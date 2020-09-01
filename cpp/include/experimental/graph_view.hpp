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

#include <utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace experimental {

// FIXME: these should better be defined somewhere else.
std::string const comm_p_row_key = "comm_p_row";
std::string const comm_p_col_key = "comm_p_key";

/**
 * @brief store vertex partitioning map
 *
 * Say P = P_row * P_col GPUs. We need to partition 1D vertex arrays (storing per vertex values) and
 * the 2D graph adjacency matrix (or transposed 2D graph adjacency matrix) of G. An 1D vertex array
 * of size V is divided to P linear partitions; each partition has the size close to V / P. We
 * consider two different strategies to partition the 2D matrix: the default strategy and the
 * hypergraph partitioning based strategy (the latter is for future extension).
 *
 * In the default case, one GPU will be responsible for 1 rectangular partition. The matrix will be
 * horizontally partitioned first to P_row slabs. Each slab will be further vertically partitioned
 * to P_col rectangles. Each rectangular partition will have the size close to V / P_row by V /
 * P_col.
 *
 * To be more specific, a GPU with (row_rank, col_rank) will be responsible for one rectangular
 * partition [a,b) by [c,d) where a = vertex_partition_offsets[P_col * row_rank], b =
 * vertex_partition_offsets[p_col * (row_rank + 1)], c = vertex_partition_offsets[P_row * col_rank],
 * and d = vertex_partition_offsets[p_row * (col_rank + 1)]
 *
 * In the future, we may apply hyper-graph partitioning to divide V vertices to P groups minimizing
 * edge cuts across groups while balancing the number of vertices in each group. We will also
 * renumber vertices so the vertices in each group are mapped to consecutive integers. Then, there
 * will be more non-zeros in the diagonal partitions of the 2D graph adjacency matrix (or the
 * transposed 2D graph adjacency matrix) than the off-diagonal partitions. The default strategy does
 * not balance the number of nonzeros if hyper-graph partitioning is applied. To solve this problem,
 * the matrix is first horizontally partitioned to P (instead of P_row) slabs, then each slab will
 * be further vertically partitioned to P_col rectangles. One GPU will be responsible P_col
 * rectangular partitions in this case.
 *
 * To be more specific, a GPU with (row_rank, col_rank) will be responsible for P_col rectangular
 * partitions [a_i,b_i) by [c,d) where a_i = vertex_partition_offsets[P_row * i + row_rank] and b_i
 * = vertex_partition_offsets[P_row * i + row_rank + 1]. c and d are same to 1) and i = [0, P_col).
 *
 * See E. G. Boman et. al., “Scalable matrix computations on large scale-free graphs using 2D graph
 * partitioning”, 2013 for additional detail.
 *
 * @tparam vertex_t Type of vertex ID
 */
template <typename vertex_t>
struct partition_t {
  std::vector<vertex_t> vertex_partition_offsets{};  // size = P + 1
  bool hypergraph_partitioned{false};
};

namespace detail {

// FIXME: threshold values require tuning
size_t constexpr low_degree_threshold{raft::warp_size()};
size_t constexpr mid_degree_threshold{1024};
size_t constexpr num_segments_per_vertex_partition{3};

struct graph_properties_t {
  bool is_symmetric{false};
  bool is_multigraph{false};
  bool is_weighted{false};
};

// Common for both graph_view_t & graph_t and both single-GPU & multi-GPU versions
template <typename vertex_t, typename edge_t, typename weight_t>
class graph_base_t {
 public:
  graph_base_t(raft::handle_t const& handle,
               vertex_t number_of_vertices,
               edge_t number_of_edges,
               bool is_symmetric,
               bool is_multigraph,
               bool is_weighted)
    : handle_ptr_(&handle),
      number_of_vertices_(number_of_vertices),
      number_of_edges_(number_of_edges),
      properties_({is_symmetric, is_multigraph, is_weighted}){};

  vertex_t get_number_of_vertices() const { return number_of_vertices_; }
  edge_t get_number_of_edges() const { return number_of_edges_; }

  bool is_symmetric() const { return properties_.is_symmetric; }
  bool is_multigraph() const { return properties_.is_multigraph; }
  bool is_weighted() const { return properties_.is_weighted; }

 protected:
  raft::handle_t const* get_handle_ptr() const { return handle_ptr_; };

 private:
  raft::handle_t const* handle_ptr_{nullptr};

  vertex_t number_of_vertices_{0};
  edge_t number_of_edges_{0};

  graph_properties_t properties_{};
};

}  // namespace detail

// graph_view_t is a non-owning graph class (note that graph_t is an owning graph class)
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu,
          typename Enable = void>
class graph_view_t;

// multi-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
class graph_view_t<vertex_t,
                   edge_t,
                   weight_t,
                   store_transposed,
                   multi_gpu,
                   std::enable_if_t<multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t, weight_t> {
 public:
  using vertex_type                              = vertex_t;
  using edge_type                                = edge_t;
  using weight_type                              = weight_t;
  static constexpr bool is_adj_matrix_transposed = store_transposed;
  static constexpr bool is_multi_gpu             = multi_gpu;

  graph_view_t(raft::handle_t const& handle,
               std::vector<edge_t const*> const& adj_matrix_partition_offsets,
               std::vector<vertex_t const*> const& adj_matrix_partition_indices,
               std::vector<weight_t const*> const& adj_matrix_partition_weights,
               std::vector<vertex_t> const& vertex_partition_segment_offsets,
               partition_t<vertex_t> const& partition,
               vertex_t number_of_vertices,
               edge_t number_of_edges,
               bool is_symmetric,
               bool is_multigraph,
               bool is_weighted,
               bool sorted_by_global_degree_within_vertex_partition,
               bool do_expensive_check = false);

  vertex_t get_number_of_local_vertices() const
  {
    auto comm_p_rank = this->get_handle_ptr()->get_comms().get_rank();
    return partition_.vertex_partition_offsets[comm_p_rank + 1] -
           partition_.vertex_partition_offsets[comm_p_rank];
  }

  size_t get_number_of_adj_matrix_partitions() { return adj_matrix_partition_offsets_.size(); }

  // Better avoid direct invocation in application code.
  // This is mainly for pattern accelerator implementation.
  // This function may disappear in the future if we switch to CSR + DCSR (or CSC + DCSC)
  edge_t const* offsets(size_t adj_matrix_partition_idx) const
  {
    return adj_matrix_partition_offsets_[adj_matrix_partition_idx];
  }

  // Better avoid direct invocation in application code.
  // This is mainly for pattern accelerator implementation.
  // This function may disappear in the future if we switch to CSR + DCSR (or CSC + DCSC)
  vertex_t const* indices(size_t adj_matrix_partition_idx) const
  {
    return adj_matrix_partition_indices_[adj_matrix_partition_idx];
  }

  // Better avoid direct invocation in application code.
  // This is mainly for pattern accelerator implementation.
  // This function may disappear in the future if we switch to CSR + DCSR (or CSC + DCSC)
  weight_t const* weights(size_t adj_matrix_partition_idx) const
  {
    return adj_matrix_partition_weights_[adj_matrix_partition_idx];
  }

 private:
  std::vector<edge_t const*> adj_matrix_partition_offsets_{};
  std::vector<vertex_t const*> adj_matrix_partition_indices_{};
  std::vector<weight_t const*> adj_matrix_partition_weights_{};

  partition_t<vertex_t> partition_{};

  std::vector<vertex_t>
    vertex_partition_segment_offsets_{};  // segment offsets within the vertex partition based on
                                          // vertex degree, relevant only if
                                          // sorted_by_global_degree_within_vertex_partition is true
};

// single-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
class graph_view_t<vertex_t,
                   edge_t,
                   weight_t,
                   store_transposed,
                   multi_gpu,
                   std::enable_if_t<!multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t, weight_t> {
 public:
  using vertex_type                              = vertex_t;
  using edge_type                                = edge_t;
  using weight_type                              = weight_t;
  static constexpr bool is_adj_matrix_transposed = store_transposed;
  static constexpr bool is_multi_gpu             = multi_gpu;

  graph_view_t(raft::handle_t const& handle,
               edge_t const* offsets,
               vertex_t const* indices,
               weight_t const* weights,
               std::vector<vertex_t> const& segment_offsets,
               vertex_t number_of_vertices,
               edge_t number_of_edges,
               bool is_symmetric,
               bool is_multigraph,
               bool is_weighted,
               bool sorted_by_degree,
               bool do_expensive_check = false);

  vertex_t get_number_of_local_vertices() const { return this->get_number_of_vertices(); }

  // Better avoid direct invocation in application code.
  // This is mainly for pattern accelerator implementation.
  edge_t const* offsets() const { return offsets_; }

  // Better avoid direct invocation in application code.
  // This is mainly for pattern accelerator implementation.
  vertex_t const* indices() const { return indices_; }

  // Better avoid direct invocation in application code.
  // This is mainly for pattern accelerator implementation.
  weight_t const* weights() const { return weights_; }

 private:
  edge_t const* offsets_{nullptr};
  vertex_t const* indices_{nullptr};
  weight_t const* weights_{nullptr};
  std::vector<vertex_t> segment_offsets_{};  // segment offsets based on vertex degree, relevant
                                             // only if sorted_by_global_degree is true
};

}  // namespace experimental
}  // namespace cugraph