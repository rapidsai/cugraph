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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace experimental {

/**
 * @brief store vertex partitioning map
 *
 * Say P = P_row * P_col GPUs. For communication, we need P_row row communicators of size P_col and
 * P_col column communicators of size P_row. row_comm_size = P_col and col_comm_size = P_row.
 * row_comm_rank & col_comm_rank are ranks within the row & column communicators, respectively.
 *
 * We need to partition 1D vertex arrays (storing per vertex values) and the 2D graph adjacency
 * matrix (or transposed 2D graph adjacency matrix) of G. An 1D vertex array of size V is divided to
 * P linear partitions; each partition has the size close to V / P. We consider two different
 * strategies to partition the 2D matrix: the default strategy and the hypergraph partitioning based
 * strategy (the latter is for future extension).
 * FIXME: in the future we may use the latter for both as this leads to simpler communication
 * patterns and better control over parallelism vs memory footprint trade-off.
 *
 * In the default case, one GPU will be responsible for 1 rectangular partition. The matrix will be
 * horizontally partitioned first to P_row slabs. Each slab will be further vertically partitioned
 * to P_col rectangles. Each rectangular partition will have the size close to V / P_row by V /
 * P_col.
 *
 * To be more specific, a GPU with (col_comm_rank, row_comm_rank) will be responsible for one
 * rectangular partition [a,b) by [c,d) where a = vertex_partition_offsets[row_comm_size *
 * col_comm_rank], b = vertex_partition_offsets[row_comm_size * (col_comm_rank + 1)], c =
 * vertex_partition_offsets[col_comm_size * row_comm_rank], and d =
 * vertex_partition_offsets[col_comm_size * (row_comm_rank + 1)].
 *
 * In the future, we may apply hyper-graph partitioning to divide V vertices to P groups minimizing
 * edge cuts across groups while balancing the number of vertices in each group. We will also
 * renumber vertices so the vertices in each group are mapped to consecutive integers. Then, there
 * will be more non-zeros in the diagonal partitions of the 2D graph adjacency matrix (or the
 * transposed 2D graph adjacency matrix) than the off-diagonal partitions. The default strategy does
 * not balance the number of nonzeros if hyper-graph partitioning is applied. To solve this problem,
 * the matrix is first horizontally partitioned to P slabs, then each slab will be further
 * vertically partitioned to P_row (instead of P_col in the default case) rectangles. One GPU will
 * be responsible col_comm_size rectangular partitions in this case.
 *
 * To be more specific, a GPU with (col_comm_rank, row_comm_rank) will be responsible for
 * col_comm_size rectangular partitions [a_i,b_i) by [c,d) where a_i =
 * vertex_partition_offsets[row_comm_size * i + row_comm_rank] and b_i =
 * vertex_partition_offsets[row_comm_size * i + row_comm_rank + 1]. c is
 * vertex_partition_offsets[row_comm_size * col_comm_rank] and d =
 * vertex_partition_offsests[row_comm_size * (col_comm_rank + 1)].
 *
 * See E. G. Boman et. al., “Scalable matrix computations on large scale-free graphs using 2D graph
 * partitioning”, 2013 for additional detail.
 *
 * @tparam vertex_t Type of vertex ID
 */
template <typename vertex_t>
class partition_t {
 public:
  partition_t(std::vector<vertex_t> const& vertex_partition_offsets,
              bool hypergraph_partitioned,
              int row_comm_size,
              int col_comm_size,
              int row_comm_rank,
              int col_comm_rank)
    : vertex_partition_offsets_(vertex_partition_offsets),
      hypergraph_partitioned_(hypergraph_partitioned),
      comm_rank_(col_comm_rank * row_comm_size + row_comm_rank),
      row_comm_size_(row_comm_size),
      col_comm_size_(col_comm_size),
      row_comm_rank_(row_comm_rank),
      col_comm_rank_(col_comm_rank)
  {
    CUGRAPH_EXPECTS(
      vertex_partition_offsets.size() == static_cast<size_t>(row_comm_size * col_comm_size + 1),
      "Invalid API parameter: erroneous vertex_partition_offsets.size().");

    CUGRAPH_EXPECTS(
      std::is_sorted(vertex_partition_offsets_.begin(), vertex_partition_offsets_.end()),
      "Invalid API parameter: partition.vertex_partition_offsets values should be non-descending.");
    CUGRAPH_EXPECTS(vertex_partition_offsets_[0] == vertex_t{0},
                    "Invalid API parameter: partition.vertex_partition_offsets[0] should be 0.");

    vertex_t start_offset{0};
    matrix_partition_major_value_start_offsets_.assign(get_number_of_matrix_partitions(), 0);
    for (size_t i = 0; i < matrix_partition_major_value_start_offsets_.size(); ++i) {
      matrix_partition_major_value_start_offsets_[i] = start_offset;
      start_offset += get_matrix_partition_major_last(i) - get_matrix_partition_major_first(i);
    }
  }

  int get_row_size() const { return row_comm_size_; }

  int get_col_size() const { return col_comm_size_; }

  int get_comm_rank() const { return comm_rank_; }

  std::vector<vertex_t> const& get_vertex_partition_offsets() const
  {
    return vertex_partition_offsets_;
  }

  std::tuple<vertex_t, vertex_t> get_local_vertex_range() const
  {
    return std::make_tuple(vertex_partition_offsets_[comm_rank_],
                           vertex_partition_offsets_[comm_rank_ + 1]);
  }

  vertex_t get_local_vertex_first() const { return vertex_partition_offsets_[comm_rank_]; }

  vertex_t get_local_vertex_last() const { return vertex_partition_offsets_[comm_rank_ + 1]; }

  std::tuple<vertex_t, vertex_t> get_vertex_partition_range(size_t vertex_partition_idx) const
  {
    return std::make_tuple(vertex_partition_offsets_[vertex_partition_idx],
                           vertex_partition_offsets_[vertex_partition_idx + 1]);
  }

  vertex_t get_vertex_partition_first(size_t vertex_partition_idx) const
  {
    return vertex_partition_offsets_[vertex_partition_idx];
  }

  vertex_t get_vertex_partition_last(size_t vertex_partition_idx) const
  {
    return vertex_partition_offsets_[vertex_partition_idx + 1];
  }

  vertex_t get_vertex_partition_size(size_t vertex_partition_idx) const
  {
    return get_vertex_partition_last(vertex_partition_idx) -
           get_vertex_partition_first(vertex_partition_idx);
  }

  size_t get_number_of_matrix_partitions() const
  {
    return hypergraph_partitioned_ ? col_comm_size_ : 1;
  }

  // major: row of the graph adjacency matrix (if the graph adjacency matrix is stored as is) or
  // column of the graph adjacency matrix (if the transposed graph adjacency matrix is stored).
  std::tuple<vertex_t, vertex_t> get_matrix_partition_major_range(size_t partition_idx) const
  {
    auto major_first = get_matrix_partition_major_first(partition_idx);
    auto major_last  = get_matrix_partition_major_last(partition_idx);
    return std::make_tuple(major_first, major_last);
  }

  vertex_t get_matrix_partition_major_first(size_t partition_idx) const
  {
    return hypergraph_partitioned_
             ? vertex_partition_offsets_[row_comm_size_ * partition_idx + row_comm_rank_]
             : vertex_partition_offsets_[col_comm_rank_ * row_comm_size_];
  }

  vertex_t get_matrix_partition_major_last(size_t partition_idx) const
  {
    return hypergraph_partitioned_
             ? vertex_partition_offsets_[row_comm_size_ * partition_idx + row_comm_rank_ + 1]
             : vertex_partition_offsets_[(col_comm_rank_ + 1) * row_comm_size_];
  }

  vertex_t get_matrix_partition_major_value_start_offset(size_t partition_idx) const
  {
    return matrix_partition_major_value_start_offsets_[partition_idx];
  }

  // minor: column of the graph adjacency matrix (if the graph adjacency matrix is stored as is) or
  // row of the graph adjacency matrix (if the transposed graph adjacency matrix is stored).
  std::tuple<vertex_t, vertex_t> get_matrix_partition_minor_range() const
  {
    auto minor_first = get_matrix_partition_minor_first();
    auto minor_last  = get_matrix_partition_minor_last();

    return std::make_tuple(minor_first, minor_last);
  }

  vertex_t get_matrix_partition_minor_first() const
  {
    return hypergraph_partitioned_ ? vertex_partition_offsets_[col_comm_rank_ * row_comm_size_]
                                   : vertex_partition_offsets_[row_comm_rank_ * col_comm_size_];
  }

  vertex_t get_matrix_partition_minor_last() const
  {
    return hypergraph_partitioned_
             ? vertex_partition_offsets_[(col_comm_rank_ + 1) * row_comm_size_]
             : vertex_partition_offsets_[(row_comm_rank_ + 1) * col_comm_size_];
  }

  // FIXME: this function may be removed if we use the same partitioning strategy whether hypergraph
  // partitioning is applied or not
  bool is_hypergraph_partitioned() const { return hypergraph_partitioned_; }

 private:
  std::vector<vertex_t> vertex_partition_offsets_{};  // size = P + 1
  bool hypergraph_partitioned_{false};

  int comm_rank_{0};
  int row_comm_size_{0};
  int col_comm_size_{0};
  int row_comm_rank_{0};
  int col_comm_rank_{0};

  std::vector<vertex_t>
    matrix_partition_major_value_start_offsets_{};  // size = get_number_of_matrix_partitions()
};

struct graph_properties_t {
  bool is_symmetric{false};
  bool is_multigraph{false};
};

namespace detail {

// FIXME: threshold values require tuning
size_t constexpr low_degree_threshold{raft::warp_size()};
size_t constexpr mid_degree_threshold{1024};
size_t constexpr num_segments_per_vertex_partition{3};

// Common for both graph_view_t & graph_t and both single-GPU & multi-GPU versions
template <typename vertex_t, typename edge_t, typename weight_t>
class graph_base_t {
 public:
  graph_base_t(raft::handle_t const& handle,
               vertex_t number_of_vertices,
               edge_t number_of_edges,
               graph_properties_t properties)
    : handle_ptr_(&handle),
      number_of_vertices_(number_of_vertices),
      number_of_edges_(number_of_edges),
      properties_(properties){};

  vertex_t get_number_of_vertices() const { return number_of_vertices_; }
  edge_t get_number_of_edges() const { return number_of_edges_; }

  template <typename vertex_type = vertex_t>
  std::enable_if_t<std::is_signed<vertex_type>::value, bool> is_valid_vertex(vertex_type v) const
  {
    return ((v >= 0) && (v < number_of_vertices_));
  }

  template <typename vertex_type = vertex_t>
  std::enable_if_t<std::is_unsigned<vertex_type>::value, bool> is_valid_vertex(vertex_type v) const
  {
    return (v < number_of_vertices_);
  }

  bool is_symmetric() const { return properties_.is_symmetric; }
  bool is_multigraph() const { return properties_.is_multigraph; }

 protected:
  raft::handle_t const* get_handle_ptr() const { return handle_ptr_; };
  graph_properties_t get_graph_properties() const { return properties_; }

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
               graph_properties_t properties,
               bool sorted_by_global_degree_within_vertex_partition,
               bool do_expensive_check = false);

  bool is_weighted() const { return adj_matrix_partition_weights_.size() > 0; }

  partition_t<vertex_t> get_partition() const { return partition_; }

  vertex_t get_number_of_local_vertices() const
  {
    return partition_.get_local_vertex_last() - partition_.get_local_vertex_first();
  }

  vertex_t get_local_vertex_first() const { return partition_.get_local_vertex_first(); }

  vertex_t get_local_vertex_last() const { return partition_.get_local_vertex_last(); }

  vertex_t get_vertex_partition_first(size_t vertex_partition_idx) const
  {
    return partition_.get_vertex_partition_first(vertex_partition_idx);
  }

  vertex_t get_vertex_partition_last(size_t vertex_partition_idx) const
  {
    return partition_.get_vertex_partition_last(vertex_partition_idx);
  }

  vertex_t get_vertex_partition_size(size_t vertex_partition_idx) const
  {
    return get_vertex_partition_last(vertex_partition_idx) -
           get_vertex_partition_first(vertex_partition_idx);
  }

  bool is_local_vertex_nocheck(vertex_t v) const
  {
    return (v >= get_local_vertex_first()) && (v < get_local_vertex_last());
  }

  size_t get_number_of_local_adj_matrix_partitions() const
  {
    return adj_matrix_partition_offsets_.size();
  }

  vertex_t get_number_of_local_adj_matrix_partition_rows() const
  {
    if (!store_transposed) {
      vertex_t ret{0};
      for (size_t i = 0; i < partition_.get_number_of_matrix_partitions(); ++i) {
        ret += partition_.get_matrix_partition_major_last(i) -
               partition_.get_matrix_partition_major_first(i);
      }
      return ret;
    } else {
      return partition_.get_matrix_partition_minor_last() -
             partition_.get_matrix_partition_minor_first();
    }
  }

  vertex_t get_number_of_local_adj_matrix_partition_cols() const
  {
    if (store_transposed) {
      vertex_t ret{0};
      for (size_t i = 0; i < partition_.get_number_of_matrix_partitions(); ++i) {
        ret += partition_.get_matrix_partition_major_last(i) -
               partition_.get_matrix_partition_major_first(i);
      }
      return ret;
    } else {
      return partition_.get_matrix_partition_minor_last() -
             partition_.get_matrix_partition_minor_first();
    }
  }

  vertex_t get_local_adj_matrix_partition_row_first(size_t adj_matrix_partition_idx) const
  {
    return store_transposed ? partition_.get_matrix_partition_minor_first()
                            : partition_.get_matrix_partition_major_first(adj_matrix_partition_idx);
  }

  vertex_t get_local_adj_matrix_partition_row_last(size_t adj_matrix_partition_idx) const
  {
    return store_transposed ? partition_.get_matrix_partition_minor_last()
                            : partition_.get_matrix_partition_major_last(adj_matrix_partition_idx);
  }

  vertex_t get_local_adj_matrix_partition_row_value_start_offset(
    size_t adj_matrix_partition_idx) const
  {
    return store_transposed
             ? vertex_t{0}
             : partition_.get_matrix_partition_major_value_start_offset(adj_matrix_partition_idx);
  }

  vertex_t get_local_adj_matrix_partition_col_first(size_t adj_matrix_partition_idx) const
  {
    return store_transposed ? partition_.get_matrix_partition_major_first(adj_matrix_partition_idx)
                            : partition_.get_matrix_partition_minor_first();
  }

  vertex_t get_local_adj_matrix_partition_col_last(size_t adj_matrix_partition_idx) const
  {
    return store_transposed ? partition_.get_matrix_partition_major_last(adj_matrix_partition_idx)
                            : partition_.get_matrix_partition_minor_last();
  }

  vertex_t get_local_adj_matrix_partition_col_value_start_offset(
    size_t adj_matrix_partition_idx) const
  {
    return store_transposed
             ? partition_.get_matrix_partition_major_value_start_offset(adj_matrix_partition_idx)
             : vertex_t{0};
  }

  bool is_hypergraph_partitioned() const { return partition_.is_hypergraph_partitioned(); }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private or even disappear if we switch to CSR + DCSR (or CSC + DCSC).
  edge_t const* offsets() const { return offsets(0); }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private or even disappear if we switch to CSR + DCSR (or CSC + DCSC).
  vertex_t const* indices() const { return indices(0); }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private or even disappear if we switch to CSR + DCSR (or CSC + DCSC).
  weight_t const* weights() const { return weights(0); }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private or even disappear if we switch to CSR + DCSR (or CSC + DCSC).
  edge_t const* offsets(size_t adj_matrix_partition_idx) const
  {
    return adj_matrix_partition_offsets_[adj_matrix_partition_idx];
  }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private or even disappear if we switch to CSR + DCSR (or CSC + DCSC).
  vertex_t const* indices(size_t adj_matrix_partition_idx) const
  {
    return adj_matrix_partition_indices_[adj_matrix_partition_idx];
  }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private or even disappear if we switch to CSR + DCSR (or CSC + DCSC).
  weight_t const* weights(size_t adj_matrix_partition_idx) const
  {
    return adj_matrix_partition_weights_.size() > 0
             ? adj_matrix_partition_weights_[adj_matrix_partition_idx]
             : static_cast<weight_t const*>(nullptr);
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
               graph_properties_t properties,
               bool sorted_by_degree,
               bool do_expensive_check = false);

  bool is_weighted() const { return weights_ != nullptr; }

  vertex_t get_number_of_local_vertices() const { return this->get_number_of_vertices(); }

  constexpr vertex_t get_local_vertex_first() const { return vertex_t{0}; }

  vertex_t get_local_vertex_last() const { return this->get_number_of_vertices(); }

  vertex_t get_vertex_partition_first(size_t vertex_partition_idx) const { return vertex_t{0}; }

  vertex_t get_vertex_partition_last(size_t vertex_partition_idx) const
  {
    return this->get_number_of_vertices();
  }

  vertex_t get_vertex_partition_size(size_t vertex_partition_idx) const
  {
    return get_vertex_partition_last(vertex_partition_idx) -
           get_vertex_partition_first(vertex_partition_idx);
  }

  constexpr bool is_local_vertex_nocheck(vertex_t v) const { return true; }

  constexpr size_t get_number_of_local_adj_matrix_partitions() const { return size_t(1); }

  vertex_t get_number_of_local_adj_matrix_partition_rows() const
  {
    return this->get_number_of_vertices();
  }

  vertex_t get_number_of_local_adj_matrix_partition_cols() const
  {
    return this->get_number_of_vertices();
  }

  vertex_t get_local_adj_matrix_partition_row_first(size_t adj_matrix_partition_idx) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t get_local_adj_matrix_partition_row_last(size_t adj_matrix_partition_idx) const
  {
    assert(adj_matrix_partition_idx == 0);
    return this->get_number_of_vertices();
  }

  vertex_t get_local_adj_matrix_partition_row_value_start_offset(
    size_t adj_matrix_partition_idx) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t get_local_adj_matrix_partition_col_first(size_t adj_matrix_partition_idx) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t get_local_adj_matrix_partition_col_last(size_t adj_matrix_partition_idx) const
  {
    assert(adj_matrix_partition_idx == 0);
    return this->get_number_of_vertices();
  }

  vertex_t get_local_adj_matrix_partition_col_value_start_offset(
    size_t adj_matrix_partition_idx) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  bool is_hypergraph_partitioned() const { return false; }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private.
  edge_t const* offsets() const { return offsets_; }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private.
  vertex_t const* indices() const { return indices_; }

  // FIXME: this function is not part of the public stable API.This function is mainly for pattern
  // accelerator implementation. This function is currently public to support the legacy
  // implementations directly accessing CSR/CSC data, but this function will eventually become
  // private.
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
