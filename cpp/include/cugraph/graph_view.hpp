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

#include <cugraph/matrix_partition_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_view.hpp>

// visitor logic:
//
#include <cugraph/visitors/graph_envelope.hpp>

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
namespace serializer {
class serializer_t;  // forward...
}

/**
 * @brief store vertex partitioning map
 *
 * Say P = P_row * P_col GPUs. For communication, we need P_row row communicators of size P_col and
 * P_col column communicators of size P_row. row_comm_size = P_col and col_comm_size = P_row.
 * row_comm_rank & col_comm_rank are ranks within the row & column communicators, respectively.
 *
 * We need to partition 1D vertex arrays (storing per vertex values) and the 2D graph adjacency
 * matrix (or transposed 2D graph adjacency matrix) of G. An 1D vertex array of size V is divided to
 * P linear partitions; each partition has the size close to V / P.
 *
 * The 2D graph adjacency matrix is first horizontally partitioned to P slabs, then each slab will
 * be further vertically partitioned to P_row (instead of P_col in the default case) rectangles. One
 * GPU will be responsible col_comm_size rectangular partitions.
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
  partition_t() = default;

  partition_t(std::vector<vertex_t> const& vertex_partition_offsets,
              int row_comm_size,
              int col_comm_size,
              int row_comm_rank,
              int col_comm_rank)
    : vertex_partition_offsets_(vertex_partition_offsets),
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

  std::vector<vertex_t> get_vertex_partition_lasts() const
  {
    return std::vector<vertex_t>(vertex_partition_offsets_.begin() + 1,
                                 vertex_partition_offsets_.end());
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

  size_t get_number_of_matrix_partitions() const { return col_comm_size_; }

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
    return vertex_partition_offsets_[row_comm_size_ * partition_idx + row_comm_rank_];
  }

  vertex_t get_matrix_partition_major_last(size_t partition_idx) const
  {
    return vertex_partition_offsets_[row_comm_size_ * partition_idx + row_comm_rank_ + 1];
  }

  vertex_t get_matrix_partition_major_size(size_t partition_idx) const
  {
    return get_matrix_partition_major_last(partition_idx) -
           get_matrix_partition_major_first(partition_idx);
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
    return vertex_partition_offsets_[col_comm_rank_ * row_comm_size_];
  }

  vertex_t get_matrix_partition_minor_last() const
  {
    return vertex_partition_offsets_[(col_comm_rank_ + 1) * row_comm_size_];
  }

  vertex_t get_matrix_partition_minor_size() const
  {
    return get_matrix_partition_minor_last() - get_matrix_partition_minor_first();
  }

 private:
  std::vector<vertex_t> vertex_partition_offsets_{};  // size = P + 1

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

using namespace cugraph::visitors;

// FIXME: threshold values require tuning (currently disabled)
// use (key, value) pairs to store row/column properties if (unique edge rows/cols) over (V /
// row_comm_size|col_comm_size) is smaller than the threshold value
double constexpr row_col_properties_kv_pair_fill_ratio_threshold = 0.0;

// FIXME: threshold values require tuning
// use the hypersparse format (currently, DCSR or DCSC) for the vertices with their degrees smaller
// than col_comm_size * hypersparse_threshold_ratio, should be less than 1.0
double constexpr hypersparse_threshold_ratio = 0.5;
size_t constexpr low_degree_threshold{raft::warp_size()};
size_t constexpr mid_degree_threshold{1024};
size_t constexpr num_sparse_segments_per_vertex_partition{3};

// Common for both graph_view_t & graph_t and both single-GPU & multi-GPU versions
template <typename vertex_t, typename edge_t, typename weight_t>
class graph_base_t : public graph_envelope_t::base_graph_t /*<- visitor logic*/ {
 public:
  graph_base_t() = default;  // Note: required by visitor logic

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

  void apply(visitor_t& v) const override  // <- visitor logic
  {
    v.visit_graph(*this);
  }

 protected:
  friend class cugraph::serializer::serializer_t;

  raft::handle_t const* get_handle_ptr() const { return handle_ptr_; };
  graph_properties_t get_graph_properties() const { return properties_; }

 private:
  raft::handle_t const* handle_ptr_{nullptr};

  vertex_t number_of_vertices_{0};
  edge_t number_of_edges_{0};

  graph_properties_t properties_{};
};

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu, typename Enable = void>
struct graph_view_meta_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
struct graph_view_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>> {
  vertex_t number_of_vertices;
  edge_t number_of_edges;
  graph_properties_t properties;

  partition_t<vertex_t> partition{};

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> adj_matrix_partition_segment_offsets{};

  std::optional<vertex_t const*> local_sorted_unique_edge_row_first{std::nullopt};
  std::optional<vertex_t const*> local_sorted_unique_edge_row_last{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_row_offsets{std::nullopt};
  std::optional<vertex_t const*> local_sorted_unique_edge_col_first{std::nullopt};
  std::optional<vertex_t const*> local_sorted_unique_edge_col_last{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_col_offsets{std::nullopt};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
struct graph_view_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>> {
  vertex_t number_of_vertices;
  edge_t number_of_edges;
  graph_properties_t properties;

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> segment_offsets{std::nullopt};
};

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

  graph_view_t(
    raft::handle_t const& handle,
    std::vector<edge_t const*> const& adj_matrix_partition_offsets,
    std::vector<vertex_t const*> const& adj_matrix_partition_indices,
    std::optional<std::vector<weight_t const*>> const& adj_matrix_partition_weights,
    std::optional<std::vector<vertex_t const*>> const& adj_matrix_partition_dcs_nzd_vertices,
    std::optional<std::vector<vertex_t>> const& adj_matrix_partition_dcs_nzd_vertex_counts,
    graph_view_meta_t<vertex_t, edge_t, multi_gpu> meta);

  bool is_weighted() const { return adj_matrix_partition_weights_.has_value(); }

  std::vector<vertex_t> get_vertex_partition_lasts() const
  {
    return partition_.get_vertex_partition_lasts();
  }

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

  edge_t get_number_of_local_adj_matrix_partition_edges(size_t adj_matrix_partition_idx) const
  {
    return adj_matrix_partition_number_of_edges_[adj_matrix_partition_idx];
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<transposed, vertex_t> get_local_adj_matrix_partition_row_first() const
  {
    return partition_.get_matrix_partition_minor_first();
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<transposed, vertex_t> get_local_adj_matrix_partition_row_last() const
  {
    return partition_.get_matrix_partition_minor_last();
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<transposed, vertex_t> get_number_of_local_adj_matrix_partition_rows() const
  {
    return get_local_adj_matrix_partition_row_last() - get_local_adj_matrix_partition_row_first();
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

  vertex_t get_number_of_local_adj_matrix_partition_rows(size_t adj_matrix_partition_idx) const
  {
    return get_local_adj_matrix_partition_row_last(adj_matrix_partition_idx) -
           get_local_adj_matrix_partition_row_first(adj_matrix_partition_idx);
  }

  vertex_t get_local_adj_matrix_partition_row_value_start_offset(
    size_t adj_matrix_partition_idx) const
  {
    return store_transposed
             ? vertex_t{0}
             : partition_.get_matrix_partition_major_value_start_offset(adj_matrix_partition_idx);
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<!transposed, vertex_t> get_local_adj_matrix_partition_col_first() const
  {
    return partition_.get_matrix_partition_minor_first();
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<!transposed, vertex_t> get_local_adj_matrix_partition_col_last() const
  {
    return partition_.get_matrix_partition_minor_last();
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<!transposed, vertex_t> get_number_of_local_adj_matrix_partition_cols() const
  {
    return get_local_adj_matrix_partition_col_last() - get_local_adj_matrix_partition_col_first();
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

  vertex_t get_number_of_local_adj_matrix_partition_cols(size_t adj_matrix_partition_idx) const
  {
    return get_local_adj_matrix_partition_col_last(adj_matrix_partition_idx) -
           get_local_adj_matrix_partition_col_first(adj_matrix_partition_idx);
  }

  vertex_t get_local_adj_matrix_partition_col_value_start_offset(
    size_t adj_matrix_partition_idx) const
  {
    return store_transposed
             ? partition_.get_matrix_partition_major_value_start_offset(adj_matrix_partition_idx)
             : vertex_t{0};
  }

  std::optional<std::vector<vertex_t>> get_local_adj_matrix_partition_segment_offsets(
    size_t partition_idx) const
  {
    if (adj_matrix_partition_segment_offsets_) {
      auto size_per_partition =
        (*adj_matrix_partition_segment_offsets_).size() / partition_.get_col_size();
      return std::vector<vertex_t>(
        (*adj_matrix_partition_segment_offsets_).begin() + partition_idx * size_per_partition,
        (*adj_matrix_partition_segment_offsets_).begin() +
          (partition_idx + 1) * size_per_partition);
    } else {
      return std::nullopt;
    }
  }

  vertex_partition_view_t<vertex_t, true> get_vertex_partition_view() const
  {
    return vertex_partition_view_t<vertex_t, true>(this->get_number_of_vertices(),
                                                   this->get_local_vertex_first(),
                                                   this->get_local_vertex_last());
  }

  matrix_partition_view_t<vertex_t, edge_t, weight_t, true> get_matrix_partition_view(
    size_t adj_matrix_partition_idx) const
  {
    return matrix_partition_view_t<vertex_t, edge_t, weight_t, true>(
      adj_matrix_partition_offsets_[adj_matrix_partition_idx],
      adj_matrix_partition_indices_[adj_matrix_partition_idx],
      adj_matrix_partition_weights_
        ? std::optional<weight_t const*>{(*adj_matrix_partition_weights_)[adj_matrix_partition_idx]}
        : std::nullopt,
      adj_matrix_partition_dcs_nzd_vertices_
        ? std::optional<vertex_t const*>{(
            *adj_matrix_partition_dcs_nzd_vertices_)[adj_matrix_partition_idx]}
        : std::nullopt,
      adj_matrix_partition_dcs_nzd_vertex_counts_
        ? std::optional<vertex_t>{(
            *adj_matrix_partition_dcs_nzd_vertex_counts_)[adj_matrix_partition_idx]}
        : std::nullopt,
      this->get_number_of_local_adj_matrix_partition_edges(adj_matrix_partition_idx),
      store_transposed ? this->get_local_adj_matrix_partition_col_first(adj_matrix_partition_idx)
                       : this->get_local_adj_matrix_partition_row_first(adj_matrix_partition_idx),
      store_transposed ? this->get_local_adj_matrix_partition_col_last(adj_matrix_partition_idx)
                       : this->get_local_adj_matrix_partition_row_last(adj_matrix_partition_idx),
      store_transposed ? this->get_local_adj_matrix_partition_row_first(adj_matrix_partition_idx)
                       : this->get_local_adj_matrix_partition_col_first(adj_matrix_partition_idx),
      store_transposed ? this->get_local_adj_matrix_partition_row_last(adj_matrix_partition_idx)
                       : this->get_local_adj_matrix_partition_col_last(adj_matrix_partition_idx),
      store_transposed
        ? this->get_local_adj_matrix_partition_col_value_start_offset(adj_matrix_partition_idx)
        : this->get_local_adj_matrix_partition_row_value_start_offset(adj_matrix_partition_idx));
  }

  rmm::device_uvector<edge_t> compute_in_degrees(raft::handle_t const& handle) const;
  rmm::device_uvector<edge_t> compute_out_degrees(raft::handle_t const& handle) const;

  rmm::device_uvector<weight_t> compute_in_weight_sums(raft::handle_t const& handle) const;
  rmm::device_uvector<weight_t> compute_out_weight_sums(raft::handle_t const& handle) const;

  edge_t compute_max_in_degree(raft::handle_t const& handle) const;
  edge_t compute_max_out_degree(raft::handle_t const& handle) const;

  weight_t compute_max_in_weight_sum(raft::handle_t const& handle) const;
  weight_t compute_max_out_weight_sum(raft::handle_t const& handle) const;

  edge_t count_self_loops(raft::handle_t const& handle) const;
  edge_t count_multi_edges(raft::handle_t const& handle) const;

  std::optional<vertex_t const*> get_local_sorted_unique_edge_row_begin() const
  {
    return local_sorted_unique_edge_row_first_;
  }

  std::optional<vertex_t const*> get_local_sorted_unique_edge_row_end() const
  {
    return local_sorted_unique_edge_row_last_;
  }

  std::optional<std::vector<vertex_t>> get_local_sorted_unique_edge_row_offsets() const
  {
    return local_sorted_unique_edge_row_offsets_;
  }

  std::optional<vertex_t const*> get_local_sorted_unique_edge_col_begin() const
  {
    return local_sorted_unique_edge_col_first_;
  }

  std::optional<vertex_t const*> get_local_sorted_unique_edge_col_end() const
  {
    return local_sorted_unique_edge_col_last_;
  }

  std::optional<std::vector<vertex_t>> get_local_sorted_unique_edge_col_offsets() const
  {
    return local_sorted_unique_edge_col_offsets_;
  }

 private:
  std::vector<edge_t const*> adj_matrix_partition_offsets_{};
  std::vector<vertex_t const*> adj_matrix_partition_indices_{};
  std::optional<std::vector<weight_t const*>> adj_matrix_partition_weights_{};

  // relevant only if we use the CSR + DCSR (or CSC + DCSC) hybrid format
  std::optional<std::vector<vertex_t const*>> adj_matrix_partition_dcs_nzd_vertices_{};
  std::optional<std::vector<vertex_t>> adj_matrix_partition_dcs_nzd_vertex_counts_{};

  std::vector<edge_t> adj_matrix_partition_number_of_edges_{};

  partition_t<vertex_t> partition_{};

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> adj_matrix_partition_segment_offsets_{};

  // if valid, store row/column properties in key/value pairs (this saves memory if # unique edge
  // rows/cols << V / row_comm_size|col_comm_size).
  std::optional<vertex_t const*> local_sorted_unique_edge_row_first_{std::nullopt};
  std::optional<vertex_t const*> local_sorted_unique_edge_row_last_{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_row_offsets_{std::nullopt};
  std::optional<vertex_t const*> local_sorted_unique_edge_col_first_{std::nullopt};
  std::optional<vertex_t const*> local_sorted_unique_edge_col_last_{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_col_offsets_{std::nullopt};
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
               std::optional<weight_t const*> weights,
               graph_view_meta_t<vertex_t, edge_t, multi_gpu> meta);

  bool is_weighted() const { return weights_.has_value(); }

  vertex_t get_number_of_local_vertices() const { return this->get_number_of_vertices(); }

  std::vector<vertex_t> get_vertex_partition_lasts() const
  {
    return std::vector<vertex_t>{get_local_vertex_last()};
  }

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

  edge_t get_number_of_local_adj_matrix_partition_edges(size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return this->get_number_of_edges();
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<transposed, vertex_t> get_local_adj_matrix_partition_row_first() const
  {
    return get_local_adj_matrix_partition_row_first(0);
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<transposed, vertex_t> get_local_adj_matrix_partition_row_last() const
  {
    return get_local_adj_matrix_partition_row_last(0);
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<transposed, vertex_t> get_number_of_local_adj_matrix_partition_rows() const
  {
    return get_number_of_local_adj_matrix_partition_rows(0);
  }

  vertex_t get_local_adj_matrix_partition_row_first(size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t get_local_adj_matrix_partition_row_last(size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return this->get_number_of_vertices();
  }

  vertex_t get_local_adj_matrix_partition_row_value_start_offset(
    size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<!transposed, vertex_t> get_local_adj_matrix_partition_col_first() const
  {
    return get_local_adj_matrix_partition_col_first(0);
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<!transposed, vertex_t> get_local_adj_matrix_partition_col_last() const
  {
    return get_local_adj_matrix_partition_col_last(0);
  }

  template <bool transposed = is_adj_matrix_transposed>
  std::enable_if_t<!transposed, vertex_t> get_number_of_local_adj_matrix_partition_cols() const
  {
    return get_number_of_local_adj_matrix_partition_cols(0);
  }

  vertex_t get_local_adj_matrix_partition_col_first(size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t get_local_adj_matrix_partition_col_last(size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return this->get_number_of_vertices();
  }

  vertex_t get_local_adj_matrix_partition_col_value_start_offset(
    size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return vertex_t{0};
  }

  std::optional<std::vector<vertex_t>> get_local_adj_matrix_partition_segment_offsets(
    size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);
    return segment_offsets_;
  }

  vertex_partition_view_t<vertex_t, false> get_vertex_partition_view() const
  {
    return vertex_partition_view_t<vertex_t, false>(this->get_number_of_vertices());
  }

  matrix_partition_view_t<vertex_t, edge_t, weight_t, false> get_matrix_partition_view(
    size_t adj_matrix_partition_idx = 0) const
  {
    assert(adj_matrix_partition_idx == 0);  // there is only one matrix partition in single-GPU
    return matrix_partition_view_t<vertex_t, edge_t, weight_t, false>(
      offsets_, indices_, weights_, this->get_number_of_vertices(), this->get_number_of_edges());
  }

  rmm::device_uvector<edge_t> compute_in_degrees(raft::handle_t const& handle) const;
  rmm::device_uvector<edge_t> compute_out_degrees(raft::handle_t const& handle) const;

  rmm::device_uvector<weight_t> compute_in_weight_sums(raft::handle_t const& handle) const;
  rmm::device_uvector<weight_t> compute_out_weight_sums(raft::handle_t const& handle) const;

  edge_t compute_max_in_degree(raft::handle_t const& handle) const;
  edge_t compute_max_out_degree(raft::handle_t const& handle) const;

  weight_t compute_max_in_weight_sum(raft::handle_t const& handle) const;
  weight_t compute_max_out_weight_sum(raft::handle_t const& handle) const;

  edge_t count_self_loops(raft::handle_t const& handle) const;
  edge_t count_multi_edges(raft::handle_t const& handle) const;

  std::optional<vertex_t const*> get_local_sorted_unique_edge_row_begin() const
  {
    return std::nullopt;
  }

  std::optional<vertex_t const*> get_local_sorted_unique_edge_row_end() const
  {
    return std::nullopt;
  }

  std::optional<vertex_t const*> get_local_sorted_unique_edge_col_begin() const
  {
    return std::nullopt;
  }

  std::optional<vertex_t const*> get_local_sorted_unique_edge_col_end() const
  {
    return std::nullopt;
  }

 private:
  edge_t const* offsets_{nullptr};
  vertex_t const* indices_{nullptr};
  std::optional<weight_t const*> weights_{std::nullopt};

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> segment_offsets_{std::nullopt};
};

}  // namespace cugraph
