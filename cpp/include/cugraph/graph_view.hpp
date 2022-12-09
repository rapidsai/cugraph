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

#include <cugraph/edge_partition_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_view.hpp>

// visitor logic:
//
#include <cugraph/visitors/graph_envelope.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/host_span.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cugraph/graph_mask.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {

/**
 * @brief store vertex partitioning map
 *
 * Say P = P_row * P_col GPUs. For communication, we need P_row row communicators of size P_col and
 * P_col column communicators of size P_row. row_comm_size = P_col and col_comm_size = P_row.
 * row_comm_rank & col_comm_rank are ranks within the row & column communicators, respectively.
 *
 * We need to partition 1D vertex array and the 2D edge matrix (major = source if not transposed,
 * destination if transposed).
 *
 * An 1D vertex array of size V is divided to P linear partitions; each partition has the size close
 * to V / P.
 *
 * The 2D edge matrix is first horizontally partitioned to P slabs, then each slab will be further
 * vertically partitioned to P_row rectangles. One GPU will be responsible col_comm_size rectangular
 * partitions.
 *
 * To be more specific, a GPU with (col_comm_rank, row_comm_rank) will be responsible for
 * col_comm_size rectangular partitions [a_i,b_i) by [c,d) where a_i =
 * vertex_partition_range_offsets[row_comm_size * i + row_comm_rank] and b_i =
 * vertex_partition_range_offsets[row_comm_size * i + row_comm_rank + 1]. c is
 * vertex_partition_range_offsets[row_comm_size * col_comm_rank] and d =
 * vertex_partition_range_offsets[row_comm_size * (col_comm_rank + 1)].
 * Here, vertex_partition_range_offsets (size = P + 1) stores the 1D partitioning of the vertex ID
 * range [0, # vertices). The first P values store the beginning (inclusive) of each GPU partition.
 * The last value marks the end (exclusive) of the last GPU partition (which coincides with #
 * vertices).
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

  partition_t(std::vector<vertex_t> const& vertex_partition_range_offsets,
              int row_comm_size,
              int col_comm_size,
              int row_comm_rank,
              int col_comm_rank)
    : vertex_partition_range_offsets_(vertex_partition_range_offsets),
      comm_rank_(col_comm_rank * row_comm_size + row_comm_rank),
      row_comm_size_(row_comm_size),
      col_comm_size_(col_comm_size),
      row_comm_rank_(row_comm_rank),
      col_comm_rank_(col_comm_rank)
  {
    CUGRAPH_EXPECTS(vertex_partition_range_offsets.size() ==
                      static_cast<size_t>(row_comm_size * col_comm_size + 1),
                    "Invalid API parameter: erroneous vertex_partition_range_offsets.size().");

    CUGRAPH_EXPECTS(std::is_sorted(vertex_partition_range_offsets_.begin(),
                                   vertex_partition_range_offsets_.end()),
                    "Invalid API parameter: partition.vertex_partition_range_offsets values should "
                    "be non-descending.");
    CUGRAPH_EXPECTS(
      vertex_partition_range_offsets_[0] == vertex_t{0},
      "Invalid API parameter: partition.vertex_partition_range_offsets[0] should be 0.");

    vertex_t start_offset{0};
    edge_partition_major_value_start_offsets_.assign(number_of_local_edge_partitions(), 0);
    for (size_t i = 0; i < edge_partition_major_value_start_offsets_.size(); ++i) {
      edge_partition_major_value_start_offsets_[i] = start_offset;
      start_offset +=
        local_edge_partition_major_range_last(i) - local_edge_partition_major_range_first(i);
    }
  }

  // FIXME: these are used only in cugraph/utilities/cython.hpp, better delete once we fully switch
  // to the pylibcugraph path
  int row_comm_size() const { return row_comm_size_; }
  int col_comm_size() const { return col_comm_size_; }
  int comm_rank() const { return comm_rank_; }

  std::vector<vertex_t> const& vertex_partition_range_offsets() const
  {
    return vertex_partition_range_offsets_;
  }

  std::vector<vertex_t> vertex_partition_range_lasts() const
  {
    return std::vector<vertex_t>(vertex_partition_range_offsets_.begin() + 1,
                                 vertex_partition_range_offsets_.end());
  }

  std::tuple<vertex_t, vertex_t> local_vertex_partition_range() const
  {
    return std::make_tuple(vertex_partition_range_offsets_[comm_rank_],
                           vertex_partition_range_offsets_[comm_rank_ + 1]);
  }

  vertex_t local_vertex_partition_range_first() const
  {
    return vertex_partition_range_offsets_[comm_rank_];
  }

  vertex_t local_vertex_partition_range_last() const
  {
    return vertex_partition_range_offsets_[comm_rank_ + 1];
  }

  vertex_t local_vertex_partition_range_size() const
  {
    return local_vertex_partition_range_last() - local_vertex_partition_range_first();
  }

  std::tuple<vertex_t, vertex_t> vertex_partition_range(size_t partition_idx) const
  {
    return std::make_tuple(vertex_partition_range_offsets_[partition_idx],
                           vertex_partition_range_offsets_[partition_idx + 1]);
  }

  vertex_t vertex_partition_range_first(size_t partition_idx) const
  {
    return vertex_partition_range_offsets_[partition_idx];
  }

  vertex_t vertex_partition_range_last(size_t partition_idx) const
  {
    return vertex_partition_range_offsets_[partition_idx + 1];
  }

  vertex_t vertex_partition_range_size(size_t partition_idx) const
  {
    return vertex_partition_range_last(partition_idx) - vertex_partition_range_first(partition_idx);
  }

  size_t number_of_local_edge_partitions() const { return col_comm_size_; }

  // major: source of the edge partition (if not transposed) or destination of the edge partition
  // (if transposed).
  std::tuple<vertex_t, vertex_t> local_edge_partition_major_range(size_t partition_idx) const
  {
    return std::make_tuple(local_edge_partition_major_range_first(partition_idx),
                           local_edge_partition_major_range_last(partition_idx));
  }

  vertex_t local_edge_partition_major_range_first(size_t partition_idx) const
  {
    return vertex_partition_range_offsets_[row_comm_size_ * partition_idx + row_comm_rank_];
  }

  vertex_t local_edge_partition_major_range_last(size_t partition_idx) const
  {
    return vertex_partition_range_offsets_[row_comm_size_ * partition_idx + row_comm_rank_ + 1];
  }

  vertex_t local_edge_partition_major_range_size(size_t partition_idx) const
  {
    return local_edge_partition_major_range_last(partition_idx) -
           local_edge_partition_major_range_first(partition_idx);
  }

  vertex_t local_edge_partition_major_value_start_offset(size_t partition_idx) const
  {
    return edge_partition_major_value_start_offsets_[partition_idx];
  }

  // minor: destination of the edge partition (if not transposed) or source of the edge partition
  // (if transposed).
  std::tuple<vertex_t, vertex_t> local_edge_partition_minor_range() const
  {
    return std::make_tuple(local_edge_partition_minor_range_first(),
                           local_edge_partition_minor_range_last());
  }

  vertex_t local_edge_partition_minor_range_first() const
  {
    return vertex_partition_range_offsets_[col_comm_rank_ * row_comm_size_];
  }

  vertex_t local_edge_partition_minor_range_last() const
  {
    return vertex_partition_range_offsets_[(col_comm_rank_ + 1) * row_comm_size_];
  }

  vertex_t local_edge_partition_minor_range_size() const
  {
    return local_edge_partition_minor_range_last() - local_edge_partition_minor_range_first();
  }

 private:
  std::vector<vertex_t> vertex_partition_range_offsets_{};  // size = P + 1

  int comm_rank_{0};
  int row_comm_size_{0};
  int col_comm_size_{0};
  int row_comm_rank_{0};
  int col_comm_rank_{0};

  std::vector<vertex_t>
    edge_partition_major_value_start_offsets_{};  // size = number_of_local_edge_partitions()
};

struct graph_properties_t {
  bool is_symmetric{false};
  bool is_multigraph{false};
};

namespace detail {

using namespace cugraph::visitors;

// use (key, value) pairs to store source/destination properties if (unique edge
// sources/destinations) over (V / row_comm_size|col_comm_size) is smaller than the threshold value
double constexpr edge_partition_src_dst_property_values_kv_pair_fill_ratio_threshold = 0.1;

// FIXME: threshold values require tuning
// use the hypersparse format (currently, DCSR or DCSC) for the vertices with their degrees smaller
// than col_comm_size * hypersparse_threshold_ratio, should be less than 1.0
double constexpr hypersparse_threshold_ratio = 0.5;
size_t constexpr low_degree_threshold{raft::warp_size()};
size_t constexpr mid_degree_threshold{1024};
size_t constexpr num_sparse_segments_per_vertex_partition{3};

// Common for both graph_view_t & graph_t and both single-GPU & multi-GPU versions
template <typename vertex_t, typename edge_t>
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

  vertex_t number_of_vertices() const { return number_of_vertices_; }
  edge_t number_of_edges() const { return number_of_edges_; }

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
  raft::handle_t const* handle_ptr() const { return handle_ptr_; };
  graph_properties_t graph_properties() const { return properties_; }

 private:
  raft::handle_t const* handle_ptr_{nullptr};

  vertex_t number_of_vertices_{0};
  edge_t number_of_edges_{0};

  graph_properties_t properties_{};
};

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          bool store_transposed,
          bool multi_gpu,
          typename Enable = void>
struct graph_view_meta_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
struct graph_view_meta_t<vertex_t,
                         edge_t,
                         store_transposed,
                         multi_gpu,
                         std::enable_if_t<multi_gpu>> {
  vertex_t number_of_vertices;
  edge_t number_of_edges;
  graph_properties_t properties;

  partition_t<vertex_t> partition{};

  // segment offsets based on vertex degree
  std::vector<vertex_t> edge_partition_segment_offsets{};

  std::conditional_t<store_transposed,
                     std::optional<raft::device_span<vertex_t const>>,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>>
    local_sorted_unique_edge_srcs{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<raft::device_span<vertex_t const>>,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>>
    local_sorted_unique_edge_src_chunk_start_offsets{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_src_chunk_size{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<raft::host_span<vertex_t const>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_src_vertex_partition_offsets{std::nullopt};

  std::conditional_t<store_transposed,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>,
                     std::optional<raft::device_span<vertex_t const>>>
    local_sorted_unique_edge_dsts{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>,
                     std::optional<raft::device_span<vertex_t const>>>
    local_sorted_unique_edge_dst_chunk_start_offsets{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_dst_chunk_size{std::nullopt};
  std::conditional_t<!store_transposed,
                     std::optional<raft::host_span<vertex_t const>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_dst_vertex_partition_offsets{std::nullopt};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
struct graph_view_meta_t<vertex_t,
                         edge_t,
                         store_transposed,
                         multi_gpu,
                         std::enable_if_t<!multi_gpu>> {
  vertex_t number_of_vertices;
  edge_t number_of_edges;
  graph_properties_t properties;

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> segment_offsets{std::nullopt};
};

// graph_view_t is a non-owning graph class (note that graph_t is an owning graph class)
template <typename vertex_t,
          typename edge_t,
          bool store_transposed,
          bool multi_gpu,
          typename Enable = void>
class graph_view_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
class graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t> {
 public:
  using vertex_type                           = vertex_t;
  using edge_type                             = edge_t;
  static constexpr bool is_storage_transposed = store_transposed;
  static constexpr bool is_multi_gpu          = multi_gpu;

  graph_view_t(raft::handle_t const& handle,
               std::vector<edge_t const*> const& edge_partition_offsets,
               std::vector<vertex_t const*> const& edge_partition_indices,
               std::optional<std::vector<vertex_t const*>> const& edge_partition_dcs_nzd_vertices,
               std::optional<std::vector<vertex_t>> const& edge_partition_dcs_nzd_vertex_counts,
               graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu> meta);

  std::vector<vertex_t> vertex_partition_range_offsets() const
  {
    return partition_.vertex_partition_range_offsets();
  }

  std::vector<vertex_t> vertex_partition_range_lasts() const
  {
    return partition_.vertex_partition_range_lasts();
  }

  vertex_t local_vertex_partition_range_size() const
  {
    return partition_.local_vertex_partition_range_size();
  }

  vertex_t local_vertex_partition_range_first() const
  {
    return partition_.local_vertex_partition_range_first();
  }

  vertex_t local_vertex_partition_range_last() const
  {
    return partition_.local_vertex_partition_range_last();
  }

  vertex_t vertex_partition_range_first(size_t partition_idx) const
  {
    return partition_.vertex_partition_range_first(partition_idx);
  }

  vertex_t vertex_partition_range_last(size_t partition_idx) const
  {
    return partition_.vertex_partition_range_last(partition_idx);
  }

  vertex_t vertex_partition_range_size(size_t partition_idx) const
  {
    return partition_.vertex_partition_range_size(partition_idx);
  }

  bool in_local_vertex_partition_range_nocheck(vertex_t v) const
  {
    return (v >= local_vertex_partition_range_first()) && (v < local_vertex_partition_range_last());
  }

  size_t number_of_local_edge_partitions() const { return edge_partition_offsets_.size(); }

  edge_t number_of_local_edge_partition_edges(size_t partition_idx) const
  {
    return edge_partition_number_of_edges_[partition_idx];
  }

  vertex_t local_edge_partition_src_range_size() const
  {
    if constexpr (!store_transposed) {  // source range can be non-contiguous
      vertex_t ret{0};
      for (size_t i = 0; i < partition_.number_of_local_edge_partitions(); ++i) {
        ret += partition_.local_edge_partition_major_range_size(i);
      }
      return ret;
    } else {
      return partition_.local_edge_partition_minor_range_size();
    }
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, vertex_t> local_edge_partition_src_range_size(
    size_t partition_idx) const
  {
    return local_edge_partition_src_range_last(partition_idx) -
           local_edge_partition_src_range_first(partition_idx);
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, vertex_t> local_edge_partition_src_range_first() const
  {
    return partition_.local_edge_partition_minor_range_first();
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, vertex_t> local_edge_partition_src_range_first(
    size_t partition_idx) const
  {
    return partition_.local_edge_partition_major_range_first(partition_idx);
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, vertex_t> local_edge_partition_src_range_last() const
  {
    return partition_.local_edge_partition_minor_range_last();
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, vertex_t> local_edge_partition_src_range_last(
    size_t partition_idx) const
  {
    return partition_.local_edge_partition_major_range_last(partition_idx);
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, vertex_t> local_edge_partition_src_value_start_offset(
    size_t partition_idx) const
  {
    return partition_.local_edge_partition_major_value_start_offset(partition_idx);
  }

  vertex_t local_edge_partition_dst_range_size() const
  {
    if constexpr (store_transposed) {  // destination range can be non-contiguous
      vertex_t ret{0};
      for (size_t i = 0; i < partition_.number_of_local_edge_partitions(); ++i) {
        ret += partition_.local_edge_partition_major_range_size(i);
      }
      return ret;
    } else {
      return partition_.local_edge_partition_minor_range_size();
    }
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, vertex_t> local_edge_partition_dst_range_size(
    size_t partition_idx) const
  {
    return local_edge_partition_dst_range_last(partition_idx) -
           local_edge_partition_dst_range_first(partition_idx);
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, vertex_t> local_edge_partition_dst_range_first() const
  {
    return partition_.local_edge_partition_minor_range_first();
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, vertex_t> local_edge_partition_dst_range_first(
    size_t partition_idx) const
  {
    return partition_.local_edge_partition_major_range_first(partition_idx);
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, vertex_t> local_edge_partition_dst_range_last() const
  {
    return partition_.local_edge_partition_minor_range_last();
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, vertex_t> local_edge_partition_dst_range_last(
    size_t partition_idx) const
  {
    return partition_.local_edge_partition_major_range_last(partition_idx);
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, vertex_t> local_edge_partition_dst_value_start_offset(
    size_t partition_idx) const
  {
    return store_transposed
             ? partition_.local_edge_partition_major_value_start_offset(partition_idx)
             : vertex_t{0};
  }

  bool use_dcs() const
  {
    auto num_segments_per_vertex_partition =
      edge_partition_segment_offsets_.size() / edge_partition_offsets_.size();
    return num_segments_per_vertex_partition >
           (detail::num_sparse_segments_per_vertex_partition + size_t{2});
  }

  std::optional<std::vector<vertex_t>> local_edge_partition_segment_offsets(
    size_t partition_idx) const
  {
    auto num_segments_per_vertex_partition =
      edge_partition_segment_offsets_.size() / edge_partition_offsets_.size();
    return std::vector<vertex_t>(
      edge_partition_segment_offsets_.begin() + partition_idx * num_segments_per_vertex_partition,
      edge_partition_segment_offsets_.begin() +
        (partition_idx + 1) * num_segments_per_vertex_partition);
  }

  vertex_partition_view_t<vertex_t, true> local_vertex_partition_view() const
  {
    return vertex_partition_view_t<vertex_t, true>(this->number_of_vertices(),
                                                   this->local_vertex_partition_range_first(),
                                                   this->local_vertex_partition_range_last());
  }

  edge_partition_view_t<vertex_t, edge_t, true> local_edge_partition_view(
    size_t partition_idx) const
  {
    vertex_t major_range_first{};
    vertex_t major_range_last{};
    vertex_t minor_range_first{};
    vertex_t minor_range_last{};
    vertex_t major_value_range_start_offset{};
    if constexpr (store_transposed) {
      major_range_first = this->local_edge_partition_dst_range_first(partition_idx);
      major_range_last  = this->local_edge_partition_dst_range_last(partition_idx);
      minor_range_first = this->local_edge_partition_src_range_first();
      minor_range_last  = this->local_edge_partition_src_range_last();
      major_value_range_start_offset =
        this->local_edge_partition_dst_value_start_offset(partition_idx);
    } else {
      major_range_first = this->local_edge_partition_src_range_first(partition_idx);
      major_range_last  = this->local_edge_partition_src_range_last(partition_idx);
      minor_range_first = this->local_edge_partition_dst_range_first();
      minor_range_last  = this->local_edge_partition_dst_range_last();
      major_value_range_start_offset =
        this->local_edge_partition_src_value_start_offset(partition_idx);
    }
    std::optional<vertex_t> major_hypersparse_first{std::nullopt};
    vertex_t offset_size = (major_range_last - major_range_first) + 1;
    if (this->use_dcs()) {
      major_hypersparse_first =
        major_range_first + (*(this->local_edge_partition_segment_offsets(
                              partition_idx)))[detail::num_sparse_segments_per_vertex_partition];
      offset_size = ((*major_hypersparse_first) - major_range_first) +
                    (*edge_partition_dcs_nzd_vertex_counts_)[partition_idx] + 1;
    }
    return edge_partition_view_t<vertex_t, edge_t, true>(
      raft::device_span<edge_t const>(edge_partition_offsets_[partition_idx],
                                      edge_partition_offsets_[partition_idx] + offset_size),
      raft::device_span<vertex_t const>(
        edge_partition_indices_[partition_idx],
        edge_partition_indices_[partition_idx] + edge_partition_number_of_edges_[partition_idx]),
      edge_partition_dcs_nzd_vertices_
        ? std::make_optional<raft::device_span<vertex_t const>>(
            (*edge_partition_dcs_nzd_vertices_)[partition_idx],
            (*edge_partition_dcs_nzd_vertices_)[partition_idx] +
              (*edge_partition_dcs_nzd_vertex_counts_)[partition_idx])
        : std::nullopt,
      major_hypersparse_first,
      major_range_first,
      major_range_last,
      minor_range_first,
      minor_range_last,
      major_value_range_start_offset);
  }

  rmm::device_uvector<edge_t> compute_in_degrees(raft::handle_t const& handle) const;
  rmm::device_uvector<edge_t> compute_out_degrees(raft::handle_t const& handle) const;

  edge_t compute_max_in_degree(raft::handle_t const& handle) const;
  edge_t compute_max_out_degree(raft::handle_t const& handle) const;

  edge_t count_self_loops(raft::handle_t const& handle) const;
  edge_t count_multi_edges(raft::handle_t const& handle) const;

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_srcs() const
  {
    return local_sorted_unique_edge_srcs_;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_srcs(size_t partition_idx) const
  {
    return local_sorted_unique_edge_srcs_ ? std::optional<raft::device_span<vertex_t const>>{(
                                              *local_sorted_unique_edge_srcs_)[partition_idx]}
                                          : std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_src_chunk_start_offsets() const
  {
    return local_sorted_unique_edge_src_chunk_start_offsets_;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_src_chunk_start_offsets(size_t partition_idx) const
  {
    return local_sorted_unique_edge_src_chunk_start_offsets_
             ? std::optional<raft::device_span<vertex_t const>>{(
                 *local_sorted_unique_edge_src_chunk_start_offsets_)[partition_idx]}
             : std::nullopt;
  }

  std::optional<vertex_t> local_sorted_unique_edge_src_chunk_size() const
  {
    return local_sorted_unique_edge_src_chunk_size_;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::host_span<vertex_t const>>>
  local_sorted_unique_edge_src_vertex_partition_offsets() const
  {
    return local_sorted_unique_edge_src_vertex_partition_offsets_;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dsts() const
  {
    return local_sorted_unique_edge_dsts_;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dsts(size_t partition_idx) const
  {
    return local_sorted_unique_edge_dsts_ ? std::optional<raft::device_span<vertex_t const>>{(
                                              *local_sorted_unique_edge_dsts_)[partition_idx]}
                                          : std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dst_chunk_start_offsets() const
  {
    return local_sorted_unique_edge_dst_chunk_start_offsets_;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dst_chunk_start_offsets(size_t partition_idx) const
  {
    return local_sorted_unique_edge_dst_chunk_start_offsets_
             ? std::optional<raft::device_span<vertex_t const>>{(
                 *local_sorted_unique_edge_dst_chunk_start_offsets_)[partition_idx]}
             : std::nullopt;
  }

  std::optional<vertex_t> local_sorted_unique_edge_dst_chunk_size() const
  {
    return local_sorted_unique_edge_dst_chunk_size_;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::host_span<vertex_t const>>>
  local_sorted_unique_edge_dst_vertex_partition_offsets() const
  {
    return local_sorted_unique_edge_dst_vertex_partition_offsets_;
  }

 private:
  std::vector<edge_t const*> edge_partition_offsets_{};
  std::vector<vertex_t const*> edge_partition_indices_{};

  // relevant only if we use the CSR + DCSR (or CSC + DCSC) hybrid format
  std::optional<std::vector<vertex_t const*>> edge_partition_dcs_nzd_vertices_{};
  std::optional<std::vector<vertex_t>> edge_partition_dcs_nzd_vertex_counts_{};

  std::vector<edge_t> edge_partition_number_of_edges_{};

  partition_t<vertex_t> partition_{};

  // segment offsets based on vertex degree
  std::vector<vertex_t> edge_partition_segment_offsets_{};

  // if valid, store source/destination property values in key/value pairs (this saves memory if #
  // unique edge sources/destinations << V / row_comm_size|col_comm_size).

  std::conditional_t<store_transposed,
                     std::optional<raft::device_span<vertex_t const>>,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>>
    local_sorted_unique_edge_srcs_{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<raft::device_span<vertex_t const>>,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>>
    local_sorted_unique_edge_src_chunk_start_offsets_{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_src_chunk_size_{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<raft::host_span<vertex_t const>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_src_vertex_partition_offsets_{std::nullopt};

  std::conditional_t<store_transposed,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>,
                     std::optional<raft::device_span<vertex_t const>>>
    local_sorted_unique_edge_dsts_{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<std::vector<raft::device_span<vertex_t const>>>,
                     std::optional<raft::device_span<vertex_t const>>>
    local_sorted_unique_edge_dst_chunk_start_offsets_{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_dst_chunk_size_{std::nullopt};
  std::conditional_t<!store_transposed,
                     std::optional<raft::host_span<vertex_t const>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_dst_vertex_partition_offsets_{std::nullopt};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
class graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t> {
 public:
  using vertex_type                           = vertex_t;
  using edge_type                             = edge_t;
  static constexpr bool is_storage_transposed = store_transposed;
  static constexpr bool is_multi_gpu          = multi_gpu;

  graph_view_t(raft::handle_t const& handle,
               edge_t const* offsets,
               vertex_t const* indices,
               graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu> meta);

  std::vector<vertex_t> vertex_partition_range_offsets() const
  {
    return std::vector<vertex_t>{local_vertex_partition_range_first(),
                                 local_vertex_partition_range_last()};
  }

  std::vector<vertex_t> vertex_partition_range_lasts() const
  {
    return std::vector<vertex_t>{local_vertex_partition_range_last()};
  }

  vertex_t local_vertex_partition_range_size() const { return this->number_of_vertices(); }

  constexpr vertex_t local_vertex_partition_range_first() const { return vertex_t{0}; }

  vertex_t local_vertex_partition_range_last() const { return this->number_of_vertices(); }

  vertex_t vertex_partition_range_first(size_t partition_idx) const
  {
    assert(partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t vertex_partition_range_last(size_t partition_idx) const
  {
    assert(partition_idx == 0);
    return this->number_of_vertices();
  }

  vertex_t vertex_partition_range_size(size_t partition_idx) const
  {
    assert(partition_idx == 0);
    return this->number_of_vertices();
  }

  constexpr bool in_local_vertex_partition_range_nocheck(vertex_t v) const { return true; }

  constexpr size_t number_of_local_edge_partitions() const { return size_t(1); }

  edge_t number_of_local_edge_partition_edges(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return this->number_of_edges();
  }

  vertex_t local_edge_partition_src_range_size(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return this->number_of_vertices();
  }

  vertex_t local_edge_partition_src_range_first(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t local_edge_partition_src_range_last(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return this->number_of_vertices();
  }

  vertex_t local_edge_partition_src_value_start_offset(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t local_edge_partition_dst_range_size(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return this->number_of_vertices();
  }

  vertex_t local_edge_partition_dst_range_first(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return vertex_t{0};
  }

  vertex_t local_edge_partition_dst_range_last(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return this->number_of_vertices();
  }

  vertex_t local_edge_partition_dst_value_start_offset(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return vertex_t{0};
  }

  bool use_dcs() const { return false; }

  std::optional<std::vector<vertex_t>> local_edge_partition_segment_offsets(
    size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return segment_offsets_;
  }

  vertex_partition_view_t<vertex_t, false> local_vertex_partition_view() const
  {
    return vertex_partition_view_t<vertex_t, false>(this->number_of_vertices());
  }

  edge_partition_view_t<vertex_t, edge_t, false> local_edge_partition_view(
    size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);  // there is only one edge partition in single-GPU
    return edge_partition_view_t<vertex_t, edge_t, false>(
      raft::device_span<edge_t const>(offsets_, offsets_ + (this->number_of_vertices() + 1)),
      raft::device_span<vertex_t const>(indices_, indices_ + this->number_of_edges()),
      this->number_of_vertices());
  }

  rmm::device_uvector<edge_t> compute_in_degrees(raft::handle_t const& handle) const;
  rmm::device_uvector<edge_t> compute_out_degrees(raft::handle_t const& handle) const;

  edge_t compute_max_in_degree(raft::handle_t const& handle) const;
  edge_t compute_max_out_degree(raft::handle_t const& handle) const;

  edge_t count_self_loops(raft::handle_t const& handle) const;
  edge_t count_multi_edges(raft::handle_t const& handle) const;

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_srcs() const
  {
    return std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_srcs(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_src_chunk_start_offsets() const
  {
    return std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_src_chunk_start_offsets(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return std::nullopt;
  }

  std::optional<vertex_t> local_sorted_unique_edge_src_chunk_size() const { return std::nullopt; }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::host_span<vertex_t const>>>
  local_sorted_unique_edge_src_vertex_partition_offsets() const
  {
    return std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dsts() const
  {
    return std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dsts(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dst_chunk_start_offsets() const
  {
    return std::nullopt;
  }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<transposed, std::optional<raft::device_span<vertex_t const>>>
  local_sorted_unique_edge_dst_chunk_start_offsets(size_t partition_idx = 0) const
  {
    assert(partition_idx == 0);
    return std::nullopt;
  }

  std::optional<vertex_t> local_sorted_unique_edge_dst_chunk_size() const { return std::nullopt; }

  template <bool transposed = is_storage_transposed>
  std::enable_if_t<!transposed, std::optional<raft::host_span<vertex_t const>>>
  local_sorted_unique_edge_dst_vertex_partition_offsets() const
  {
    return std::nullopt;
  }

 private:
  edge_t const* offsets_{nullptr};
  vertex_t const* indices_{nullptr};

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> segment_offsets_{std::nullopt};
};

}  // namespace cugraph
