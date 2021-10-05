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

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t>
struct edgelist_t {
  vertex_t const* p_src_vertices{nullptr};
  vertex_t const* p_dst_vertices{nullptr};
  std::optional<weight_t const*> p_edge_weights{std::nullopt};
  edge_t number_of_edges{0};
};

template <typename vertex_t, typename edge_t, bool multi_gpu, typename Enable = void>
struct graph_meta_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
struct graph_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>> {
  vertex_t number_of_vertices{};
  edge_t number_of_edges{};

  bool store_transposed{};
  graph_properties_t properties{};

  partition_t<vertex_t> partition{};

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> segment_offsets{std::nullopt};

  vertex_t num_local_unique_edge_rows{};
  vertex_t num_local_unique_edge_cols{};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
struct graph_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>> {
  vertex_t number_of_vertices{};

  bool store_transposed{};
  graph_properties_t properties{};

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> segment_offsets{std::nullopt};
};

// graph_t is an owning graph class (note that graph_view_t is a non-owning graph class)
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename Enable = void>
class graph_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
class graph_t<vertex_t, edge_t, weight_t, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t, weight_t> {
 public:
  using vertex_type                  = vertex_t;
  using edge_type                    = edge_t;
  using weight_type                  = weight_t;
  static constexpr bool is_multi_gpu = multi_gpu;

  graph_t(raft::handle_t const& handle) : detail::graph_base_t<vertex_t, edge_t, weight_t>() {}

  graph_t(raft::handle_t const& handle,
          std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const& edgelists,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check = false);

  bool is_weighted() const { return adj_matrix_partition_weights_.has_value(); }

  graph_view_t<vertex_t, edge_t, weight_t, multi_gpu> view() const
  {
    std::vector<edge_t const*> offsets(adj_matrix_partition_offsets_.size(), nullptr);
    std::vector<vertex_t const*> indices(adj_matrix_partition_indices_.size(), nullptr);
    auto weights          = adj_matrix_partition_weights_
                              ? std::make_optional<std::vector<weight_t const*>>(
                         (*adj_matrix_partition_weights_).size(), nullptr)
                              : std::nullopt;
    auto dcs_nzd_vertices = adj_matrix_partition_dcs_nzd_vertices_
                              ? std::make_optional<std::vector<vertex_t const*>>(
                                  (*adj_matrix_partition_dcs_nzd_vertices_).size(), nullptr)
                              : std::nullopt;
    auto dcs_nzd_vertex_counts =
      adj_matrix_partition_dcs_nzd_vertex_counts_
        ? std::make_optional<std::vector<vertex_t>>(
            (*adj_matrix_partition_dcs_nzd_vertex_counts_).size(), vertex_t{0})
        : std::nullopt;
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] = adj_matrix_partition_offsets_[i].data();
      indices[i] = adj_matrix_partition_indices_[i].data();
      if (weights) { (*weights)[i] = (*adj_matrix_partition_weights_)[i].data(); }
      if (dcs_nzd_vertices) {
        (*dcs_nzd_vertices)[i]      = (*adj_matrix_partition_dcs_nzd_vertices_)[i].data();
        (*dcs_nzd_vertex_counts)[i] = (*adj_matrix_partition_dcs_nzd_vertex_counts_)[i];
      }
    }

    return graph_view_t<vertex_t, edge_t, weight_t, multi_gpu>(
      *(this->get_handle_ptr()),
      offsets,
      indices,
      weights,
      dcs_nzd_vertices,
      dcs_nzd_vertex_counts,
      graph_view_meta_t<vertex_t, edge_t, multi_gpu>{
        this->get_number_of_vertices(),
        this->get_number_of_edges(),
        this->get_graph_properties(),
        partition_,
        adj_matrix_partition_segment_offsets_,
        local_sorted_unique_edge_rows_
          ? std::optional<vertex_t const*>{(*local_sorted_unique_edge_rows_).data()}
          : std::nullopt,
        local_sorted_unique_edge_rows_
          ? std::optional<vertex_t const*>{(*local_sorted_unique_edge_rows_).data() +
                                           (*local_sorted_unique_edge_rows_).size()}
          : std::nullopt,
        local_sorted_unique_edge_row_offsets_,
        local_sorted_unique_edge_cols_
          ? std::optional<vertex_t const*>{(*local_sorted_unique_edge_cols_).data()}
          : std::nullopt,
        local_sorted_unique_edge_cols_
          ? std::optional<vertex_t const*>{(*local_sorted_unique_edge_cols_).data() +
                                           (*local_sorted_unique_edge_cols_).size()}
          : std::nullopt,
        local_sorted_unique_edge_col_offsets_,
      },
      false);
  }

 private:
  std::vector<rmm::device_uvector<edge_t>> adj_matrix_partition_offsets_{};
  std::vector<rmm::device_uvector<vertex_t>> adj_matrix_partition_indices_{};
  std::optional<std::vector<rmm::device_uvector<weight_t>>> adj_matrix_partition_weights_{
    std::nullopt};

  // nzd: nonzero (local) degree, relevant only if segment_offsets.size() > 0
  std::optional<std::vector<rmm::device_uvector<vertex_t>>> adj_matrix_partition_dcs_nzd_vertices_{
    std::nullopt};
  std::optional<std::vector<vertex_t>> adj_matrix_partition_dcs_nzd_vertex_counts_{std::nullopt};
  partition_t<vertex_t> partition_{};

  // segment offsets within the vertex partition based on vertex degree, relevant only if
  // segment_offsets.size() > 0
  std::optional<std::vector<vertex_t>> adj_matrix_partition_segment_offsets_{std::nullopt};

  // if valid, store row/column properties in key/value pairs (this saves memory if # unique edge
  // rows/cols << V / row_comm_size|col_comm_size).
  std::optional<rmm::device_uvector<vertex_t>> local_sorted_unique_edge_rows_{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> local_sorted_unique_edge_cols_{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_row_offsets_{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_col_offsets_{std::nullopt};
};

// single-GPU version
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
class graph_t<vertex_t, edge_t, weight_t, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t, weight_t> {
 public:
  using vertex_type                  = vertex_t;
  using edge_type                    = edge_t;
  using weight_type                  = weight_t;
  static constexpr bool is_multi_gpu = multi_gpu;

  graph_t(raft::handle_t const& handle)
    : detail::graph_base_t<vertex_t, edge_t, weight_t>(),
      offsets_(0, handle.get_stream()),
      indices_(0, handle.get_stream()){};

  graph_t(raft::handle_t const& handle,
          edgelist_t<vertex_t, edge_t, weight_t> const& edgelist,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check = false);

  bool is_weighted() const { return weights_.has_value(); }

  graph_view_t<vertex_t, edge_t, weight_t, multi_gpu> view() const
  {
    return graph_view_t<vertex_t, edge_t, weight_t, multi_gpu>(
      *(this->get_handle_ptr()),
      offsets_.data(),
      indices_.data(),
      weights_ ? std::optional<weight_t const*>{(*weights_).data()} : std::nullopt,
      graph_view_meta_t<vertex_t, edge_t, multi_gpu>{this->get_number_of_vertices(),
                                                     this->get_number_of_edges(),
                                                     this->get_graph_properties(),
                                                     segment_offsets_},
      false);
  }

  // FIXME: possibley to be added later;
  // for now it's unnecessary;
  // (commented out, per reviewer request)
  //
  // generic in-place sorter on CSR structure;
  // this is specifically targetting
  // segmented-sort by weigths; but
  // must be generic enough to support future
  // types of sorting;
  // Notes:
  // (1.) criterion is mutable (non-const)
  //      to allow for sorter obejcts for which
  //      the sorting operation fills additional structures
  //      (later to be retrieved; e.g., for debugging);
  // (2.) sorting object is responsible for updating "in-place"
  //      any of the (offsets, indices, weights) arrays;
  //
  // template <typename in_place_sorter_t>
  // void sort(in_place_sorter_t& criterion)
  // {
  //   criterion(offsets_, indices_, weights_);
  // }

 private:
  friend class cugraph::serializer::serializer_t;

  // cnstr. to be used _only_ for un/serialization purposes:
  //
  graph_t(raft::handle_t const& handle,
          vertex_t number_of_vertices,
          edge_t number_of_edges,
          graph_properties_t properties,
          rmm::device_uvector<edge_t>&& offsets,
          rmm::device_uvector<vertex_t>&& indices,
          std::optional<rmm::device_uvector<weight_t>>&& weights,
          std::optional<std::vector<vertex_t>>&& segment_offsets)
    : detail::graph_base_t<vertex_t, edge_t, weight_t>(
        handle, number_of_vertices, number_of_edges, properties),
      offsets_(std::move(offsets)),
      indices_(std::move(indices)),
      weights_(std::move(weights)),
      segment_offsets_(std::move(segment_offsets))
  {
  }

  rmm::device_uvector<edge_t> offsets_;
  rmm::device_uvector<vertex_t> indices_;
  std::optional<rmm::device_uvector<weight_t>> weights_{std::nullopt};

  // segment offsets based on vertex degree, relevant only if sorted_by_global_degree is true
  std::optional<std::vector<vertex_t>> segment_offsets_{};
};

template <typename T, typename Enable = void>
struct invalid_idx;

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>>
  : std::integral_constant<T, -1> {
};

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>>
  : std::integral_constant<T, std::numeric_limits<T>::max()> {
};

template <typename vertex_t>
struct invalid_vertex_id : invalid_idx<vertex_t> {
};

template <typename edge_t>
struct invalid_edge_id : invalid_idx<edge_t> {
};

template <typename vertex_t>
struct invalid_component_id : invalid_idx<vertex_t> {
};

template <typename vertex_t>
__host__ __device__ std::enable_if_t<std::is_signed<vertex_t>::value, bool> is_valid_vertex(
  vertex_t num_vertices, vertex_t v)
{
  return (v >= 0) && (v < num_vertices);
}

template <typename vertex_t>
__host__ __device__ std::enable_if_t<std::is_unsigned<vertex_t>::value, bool> is_valid_vertex(
  vertex_t num_vertices, vertex_t v)
{
  return v < num_vertices;
}

}  // namespace cugraph

#include "eidecl_graph.hpp"
