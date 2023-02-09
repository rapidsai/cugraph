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

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

/** @defgroup cpp_api cuGraph C++ API
 *  @{
 */

namespace cugraph {

template <typename vertex_t, typename edge_t, bool multi_gpu, typename Enable = void>
struct graph_meta_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
struct graph_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>> {
  vertex_t number_of_vertices{};
  edge_t number_of_edges{};
  graph_properties_t properties{};

  partition_t<vertex_t> partition{};

  std::vector<vertex_t> edge_partition_segment_offsets{};

  vertex_t num_local_unique_edge_srcs{};
  vertex_t num_local_unique_edge_dsts{};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
struct graph_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>> {
  vertex_t number_of_vertices{};
  graph_properties_t properties{};

  // segment offsets based on vertex degree, relevant only if vertex IDs are renumbered
  std::optional<std::vector<vertex_t>> segment_offsets{std::nullopt};
};

// graph_t is an owning graph class (note that graph_view_t is a non-owning graph class)
template <typename vertex_t,
          typename edge_t,
          bool store_transposed,
          bool multi_gpu,
          typename Enable = void>
class graph_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
class graph_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t> {
 public:
  using vertex_type                           = vertex_t;
  using edge_type                             = edge_t;
  static constexpr bool is_storage_transposed = store_transposed;
  static constexpr bool is_multi_gpu          = multi_gpu;

  graph_t(raft::handle_t const& handle) : detail::graph_base_t<vertex_t, edge_t>() {}

  graph_t(
    raft::handle_t const& handle,
    std::vector<rmm::device_uvector<edge_t>>&& edge_partition_offsets,
    std::vector<rmm::device_uvector<vertex_t>>&& edge_partition_indices,
    std::optional<std::vector<rmm::device_uvector<vertex_t>>>&& edge_partition_dcs_nzd_vertices,
    graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
    bool do_expensive_check = false);

  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> view() const
  {
    std::vector<edge_t const*> offsets(edge_partition_offsets_.size(), nullptr);
    std::vector<vertex_t const*> indices(edge_partition_indices_.size(), nullptr);
    auto dcs_nzd_vertices      = edge_partition_dcs_nzd_vertices_
                                   ? std::make_optional<std::vector<vertex_t const*>>(
                                  (*edge_partition_dcs_nzd_vertices_).size(), nullptr)
                                   : std::nullopt;
    auto dcs_nzd_vertex_counts = edge_partition_dcs_nzd_vertex_counts_
                                   ? std::make_optional<std::vector<vertex_t>>(
                                       (*edge_partition_dcs_nzd_vertex_counts_).size(), vertex_t{0})
                                   : std::nullopt;
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] = edge_partition_offsets_[i].data();
      indices[i] = edge_partition_indices_[i].data();
      if (dcs_nzd_vertices) {
        (*dcs_nzd_vertices)[i]      = (*edge_partition_dcs_nzd_vertices_)[i].data();
        (*dcs_nzd_vertex_counts)[i] = (*edge_partition_dcs_nzd_vertex_counts_)[i];
      }
    }

    std::conditional_t<store_transposed,
                       std::optional<raft::device_span<vertex_t const>>,
                       std::optional<std::vector<raft::device_span<vertex_t const>>>>
      local_sorted_unique_edge_srcs{std::nullopt};
    std::conditional_t<store_transposed,
                       std::optional<raft::device_span<vertex_t const>>,
                       std::optional<std::vector<raft::device_span<vertex_t const>>>>
      local_sorted_unique_edge_src_chunk_start_offsets{std::nullopt};
    std::conditional_t<store_transposed,
                       std::optional<raft::host_span<vertex_t const>>,
                       std::optional<std::byte>>
      local_sorted_unique_edge_src_vertex_partition_offsets{std::nullopt};

    std::conditional_t<store_transposed,
                       std::optional<std::vector<raft::device_span<vertex_t const>>>,
                       std::optional<raft::device_span<vertex_t const>>>
      local_sorted_unique_edge_dsts{std::nullopt};
    std::conditional_t<store_transposed,
                       std::optional<std::vector<raft::device_span<vertex_t const>>>,
                       std::optional<raft::device_span<vertex_t const>>>
      local_sorted_unique_edge_dst_chunk_start_offsets{std::nullopt};
    std::conditional_t<!store_transposed,
                       std::optional<raft::host_span<vertex_t const>>,
                       std::optional<std::byte>>
      local_sorted_unique_edge_dst_vertex_partition_offsets{std::nullopt};

    if (local_sorted_unique_edge_srcs_) {
      if constexpr (store_transposed) {  // minor
        local_sorted_unique_edge_srcs = raft::device_span<vertex_t const>(
          (*local_sorted_unique_edge_srcs_).begin(), (*local_sorted_unique_edge_srcs_).end());
        local_sorted_unique_edge_src_chunk_start_offsets = raft::device_span<vertex_t const>(
          (*local_sorted_unique_edge_src_chunk_start_offsets_).begin(),
          (*local_sorted_unique_edge_src_chunk_start_offsets_).end());
        local_sorted_unique_edge_src_vertex_partition_offsets = raft::host_span<vertex_t const>(
          (*local_sorted_unique_edge_src_vertex_partition_offsets_).data(),
          (*local_sorted_unique_edge_src_vertex_partition_offsets_).data() +
            (*local_sorted_unique_edge_src_vertex_partition_offsets_).size());
      } else {  // major
        local_sorted_unique_edge_srcs =
          std::vector<raft::device_span<vertex_t const>>((*local_sorted_unique_edge_srcs_).size());
        local_sorted_unique_edge_src_chunk_start_offsets =
          std::vector<raft::device_span<vertex_t const>>(
            (*local_sorted_unique_edge_src_chunk_start_offsets_).size());
        for (size_t i = 0; i < (*local_sorted_unique_edge_srcs).size(); ++i) {
          (*local_sorted_unique_edge_srcs)[i] =
            raft::device_span<vertex_t const>((*local_sorted_unique_edge_srcs_)[i].begin(),
                                              (*local_sorted_unique_edge_srcs_)[i].end());
          (*local_sorted_unique_edge_src_chunk_start_offsets)[i] =
            raft::device_span<vertex_t const>(
              (*local_sorted_unique_edge_src_chunk_start_offsets_)[i].begin(),
              (*local_sorted_unique_edge_src_chunk_start_offsets_)[i].end());
        }
      }
    }

    if (local_sorted_unique_edge_dsts_) {
      if constexpr (store_transposed) {  // major
        local_sorted_unique_edge_dsts =
          std::vector<raft::device_span<vertex_t const>>((*local_sorted_unique_edge_dsts_).size());
        local_sorted_unique_edge_dst_chunk_start_offsets =
          std::vector<raft::device_span<vertex_t const>>(
            (*local_sorted_unique_edge_dst_chunk_start_offsets_).size());
        for (size_t i = 0; i < (*local_sorted_unique_edge_dsts).size(); ++i) {
          (*local_sorted_unique_edge_dsts)[i] =
            raft::device_span<vertex_t const>((*local_sorted_unique_edge_dsts_)[i].begin(),
                                              (*local_sorted_unique_edge_dsts_)[i].end());
          (*local_sorted_unique_edge_dst_chunk_start_offsets)[i] =
            raft::device_span<vertex_t const>(
              (*local_sorted_unique_edge_dst_chunk_start_offsets_)[i].begin(),
              (*local_sorted_unique_edge_dst_chunk_start_offsets_)[i].end());
        }
      } else {  // minor
        local_sorted_unique_edge_dsts = raft::device_span<vertex_t const>(
          (*local_sorted_unique_edge_dsts_).begin(), (*local_sorted_unique_edge_dsts_).end());
        local_sorted_unique_edge_dst_chunk_start_offsets = raft::device_span<vertex_t const>(
          (*local_sorted_unique_edge_dst_chunk_start_offsets_).begin(),
          (*local_sorted_unique_edge_dst_chunk_start_offsets_).end());
        local_sorted_unique_edge_dst_vertex_partition_offsets = raft::host_span<vertex_t const>(
          (*local_sorted_unique_edge_dst_vertex_partition_offsets_).data(),
          (*local_sorted_unique_edge_dst_vertex_partition_offsets_).data() +
            (*local_sorted_unique_edge_dst_vertex_partition_offsets_).size());
      }
    }

    return graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>(
      *(this->handle_ptr()),
      offsets,
      indices,
      dcs_nzd_vertices,
      dcs_nzd_vertex_counts,
      graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu>{
        this->number_of_vertices(),
        this->number_of_edges(),
        this->graph_properties(),
        partition_,
        edge_partition_segment_offsets_,
        local_sorted_unique_edge_srcs,
        local_sorted_unique_edge_src_chunk_start_offsets,
        local_sorted_unique_edge_src_chunk_size_,
        local_sorted_unique_edge_src_vertex_partition_offsets,
        local_sorted_unique_edge_dsts,
        local_sorted_unique_edge_dst_chunk_start_offsets,
        local_sorted_unique_edge_dst_chunk_size_,
        local_sorted_unique_edge_dst_vertex_partition_offsets});
  }

 private:
  std::vector<rmm::device_uvector<edge_t>> edge_partition_offsets_{};
  std::vector<rmm::device_uvector<vertex_t>> edge_partition_indices_{};

  // nzd: nonzero (local) degree
  std::optional<std::vector<rmm::device_uvector<vertex_t>>> edge_partition_dcs_nzd_vertices_{
    std::nullopt};
  std::optional<std::vector<vertex_t>> edge_partition_dcs_nzd_vertex_counts_{std::nullopt};
  partition_t<vertex_t> partition_{};

  // segment offsets within the vertex partition based on vertex degree
  std::vector<vertex_t> edge_partition_segment_offsets_{};

  // if valid, store row/column properties in key/value pairs (this saves memory if # unique edge
  // rows/cols << V / row_comm_size|col_comm_size).

  std::conditional_t<store_transposed,
                     std::optional<rmm::device_uvector<vertex_t>>,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>>
    local_sorted_unique_edge_srcs_{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<rmm::device_uvector<vertex_t>>,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>>
    local_sorted_unique_edge_src_chunk_start_offsets_{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_src_chunk_size_{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<std::vector<vertex_t>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_src_vertex_partition_offsets_{std::nullopt};

  std::conditional_t<store_transposed,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>,
                     std::optional<rmm::device_uvector<vertex_t>>>
    local_sorted_unique_edge_dsts_{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>,
                     std::optional<rmm::device_uvector<vertex_t>>>
    local_sorted_unique_edge_dst_chunk_start_offsets_{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_dst_chunk_size_{std::nullopt};
  std::conditional_t<!store_transposed,
                     std::optional<std::vector<vertex_t>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_dst_vertex_partition_offsets_{std::nullopt};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
class graph_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::graph_base_t<vertex_t, edge_t> {
 public:
  using vertex_type                           = vertex_t;
  using edge_type                             = edge_t;
  static constexpr bool is_storage_transposed = store_transposed;
  static constexpr bool is_multi_gpu          = multi_gpu;

  graph_t(raft::handle_t const& handle)
    : detail::graph_base_t<vertex_t, edge_t>(),
      offsets_(0, handle.get_stream()),
      indices_(0, handle.get_stream()){};

  graph_t(raft::handle_t const& handle,
          rmm::device_uvector<edge_t>&& offsets,
          rmm::device_uvector<vertex_t>&& indices,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check = false);

  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> view() const
  {
    return graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>(
      *(this->handle_ptr()),
      offsets_.data(),
      indices_.data(),
      graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu>{this->number_of_vertices(),
                                                                       this->number_of_edges(),
                                                                       this->graph_properties(),
                                                                       segment_offsets_});
  }

 private:
  rmm::device_uvector<edge_t> offsets_;
  rmm::device_uvector<vertex_t> indices_;

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

/**
 * @}
 */
