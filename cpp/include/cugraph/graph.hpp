/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
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

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

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
  std::optional<std::vector<vertex_t>> edge_partition_hypersparse_degree_offsets{};

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

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
struct static_graph_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed>
struct static_graph_t<vertex_t, edge_t, store_transposed, true> {
  static_graph_t(raft::handle_t const&) {}

  std::vector<rmm::device_uvector<edge_t>> edge_partition_offsets{};
  std::vector<rmm::device_uvector<vertex_t>> edge_partition_indices{};

  // nzd: nonzero (local) degree
  std::optional<std::vector<rmm::device_uvector<vertex_t>>> edge_partition_dcs_nzd_vertices{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<uint32_t>>> edge_partition_dcs_nzd_range_bitmaps{
    std::nullopt};
  partition_t<vertex_t> partition{};

  // segment offsets within the vertex partition based on vertex degree
  std::vector<vertex_t> edge_partition_segment_offsets{};
  std::optional<std::vector<vertex_t>> edge_partition_hypersparse_degree_offsets{};

  // if valid, store row/column properties in key/value pairs (this saves memory if # unique edge
  // sources/destinations << V / major_comm_size|minor_comm_size).

  std::conditional_t<store_transposed,
                     std::optional<rmm::device_uvector<vertex_t>>,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>>
    local_sorted_unique_edge_srcs{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<rmm::device_uvector<vertex_t>>,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>>
    local_sorted_unique_edge_src_chunk_start_offsets{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_src_chunk_size{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<std::vector<vertex_t>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_src_vertex_partition_offsets{std::nullopt};

  std::conditional_t<store_transposed,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>,
                     std::optional<rmm::device_uvector<vertex_t>>>
    local_sorted_unique_edge_dsts{std::nullopt};
  std::conditional_t<store_transposed,
                     std::optional<std::vector<rmm::device_uvector<vertex_t>>>,
                     std::optional<rmm::device_uvector<vertex_t>>>
    local_sorted_unique_edge_dst_chunk_start_offsets{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_dst_chunk_size{std::nullopt};
  std::conditional_t<!store_transposed,
                     std::optional<std::vector<vertex_t>>,
                     std::optional<std::byte> /* dummy */>
    local_sorted_unique_edge_dst_vertex_partition_offsets{std::nullopt};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed>
struct static_graph_t<vertex_t, edge_t, store_transposed, false> {
  static_graph_t(raft::handle_t const& handle)
    : offsets(0, handle.get_stream()), indices(0, handle.get_stream())
  {
  }

  rmm::device_uvector<edge_t> offsets;
  rmm::device_uvector<vertex_t> indices;

  // segment offsets based on vertex degree, relevant only if sorted_by_global_degree is true
  std::optional<std::vector<vertex_t>> segment_offsets{};
};

// placeholder, dynamic graphs are not supported yet
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
struct dynamic_graph_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed>
struct dynamic_graph_t<vertex_t, edge_t, store_transposed, true> {
  dynamic_graph_t(raft::handle_t const&) {}
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed>
struct dynamic_graph_t<vertex_t, edge_t, store_transposed, false> {
  dynamic_graph_t(raft::handle_t const&) {}
};

}  // namespace detail

// graph_t is an owning graph class (note that graph_view_t is a non-owning graph class)
template <typename vertex_t,
          typename edge_t,
          bool store_transposed,
          bool multi_gpu,
          bool dynamic = false>
class graph_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool dynamic>
class graph_t<vertex_t, edge_t, store_transposed, true, dynamic>
  : public detail::graph_base_t<vertex_t, edge_t> {
 public:
  using vertex_type                           = vertex_t;
  using edge_type                             = edge_t;
  static constexpr bool is_storage_transposed = store_transposed;
  static constexpr bool is_multi_gpu          = true;
  static constexpr bool is_dynamic            = dynamic;

  graph_t(raft::handle_t const& handle) : detail::graph_base_t<vertex_t, edge_t>(), graph_(handle)
  {
  }

  graph_t(
    raft::handle_t const& handle,
    std::vector<rmm::device_uvector<edge_t>>&& edge_partition_offsets,
    std::vector<rmm::device_uvector<vertex_t>>&& edge_partition_indices,
    std::optional<std::vector<rmm::device_uvector<vertex_t>>>&& edge_partition_dcs_nzd_vertices,
    detail::graph_meta_t<vertex_t, edge_t, true> meta,
    bool do_expensive_check = false) requires(!dynamic);

  edge_t number_of_edges() const { return this->number_of_edges_; }

  graph_view_t<vertex_t, edge_t, store_transposed, true> view() const requires(!dynamic)
  {
    std::vector<raft::device_span<edge_t const>> offsets(graph_.edge_partition_offsets.size());
    std::vector<raft::device_span<vertex_t const>> indices(graph_.edge_partition_indices.size());
    auto dcs_nzd_vertices = graph_.edge_partition_dcs_nzd_vertices
                              ? std::make_optional<std::vector<raft::device_span<vertex_t const>>>(
                                  graph_.edge_partition_dcs_nzd_vertices->size())
                              : std::nullopt;
    auto dcs_nzd_range_bitmaps =
      graph_.edge_partition_dcs_nzd_range_bitmaps
        ? std::make_optional<std::vector<raft::device_span<uint32_t const>>>(
            graph_.edge_partition_dcs_nzd_range_bitmaps->size())
        : std::nullopt;
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] = raft::device_span<edge_t const>(graph_.edge_partition_offsets[i].data(),
                                                   graph_.edge_partition_offsets[i].size());
      indices[i] = raft::device_span<vertex_t const>(graph_.edge_partition_indices[i].data(),
                                                     graph_.edge_partition_indices[i].size());
      if (dcs_nzd_vertices) {
        (*dcs_nzd_vertices)[i] =
          raft::device_span<vertex_t const>((*graph_.edge_partition_dcs_nzd_vertices)[i].data(),
                                            (*graph_.edge_partition_dcs_nzd_vertices)[i].size());
      }
      if (dcs_nzd_range_bitmaps) {
        (*dcs_nzd_range_bitmaps)[i] = raft::device_span<uint32_t const>(
          (*graph_.edge_partition_dcs_nzd_range_bitmaps)[i].data(),
          (*graph_.edge_partition_dcs_nzd_range_bitmaps)[i].size());
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

    if (graph_.local_sorted_unique_edge_srcs) {
      if constexpr (store_transposed) {  // minor
        local_sorted_unique_edge_srcs =
          raft::device_span<vertex_t const>(graph_.local_sorted_unique_edge_srcs->begin(),
                                            graph_.local_sorted_unique_edge_srcs->end());
        local_sorted_unique_edge_src_chunk_start_offsets = raft::device_span<vertex_t const>(
          graph_.local_sorted_unique_edge_src_chunk_start_offsets->begin(),
          graph_.local_sorted_unique_edge_src_chunk_start_offsets->end());
        local_sorted_unique_edge_src_vertex_partition_offsets = raft::host_span<vertex_t const>(
          graph_.local_sorted_unique_edge_src_vertex_partition_offsets->data(),
          graph_.local_sorted_unique_edge_src_vertex_partition_offsets->data() +
            graph_.local_sorted_unique_edge_src_vertex_partition_offsets->size());
      } else {  // major
        local_sorted_unique_edge_srcs = std::vector<raft::device_span<vertex_t const>>(
          graph_.local_sorted_unique_edge_srcs->size());
        local_sorted_unique_edge_src_chunk_start_offsets =
          std::vector<raft::device_span<vertex_t const>>(
            graph_.local_sorted_unique_edge_src_chunk_start_offsets->size());
        for (size_t i = 0; i < local_sorted_unique_edge_srcs->size(); ++i) {
          (*local_sorted_unique_edge_srcs)[i] =
            raft::device_span<vertex_t const>((*graph_.local_sorted_unique_edge_srcs)[i].begin(),
                                              (*graph_.local_sorted_unique_edge_srcs)[i].end());
          (*local_sorted_unique_edge_src_chunk_start_offsets)[i] =
            raft::device_span<vertex_t const>(
              (*graph_.local_sorted_unique_edge_src_chunk_start_offsets)[i].begin(),
              (*graph_.local_sorted_unique_edge_src_chunk_start_offsets)[i].end());
        }
      }
    }

    if (graph_.local_sorted_unique_edge_dsts) {
      if constexpr (store_transposed) {  // major
        local_sorted_unique_edge_dsts = std::vector<raft::device_span<vertex_t const>>(
          graph_.local_sorted_unique_edge_dsts->size());
        local_sorted_unique_edge_dst_chunk_start_offsets =
          std::vector<raft::device_span<vertex_t const>>(
            graph_.local_sorted_unique_edge_dst_chunk_start_offsets->size());
        for (size_t i = 0; i < local_sorted_unique_edge_dsts->size(); ++i) {
          (*local_sorted_unique_edge_dsts)[i] =
            raft::device_span<vertex_t const>((*graph_.local_sorted_unique_edge_dsts)[i].begin(),
                                              (*graph_.local_sorted_unique_edge_dsts)[i].end());
          (*local_sorted_unique_edge_dst_chunk_start_offsets)[i] =
            raft::device_span<vertex_t const>(
              (*graph_.local_sorted_unique_edge_dst_chunk_start_offsets)[i].begin(),
              (*graph_.local_sorted_unique_edge_dst_chunk_start_offsets)[i].end());
        }
      } else {  // minor
        local_sorted_unique_edge_dsts =
          raft::device_span<vertex_t const>(graph_.local_sorted_unique_edge_dsts->begin(),
                                            graph_.local_sorted_unique_edge_dsts->end());
        local_sorted_unique_edge_dst_chunk_start_offsets = raft::device_span<vertex_t const>(
          graph_.local_sorted_unique_edge_dst_chunk_start_offsets->begin(),
          graph_.local_sorted_unique_edge_dst_chunk_start_offsets->end());
        local_sorted_unique_edge_dst_vertex_partition_offsets = raft::host_span<vertex_t const>(
          graph_.local_sorted_unique_edge_dst_vertex_partition_offsets->data(),
          graph_.local_sorted_unique_edge_dst_vertex_partition_offsets->data() +
            graph_.local_sorted_unique_edge_dst_vertex_partition_offsets->size());
      }
    }

    return graph_view_t<vertex_t, edge_t, store_transposed, true>(
      offsets,
      indices,
      dcs_nzd_vertices,
      dcs_nzd_range_bitmaps,
      graph_view_meta_t<vertex_t, edge_t, store_transposed, true>{
        this->number_of_vertices(),
        this->number_of_edges(),
        this->properties_,
        graph_.partition,
        graph_.edge_partition_segment_offsets,
        graph_.edge_partition_hypersparse_degree_offsets,
        local_sorted_unique_edge_srcs,
        local_sorted_unique_edge_src_chunk_start_offsets,
        graph_.local_sorted_unique_edge_src_chunk_size,
        local_sorted_unique_edge_src_vertex_partition_offsets,
        local_sorted_unique_edge_dsts,
        local_sorted_unique_edge_dst_chunk_start_offsets,
        graph_.local_sorted_unique_edge_dst_chunk_size,
        local_sorted_unique_edge_dst_vertex_partition_offsets});
  }

 private:
  std::conditional_t<dynamic,
                     detail::dynamic_graph_t<vertex_t, edge_t, store_transposed, true>,
                     detail::static_graph_t<vertex_t, edge_t, store_transposed, true>>
    graph_;
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool store_transposed, bool dynamic>
class graph_t<vertex_t, edge_t, store_transposed, false, dynamic>
  : public detail::graph_base_t<vertex_t, edge_t> {
 public:
  using vertex_type                           = vertex_t;
  using edge_type                             = edge_t;
  static constexpr bool is_storage_transposed = store_transposed;
  static constexpr bool is_multi_gpu          = false;
  static constexpr bool is_dynamic            = dynamic;

  graph_t(raft::handle_t const& handle) : detail::graph_base_t<vertex_t, edge_t>(), graph_(handle)
  {
  }

  graph_t(raft::handle_t const& handle,
          rmm::device_uvector<edge_t>&& offsets,
          rmm::device_uvector<vertex_t>&& indices,
          detail::graph_meta_t<vertex_t, edge_t, false> meta,
          bool do_expensive_check = false) requires(!dynamic);

  edge_t number_of_edges() const { return this->number_of_edges_; }

  graph_view_t<vertex_t, edge_t, store_transposed, false> view() const requires(!dynamic)
  {
    return graph_view_t<vertex_t, edge_t, store_transposed, false>(
      raft::device_span<edge_t const>(graph_.offsets.data(), graph_.offsets.size()),
      raft::device_span<vertex_t const>(graph_.indices.data(), graph_.indices.size()),
      graph_view_meta_t<vertex_t, edge_t, store_transposed, false>{this->number_of_vertices(),
                                                                   this->number_of_edges(),
                                                                   this->properties_,
                                                                   graph_.segment_offsets});
  }

 private:
  std::conditional_t<dynamic,
                     detail::dynamic_graph_t<vertex_t, edge_t, store_transposed, false>,
                     detail::static_graph_t<vertex_t, edge_t, store_transposed, false>>
    graph_;
};

template <typename T, typename Enable = void>
struct invalid_idx;

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>>
  : std::integral_constant<T, -1> {};

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>>
  : std::integral_constant<T, std::numeric_limits<T>::max()> {};

template <typename vertex_t>
struct invalid_vertex_id : invalid_idx<vertex_t> {};

template <typename vertex_t>
inline constexpr vertex_t invalid_vertex_id_v = invalid_vertex_id<vertex_t>::value;

template <typename edge_t>
struct invalid_edge_id : invalid_idx<edge_t> {};

template <typename edge_t>
inline constexpr edge_t invalid_edge_id_v = invalid_edge_id<edge_t>::value;

template <typename component_t>
struct invalid_component_id : invalid_idx<component_t> {};

template <typename component_t>
inline constexpr component_t invalid_component_id_v = invalid_component_id<component_t>::value;

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

}  // namespace CUGRAPH_EXPORT cugraph

/**
 * @}
 */
