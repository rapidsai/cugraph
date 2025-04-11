/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

/** @defgroup shuffle_functions_cpp C++ Vertex/Edge Shuffle Funtions
 */

namespace cugraph {

/**
 * @ingroup shuffle_functions_cpp
 * @brief Shuffle external vertex IDs to the owning GPUs (by vertex partitioning)
 *
 * @deprecated Replaced with shuffle_ext_vertices
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  List of vertex ids
 * @return Vector of vertex ids mapped to this GPU.
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_external_vertices(raft::handle_t const& handle,
                                                        rmm::device_uvector<vertex_t>&& vertices);

/**
 * @ingroup shuffle_functions_cpp
 * @brief Shuffle external vertex IDs to the owning GPUs (by vertex partitioning)
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  Vector of vertex ids
 * @return Vector of vertex ids mapped to this GPU.
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices(raft::handle_t const& handle,
                                                   rmm::device_uvector<vertex_t>&& vertices);

/**
 * @ingroup shuffle_functions_cpp
 * @brief Shuffle external vertex ID & value pairs to the owning GPUs (by vertex partitioning)
 *
 * @deprecated Replaced with shuffle_ext_vertex_value_pairs
 *
 * @tparam vertex_t   Type of vertex identifiers. Needs to be an integral type.
 * @tparam value_t    Type of values. currently supported types are int32_t,
 * int64_t, size_t, float and double.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  List of vertex ids
 * @param values List of values
 * @return Tuple of vectors storing vertex ids and values mapped to this GPU.
 */
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_external_vertex_value_pairs(raft::handle_t const& handle,
                                    rmm::device_uvector<vertex_t>&& vertices,
                                    rmm::device_uvector<value_t>&& values);

/**
 * @ingroup shuffle_functions_cpp
 * @brief Shuffle external vertex ID & value pairs to the owning GPUs (by vertex partitioning)
 *
 * @tparam vertex_t   Type of vertex identifiers. Needs to be an integral type.
 * @tparam value_t    Type of values. currently supported types are int32_t, int64_t, size_t, float
 * and double.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  Vector of vertex ids
 * @param values Vector of values
 * @return Tuple of vectors storing vertex ids and values mapped to this GPU.
 */
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<vertex_t>&& vertices,
                               rmm::device_uvector<value_t>&& values);

/**
 * @ingroup graph_functions_cpp
 * @brief Shuffle external edges to the owning GPUs (by edge partitioning)
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t      Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t    Type of edge weight. Currently float and double are supported.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type, currently only int32_t is
 * supported.
 * @tparam edge_time_t Type of edge time.  Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  Vector of source vertex ids
 * @param edge_dsts  Vector of destination vertex ids
 * @param edge_weights  Optional vector of edge weights
 * @param edge_ids  Optional vector of edge ids
 * @param edge_types Optional vector of edge types
 * @param edge_start_times Optional vector of edge start times
 * @param edge_end_times Optional vector of edge end times
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Tuple of vectors storing edge sources, destinations, optional weights,
 *          optional edge ids, optional edge types, optional edge start times, optional edge end
 * times mapped to this GPU and a vector storing the number of edges received from each GPU.
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::vector<size_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edge_srcs,
                  rmm::device_uvector<vertex_t>&& edge_dsts,
                  std::optional<rmm::device_uvector<weight_t>>&& edge_weights,
                  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
                  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
                  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
                  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
                  bool store_transposed);

/**
 * @brief Shuffle local edge sources (already placed by edge partitioning) to the owning GPUs (by
 * vertex partitioning).
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  Vector of local edge source IDs
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Vector of shuffled edge source vertex IDs (shuffled by vertex partitioning).
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_local_edge_srcs(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_srcs,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  bool store_transposed);

/**
 * @brief Shuffle local edge source & value pairs (already placed by edge partitioning) to the
 * owning GPUs (by vertex partitioning).
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  Vector of local edge source IDs
 * @param edge_values  Vector of edge values
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Tuple of vectors storing shuffled edge source vertex IDs and values (shuffled by vertex
 * partitioning).
 */
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, dataframe_buffer_type_t<value_t>>
shuffle_local_edge_src_value_pairs(raft::handle_t const& handle,
                                   rmm::device_uvector<vertex_t>&& edge_srcs,
                                   dataframe_buffer_type_t<value_t>&& edge_values,
                                   raft::host_span<vertex_t const> vertex_partition_range_lasts,
                                   bool store_transposed);

/**
 * @brief Shuffle local edge destinations (already placed by edge partitioning) to the owning GPUs
 * (by vertex partitioning).
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_dsts  Vector of local edge destination IDs
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Vector of shuffled edge destination vertex IDs (shuffled by vertex partitioning).
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_local_edge_dsts(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_dsts,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  bool store_transposed);

/**
 * @brief Shuffle local edge destination & value pairs (already placed by edge partitioning) to the
 * owning GPUs (by vertex partitioning).
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_dsts  Vector of local edge destination IDs
 * @param edge_values  Vector of edge values
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Tuple of vectors storing shuffled edge destination vertex IDs and values (shuffled by
 * vertex partitioning).
 */
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, dataframe_buffer_type_t<value_t>>
shuffle_local_edge_dst_value_pairs(raft::handle_t const& handle,
                                   rmm::device_uvector<vertex_t>&& edge_dsts,
                                   dataframe_buffer_type_t<value_t>&& edge_values,
                                   raft::host_span<vertex_t const> vertex_partition_range_lasts,
                                   bool store_transposed);

}  // namespace cugraph
