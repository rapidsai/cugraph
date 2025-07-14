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

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>

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
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  Vector of vertex ids
 * @return Vector of vertex ids mapped to this GPU.
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

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
                               rmm::device_uvector<value_t>&& values,
                               std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @ingroup graph_functions_cpp
 * @brief Shuffle external edges to the owning GPUs (by edge partitioning)
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  Vector of source vertex ids
 * @param edge_dsts  Vector of destination vertex ids
 * @param edge_properties  Vector of edge properties, each element is an arithmetic device vector
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Tuple of vectors storing edge sources, destinations, and edge properties
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::arithmetic_device_uvector_t>,
           std::vector<size_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edge_srcs,
                  rmm::device_uvector<vertex_t>&& edge_dsts,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @ingroup graph_functions_cpp
 * @brief Shuffle internal edges to the owning GPUs (by edge partitioning)
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  Vector of source vertex ids
 * @param edge_dsts  Vector of destination vertex ids
 * @param edge_properties  Vector of edge properties, each element is an arithmetic device vector
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Tuple of vectors storing edge sources, destinations, and edge properties
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::arithmetic_device_uvector_t>,
           std::vector<size_t>>
shuffle_int_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& majors,
                  rmm::device_uvector<vertex_t>&& minors,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  raft::host_span<vertex_t const> vertex_partition_range_lasts,
                  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @brief Shuffle local edge sources (already placed by edge partitioning) to the owning GPUs (by
 * vertex partitioning).
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  Vector of local edge source IDs
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of
 * GPUs)
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
