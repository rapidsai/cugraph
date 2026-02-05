/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 * @param vertex_properties  Vector of vertex properties, each element is an arithmetic device
 * vector
 *
 * @return Vector of vertex ids mapped to this GPU.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_ext_vertices(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>&& vertices,
                     std::vector<cugraph::arithmetic_device_uvector_t>&& vertex_properties,
                     std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @ingroup shuffle_functions_cpp
 * @brief Shuffle internal vertex IDs to the owning GPUs (by vertex partitioning)
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  Vector of vertex ids
 * @param vertex_properties  Vector of vertex properties, each element is an arithmetic device
 * vector
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 *
 * @return Vector of vertex ids mapped to this GPU.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_int_vertices(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>&& vertices,
                     std::vector<cugraph::arithmetic_device_uvector_t>&& vertex_properties,
                     raft::host_span<vertex_t const> vertex_partition_range_lasts,
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
           std::vector<cugraph::arithmetic_device_uvector_t>>
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
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 * @return Tuple of vectors storing edge sources, destinations, and edge properties
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_int_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& majors,
                  rmm::device_uvector<vertex_t>&& minors,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  raft::host_span<vertex_t const> vertex_partition_range_lasts,
                  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @brief Shuffle local edge source & value pairs (already placed by edge partitioning) to the
 * owning GPUs (by vertex partitioning).
 *
 * @tparam vertex_t    Type of vertex identifiers. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edge_srcs  Vector of local edge source IDs
 * @param edge_src_properties  Vector of local edge source properties, each element is an arithmetic
 * device vector
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Tuple of vectors storing shuffled edge source vertex IDs and values (shuffled by vertex
 * partitioning).
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_local_edge_srcs(raft::handle_t const& handle,
                        rmm::device_uvector<vertex_t>&& edge_srcs,
                        std::vector<cugraph::arithmetic_device_uvector_t>&& edge_src_properties,
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
 * @param edge_dst_properties  Vector of local edge destination properties, each element is an
 * arithmetic device vector
 * @param vertex_partition_range_lasts  Span of vertex partition range lasts (size = number of GPUs)
 * @param store_transposed Should be true if shuffled edges will be used with a cugraph::graph_t
 * object with store_tranposed = true. Should be false otherwise.
 * @return Tuple of vectors storing shuffled edge destination vertex IDs and values (shuffled by
 * vertex partitioning).
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_local_edge_dsts(raft::handle_t const& handle,
                        rmm::device_uvector<vertex_t>&& edge_dsts,
                        std::vector<cugraph::arithmetic_device_uvector_t>&& edge_dst_properties,
                        raft::host_span<vertex_t const> vertex_partition_range_lasts,
                        bool store_transposed);

}  // namespace cugraph
