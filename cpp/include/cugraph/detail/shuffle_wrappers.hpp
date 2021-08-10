/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

/**
 * @brief Shuffle edgelist using the edge key function which returns the target GPU ID.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] d_edgelist_majors Vertex IDs for rows (if the graph adjacency matrix is stored as is)
 * or columns (if the graph adjacency matrix is stored transposed)
 * @param[in] d_edgelist_minors Vertex IDs for columns (if the graph adjacency matrix is stored as
 * is) or rows (if the graph adjacency matrix is stored transposed)
 * @param[in] d_edgelist_weights Optional edge weights
 *
 * @return Tuple of shuffled major vertices, minor vertices and optional weights
 */
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<vertex_t>&& d_edgelist_majors,
                           rmm::device_uvector<vertex_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<weight_t>>&& d_edgelist_weights);

/**
 * @brief Shuffle vertices using the vertex key function which returns the target GPU ID.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * @param[in] d_vertices Vertex IDs to shuffle
 *
 * @return device vector of shuffled vertices
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_vertices_by_gpu_id(
  raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& d_vertices);

/**
 * @brief Groupby and count edgelist using the key function which returns the target local partition
 * ID for an edge.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in/out] d_edgelist_majors Vertex IDs for rows (if the graph adjacency matrix is stored as
 * is) or columns (if the graph adjacency matrix is stored transposed)
 * @param[in/out] d_edgelist_minors Vertex IDs for columns (if the graph adjacency matrix is stored
 * as is) or rows (if the graph adjacency matrix is stored transposed)
 * @param[in/out] d_edgelist_weights Optional edge weights
 * @param[in] groupby_and_count_local_partition If set to true, groupby and count edges based on
 * (local partition ID, GPU ID) pairs (where GPU IDs are computed by applying the
 * compute_gpu_id_from_vertex_t function to the minor vertex ID). If set to false, groupby and count
 * edges by just local partition ID.
 *
 * @return A vector containing the number of edges in each local partition (if
 * groupby_and_count_local_partition is false) or in each segment with the same (local partition ID,
 * GPU ID) pair.
 */
template <typename vertex_t, typename weight_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_edgelist_weights,
  bool groupby_and_count_local_partition = false);

}  // namespace detail
}  // namespace cugraph
