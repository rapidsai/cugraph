/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <thrust/optional.h>

namespace cugraph {
namespace detail {

// FIXME: Several of the functions in this file assume that store_transposed=false,
//    in implementation, naming and documentation.  We should review these and
//    consider updating things to support an arbitrary value for store_transposed

/**
 * @brief Compute local out degrees of the majors belonging to the adjacency matrices
 * stored on each gpu
 *
 * Iterate through partitions and store their local degrees
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return A single vector containing the local out degrees of the majors belong to the adjacency
 * matrices
 */
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> compute_local_major_degrees(
  raft::handle_t const& handle, GraphViewType const& graph_view);

/**
 * @brief Calculate global degree information for all vertices represented by current gpu
 *
 * Calculate local degree and perform row wise exclusive scan over all gpus in column
 * communicator.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return Tuple of two device vectors. The first one contains per source edge-count encountered
 * by gpus in the column communicator before current gpu. The second device vector contains the
 * global out degree for every source represented by current gpu
 */
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
get_global_degree_information(raft::handle_t const& handle, GraphViewType const& graph_view);

/**
 * @brief Calculate global adjacency offset for all majors represented by current gpu
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param[in] global_degree_offsets Global degree offset to local adjacency list for every major
 * represented by current gpu
 * @param global_out_degrees Global out degrees for every source represented by current gpu
 * @return Device vector containing the number of edges that are prior to the adjacency list of
 * every major that can be represented by the current gpu
 */
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> get_global_adjacency_offset(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_degree_offsets,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_out_degrees);

/**
 * @brief Gather active majors across gpus in a column communicator
 *
 * Collect all the vertex ids and client gpu ids to be processed by every gpu in
 * the column communicator and sort the list.
 *
 * @tparam vertex_t Type of vertex indices.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param d_in Device vector containing vertices local to this GPU
 * @return Device vector containing all the vertices that are to be processed by every gpu
 * in the column communicator
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> allgather_active_majors(raft::handle_t const& handle,
                                                      rmm::device_uvector<vertex_t>&& d_in);

/**
 * @brief Return global out degrees of active majors
 *
 * Get partition information of all graph partitions on the gpu and select
 * global degrees of all active majors
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param global_out_degrees Global out degrees for every source represented by current gpu
 * @return Global out degrees of all majors in active_majors
 */
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> get_active_major_global_degrees(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_out_degrees);

/**
 * @brief Gather specified edges present on the current gpu
 *
 * Collect all the edges that are present in the adjacency lists on the current gpu
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] graph_view Non-owning graph object.
 * @param[in] active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param[in] minor_map Device vector of minor indices (modifiable in-place) corresponding to
 * vertex IDs being returned
 * @param[in] indices_per_major Number of indices supplied for every major in the range
 * [vertex_input_first, vertex_input_last)
 * @param[in] global_degree_offsets Global degree offset to local adjacency list for every major
 * represented by current gpu
 * @return A tuple of device vector containing the majors, minors, and weights gathered
 * locally
 */
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>>
gather_local_edges(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  rmm::device_uvector<typename GraphViewType::edge_type>&& minor_map,
  typename GraphViewType::edge_type indices_per_major,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_degree_offsets,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_adjacency_list_offsets);

/**
 * @brief Gather edge list for specified vertices
 *
 * Collect all the edges that are present in the adjacency lists on the current gpu
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @return A tuple of device vector containing the majors, minors and weights gathered locally
 */
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors);

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<edge_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<vertex_t>&& src,
                            rmm::device_uvector<vertex_t>&& dst,
                            rmm::device_uvector<weight_t>&& wgt);

}  // namespace detail

}  // namespace cugraph
