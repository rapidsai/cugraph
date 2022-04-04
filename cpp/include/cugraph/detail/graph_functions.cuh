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

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/handle.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <rmm/device_uvector.hpp>

#include <numeric>
#include <vector>

namespace cugraph {

namespace detail {

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
 * @brief Gather active majors and associated client gpu ids across gpus in a
 * column communicator
 *
 * Collect all the vertex ids and client gpu ids to be processed by every gpu in
 * the column communicator and call sort on the list.
 *
 * @tparam vertex_t Type of vertex indices.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam GPUIdIterator  Type of the iterator for gpu id identifiers.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertex_input_first Iterator pointing to the first vertex id to be processed
 * @param vertex_input_last Iterator pointing to the last (exclusive) vertex id to be processed
 * @param gpu_id_first Iterator pointing to the first gpu id to be processed
 * @return Device vector containing all the vertices that are to be processed by every gpu
 * in the column communicator
 */
template <typename GraphViewType, typename VertexIterator, typename GPUIdIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename std::iterator_traits<GPUIdIterator>::value_type>>
gather_active_majors(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     VertexIterator vertex_input_first,
                     VertexIterator vertex_input_last,
                     GPUIdIterator gpu_id_first);

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
 * @brief Return partition information of all vertex ids of all the partitions belonging to a gpu
 *
 * Iterate through partitions and store the starting vertex ids, exclusive scan of vertex counts,
 * offsets and indices of the partitions csr structure
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return Tuple of device vectors. The first vector contains all the partitions related to the
 * gpu. The second and third vectors contain starting and ending vertex ids of all the partitions
 * belonging to the gpu. The fourth vector contains the starting vertex id of the hypersparse
 * region in each partition. The fifth vector denotes the vertex count offset (how many vertices
 * are dealt with by the previous partitions.
 */
template <typename GraphViewType>
std::tuple<rmm::device_uvector<edge_partition_device_view_t<typename GraphViewType::vertex_type,
                                                            typename GraphViewType::edge_type,
                                                            typename GraphViewType::weight_type,
                                                            GraphViewType::is_multi_gpu>>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
partition_information(raft::handle_t const& handle, GraphViewType const& graph_view);

/**
 * @brief Gather valid edges present on the current gpu
 *
 * Collect all the edges that are present in the adjacency lists on the current gpu
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam gpu_t  Type of gpu id identifiers.
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] graph_view Non-owning graph object.
 * @param[in] active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param[in] active_major_gpu_ids Device vector containing the gpu id associated by every vertex
 * present in active_majors
 * @param[in] minor_map Device vector of minor indices (modifiable in-place) corresponding to
 * vertex IDs being returned
 * @param[in] indices_per_major Number of indices supplied for every major in the range
 * [vertex_input_first, vertex_input_last)
 * @param[in] global_degree_offsets Global degree offset to local adjacency list for every major
 * represented by current gpu
 * @return A tuple of device vector containing the majors, minors, gpu_ids and indices gathered
 * locally
 */
template <typename GraphViewType, typename gpu_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<gpu_t>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
gather_local_edges(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  const rmm::device_uvector<gpu_t>& active_major_gpu_ids,
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
 * @tparam prop_t  Type of the property associated with the majors.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param active_major_property Device vector containing the property values associated by every
 * vertex present in active_majors
 * @return A tuple of device vector containing the majors, minors and properties gathered locally
 */
template <typename GraphViewType, typename prop_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<prop_t>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  const rmm::device_uvector<prop_t>& active_major_property,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_adjacency_list_offsets);

}  // namespace detail

}  // namespace cugraph
