/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <rmm/device_uvector.hpp>

#include <optional>

namespace cugraph {
namespace detail {

/**
 * @brief Shuffle external (i.e. before renumbering) vertex pairs (which can be edge end points) to
 * their local GPUs based on edge partitioning.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type identifiers. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] majors Vector of first elemetns in vertex pairs. To determine the local GPU of a
 * (major, minor) pair, we assume there exists an edge from major=>minor (if we store edges in the
 * sparse 2D matrix using sources as major indices) or minor=>major (otherwise) and apply the edge
 * partitioning to determine the local GPU.
 * @param[in] minors Vector of second elements in vertex pairs.
 * @param[in] weights Optional vector of vertex pair weight values.
 * @param[in] edge_id_type_tuple Optional tuple of vectors of edge id and edge type values
 *
 * @return Tuple of vectors storing shuffled major vertices, minor vertices and optional weights.
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_id_t>
std::tuple<
  rmm::device_uvector<vertex_t>,
  rmm::device_uvector<vertex_t>,
  std::optional<rmm::device_uvector<weight_t>>,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_id_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_id_t>>>&&
    edge_id_type_tuple);

/**
 * @brief Shuffle internal (i.e. renumbered) vertex pairs (which can be edge end points) to their
 * local GPUs based on edge partitioning.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type identifiers. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] majors Vector of first elemetns in vertex pairs. To determine the local GPU of a
 * (major, minor) pair, we assume there exists an edge from major=>minor (if we store edges in the
 * sparse 2D matrix using sources as major indices) or minor=>major (otherwise) and apply the edge
 * partitioning to determine the local GPU.
 * @param[in] minors Vector of second elements in vertex pairs.
 * @param[in] weights Optional vector of vertex pair weight values.
 * @param[in] edge_id_type_tuple Optional tuple of vectors of edge id and edge type values
 * @param[in] vertex_partition_range_lasts Vector of each GPU's vertex partition range's last
 * (exclusive) vertex ID.
 *
 * @return Tuple of vectors storing shuffled major vertices, minor vertices and optional weights.
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_id_t>
std::tuple<
  rmm::device_uvector<vertex_t>,
  rmm::device_uvector<vertex_t>,
  std::optional<rmm::device_uvector<weight_t>>,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_id_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_id_t>>>&&
    edge_id_type_tuple,
  std::vector<vertex_t> const& vertex_partition_range_lasts);

/**
 * @brief Shuffle external (i.e. before renumbering) vertices to their local GPU based on vertex
 * partitioning.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] vertices Vertices to shuffle.
 *
 * @return Vector of shuffled vertices.
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& vertices);

/**
 * @brief Shuffle external (i.e. before renumbering) vertex & value pairs to their local GPU based
 * on vertex partitioning.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam value_t Type of values.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] vertices Vertices to shuffle.
 * @param[in] values Values to shuffle.
 *
 * @return Tuple of vectors storing shuffled vertex & value pairs.
 */
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values);

/**
 * @brief Shuffle internal (i.e. renumbered) vertices to their local GPUs based on vertex
 * partitioning.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] vertices Vertices to shuffle.
 * @param[in] vertex_partition_range_lasts Vector of each GPU's vertex partition range's last
 * (exclusive) vertex ID.
 *
 * @return Vector of shuffled vertices.
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  std::vector<vertex_t> const& vertex_partition_range_lasts);

/**
 * @brief Shuffle vertices using the internal vertex key function which returns the target GPU ID.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam value_t Type of vertex values. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * @param[in] vertices Vertex IDs to shuffle
 * @param[in] values Vertex Values to shuffle
 * @param[in] vertex_partition_range_lasts From graph view, vector of last vertex id for each gpu
 *
 * @return tuple containing device vector of shuffled vertices and device vector of corresponding
 *         values
 */
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values,
  std::vector<vertex_t> const& vertex_partition_range_lasts);

/**
 * @brief Groupby and count edgelist using the key function which returns the target local partition
 * ID for an edge.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in/out] d_edgelist_majors Vertex IDs for sources (if we are internally storing edges in
 * the sparse 2D matrix using sources as major indices) or destinations (otherwise)
 * @param[in/out] d_edgelist_minors Vertex IDs for destinations (if we are internally storing edges
 * in the sparse 2D matrix using sources as major indices) or sources (otherwise)
 * @param[in/out] d_edgelist_weights Optional edge weights
 * @param[in/out] d_edgelist_id_type_pairs Optional edge (ID, type) pairs
 * @param[in] groupby_and_count_local_partition_by_minor If set to true, groupby and count edges
 * based on (local partition ID, GPU ID) pairs (where GPU IDs are computed by applying the
 * compute_gpu_id_from_vertex_t function to the minor vertex ID). If set to false, groupby and count
 * edges by just local partition ID.
 *
 * @return A vector containing the number of edges in each local partition (if
 * groupby_and_count_local_partition is false) or in each segment with the same (local partition ID,
 * GPU ID) pair.
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_count_local_partition_by_minor = false);

/**
 * @brief Collect vertex values (represented as k/v pairs across cluster) and return
 *        local value arrays on the GPU responsible for each vertex.
 *
 * Data will be shuffled, renumbered and initialized with the default value,
 * then:  return_array[d_vertices[i]] = d_values[i]
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam value_t  Type of value associated with the vertex.
 * @tparam bool     multi_gpu flag
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] d_vertices Vertex IDs for the k/v pair
 * @param[in] d_values Values for the k/v pair
 * @param[in] The number map for performing renumbering
 * @param[in] local_vertex_first The first vertex id assigned to the local GPU
 * @param[in] local_vertex_last The last vertex id assigned to the local GPU
 * @param[in] default_value The default value the return array will be initialized by
 * @param[in] do_expensive_check If true, enable expensive validation checks
 * @return device vector of values, where return[i] is the value associated with
 *         vertex (i + local_vertex_first)
 */
template <typename vertex_t, typename value_t, bool multi_gpu>
rmm::device_uvector<value_t> collect_local_vertex_values_from_ext_vertex_value_pairs(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& d_vertices,
  rmm::device_uvector<value_t>&& d_values,
  rmm::device_uvector<vertex_t> const& number_map,
  vertex_t local_vertex_first,
  vertex_t local_vertex_last,
  value_t default_value,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph
