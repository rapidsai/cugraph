/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>

namespace cugraph {
namespace detail {

/** @defgroup shuffle_wrappers_cpp C++ Shuffle Wrappers
 */

/**
 * @ingroup shuffle_wrappers_cpp
 * @brief Shuffle external (i.e. before renumbering) vertex pairs (which can be edge end points) to
 * their local GPUs based on edge partitioning.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type identifiers. Needs to be an integral type.
 * @tparam edge_time_t The type of the edge time stamp.  Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] majors Vector of first elemetns in vertex pairs. To determine the local GPU of a
 * (major, minor) pair, we assume there exists an edge from major=>minor (if we store edges in the
 * sparse 2D matrix using sources as major indices) or minor=>major (otherwise) and apply the edge
 * partitioning to determine the local GPU.
 * @param[in] minors Vector of second elements in vertex pairs.
 * @param[in] weights Optional vector of vertex pair weight values.
 * @param[in] edge_ids Optional vector of vertex pair edge id values.
 * @param[in] edge_types Optional vector of vertex pair edge type values.
 * @param[in] edge_start_times Optional vector of vertex pair edge start time values.
 * @param[in] edge_end_times Optional vector of vertex pair edge end time values.
 * @param[in] large_buffer_type Dictates the large buffer type to use in storing the shuffled vertex
 * value pairs (if the value is std::nullopt, the default RMM per-device memory resource is used).
 *
 * @return Tuple of vectors storing shuffled major vertices, minor vertices and optional weights,
 * edge ids and edge types
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
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @ingroup shuffle_wrappers_cpp
 * @brief Shuffle internal (i.e. renumbered) vertex pairs (which can be edge end points) to their
 * local GPUs based on edge partitioning.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type identifiers. Needs to be an integral type.
 * @tparam edge_time_t Type of edge time. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] majors Vector of first elemetns in vertex pairs. To determine the local GPU of a
 * (major, minor) pair, we assume there exists an edge from major=>minor (if we store edges in the
 * sparse 2D matrix using sources as major indices) or minor=>major (otherwise) and apply the edge
 * partitioning to determine the local GPU.
 * @param[in] minors Vector of second elements in vertex pairs.
 * @param[in] weights Optional vector of vertex pair weight values.
 * @param[in] edge_ids Optional vector of vertex pair edge id values.
 * @param[in] edge_types Optional vector of vertex pair edge type values.
 * @param[in] edge_start_times Optional vector of vertex pair edge start time values.
 * @param[in] edge_end_times Optional vector of vertex pair edge end time values.
 * @param[in] vertex_partition_range_lasts Vector of each GPU's vertex partition range's last
 * (exclusive) vertex ID.
 * @param[in] large_buffer_type Dictates the large buffer type to use in storing the shuffled vertex
 * pairs (if the value is std::nullopt, the default RMM per-device memory resource is used).
 *
 * @return Tuple of vectors storing shuffled major vertices, minor vertices and optional weights,
 * edge ids and edge types and rx counts
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
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @ingroup shuffle_wrappers_cpp
 * @brief Permute a range.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in]  rng_state The RngState instance holding pseudo-random number generator state.
 * @param[in] local_range_size Size of local range assigned to this process.
 * @param[in] local_start Start of local range assigned to this process.
 *
 * @return permuted range.
 */

template <typename vertex_t>
rmm::device_uvector<vertex_t> permute_range(raft::handle_t const& handle,
                                            raft::random::RngState& rng_state,
                                            vertex_t local_start,
                                            vertex_t local_range_size,
                                            bool multi_gpu          = false,
                                            bool do_expensive_check = false);

/**
 * @ingroup shuffle_wrappers_cpp
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
 * @param[in] large_buffer_type Dictates the large buffer type to use in storing the shuffled
 * vertices (if the value is std::nullopt, the default RMM per-device memory resource is used).
 *
 * @return Vector of shuffled vertices.
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @ingroup shuffle_wrappers_cpp
 * @brief Shuffle vertices using the internal vertex key function which returns the target GPU ID.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam value_t Type of vertex values. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * @param[in] vertices Vertex IDs to shuffle
 * @param[in] values Vertex Values to shuffle
 * @param[in] vertex_partition_range_lasts From graph view, vector of last vertex id for each gpu
 * @param[in] large_buffer_type Dictates the large buffer type to use in storing the shuffled vertex
 * value pairs (if the value is std::nullopt, the default RMM per-device memory resource is used).
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
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

/**
 * @ingroup shuffle_wrappers_cpp
 * @brief Groupby and count edgelist using the key function which returns the target local partition
 * ID for an edge.  The specified spans are reordered in place.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in,out] edgelist_majors Vertex IDs for sources (if we are internally storing edges in
 * the sparse 2D matrix using sources as major indices) or destinations (otherwise)
 * @param[in,out] edgelist_minors Vertex IDs for destinations (if we are internally storing edges
 * in the sparse 2D matrix using sources as major indices) or sources (otherwise)
 * @param[in,out] edgelist_properties Span of edge properties, each element a device span of an
 * edge property.
 * @param[in] groupby_and_count_local_partition_by_minor If set to true, groupby and count edges
 * based on (local partition ID, GPU ID) pairs (where GPU IDs are computed by applying the
 * compute_gpu_id_from_vertex_t function to the minor vertex ID). If set to false, groupby and count
 * edges by just local partition ID.
 *
 * @return A vector containing the number of edges in each local partition (if
 * groupby_and_count_local_partition is false) or in each segment with the same (local partition ID,
 * GPU ID) pair.
 */
template <typename vertex_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  raft::device_span<vertex_t> edgelist_majors,
  raft::device_span<vertex_t> edgelist_minors,
  raft::host_span<arithmetic_device_span_t> edgelist_properties,
  bool groupby_and_count_local_partition_by_minor = false);

/**
 * @ingroup shuffle_wrappers_cpp
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

/**
 * @ingroup shuffle_wrappers_cpp
 * @brief Shuffle keys and their associated properties to the proper GPU based on a partitioning
 * function.  Keys for this function are assumed to be an arithmetic type
 *
 * @tparam key_t type of key
 * @tparam key_to_gpu_op_t Function to convert key to GPU id
 *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param [in] key Device vector of keys
 * @param [in] properties Vector of device vectors of properties associated with each key
 * @param [in] key_to_gpu_op function converting key to a GPU id
 *
 * @return tuple of device vector of keys and vector of device vector of properties associated with
 * the key.
 */
template <typename key_t, typename key_to_gpu_op_t>
std::tuple<rmm::device_uvector<key_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_keys_with_properties(raft::handle_t const& handle,
                             rmm::device_uvector<key_t>&& keys,
                             std::vector<arithmetic_device_uvector_t>&& properties,
                             key_to_gpu_op_t key_to_gpu_op);

/**
 * @ingroup shuffle_wrappers_cpp
 * @brief Shuffle properties to the proper GPU
 * *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param [in] gpu Device vector of gpu ids
 * @param [in] properties Vector of device vectors of properties
 *
 * @return vector of device vector of properties shuffled to the proper GPU
 */
std::vector<arithmetic_device_uvector_t> shuffle_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int>&& gpus,
  std::vector<arithmetic_device_uvector_t>&& properties);

}  // namespace cugraph
