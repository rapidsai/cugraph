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
 * @brief Groupby and count edgelist using the key function which returns the target local
 * partition ID for an edge.  The specified spans are reordered in place.
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
 * @param[in] large_buffer_type Flag indicating the large buffer type to use when we need to create
 * a large device-accessible vector object (if the value is std::nullopt, the default RMM per-device
 * memory resource is used).
 *
 * @return A vector containing the number of edges in each local partition (if
 * groupby_and_count_local_partition is false) or in each segment with the same (local partition
 * ID, GPU ID) pair.
 */
template <typename vertex_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  raft::device_span<vertex_t> edgelist_majors,
  raft::device_span<vertex_t> edgelist_minors,
  raft::host_span<arithmetic_device_span_t> edgelist_properties,
  bool groupby_and_count_local_partition_by_minor      = false,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

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
 * @brief Shuffle properties to the proper GPU
 * *
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param [in] gpu Device vector of gpu ids
 * @param [in] properties Vector of device vectors of properties
 * @param[in] large_buffer_type Flag indicating the large buffer type to use when we need to create
 * a large device-accessible vector object (if the value is std::nullopt, the default RMM per-device
 * memory resource is used). The shuffled property values will also be stored in the buffer type
 * dictated by this parameter.
 *
 * @return vector of device vector of properties shuffled to the proper GPU
 */
std::vector<arithmetic_device_uvector_t> shuffle_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int>&& gpus,
  std::vector<arithmetic_device_uvector_t>&& properties,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt);

}  // namespace cugraph
