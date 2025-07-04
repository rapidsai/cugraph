/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <cugraph/sampling_functions.hpp>

#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>

namespace cugraph {
namespace detail {

// FIXME: Functions in this file assume that store_transposed=false,
//    in implementation, naming and documentation.  We should review these and
//    consider updating things to support an arbitrary value for store_transposed

/**
 * @brief Check edge bias values.
 *
 * Count the number of negative edge bias values & the number of vertices with the sum of their
 * outgoing edge bias values exceeding std::numeric_limits<bias_t>::max().
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam bias_t Type of edge bias values. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate neighbor sampling on.
 * @param edge_weight_view View object holding edge bias values for @p graph_view.
 */
template <typename vertex_t, typename edge_t, typename bias_t, bool multi_gpu>
std::tuple<size_t, size_t> check_edge_bias_values(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view);

/**
 * @brief Gather edge list for specified vertices
 *
 * Collect all the edges that are present in the adjacency lists on the current gpu
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 * @tparam label_t Type of label. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate neighbor sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param active_major_labels Optional device vector containing labels for each device vector
 * @param gather_flags Optional host span indicating whether to gather edge or not for each edge
 * type. @p gather_flags.has_value() should coincide with @p edge_type_view.has_value().
 * @return A tuple of device vectors containing the majors, minors, optional weights,
 *  optional edge ids, optional edge types and optional label
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<label_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<label_t const>> active_major_labels,
  std::optional<raft::host_span<uint8_t const>> gather_flags,
  bool do_expensive_check = false);

/**
 * @brief Randomly sample edges from the adjacency list of specified vertices
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 * @tparam label_t Type of label. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate neighbor sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param rng_state Random number generator state
 * @param active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param active_major_labels Optional device vector containing labels for each device vector
 * @param fan_out How many edges to sample for each vertex per edge type
 * @param with_replacement If true sample with replacement, otherwise sample without replacement
 * @param invalid_vertex_id Value to use for an invalid vertex
 * @return A tuple of device vectors containing the majors, minors, optional weights,
 *  optional edge ids, optional edge types and optional label
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename bias_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<label_t>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
             std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
             std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
             raft::random::RngState& rng_state,
             raft::device_span<vertex_t const> active_majors,
             std::optional<raft::device_span<label_t const>> active_major_labels,
             raft::host_span<size_t const> fan_out,
             bool with_replacement);

/**
 * @brief Use the sampling results from hop N to populate the new frontier for hop N+1.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam label_t Type of label. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param sampled_src_vertices The source vertices for the current sampling
 * @param sampled_src_vertex_labels Optional labels for the vertices for the current sampling
 * @param sampled_dst_vertices Vertices for the next frontier
 * @param sampled_dst_vertex_labels Optional labels for the next frontier
 * @param vertex_used_as_source Optional. If specified then we want to exclude vertices that
 * were previously used as sources.  These vertices (and optional labels) will be updated based
 * on the contents of sampled_src_vertices/sampled_src_vertex_labels and the update will be part
 * of the return value.
 * @param vertex_partition_range_lasts End of range information from graph view
 * @param prior_sources_behavior Identifies how to treat sources in each hop
 * @param dedupe_sources boolean flag, if true then if a vertex v appears as a destination in hop X
 * multiple times with the same label, it will only be passed once (for each label) as a source
 * for the next hop.  Default is false.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 *
 * @return A tuple of device vectors containing the vertices for the next frontier expansion and
 *  optional labels associated with the vertices, along with the updated value for
 *  @p vertex_used_as_sources
 */
template <typename vertex_t, typename label_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                                    std::optional<rmm::device_uvector<label_t>>>>>
prepare_next_frontier(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> sampled_src_vertices,
  std::optional<raft::device_span<label_t const>> sampled_src_vertex_labels,
  raft::device_span<vertex_t const> sampled_dst_vertices,
  std::optional<raft::device_span<label_t const>> sampled_dst_vertex_labels,
  std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                           std::optional<rmm::device_uvector<label_t>>>>&& vertex_used_as_source,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check);

/**
 * @brief Remove from the frontier any vertices that have already been used as a source
 *
 * @tparam vertex_t Type of vertex identifiers.  Needs to be an integral type.
 * @tparam label_t Type of label.  Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param frontier_vertices Vertices discovered in the current hop
 * @param frontier_vertex_labels Labels for the vertices discovered in the current hop
 * @param vertices_used_as_source Device vector containing all vertices used in previous hops as a
 * source
 * @param vertex_labels_used_as_source Device vector containing vertex labels associated with
 * the @p vertices_used_as_source used in previous hops as a source vertex label
 *
 * @return tuple containing the modified frontier_vertices and frontier_vertex_labels
 */
template <typename vertex_t, typename label_t>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<label_t>>>
remove_visited_vertices_from_frontier(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& frontier_vertices,
  std::optional<rmm::device_uvector<label_t>>&& frontier_vertex_labels,
  raft::device_span<vertex_t const> vertices_used_as_source,
  std::optional<raft::device_span<label_t const>> vertex_labels_used_as_source);

/**
 * @brief Organize sampling results by shuffling to the proper GPU (if necessary as identified by
 * labels and label_to_output_comm_rank) and sorting the vertices by label and hop
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param property_edges vector of arithmetic device vectors of the edge data and the
 * properties.  This should include the src and dst vertices, any edge properties that exist for the
 * sampled edge and optionally the hop where the edge was sampled
 * @param labels Optional labels associated with each edge.  If labels are not specified this
 * function is a noop.
 * @param hops Optional hops associated with each edge.  If hops are specified the result is sorted
 * by label and hop
 * @param label_to_output_comm_rank Optional map associating each label to a comm rank.  If
 * specified this will result in shuffling the data, if not specified this will skip the shuffling
 * step and only consider sorting the results
 */
std::tuple<std::vector<cugraph::arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  std::vector<cugraph::arithmetic_device_uvector_t>&& property_edges,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank);

/**
 * @brief   Convert the starting vertex offsets into starting vertex labels
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param starting_vertex_label_offsets Offsets array defining where each vertex label begins
 *
 * @returns device vector containing labels for each starting vertex
 */
rmm::device_uvector<int32_t> convert_starting_vertex_label_offsets_to_labels(
  raft::handle_t const& handle, raft::device_span<size_t const> starting_vertex_label_offsets);

/**
 * @brief   Flatten the legacy label_to_output_comm_rank into the new structure
 *
 * Legacy structure supported arbitrary labels, the new structure is a dense mapping of labels from
 * [0,n).
 *
 * @tparam label_t typename for the label
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param label_to_output_comm_rank  A tuple containing label ids and the comm rank each label
 * should be assigned to
 *
 * @returns device vector containing the mapping to comm_rank.  Entry `i` will be the comm rank
 * destination for label `i`.
 */
template <typename label_t>
rmm::device_uvector<int32_t> flatten_label_map(
  raft::handle_t const& handle,
  std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>
    label_to_output_comm_rank);

}  // namespace detail
}  // namespace cugraph
