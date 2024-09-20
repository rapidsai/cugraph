/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cugraph/graph_view.hpp>
#include <cugraph/src_dst_lookup_container.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {

/**
 * @brief Controls how we treat prior sources in sampling
 *
 * @param DEFAULT    Add vertices encountered while sampling to the new frontier
 * @param CARRY_OVER In addition to newly encountered vertices, include vertices
 *                   used as sources in any previous frontier in the new frontier
 * @param EXCLUDE    Filter the new frontier to exclude any vertex that was
 *                   used as a source in a previous frontier
 */
enum class prior_sources_behavior_t { DEFAULT = 0, CARRY_OVER, EXCLUDE };

/**
 * @brief Uniform Neighborhood Sampling.
 * 
 * @deprecated  This API will be deleted, use cugraph_homogeneous_neighbor_sample with
 * 'is_biased' set to false instead
 *
 * @deprecated Replaced with homogeneous_uniform_neighbor_sample
 *
 * This function traverses from a set of starting vertices, traversing outgoing edges and
 * randomly selects from these outgoing neighbors to extract a subgraph.
 *
 * Output from this function is a tuple of vectors (src, dst, weight, edge_id, edge_type, hop,
 * label, offsets), identifying the randomly selected edges.  src is the source vertex, dst is the
 * destination vertex, weight (optional) is the edge weight, edge_id (optional) identifies the edge
 * id, edge_type (optional) identifies the edge type, hop identifies which hop the edge was
 * encountered in.  The label output (optional) identifes the vertex label.  The offsets array
 * (optional) will be described below and is dependent upon the input parameters.
 *
 * If @p starting_vertex_labels is not specified then no organization is applied to the output, the
 * label and offsets values in the return set will be std::nullopt.
 *
 * If @p starting_vertex_labels is specified and @p label_to_output_comm_rank is not specified then
 * the label output has values.  This will also result in the output being sorted by vertex label.
 * The offsets array in the return will be a CSR-style offsets array to identify the beginning of
 * each label range in the data.  `labels.size() == (offsets.size() - 1)`.
 *
 * If @p starting_vertex_labels is specified and @p label_to_output_comm_rank is specified then the
 * label output has values.  This will also result in the output being sorted by vertex label.  The
 * offsets array in the return will be a CSR-style offsets array to identify the beginning of each
 * label range in the data.  `labels.size() == (offsets.size() - 1)`.  Additionally, the data will
 * be shuffled so that all data with a particular label will be on the specified rank.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 * @tparam label_t Type of label. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether sources (if false) or destinations (if
 * true) are major indices
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param starting_vertices Device span of starting vertex IDs for the sampling.
 * In a multi-gpu context the starting vertices should be local to this GPU.
 * @param starting_vertex_labels Optional device span of labels associted with each starting vertex
 * for the sampling.
 * @param label_to_output_comm_rank Optional tuple of device spans mapping label to a particular
 * output rank.  Element 0 of the tuple identifes the label, Element 1 of the tuple identifies the
 * output rank.  The label span must be sorted in ascending order.
 * @param fan_out Host span defining branching out (fan-out) degree per source vertex for each
 * level
 * @param rng_state A pre-initialized raft::RngState object for generating random numbers
 * @param return_hops boolean flag specifying if the hop information should be returned
 * @param prior_sources_behavior Enum type defining how to handle prior sources, (defaults to
 * DEFAULT)
 * @param dedupe_sources boolean flag, if true then if a vertex v appears as a destination in hop X
 * multiple times with the same label, it will only be passed once (for each label) as a source
 * for the next hop.  Default is false.
 * @param with_replacement boolean flag specifying if random sampling is done with replacement
 * (true); or, without replacement (false); default = true;
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return tuple device vectors (vertex_t source_vertex, vertex_t destination_vertex,
 * optional weight_t weight, optional edge_t edge id, optional edge_type_t edge type,
 * optional int32_t hop, optional label_t label, optional size_t offsets)
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<size_t>>>
uniform_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool return_hops,
  bool with_replacement                           = true,
  prior_sources_behavior_t prior_sources_behavior = prior_sources_behavior_t::DEFAULT,
  bool dedupe_sources                             = false,
  bool do_expensive_check                         = false);

/**
 * @brief Biased Neighborhood Sampling.
 * 
 * @deprecated  This API will be deleted, use cugraph_homogeneous_neighbor_sample with
 * 'is_biased' set to true instead
 *
 * @deprecated Replaced with homogeneous_biased_neighbor_sample
 *
 * This function traverses from a set of starting vertices, traversing outgoing edges and
 * randomly selects (with edge biases) from these outgoing neighbors to extract a subgraph.
 *
 * Output from this function is a tuple of vectors (src, dst, weight, edge_id, edge_type, hop,
 * label, offsets), identifying the randomly selected edges.  src is the source vertex, dst is the
 * destination vertex, weight (optional) is the edge weight, edge_id (optional) identifies the edge
 * id, edge_type (optional) identifies the edge type, hop identifies which hop the edge was
 * encountered in.  The label output (optional) identifes the vertex label.  The offsets array
 * (optional) will be described below and is dependent upon the input parameters.
 *
 * If @p starting_vertex_labels is not specified then no organization is applied to the output, the
 * label and offsets values in the return set will be std::nullopt.
 *
 * If @p starting_vertex_labels is specified and @p label_to_output_comm_rank is not specified then
 * the label output has values.  This will also result in the output being sorted by vertex label.
 * The offsets array in the return will be a CSR-style offsets array to identify the beginning of
 * each label range in the data.  `labels.size() == (offsets.size() - 1)`.
 *
 * If @p starting_vertex_labels is specified and @p label_to_output_comm_rank is specified then the
 * label output has values.  This will also result in the output being sorted by vertex label.  The
 * offsets array in the return will be a CSR-style offsets array to identify the beginning of each
 * label range in the data.  `labels.size() == (offsets.size() - 1)`.  Additionally, the data will
 * be shuffled so that all data with a particular label will be on the specified rank.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 * @tparam bias_t Type of bias. Needs to be an integral type.
 * @tparam label_t Type of label. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether sources (if false) or destinations (if
 * true) are major indices
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param edge_bias_view View object holding edge biases (to be used in biased sampling) for @p
 * graph_view. Bias values should be non-negative and the sum of edge bias values from any vertex
 * should not exceed std::numeric_limits<bias_t>::max(). 0 bias value indicates that the
 * corresponding edge can never be selected.
 * @param starting_vertices Device span of starting vertex IDs for the sampling.
 * In a multi-gpu context the starting vertices should be local to this GPU.
 * @param starting_vertex_labels Optional device span of labels associted with each starting vertex
 * for the sampling.
 * @param label_to_output_comm_rank Optional tuple of device spans mapping label to a particular
 * output rank.  Element 0 of the tuple identifes the label, Element 1 of the tuple identifies the
 * output rank.  The label span must be sorted in ascending order.
 * @param fan_out Host span defining branching out (fan-out) degree per source vertex for each
 * level
 * @param rng_state A pre-initialized raft::RngState object for generating random numbers
 * @param return_hops boolean flag specifying if the hop information should be returned
 * @param prior_sources_behavior Enum type defining how to handle prior sources, (defaults to
 * DEFAULT)
 * @param dedupe_sources boolean flag, if true then if a vertex v appears as a destination in hop X
 * multiple times with the same label, it will only be passed once (for each label) as a source
 * for the next hop.  Default is false.
 * @param with_replacement boolean flag specifying if random sampling is done with replacement
 * (true); or, without replacement (false); default = true;
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return tuple device vectors (vertex_t source_vertex, vertex_t destination_vertex,
 * optional weight_t weight, optional edge_t edge id, optional edge_type_t edge type,
 * optional int32_t hop, optional label_t label, optional size_t offsets)
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename bias_t,
          typename label_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<size_t>>>
biased_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool return_hops,
  bool with_replacement                           = true,
  prior_sources_behavior_t prior_sources_behavior = prior_sources_behavior_t::DEFAULT,
  bool dedupe_sources                             = false,
  bool do_expensive_check                         = false);

struct sampling_flags_t {
  /**
   * Specifies how to handle prior sources. Default is DEFAULT.
   */
  prior_sources_behavior_t prior_sources_behavior{};

  /**
   * Specifies if the hop information should be returned.  Default is false.
   */
  bool return_hops{false};

  /**
   * If true then if a vertex v appears as a destination in hop X multiple times
   * with the same label, it will only be passed once (for each label) as a source
   * for the next hop.  Default is false.
   */
  bool dedupe_sources{false};

  /**
   * Specifies if random sampling is done with replacement
   *   (true) or without replacement (false).  Default is true.
   */
  bool with_replacement{true};
};

/**
 * @brief Heterogeneous Neighborhood Sampling.
 *
 * This function traverses from a set of starting vertices, traversing outgoing edges and
 * randomly selects (with edge biases or not) from these outgoing neighbors to extract a subgraph.
 * The branching out to select outgoing neighbors is performed with homogeneous fanouts.
 *
 * Output from this function is a tuple of vectors (src, dst, weight, edge_id, edge_type, hop,
 * offsets), identifying the randomly selected edges.  src is the source vertex, dst is the
 * destination vertex, weight (optional) is the edge weight, edge_id (optional) identifies the edge
 * id, edge_type (optional) identifies the edge type, hop identifies which hop the edge was
 * encountered in.  The offsets array (optional) identifies the offset for each label.
 *
 * If @p starting_vertex_offsets is not specified then no organization is applied to the output, the
 * offsets values in the return set will be std::nullopt.
 *
 * If @p starting_vertex_offsets is specified the offsets array will be populated. The offsets array
 * in the return will be a CSR-style offsets array to identify the beginning of each label range in
 * the output vectors.
 *
 * If @p label_to_output_comm_rank is specified then the data will be shuffled so that all entries
 * for a particular label are returned on the specified rank.  This will result in the offsets array
 * on other GPUs indicating that there are no entries for that label (`offsets[i] == offsets[i+1]`).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 * @tparam bias_t Type of bias. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether sources (if false) or destinations (if
 * true) are major indices
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param rng_state A pre-initialized raft::RngState object for generating random numbers
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param edge_bias_view Optional view object holding edge biases (to be used in biased sampling) for @p
 * graph_view. Bias values should be non-negative and the sum of edge bias values from any vertex
 * should not exceed std::numeric_limits<bias_t>::max(). 0 bias value indicates that the
 * corresponding edge can never be selected. passing std::nullopt as the edge biases will result in
 * uniform sampling.
 * @param starting_vertices Device span of starting vertex IDs for the sampling.
 * In a multi-gpu context the starting vertices should be local to this GPU.
 * @param starting_vertex_offsets Optional device span of offsets identifying the range of
 * starting vertex values for this label.
 * @param label_to_output_comm_rank Optional device span identifying which rank should get each
 * vertex label.  This should be the same on each rank.
 * @param fan_out Host span defining branching out (fan-out) degree per source vertex for each
 * level. The fanout value at hop x is given by the expression 'fanout[x*num_edge_types + edge_type_id]'
 * @param num_edge_types Number of edge types where a value of 1 translates to homogeneous neighbor
 * sample whereas a value greater than 1 translates to heterogeneous neighbor sample.
 * @param flags A set of flags indicating which sampling features should be used.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return tuple device vectors (vertex_t source_vertex, vertex_t destination_vertex,
 * optional weight_t weight, optional edge_t edge id, optional edge_type_t edge type,
 * optional int32_t hop, optional label_t label, optional size_t offsets)
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename bias_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
heterogeneous_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<size_t const>> starting_vertex_offsets,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  edge_type_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check = false);

/**
 * @brief Homogeneous Neighborhood Sampling.
 *
 * This function traverses from a set of starting vertices, traversing outgoing edges and
 * randomly selects (with edge biases or not) from these outgoing neighbors to extract a subgraph.
 * The branching out to select outgoing neighbors is performed with homogeneous fanouts
 *
 * Output from this function is a tuple of vectors (src, dst, weight, edge_id, edge_type, hop,
 * offsets), identifying the randomly selected edges.  src is the source vertex, dst is the
 * destination vertex, weight (optional) is the edge weight, edge_id (optional) identifies the edge
 * id, edge_type (optional) identifies the edge type, hop identifies which hop the edge was
 * encountered in.  The offsets array (optional) identifies the offset for each label.
 *
 * If @p starting_vertex_offsets is not specified then no organization is applied to the output, the
 * offsets values in the return set will be std::nullopt.
 *
 * If @p starting_vertex_offsets is specified the offsets array will be populated. The offsets array
 * in the return will be a CSR-style offsets array to identify the beginning of each label range in
 * the output vectors.
 *
 * If @p label_to_output_comm_rank is specified then the data will be shuffled so that all entries
 * for a particular label are returned on the specified rank.  This will result in the offsets array
 * on other GPUs indicating that there are no entries for that label (`offsets[i] == offsets[i+1]`).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 * @tparam bias_t Type of bias. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether sources (if false) or destinations (if
 * true) are major indices
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param rng_state A pre-initialized raft::RngState object for generating random numbers
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param edge_bias_view Optional view object holding edge biases (to be used in biased sampling) for @p
 * graph_view. Bias values should be non-negative and the sum of edge bias values from any vertex
 * should not exceed std::numeric_limits<bias_t>::max(). 0 bias value indicates that the
 * corresponding edge can never be selected. passing std::nullopt as the edge biases will result in
 * uniform sampling.
 * @param starting_vertices Device span of starting vertex IDs for the sampling.
 * In a multi-gpu context the starting vertices should be local to this GPU.
  * @param starting_vertex_offsets Optional device span of offsets identifying the range of
 * starting vertex values for this label.
 * @param label_to_output_comm_rank Optional device span identifying which rank should get each
 * vertex label.  This should be the same on each rank.
 * @param fan_out Host span defining branching out (fan-out) degree per source vertex for each
 * level.
 * @param flags A set of flags indicating which sampling features should be used.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return tuple device vectors (vertex_t source_vertex, vertex_t destination_vertex,
 * optional weight_t weight, optional edge_t edge id, optional edge_type_t edge type,
 * optional int32_t hop, optional label_t label, optional size_t offsets)
 */

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename bias_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
homogeneous_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<size_t const>> starting_vertex_offsets,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check = false);

/*
 * @brief renumber sampled edge list and compress to the (D)CSR|(D)CSC format.
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) output edges fulfilling
 * the following requirements. Assume major = source if @p src_is_major is true, major = destination
 * if @p src_is_major is false.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, hop, flag=major) triplets for each
 * vertex ID in edge majors (@p edgelist_srcs if @p src_is_major is true, @p edgelist_dsts if false)
 * and (vertex ID, hop, flag=minor) triplets for each vertex ID in edge minors. From these triplets,
 * we can find the minimum (hop, flag) pairs for every unique vertex ID (hop is the primary key and
 * flag is the secondary key, flag=major is considered smaller than flag=minor if hop numbers are
 * same). Vertex IDs with smaller (hop, flag) pairs precede vertex IDs with larger (hop, flag) pairs
 * in renumbering. Ordering can be arbitrary among the vertices with the same (hop, flag) pairs. If
 * @p seed_vertices.has_value() is true, we assume (hop=0, flag=major) for every vertex in @p
 * *seed_vertices in renumbering (this is relevant when there are seed vertices with no neighbors).
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors. If @p seed_vertices.has_value() is true, vertices in @p
 * *seed_vertices precede vertex IDs that appear only in edge minors as well.
 * 3. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * The renumbered edges are compressed based on the following requirements.
 *
 * 1. If @p compress_per_hop is true, edges are compressed separately for each hop. If @p
 * compress_per_hop is false, edges with different hop numbers are compressed altogether.
 * 2. Edges are compressed independently for different labels.
 * 3. If @p doubly_compress is false, edges are compressed to CSR (if @p src_is_major is true) or
 * CSC (if @p src_is_major is false). If @p doubly_compress is true, edges are compressed to DCSR
 * (if @p src_is_major is true) or DCSC (if @p src_is_major is false). If @p doubly_compress is
 * false, the CSR/CSC offset array size is the number of vertices (which is the maximum vertex ID +
 * 1) + 1. Here, the maximum vertex ID is the maximum major vertex ID in the edges to compress if @p
 * compress_per_hop is false or for hop 0 (@p seed_vertices should be included if valid). If @p
 * compress_per_hop is true and hop number is 1 or larger, the maximum vertex ID is the larger of
 * the maximum major vertex ID for this hop and the maximum vertex ID for the edges in the previous
 * hops.
 *
 * If both @p compress_per_hop is false and @p edgelist_hops.has_value() is true, majors should be
 * non-decreasing within each label after renumbering and sorting by (hop, major, minor). Also,
 * majors in hop N should not appear in any of the previous hops. This condition is satisfied if
 * majors in hop N + 1 does not have any vertices from the previous hops excluding the minors from
 * hop N.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing edgelist source vertices.
 * @param edgelist_dsts A vector storing edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_weights An optional vector storing edgelist weights (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_ids An optional vector storing edgelist edge IDs (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_types An optional vector storing edgelist edge types (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops should be valid if @p num_hops >= 2.
 * @param seed_vertices An optional pointer to the array storing seed vertices in hop 0.
 * @param seed_vertex_label_offsets An optional pointer to the array storing label offsets to the
 * seed vertices (size = @p num_labels + 1). @p seed_vertex_label_offsets should be valid if @p
 * num_labels >= 2 and @p seed_vertices is valid and invalid otherwise.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets should be valid if @p num_labels
 * >= 2.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and compression.
 * @param compress_per_hop A flag to determine whether to compress edges with different hop numbers
 * separately (if true) or altogether (if false). If @p compress_per_hop is true, @p
 * edgelist_hops.has_value() should be true and @p doubly_compress should be false.
 * @param doubly_compress A flag to determine whether to compress to the CSR/CSC format (if false)
 * or the DCSR/DCSC format (if true).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing optional DCSR/DCSC major vertex IDs with one or more neighbors,
 * (D)CSR|(D)CSC offset values, edge minor vertex IDs, optional edge weights (valid only if @p
 * edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), optional (label, hop) offset values to the
 * (D)CSR|(D)CSC offset array (size = @p num_labels * @p num_hops + 1, valid only when @p
 * edgelist_hops.has_value() or @p edgelist_label_offsets.has_value() is true), renumber_map to
 * query original vertices (size = # unique or aggregate # unique_vertices for each label), and
 * label offsets to the renumber_map (size = num_labels + 1, valid only if @p
 * edgelist_label_offsets.has_value() is true).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>,     // dcsr/dcsc major vertices
           rmm::device_uvector<size_t>,                      // (d)csr/(d)csc offset values
           rmm::device_uvector<vertex_t>,                    // minor vertices
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,  // (label, hop) offsets to the (d)csr/(d)csc
                                                        // offset array
           rmm::device_uvector<vertex_t>,               // renumber map
           std::optional<rmm::device_uvector<size_t>>>  // label offsets to the renumber map
renumber_and_compress_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  size_t num_labels,
  size_t num_hops,
  bool src_is_major       = true,
  bool compress_per_hop   = false,
  bool doubly_compress    = false,
  bool do_expensive_check = false);

/*
 * @brief renumber sampled edge list and sort the renumbered edges.
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) output edges fulfilling
 * the following requirements. Assume major = source if @p src_is_major is true, major = destination
 * if @p src_is_major is false.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, hop, flag=major) triplets for each
 * vertex ID in edge majors (@p edgelist_srcs if @p src_is_major is true, @p edgelist_dsts if false)
 * and (vertex ID, hop, flag=minor) triplets for each vertex ID in edge minors. From these triplets,
 * we can find the minimum (hop, flag) pairs for every unique vertex ID (hop is the primary key and
 * flag is the secondary key, flag=major is considered smaller than flag=minor if hop numbers are
 * same). Vertex IDs with smaller (hop, flag) pairs precede vertex IDs with larger (hop, flag) pairs
 * in renumbering. Ordering can be arbitrary among the vertices with the same (hop, flag) pairs. If
 * @p seed_vertices.has-value() is true, we assume (hop=0, flag=major) for every vertex in @p
 * *seed_vertices in renumbering (this is relevant when there are seed vertices with no neighbors).
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors. If @p seed_vertices.has_value() is true, vertices in @p
 * *seed_vertices precede vertex IDs that appear only in edge minors as well.
 * 3. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * The renumbered edges are sorted based on the following rules.
 *
 * 1. If @p src_is_major is true, use ((hop), src, dst) as the key in sorting. If @p src_is_major is
 * false, use ((hop), dst, src) instead. hop is used only if @p edgelist_hops.has_value() is true.
 * 2. Edges in each label are sorted independently if @p edgelist_label_offsets.has_value() is true.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing edgelist source vertices.
 * @param edgelist_dsts A vector storing edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_weights An optional vector storing edgelist weights (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_ids An optional vector storing edgelist edge IDs (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_types An optional vector storing edgelist edge types (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops should be valid if @p num_hops >= 2.
 * @param seed_vertices An optional pointer to the array storing seed vertices in hop 0.
 * @param seed_vertex_label_offsets An optional pointer to the array storing label offsets to the
 * seed vertices (size = @p num_labels + 1). @p seed_vertex_label_offsets should be valid if @p
 * num_labels >= 2 and @p seed_vertices is valid and invalid otherwise.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets should be valid if @p num_labels
 * >= 2.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid
 * only if @p edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), optional (label, hop) offset values to the renumbered
 * and sorted edges (size = @p num_labels * @p num_hops + 1, valid only when @p
 * edgelist_hops.has_value() or @p edgelist_label_offsetes.has_value() is true), renumber_map to
 * query original vertices (size = # unique or aggregate # unique vertices for each label), and
 * label offsets to the renumber map (size = @p num_labels + 1, valid only if @p
 * edgelist_label_offsets.has_value() is true).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,       // (label, hop) offsets to the edges
           rmm::device_uvector<vertex_t>,                    // renumber map
           std::optional<rmm::device_uvector<size_t>>>       // label offsets to the renumber map
renumber_and_sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  size_t num_labels,
  size_t num_hops,
  bool src_is_major       = true,
  bool do_expensive_check = false);

/*
 * @brief renumber sampled edge list (vertex & edge IDs) per vertex/edge type and sort the
 * renumbered edges.
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) output edge
 * source/destination vertex IDs fulfilling the following requirements. Assume major = source if @p
 * src_is_major is true, major = destination if @p src_is_major is false.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, hop, flag=major) triplets for each
 * vertex ID in edge majors (@p edgelist_srcs if @p src_is_major is true, @p edgelist_dsts if false)
 * and (vertex ID, hop, flag=minor) triplets for each vertex ID in edge minors. From these triplets,
 * we can find the minimum (hop, flag) pair for every unique vertex ID (hop is the primary key and
 * flag is the secondary key, flag=major is considered smaller than flag=minor if hop numbers are
 * same). Vertex IDs with smaller (hop, flag) pairs precede vertex IDs with larger (hop, flag) pairs
 * in renumbering (if their vertex types are same, vertices with different types are renumbered
 * separately). Ordering can be arbitrary among the vertices with the same (vertex type, hop, flag)
 * triplets. If @p seed_vertices.has_value() is true, we assume (hop=0, flag=major) for every vertex
 * in @p *seed_vertices in renumbering (this is relevant when there are seed vertices with no
 * neighbors).
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors. If @p seed_vertices.has_value() is true, vertices in @p
 * *seed_vertices precede vertex IDs that appear only in edge minors as well.
 * 3. Vertices with different types will be renumbered separately. Unique vertex IDs for each vertex
 * type are mapped to consecutive integers starting from 0.
 * 4. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * Edge IDs are renumbered fulfilling the following requirements (This is relevant only when @p
 * edgelist_edge_ids.has_value() is true).
 *
 * 1. If @p edgelist_hops is valid, we can consider (edge ID, hop) pairs. From these pairs, we can
 * find the minimum hop value for every unique edge ID. Edge IDs with smaller hop values precede
 * edge IDs with larger hop values in renumbering (if their edge types are same, edges with
 * different edge types are renumbered separately). Ordering can be arbitrary among the edge IDs
 * with the same (edge type, hop) pairs.
 * 2. If @p edgelist_edge_hops.has_value() is false, unique edge IDs (for each edge type is @p
 * edgelist_edge_types.has_value() is true) are mapped to consecutive integers starting from 0. The
 * ordering can be arbitrary.
 * 3. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * The renumbered edges are sorted based on the following rules.
 *
 * 1. If @p src_is_major is true, use ((edge type), (hop), src, dst) as the key in sorting. If @p
 * src_is_major is false, use ((edge type), (hop), dst, src) instead. edge type is used only if @p
 * edgelist_edge_types.has_value() is true. hop is used only if @p edgelist_hops.has_value() is
 * true.
 * 2. Edges in each label are sorted independently if @p edgelist_label_offsets.has_value() is true.
 *
 * This function assumes that there is a single edge source vertex type and a single edge
 * destination vertex type for each edge. If @p edgelist_edge_types.has_value() is false (i.e. there
 * is only one edge type), there should be only one edge source vertex type and only one edge
 * destination vertex type; the source & destination vertex types may or may not coincide.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing edgelist source vertices.
 * @param edgelist_dsts A vector storing edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_weights An optional vector storing edgelist weights (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_ids An optional vector storing edgelist edge IDs (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_types An optional vector storing edgelist edge types (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops should be valid if @p num_hops >= 2.
 * @param seed_vertices An optional pointer to the array storing seed vertices in hop 0.
 * @param seed_vertex_label_offsets An optional pointer to the array storing label offsets to the
 * seed vertices (size = @p num_labels + 1). @p seed_vertex_label_offsets should be valid if @p
 * num_labels >= 2 and @p seed_vertices is valid and invalid otherwise.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets should be valid if @p num_labels
 * >= 2.
 * @param vertex_type offsets A pointer to the array storing vertex type offsets for the entire
 * vertex ID range (array size = @p num_vertex_types + 1). For example, if the array stores [0, 100,
 * 200], vertex IDs [0, 100) has vertex type 0 and vertex IDs [100, 200) has vertex type 1.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
 * @param num_vertex_types Number of vertex types.
 * @param num_edge_types Number of edge types.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing renumbered edge sources, renumbered edge destinations, optional
 * edge weights (valid only if @p edgelist_weights.has_value() is true), optional renumbered edge
 * IDs (valid only if @p edgelist_edge_ids.has_value() is true), optional (label, edge type, hop)
 * offset values to the renumbered and sorted edges (size = @p num_labels * @p num_edge_types * @p
 * num_hops + 1, valid only when @p edgelist_edge_types.has_value(), @p edgelist_hops.has_value(),
 * or @p edgelist_label_offsetes.has_value() is true), renumber_map to query original vertices (size
 * = # unique or aggregate # unique vertices for each label), (label, vertex type) offsets to the
 * vertex renumber map (size = @p num_labels * @p num_vertex_types + 1), optional renumber_map to
 * query original edge IDs (size = # unique (edge_type, edge ID) pairs, valid only if @p
 * edgelist_edge_ids.has_value() is true), and optional (label, edge type) offsets to the edge ID
 * renumber map (size = @p num_labels + @p num_edge_types + 1, valid only if @p
 * edgelist_edge_ids.has_value() is true). We do not explicitly return edge source & destination
 * vertex types as we assume that source & destination vertex type are implicilty determined for a
 * given edge type.
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<
  rmm::device_uvector<vertex_t>,                  // srcs
  rmm::device_uvector<vertex_t>,                  // dsts
  std::optional<rmm::device_uvector<weight_t>>,   // weights
  std::optional<rmm::device_uvector<edge_id_t>>,  // edge IDs
  std::optional<rmm::device_uvector<size_t>>,     // (label, edge type, hop) offsets to the edges
  rmm::device_uvector<vertex_t>,                  // vertex renumber map
  rmm::device_uvector<size_t>,  // (label, vertex type) offsets to the vertex renumber map
  std::optional<rmm::device_uvector<edge_id_t>>,  // edge ID renumber map
  std::optional<
    rmm::device_uvector<size_t>>>  // (label, edge type) offsets to the edge ID renumber map
heterogeneous_renumber_and_sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  raft::device_span<vertex_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_hops,
  size_t num_vertex_types,
  size_t num_edge_types,
  bool src_is_major       = true,
  bool do_expensive_check = false);

/*
 * @brief sort sampled edge list.
 *
 * Sampled edges are sorted based on the following rules.
 *
 * 1. If @p src_is_major is true, use ((hop), src, dst) as the key in sorting. If @p src_is_major is
 * false, use ((hop), dst, src) instead. hop is used only if @p edgelist_hops.has_value() is true.
 * 2. Edges in each label are sorted independently if @p edgelist_label_offsets.has_value() is true.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing edgelist source vertices.
 * @param edgelist_dsts A vector storing edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_weights An optional vector storing edgelist weights (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_ids An optional vector storing edgelist edge IDs (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_types An optional vector storing edgelist edge types (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops must be valid if @p num_hops >= 2.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets must be valid if @p num_labels >= 2.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid
 * only if @p edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), and optional (label, hop) offset values to the sorted
 * edges (size = @p num_labels * @p num_hops + 1, valid only when @p edgelist_hops.has_value() or @p
 * edgelist_label_offsets.has_value() is true).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>>       // (label, hop) offsets to the edges
sort_sampled_edgelist(raft::handle_t const& handle,
                      rmm::device_uvector<vertex_t>&& edgelist_srcs,
                      rmm::device_uvector<vertex_t>&& edgelist_dsts,
                      std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                      std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
                      std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                      std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
                      std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
                      size_t num_labels,
                      size_t num_hops,
                      bool src_is_major       = true,
                      bool do_expensive_check = false);
/*
 * @brief Build map to lookup source and destination using edge id and type
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam edge_type_t Type of edge types. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_id_view View object holding edge ids of the edges of the graph pointed @p graph_view
 * @param edge_type_view View object holding edge types of the edges of the graph pointed @p
 * graph_view
 * @return An object of type cugraph::lookup_container_t that encapsulates edge id and type to
 * source and destination lookup map.
 */
template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
lookup_container_t<edge_t, edge_type_t, vertex_t> build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, edge_t const*> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view);

/*
 * @brief Lookup edge sources and destinations using edge ids and a single edge type.
 * Use this function to lookup endpoints of edges belonging to the same edge type.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam edge_type_t Type of edge types. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param lookup_container Object of type cugraph::lookup_container_t that encapsulates edge id and
 * type to source and destination lookup map.
 * @param edge_ids_to_lookup Device span of edge ids to lookup
 * @param edge_type_to_lookup Type of the edges corresponding to edge ids in @p edge_ids_to_lookup
 * @return A tuple of device vector containing edge sources and destinations for edge ids
 * in @p edge_ids_to_lookup and edge type @. If an edge id in @p edge_ids_to_lookup or
 * edge type @edge_type_to_lookup is not found, the corresponding entry in the device vectors of
 * the returned tuple will contain cugraph::invalid_vertex_id<vertex_t>.
 */
template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_single_type(
  raft::handle_t const& handle,
  lookup_container_t<edge_t, edge_type_t, vertex_t> const& lookup_container,
  raft::device_span<edge_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup);

/*
 * @brief Lookup edge sources and destinations using edge ids and edge types.
 * Use this function to lookup endpoints of edges belonging to different edge types.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam edge_type_t Type of edge types. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param lookup_container Object of type cugraph::lookup_container_t that encapsulates edge id and
 * type to source and destination lookup map.
 * @param edge_ids_to_lookup Device span of edge ids to lookup
 * @param edge_types_to_lookup Device span of edge types corresponding to the edge ids
 * in @p edge_ids_to_lookup
 * @return A tuple of device vector containing edge sources and destinations for the edge ids
 * in @p edge_ids_to_lookup and the edge types in @p edge_types_to_lookup. If an edge id in
 * @p edge_ids_to_lookup or edge type in @p edge_types_to_lookup is not found, the
 * corresponding entry in the device vectors of the returned tuple will contain
 * cugraph::invalid_vertex_id<vertex_t>.
 */
template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_types(
  raft::handle_t const& handle,
  lookup_container_t<edge_t, edge_type_t, vertex_t> const& lookup_container,
  raft::device_span<edge_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup);

/**
 * @brief Negative Sampling
 *
 * This function generates negative samples for graph.
 *
 * Negative sampling is done by generating a random graph according to the specified
 * parameters and optionally removing samples that represent actual edges in the graph
 *
 * Sampling occurs by creating a list of source vertex ids from biased samping
 * of the source vertex space, and destination vertex ids from biased sampling of the
 * destination vertex space, and using this as the putative list of edges.  We
 * then can optionally remove duplicates and remove actual edges in the graph to generate
 * the final list.  If necessary we will repeat the process to end with a resulting
 * edge list of the appropriate size.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether sources (if false) or destinations (if
 * true) are major indices
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate NBR Sampling for
 * @param rng_state RNG state
 * @param src_biases Optional bias for randomly selecting source vertices.  If std::nullopt vertices
 * will be selected uniformly.  In multi-GPU environment the biases should be partitioned based
 * on the vertex partitions.
 * @param dst_biases Optional bias for randomly selecting destination vertices.  If std::nullopt
 * vertices will be selected uniformly.  In multi-GPU environment the biases should be partitioned
 * based on the vertex partitions.
 * @param num_samples Number of negative samples to generate
 * @param remove_duplicates If true, remove duplicate samples
 * @param remove_existing_edges If true, remove samples that are actually edges in the graph
 * @param exact_number_of_samples If true, repeat generation until we get the exact number of
 * negative samples
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 *
 * @return tuple containing source vertex ids and destination vertex ids for the negative samples
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> negative_sampling(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<raft::device_span<weight_t const>> src_biases,
  std::optional<raft::device_span<weight_t const>> dst_biases,
  size_t num_samples,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check);

}  // namespace cugraph
