/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

// FIXME: Functions in this file assume that store_transposed=false,
//    in implementation, naming and documentation.  We should review these and
//    consider updating things to support an arbitrary value for store_transposed

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
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param active_major_labels Optional device vector containing labels for each device vector
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
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_edge_type_view,
  rmm::device_uvector<vertex_t> const& active_majors,
  std::optional<rmm::device_uvector<label_t>> const& active_major_labels,
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
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param edge_id_view Optional view object holding edge ids for @p graph_view.
 * @param edge_type_view Optional view object holding edge types for @p graph_view.
 * @param rng_state Random number generator state
 * @param active_majors Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param active_major_labels Optional device vector containing labels for each device vector
 * @param fanout How many edges to sample for each vertex
 * @param with_replacement If true sample with replacement, otherwise sample without replacement
 * @param invalid_vertex_id Value to use for an invalid vertex
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
sample_edges(raft::handle_t const& handle,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
             std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_edge_type_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<vertex_t> const& active_majors,
             std::optional<rmm::device_uvector<label_t>> const& active_major_labels,
             size_t fanout,
             bool with_replacement);

// FIXME: Should this go in shuffle_wrappers.hpp?
/**
 * @brief Shuffle sampling results to the desired GPU based on label_to_output_gpu_mapping[label[i]]
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam edge_type_t Type of edge type. Needs to be an integral type.
 * @tparam label_t Type of label. Needs to be an integral type.
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param major Major vertices
 * @param minor Minor vertices
 * @param wgt   Optional edge weight
 * @param edge_id Optional edge id
 * @param edge_type Optional edge type
 * @param hop Optional indicator of which hop the edge was found in
 * @param label Label associated with the seed that resulted in this edge being part of the result
 * @param label_to_output_gpu_mapping Mapping that identifies the GPU that is associated with the
 * output label
 *
 * @return A tuple of device vectors containing the majors, minors, optional weights,
 *  optional edge ids, optional edge types, optional hops and optional labels after shuffling to
 *  the specified output GPU
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<label_t>>>
shuffle_sampling_results(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>&& major,
                         rmm::device_uvector<vertex_t>&& minor,
                         std::optional<rmm::device_uvector<weight_t>>&& wgt,
                         std::optional<rmm::device_uvector<edge_t>>&& edge_id,
                         std::optional<rmm::device_uvector<edge_type_t>>&& edge_type,
                         std::optional<rmm::device_uvector<int32_t>>&& hop,
                         std::optional<rmm::device_uvector<label_t>>&& label,
                         raft::device_span<int32_t const> label_to_output_gpu_mapping);

/**
 * @brief Convert a CSR-style set of offsets into a list of values (COO-style)
 *
 * @tparam label_t Type of label
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param label offsets The offsets array for labels
 *
 * @returns device vector containing labels for each seed vertex
 */
template <typename label_t>
rmm::device_uvector<label_t> expand_label_offsets(raft::handle_t const& handle,
                                                  rmm::device_uvector<size_t> const& label_offsets);

/**
 * @brief Convert a sorted list of labels into an offsets array (CSR-style)
 *
 * @tparam label_t Type of label
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param major Major vertices
 *
 * @returns device vector offsets for the specified labels
 */
template <typename label_t>
rmm::device_uvector<size_t> construct_label_offsets(raft::handle_t const& handle,
                                                    rmm::device_uvector<label_t> const& label,
                                                    size_t num_labels);

}  // namespace detail
}  // namespace cugraph
