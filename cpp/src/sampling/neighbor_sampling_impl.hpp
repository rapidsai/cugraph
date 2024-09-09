/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "prims/fill_edge_property.cuh"
#include "prims/transform_e.cuh"

#include "sampling/detail/sampling_utils.hpp"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/vertex_partition_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

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
neighbor_sample_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> this_frontier_vertices,
  std::optional<raft::device_span<label_t const>> this_frontier_vertex_labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  std::optional<raft::host_span<int32_t const>> fan_out,
  std::optional<std::tuple<raft::host_span<int32_t const>, raft::host_span<int32_t const>>>
    heterogeneous_fan_out,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  raft::random::RngState& rng_state,
  bool do_expensive_check)
{

  std::vector<cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, bool>> edge_masks_vector{};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};


  if (fan_out) {
    CUGRAPH_EXPECTS((*fan_out).size() > 0,
                    "Invalid input argument: number of levels must be non-zero.");
    CUGRAPH_EXPECTS(
      (*fan_out).size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "Invalid input argument: number of levels should not overflow int32_t");  // as we use int32_t
                                                                                // to store hops
  } else {

    CUGRAPH_EXPECTS(
      std::get<0>(*heterogeneous_fan_out).back() == std::get<1>(*heterogeneous_fan_out).size() &&
        std::get<1>(*heterogeneous_fan_out).size() != 0,
      "Invalid input argument: number of levels and size must match and should be non zero.");
    

    CUGRAPH_EXPECTS(
      std::get<0>(*heterogeneous_fan_out).size() <=
          static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
        std::get<1>(*heterogeneous_fan_out).size() <=
          static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "Invalid input argument: number of levels should not overflow int32_t");  // as we use int32_t
                                                                                // to store hops

    edge_masks_vector.reserve(std::get<0>(*heterogeneous_fan_out).size() - 1);
    
    for (int i = 0; i < std::get<0>(*heterogeneous_fan_out).size() - 1; i++) {

      cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, bool> edge_mask(handle, graph_view);
      
      cugraph::fill_edge_property(handle, graph_view, edge_mask.mutable_view(), bool{true});

      (*modified_graph_view).attach_edge_mask(edge_mask.view());

      cugraph::transform_e(
        handle,
        *modified_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        *edge_type_view,
        [valid_edge_type = i] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto edge_type) {
          return edge_type == valid_edge_type;
        },
        edge_mask.mutable_view(),
        false);
      
      edge_masks_vector.push_back(std::move(edge_mask));
    }
    
  }

  if constexpr (!multi_gpu) {
    CUGRAPH_EXPECTS(!label_to_output_comm_rank,
                    "cannot specify output GPU mapping in SG implementation");
  }

  CUGRAPH_EXPECTS(
    !label_to_output_comm_rank || this_frontier_vertex_labels,
    "cannot specify output GPU mapping without also specifying this_frontier_vertex_labels");

  if (do_expensive_check) {
    if (edge_bias_view) {
      auto [num_negative_edge_weights, num_overflows] =
        check_edge_bias_values(handle, graph_view, *edge_bias_view);

      CUGRAPH_EXPECTS(
        num_negative_edge_weights == 0,
        "Invalid input argument: input edge bias values should have non-negative values.");
      CUGRAPH_EXPECTS(num_overflows == 0,
                      "Invalid input argument: sum of neighboring edge bias values should not "
                      "exceed std::numeric_limits<bias_t>::max() for any vertex.");
    }

    if (label_to_output_comm_rank) {
      CUGRAPH_EXPECTS(cugraph::detail::is_sorted(handle, std::get<0>(*label_to_output_comm_rank)),
                      "Labels in label_to_output_comm_rank must be sorted");
    }
  }

  std::vector<rmm::device_uvector<vertex_t>> level_result_src_vectors{};
  std::vector<rmm::device_uvector<vertex_t>> level_result_dst_vectors{};
  auto level_result_weight_vectors =
    edge_weight_view ? std::make_optional(std::vector<rmm::device_uvector<weight_t>>{})
                     : std::nullopt;
  auto level_result_edge_id_vectors =
    edge_id_view ? std::make_optional(std::vector<rmm::device_uvector<edge_t>>{}) : std::nullopt;
  auto level_result_edge_type_vectors =
    edge_type_view ? std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>{})
                   : std::nullopt;
  auto level_result_label_vectors =
    this_frontier_vertex_labels ? std::make_optional(std::vector<rmm::device_uvector<label_t>>{})
                                : std::nullopt;

  level_result_src_vectors.reserve((*fan_out).size());
  level_result_dst_vectors.reserve((*fan_out).size());
  if (level_result_weight_vectors) { (*level_result_weight_vectors).reserve((*fan_out).size()); }
  if (level_result_edge_id_vectors) { (*level_result_edge_id_vectors).reserve((*fan_out).size()); }
  if (level_result_edge_type_vectors) {
    (*level_result_edge_type_vectors).reserve((*fan_out).size());
  }
  if (level_result_label_vectors) { (*level_result_label_vectors).reserve((*fan_out).size()); }

  rmm::device_uvector<vertex_t> frontier_vertices(0, handle.get_stream());
  auto frontier_vertex_labels =
    this_frontier_vertex_labels
      ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
      : std::nullopt;

  std::optional<
    std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<label_t>>>>
    vertex_used_as_source{std::nullopt};

  if (prior_sources_behavior == prior_sources_behavior_t::EXCLUDE) {
    vertex_used_as_source = std::make_optional(
      std::make_tuple(rmm::device_uvector<vertex_t>{0, handle.get_stream()},
                      this_frontier_vertex_labels
                        ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
                        : std::nullopt));
  }

  std::vector<size_t> level_sizes{};
  int32_t hop{0};
  int32_t edge_type_id_max{1}; // A value of 1 translate to homogeneous neighbor sample
  int32_t num_edge_type_per_hop{0};
  
  auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;

  if (heterogeneous_fan_out) {
    num_edge_type_per_hop = std::get<0>(*heterogeneous_fan_out).back() - 1;
  }

  while(true) {
    int32_t k_level{0};
    if (fan_out) {
      k_level = (*fan_out)[hop];
      if (hop == (*fan_out).size()) {
        break;
      }
    } else if (heterogeneous_fan_out) {
        if (num_edge_type_per_hop == 0) {
          break;
        }
        edge_type_id_max = std::get<0>(*heterogeneous_fan_out).back() - 1;
      
    }
  
    for (int i = 0; i < edge_type_id_max; i++) {
      if (heterogeneous_fan_out) { 
        // Can make the code easier to read by setting a mask for both
        // homogeneous and heterogeneous neighbor sample. For the former,
        // no edges should be masked
        cur_graph_view.attach_edge_mask(edge_masks_vector[i].view());
        auto k_level_size = (std::get<1>(*heterogeneous_fan_out)[i + 1] - std::get<1>(*heterogeneous_fan_out)[i]);
        if (k_level_size > hop) {
          k_level = i + hop;
        } else { // otherwise, k_level = 0
          --num_edge_type_per_hop ;

        }
      }
      rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
      std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
      std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

      if (k_level > 0) {
        std::tie(srcs, dsts, weights, edge_ids, edge_types, labels) =
          sample_edges(handle,
                      cur_graph_view,
                      edge_weight_view,
                      edge_id_view,
                      edge_type_view,
                      edge_bias_view,
                      rng_state,
                      this_frontier_vertices,
                      this_frontier_vertex_labels,
                      static_cast<size_t>(k_level),
                      with_replacement);
      } else {
        std::tie(srcs, dsts, weights, edge_ids, edge_types, labels) =
          gather_one_hop_edgelist(handle,
                                  cur_graph_view,
                                  edge_weight_view,
                                  edge_id_view,
                                  edge_type_view,
                                  this_frontier_vertices,
                                  this_frontier_vertex_labels);
      }

      level_sizes.push_back(srcs.size());

      level_result_src_vectors.push_back(std::move(srcs));
      level_result_dst_vectors.push_back(std::move(dsts));
      if (weights) { (*level_result_weight_vectors).push_back(std::move(*weights)); }
      if (edge_ids) { (*level_result_edge_id_vectors).push_back(std::move(*edge_ids)); }
      if (edge_types) { (*level_result_edge_type_vectors).push_back(std::move(*edge_types)); }
      if (labels) { (*level_result_label_vectors).push_back(std::move(*labels)); }
    }

    ++hop;


    // aggregation here for each hope regardless of the type  ****************************************************
    // Genrate the mask for each type and store it for each iteration. mask vector at the beginning, populate that
    // and within the inner loop, activate the right mask based and proceed.
    
    if (hop < (*fan_out).size()) {
      // FIXME:  We should modify vertex_partition_range_lasts to return a raft::host_span
      //  rather than making a copy.
      auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts();
      std::tie(frontier_vertices, frontier_vertex_labels, vertex_used_as_source) =
        prepare_next_frontier(
          handle,
          this_frontier_vertices,
          this_frontier_vertex_labels,
          raft::device_span<vertex_t const>{level_result_dst_vectors.back().data(),
                                            level_result_dst_vectors.back().size()},
          frontier_vertex_labels ? std::make_optional(raft::device_span<label_t const>(
                                     level_result_label_vectors->back().data(),
                                     level_result_label_vectors->back().size()))
                                 : std::nullopt,
          std::move(vertex_used_as_source),
          cur_graph_view.local_vertex_partition_view(),
          vertex_partition_range_lasts,
          prior_sources_behavior,
          dedupe_sources,
          do_expensive_check);

      this_frontier_vertices =
        raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size());

      if (frontier_vertex_labels) {
        this_frontier_vertex_labels = raft::device_span<label_t const>(
          frontier_vertex_labels->data(), frontier_vertex_labels->size());
      }
    }
  } // While loop end here: FIXME: remove this comment

  auto result_size = std::reduce(level_sizes.begin(), level_sizes.end());
  size_t output_offset{};

  rmm::device_uvector<vertex_t> result_srcs(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < level_result_src_vectors.size(); ++i) {
    raft::copy(result_srcs.begin() + output_offset,
               level_result_src_vectors[i].begin(),
               level_sizes[i],
               handle.get_stream());
    output_offset += level_sizes[i];
  }
  level_result_src_vectors.clear();
  level_result_src_vectors.shrink_to_fit();

  rmm::device_uvector<vertex_t> result_dsts(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < level_result_dst_vectors.size(); ++i) {
    raft::copy(result_dsts.begin() + output_offset,
               level_result_dst_vectors[i].begin(),
               level_sizes[i],
               handle.get_stream());
    output_offset += level_sizes[i];
  }
  level_result_dst_vectors.clear();
  level_result_dst_vectors.shrink_to_fit();

  auto result_weights =
    level_result_weight_vectors
      ? std::make_optional(rmm::device_uvector<weight_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_weights) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_weight_vectors).size(); ++i) {
      raft::copy((*result_weights).begin() + output_offset,
                 (*level_result_weight_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_weight_vectors = std::nullopt;
  }

  auto result_edge_ids =
    level_result_edge_id_vectors
      ? std::make_optional(rmm::device_uvector<edge_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_ids) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_edge_id_vectors).size(); ++i) {
      raft::copy((*result_edge_ids).begin() + output_offset,
                 (*level_result_edge_id_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_edge_id_vectors = std::nullopt;
  }

  auto result_edge_types =
    level_result_edge_type_vectors
      ? std::make_optional(rmm::device_uvector<edge_type_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_types) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_edge_type_vectors).size(); ++i) {
      raft::copy((*result_edge_types).begin() + output_offset,
                 (*level_result_edge_type_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_edge_type_vectors = std::nullopt;
  }

  std::optional<rmm::device_uvector<int32_t>> result_hops{std::nullopt};

  if (return_hops) {
    result_hops   = rmm::device_uvector<int32_t>(result_size, handle.get_stream());
    output_offset = 0;
    for (size_t i = 0; i < (*fan_out).size(); ++i) {
      scalar_fill(
        handle, result_hops->data() + output_offset, level_sizes[i], static_cast<int32_t>(i));
      output_offset += level_sizes[i];
    }
  }

  auto result_labels =
    level_result_label_vectors
      ? std::make_optional(rmm::device_uvector<label_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_labels) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_label_vectors).size(); ++i) {
      raft::copy((*result_labels).begin() + output_offset,
                 (*level_result_label_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_label_vectors = std::nullopt;
  }

  return detail::shuffle_and_organize_output(handle,
                                             std::move(result_srcs),
                                             std::move(result_dsts),
                                             std::move(result_weights),
                                             std::move(result_edge_ids),
                                             std::move(result_edge_types),
                                             std::move(result_hops),
                                             std::move(result_labels),
                                             label_to_output_comm_rank);
}

}  // namespace detail

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
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check)
{
  using bias_t = weight_t;  // dummy
  return detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
    handle,
    graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    std::nullopt,
    starting_vertices,
    starting_vertex_labels,
    label_to_output_comm_rank,
    std::make_optional(fan_out),
    std::nullopt,
    return_hops,
    with_replacement,
    prior_sources_behavior,
    dedupe_sources,
    rng_state,
    do_expensive_check);
}

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
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check)
{
  return detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
    handle,
    graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    edge_bias_view,
    starting_vertices,
    starting_vertex_labels,
    label_to_output_comm_rank,
    fan_out,
    std::nullopt,
    return_hops,
    with_replacement,
    prior_sources_behavior,
    dedupe_sources,
    rng_state,
    do_expensive_check);
}

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
heterogeneous_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  std::tuple<raft::host_span<int32_t const>, raft::host_span<int32_t const>>
    heterogeneous_fan_out,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check)
{
  // for loop : over the number of edgetypes /
  // inside the loop, create an edge mask and update the edge mask to mask out all the irrelevant edges
  // adn call neighborhood sampling

  // before this, there is a reordering constrain.
  // 10 labels amd 100 seeds, 3 edge type. For each seed, you provide sample of each edge type (0, 1, 2)
  // soritng criterial, label , seed, edge tyope
  // label, edge type, seed

  // Pick how to sort - requries assembling? if the natural output order based on this scheme if it doesn't meet the requirement, needs to shuffle
  // shuffling, label to be the primiary key? if you loop over the edge tyoe, edge type will be the primary key.
  
  // Goal: reorder the sample at the end - Label is the primary key? then label? then seed?
  // stable sort? to not mess up the internal order? "Use scatter operation to be more efficient?"

  // Iterate over all edge types

  cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, bool> edge_mask(handle, graph_view);
  cugraph::fill_edge_property(handle, graph_view, edge_mask.mutable_view(), bool{true});


  std::vector<rmm::device_uvector<vertex_t>> level_result_src_vectors{};
  std::vector<rmm::device_uvector<vertex_t>> level_result_dst_vectors{};
  auto level_result_weight_vectors =
    edge_weight_view ? std::make_optional(std::vector<rmm::device_uvector<weight_t>>{})
                     : std::nullopt;
  auto level_result_edge_id_vectors =
    edge_id_view ? std::make_optional(std::vector<rmm::device_uvector<edge_t>>{}) : std::nullopt;
  auto level_result_edge_type_vectors =
    edge_type_view ? std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>{})
                   : std::nullopt;
  auto level_result_label_vectors =
    starting_vertex_labels ? std::make_optional(std::vector<rmm::device_uvector<label_t>>{})
                                : std::nullopt;
  
    level_result_src_vectors.reserve(std::get<0>(heterogeneous_fan_out).size() - 1);
  level_result_dst_vectors.reserve(std::get<0>(heterogeneous_fan_out).size() - 1);
  if (level_result_weight_vectors) { (*level_result_weight_vectors).reserve(std::get<0>(heterogeneous_fan_out).size() - 1); }
  if (level_result_edge_id_vectors) { (*level_result_edge_id_vectors).reserve(std::get<0>(heterogeneous_fan_out).size() - 1); }
  if (level_result_edge_type_vectors) {
    (*level_result_edge_type_vectors).reserve(std::get<0>(heterogeneous_fan_out).size() - 1);
  }
  if (level_result_label_vectors) { (*level_result_label_vectors).reserve(std::get<0>(heterogeneous_fan_out).size() - 1); }


  








  std::vector<size_t> level_sizes{};
  for (int i = 0; i < std::get<0>(heterogeneous_fan_out).size() - 1; i++) {

    cugraph::transform_e(
        handle,
        graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        *edge_type_view,
        [valid_edge_type = i] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto edge_type) {
          return edge_type == valid_edge_type;
        },
        edge_mask.mutable_view(),
        false);
    
    auto [
      level_result_srcs,
      level_result_dsts,
      level_result_weights,
      level_result_edge_ids,
      level_result_edge_types,
      level_result_hops,
      level_result_labels,
      label_to_output_comm_rank_ // unused
      ] = detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
        handle,
        graph_view,
        edge_weight_view,
        edge_id_view,
        edge_type_view,
        edge_bias_view,
        starting_vertices,
        starting_vertex_labels,
        label_to_output_comm_rank,
        std::nullopt,
        heterogeneous_fan_out,
        return_hops,
        with_replacement,
        prior_sources_behavior,
        dedupe_sources,
        rng_state,
        do_expensive_check);
    
    level_sizes.push_back(level_result_srcs.size());

    level_result_src_vectors.push_back(std::move(level_result_srcs));
    level_result_dst_vectors.push_back(std::move(level_result_dsts));
    if (level_result_weights) { (*level_result_weight_vectors).push_back(std::move(*level_result_weights)); }
    if (level_result_edge_ids) { (*level_result_edge_id_vectors).push_back(std::move(*level_result_edge_ids)); }
    if (level_result_edge_types) { (*level_result_edge_type_vectors).push_back(std::move(*level_result_edge_types)); }
    if (level_result_labels) { (*level_result_label_vectors).push_back(std::move(*level_result_labels)); }
  }


  auto result_size = std::reduce(level_sizes.begin(), level_sizes.end());

  size_t output_offset{};

  rmm::device_uvector<vertex_t> result_srcs(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < level_result_src_vectors.size(); ++i) {
    raft::copy(result_srcs.begin() + output_offset,
               level_result_src_vectors[i].begin(),
               level_sizes[i],
               handle.get_stream());
    output_offset += level_sizes[i];
  }
  level_result_src_vectors.clear();
  level_result_src_vectors.shrink_to_fit();

  rmm::device_uvector<vertex_t> result_dsts(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < level_result_dst_vectors.size(); ++i) {
    raft::copy(result_dsts.begin() + output_offset,
               level_result_dst_vectors[i].begin(),
               level_sizes[i],
               handle.get_stream());
    output_offset += level_sizes[i];
  }
  level_result_dst_vectors.clear();
  level_result_dst_vectors.shrink_to_fit();

  auto result_weights =
    level_result_weight_vectors
      ? std::make_optional(rmm::device_uvector<weight_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_weights) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_weight_vectors).size(); ++i) {
      raft::copy((*result_weights).begin() + output_offset,
                 (*level_result_weight_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_weight_vectors = std::nullopt;
  }

  auto result_edge_ids =
    level_result_edge_id_vectors
      ? std::make_optional(rmm::device_uvector<edge_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_ids) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_edge_id_vectors).size(); ++i) {
      raft::copy((*result_edge_ids).begin() + output_offset,
                 (*level_result_edge_id_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_edge_id_vectors = std::nullopt;
  }

  auto result_edge_types =
    level_result_edge_type_vectors
      ? std::make_optional(rmm::device_uvector<edge_type_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_types) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_edge_type_vectors).size(); ++i) {
      raft::copy((*result_edge_types).begin() + output_offset,
                 (*level_result_edge_type_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_edge_type_vectors = std::nullopt;
  }

  std::optional<rmm::device_uvector<int32_t>> result_hops{std::nullopt};

  if (return_hops) {
    result_hops   = rmm::device_uvector<int32_t>(result_size, handle.get_stream());
    output_offset = 0;
    for (size_t i = 0; i < std::get<0>(heterogeneous_fan_out).size() - 1; ++i) {
      detail::scalar_fill(
        handle, result_hops->data() + output_offset, level_sizes[i], static_cast<int32_t>(i));
      output_offset += level_sizes[i];
    }
  }

  auto result_labels =
    level_result_label_vectors
      ? std::make_optional(rmm::device_uvector<label_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_labels) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_label_vectors).size(); ++i) {
      raft::copy((*result_labels).begin() + output_offset,
                 (*level_result_label_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_label_vectors = std::nullopt;
  }

  // 

  /*
  return std::make_tuple(
    std::move(result_srcs),
    std::move(result_dsts),
    std::move(result_weights),
    std::move(result_edge_ids),
    std::move(result_edge_types),
    std::move(result_hops),
    std::move(result_labels),
    //offsets
    //std::move(label_to_output_comm_rank)
  );
  */

  // Remove and pick the one on top
  return detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
    handle,
    graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    edge_bias_view,
    starting_vertices,
    starting_vertex_labels,
    label_to_output_comm_rank,
    std::nullopt,
    heterogeneous_fan_out,
    return_hops,
    with_replacement,
    prior_sources_behavior,
    dedupe_sources,
    rng_state,
    do_expensive_check);







}

  /*
  return detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
    handle,
    graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    edge_bias_view,
    starting_vertices,
    starting_vertex_labels,
    label_to_output_comm_rank,
    std::nullopt,
    heterogeneous_fan_out,
    return_hops,
    with_replacement,
    prior_sources_behavior,
    dedupe_sources,
    rng_state,
    do_expensive_check);
//}
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
homogeneous_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check)
{
  return detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
    handle,
    graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    edge_bias_view,
    starting_vertices,
    starting_vertex_labels,
    label_to_output_comm_rank,
    fan_out,
    std::nullopt,
    return_hops,
    with_replacement,
    prior_sources_behavior,
    dedupe_sources,
    rng_state,
    do_expensive_check);
}











}  // namespace cugraph
