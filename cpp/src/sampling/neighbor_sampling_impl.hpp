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
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <thrust/unique.h>

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
neighbor_sample_impl(raft::handle_t const& handle,
                     raft::random::RngState& rng_state,
                     graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
                     std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
                     std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
                     raft::device_span<vertex_t const> starting_vertices,
                     std::optional<raft::device_span<label_t const>> starting_vertex_labels,
                     std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
                     raft::host_span<int32_t const> fan_out,
                     std::optional<edge_type_t> num_edge_types,  // valid if heterogeneous sampling
                     bool return_hops,
                     bool with_replacement,
                     prior_sources_behavior_t prior_sources_behavior,
                     bool dedupe_sources,
                     bool do_expensive_check)
{
  if constexpr (multi_gpu) {
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();
    std::cout << "major_comm_rank=" << major_comm_rank << " major_comm_size=" << major_comm_size
              << " minor_comm_rank=" << minor_comm_rank << " minor_comm_size=" << minor_comm_size
              << std::endl;
  }

  static_assert(std::is_floating_point_v<bias_t>);

  if constexpr (!multi_gpu) {
    CUGRAPH_EXPECTS(!label_to_output_comm_rank,
                    "cannot specify output GPU mapping in SG implementation");
  }

  CUGRAPH_EXPECTS(
    !label_to_output_comm_rank || starting_vertex_labels,
    "cannot specify output GPU mapping without also specifying starting_vertex_labels");

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
  }

  CUGRAPH_EXPECTS(fan_out.size() > 0, "Invalid input argument: number of levels must be non-zero.");
  CUGRAPH_EXPECTS(
    fan_out.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
    "Invalid input argument: number of levels should not overflow int32_t");  // as we use int32_t
                                                                              // to store hops

  // Get the number of hop. If homogeneous neighbor sample, num_edge_types = 1.
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();

  auto num_hops = raft::div_rounding_up_safe(
    fan_out.size(), static_cast<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}));

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

  level_result_src_vectors.reserve(num_hops);
  level_result_dst_vectors.reserve(num_hops);

  if (level_result_weight_vectors) { (*level_result_weight_vectors).reserve(num_hops); }
  if (level_result_edge_id_vectors) { (*level_result_edge_id_vectors).reserve(num_hops); }
  if (level_result_edge_type_vectors) { (*level_result_edge_type_vectors).reserve(num_hops); }
  if (level_result_label_vectors) { (*level_result_label_vectors).reserve(num_hops); }

  rmm::device_uvector<vertex_t> frontier_vertices(0, handle.get_stream());

  auto frontier_vertex_labels =
    starting_vertex_labels
      ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
      : std::nullopt;

  std::optional<
    std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<label_t>>>>
    vertex_used_as_source{std::nullopt};

  if (prior_sources_behavior == prior_sources_behavior_t::EXCLUDE) {
    vertex_used_as_source = std::make_optional(
      std::make_tuple(rmm::device_uvector<vertex_t>{0, handle.get_stream()},
                      starting_vertex_labels
                        ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
                        : std::nullopt));
  }

  std::vector<size_t> level_sizes{};

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1 = std::chrono::steady_clock::now();
  for (size_t hop = 0; hop < num_hops; ++hop) {
    std::optional<std::vector<size_t>> level_Ks{std::nullopt};
    std::optional<std::vector<uint8_t>> gather_flags{std::nullopt};

    auto start_offset = hop * (num_edge_types ? *num_edge_types : edge_type_t{1});
    auto end_offset =
      start_offset +
      std::min(static_cast<size_t>((num_edge_types ? *num_edge_types : edge_type_t{1})),
               fan_out.size() -
                 hop * static_cast<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}));
    for (size_t i = start_offset; i < end_offset; ++i) {
      if (fan_out[i] > 0) {
        if (!level_Ks) {
          level_Ks = std::vector<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}, 0);
        }
        (*level_Ks)[i - start_offset] = fan_out[i];
      } else if (fan_out[i] < 0) {
        if (!gather_flags) {
          gather_flags = std::vector<uint8_t>(num_edge_types ? *num_edge_types : edge_type_t{1},
                                              static_cast<uint8_t>(false));
        }
        (*gather_flags)[i - start_offset] = static_cast<uint8_t>(true);
      }
    }

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
    std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
    std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

    if (level_Ks) {
      std::tie(srcs, dsts, weights, edge_ids, edge_types, labels) = sample_edges(
        handle,
        graph_view,
        edge_weight_view,
        edge_id_view,
        edge_type_view,
        edge_bias_view,
        rng_state,
        hop == 0
          ? starting_vertices
          : raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()),
        hop == 0 ? starting_vertex_labels
        : starting_vertex_labels
          ? std::make_optional(raft::device_span<label_t const>(frontier_vertex_labels->data(),
                                                                frontier_vertex_labels->size()))
          : std::nullopt,
        raft::host_span<size_t const>(level_Ks->data(), level_Ks->size()),
        with_replacement);
    }

    if (gather_flags) {
      auto [tmp_srcs, tmp_dsts, tmp_weights, tmp_edge_ids, tmp_edge_types, tmp_labels] =
        gather_one_hop_edgelist(
          handle,
          graph_view,
          edge_weight_view,
          edge_id_view,
          edge_type_view,
          hop == 0
            ? starting_vertices
            : raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()),
          hop == 0 ? starting_vertex_labels
          : starting_vertex_labels
            ? std::make_optional(raft::device_span<label_t const>(frontier_vertex_labels->data(),
                                                                  frontier_vertex_labels->size()))
            : std::nullopt,
          num_edge_types ? std::make_optional<raft::host_span<uint8_t const>>(gather_flags->data(),
                                                                              gather_flags->size())
                         : std::nullopt);

      auto old_size = srcs.size();
      if (old_size > 0) {
        auto new_size = old_size + tmp_srcs.size();
        srcs.resize(new_size, handle.get_stream());
        raft::copy(srcs.begin() + old_size, tmp_srcs.begin(), tmp_srcs.size(), handle.get_stream());
        dsts.resize(new_size, handle.get_stream());
        raft::copy(dsts.begin() + old_size, tmp_dsts.begin(), tmp_dsts.size(), handle.get_stream());
        if (weights) {
          weights->resize(new_size, handle.get_stream());
          raft::copy(weights->begin() + old_size,
                     tmp_weights->begin(),
                     tmp_weights->size(),
                     handle.get_stream());
        }
        if (edge_ids) {
          edge_ids->resize(new_size, handle.get_stream());
          raft::copy(edge_ids->begin() + old_size,
                     tmp_edge_ids->begin(),
                     tmp_edge_ids->size(),
                     handle.get_stream());
        }
        if (edge_types) {
          edge_types->resize(new_size, handle.get_stream());
          raft::copy(edge_types->begin() + old_size,
                     tmp_edge_types->begin(),
                     tmp_edge_types->size(),
                     handle.get_stream());
        }
        if (labels) {
          labels->resize(new_size, handle.get_stream());
          raft::copy(labels->begin() + old_size,
                     tmp_labels->begin(),
                     tmp_labels->size(),
                     handle.get_stream());
        }
      } else {
        srcs       = std::move(tmp_srcs);
        dsts       = std::move(tmp_dsts);
        weights    = std::move(tmp_weights);
        edge_ids   = std::move(tmp_edge_ids);
        edge_types = std::move(tmp_edge_types);
        labels     = std::move(tmp_labels);
      }
    }

    level_sizes.push_back(srcs.size());
    level_result_src_vectors.push_back(std::move(srcs));
    level_result_dst_vectors.push_back(std::move(dsts));

    if (weights) { (*level_result_weight_vectors).push_back(std::move(*weights)); }
    if (edge_ids) { (*level_result_edge_id_vectors).push_back(std::move(*edge_ids)); }
    if (edge_types) { (*level_result_edge_type_vectors).push_back(std::move(*edge_types)); }
    if (labels) { (*level_result_label_vectors).push_back(std::move(*labels)); }

    // FIXME:  We should modify vertex_partition_range_lasts to return a raft::host_span
    //  rather than making a copy.
    auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
    std::tie(frontier_vertices, frontier_vertex_labels, vertex_used_as_source) =
      prepare_next_frontier(
        handle,
        hop == 0
          ? starting_vertices
          : raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()),
        hop == 0 ? starting_vertex_labels
        : starting_vertex_labels
          ? std::make_optional(raft::device_span<label_t const>(frontier_vertex_labels->data(),
                                                                frontier_vertex_labels->size()))
          : std::nullopt,
        raft::device_span<vertex_t const>{level_result_dst_vectors.back().data(),
                                          level_result_dst_vectors.back().size()},
        frontier_vertex_labels
          ? std::make_optional(raft::device_span<label_t const>(
              level_result_label_vectors->back().data(), level_result_label_vectors->back().size()))
          : std::nullopt,
        std::move(vertex_used_as_source),
        graph_view.local_vertex_partition_view(),
        vertex_partition_range_lasts,
        prior_sources_behavior,
        dedupe_sources,
        do_expensive_check);
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time2 = std::chrono::steady_clock::now();

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
      if (level_sizes[i] > 0) {
        raft::copy((*result_weights).begin() + output_offset,
                   (*level_result_weight_vectors)[i].begin(),
                   level_sizes[i],
                   handle.get_stream());
        output_offset += level_sizes[i];
      }
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
      if (level_sizes[i] > 0) {
        raft::copy((*result_edge_ids).begin() + output_offset,
                   (*level_result_edge_id_vectors)[i].begin(),
                   level_sizes[i],
                   handle.get_stream());
        output_offset += level_sizes[i];
      }
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
      if (level_sizes[i] > 0) {
        raft::copy((*result_edge_types).begin() + output_offset,
                   (*level_result_edge_type_vectors)[i].begin(),
                   level_sizes[i],
                   handle.get_stream());
        output_offset += level_sizes[i];
      }
    }
    level_result_edge_type_vectors = std::nullopt;
  }

  auto result_labels =
    level_result_label_vectors
      ? std::make_optional(rmm::device_uvector<label_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_labels) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_label_vectors).size(); ++i) {
      if (level_sizes[i] > 0) {
        raft::copy((*result_labels).begin() + output_offset,
                   (*level_result_label_vectors)[i].begin(),
                   level_sizes[i],
                   handle.get_stream());
        output_offset += level_sizes[i];
      }
    }
    level_result_label_vectors = std::nullopt;
  }

  std::optional<rmm::device_uvector<int32_t>> result_hops{std::nullopt};
  if (return_hops) {
    result_hops   = rmm::device_uvector<int32_t>(result_size, handle.get_stream());
    output_offset = 0;
    for (size_t i = 0; i < num_hops; ++i) {
      scalar_fill(
        handle, result_hops->data() + output_offset, level_sizes[i], static_cast<int32_t>(i));
      output_offset += level_sizes[i];
    }
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time3                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::chrono::duration<double> dur1 = time2 - time1;
  std::chrono::duration<double> dur2 = time3 - time2;
  std::cout << "\tdetail::neighbor_sample_impl (less shuffle_and_organize_output) took ("
            << dur0.count() << "," << dur1.count() << "," << dur2.count() << ")." << std::endl;

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

// deprecated
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

  rmm::device_uvector<int32_t> label_map(0, handle.get_stream());

  if (label_to_output_comm_rank) {
    label_map = detail::flatten_label_map(handle, *label_to_output_comm_rank);
  }

  return detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
    handle,
    rng_state,
    graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    std::nullopt,
    starting_vertices,
    starting_vertex_labels,
    label_to_output_comm_rank
      ? std::make_optional(raft::device_span<int32_t const>{label_map.data(), label_map.size()})
      : std::nullopt,
    fan_out,
    std::optional<edge_type_t>{std::nullopt},
    return_hops,
    with_replacement,
    prior_sources_behavior,
    dedupe_sources,
    do_expensive_check);
}

// deprecated
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
  rmm::device_uvector<int32_t> label_map(0, handle.get_stream());

  if (label_to_output_comm_rank) {
    label_map = detail::flatten_label_map(handle, *label_to_output_comm_rank);
  }

  return detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
    handle,
    rng_state,
    graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    edge_bias_view,
    starting_vertices,
    starting_vertex_labels,
    label_to_output_comm_rank
      ? std::make_optional(raft::device_span<int32_t const>{label_map.data(), label_map.size()})
      : std::nullopt,
    fan_out,
    std::optional<edge_type_t>{std::nullopt},
    return_hops,
    with_replacement,
    prior_sources_behavior,
    dedupe_sources,
    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
homogeneous_uniform_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  using bias_t = weight_t;  // dummy

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
  auto [majors, minors, weights, edge_ids, edge_types, hops, labels, offsets] =
    detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      edge_type_view,
      std::optional<edge_property_view_t<edge_t, bias_t const*>>{
        std::nullopt},  // Optional edge_bias_view
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{std::nullopt},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::cout << "homogeneous_uniform_neighbor_sample (starting_vertices.size()="
            << starting_vertices.size() << " # edges sampled=" << majors.size() << ") took "
            << dur0.count() << std::endl;

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(hops),
                         std::move(offsets));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
heterogeneous_uniform_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  edge_type_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  using bias_t = weight_t;  // dummy

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
  auto [majors, minors, weights, edge_ids, edge_types, hops, labels, offsets] =
    detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      std::make_optional(edge_type_view),
      std::optional<edge_property_view_t<edge_t, bias_t const*>>{
        std::nullopt},  // Optional edge_bias_view
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{num_edge_types},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::cout << "heterogeneous_uniform_neighbor_sample (starting_vertices.size()="
            << starting_vertices.size() << " # edges sampled=" << majors.size() << ") took "
            << dur0.count() << std::endl;

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(hops),
                         std::move(offsets));
}

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
homogeneous_biased_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
  auto [majors, minors, weights, edge_ids, edge_types, hops, labels, offsets] =
    detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      edge_type_view,
      std::make_optional(edge_bias_view),
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{std::nullopt},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::cout << "homogeneous_biased_neighbor_sample (starting_vertices.size()="
            << starting_vertices.size() << " # edges sampled=" << majors.size() << ") took "
            << dur0.count() << std::endl;

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(hops),
                         std::move(offsets));
}

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
heterogeneous_biased_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  edge_type_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
  auto [majors, minors, weights, edge_ids, edge_types, hops, labels, offsets] =
    detail::neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      std::make_optional(edge_type_view),
      std::make_optional(edge_bias_view),
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{num_edge_types},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::cout << "heterogeneous_biased_neighbor_sample (starting_vertices.size()="
            << starting_vertices.size() << " # edges sampled=" << majors.size() << ") took "
            << dur0.count() << std::endl;

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(hops),
                         std::move(offsets));
}

}  // namespace cugraph
