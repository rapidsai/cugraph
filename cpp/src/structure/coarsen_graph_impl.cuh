/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {

namespace {

template <typename EdgeTupleType>
struct is_not_lower_triangular_t {
  __device__ bool operator()(EdgeTupleType e) const
  {
    return thrust::get<0>(e) < thrust::get<1>(e);
  }
};

template <typename EdgeTupleType>
struct is_not_self_loop_t {
  __device__ bool operator()(EdgeTupleType e) const
  {
    return thrust::get<0>(e) != thrust::get<1>(e);
  }
};

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
groupby_e_and_coarsen_edgelist(rmm::device_uvector<vertex_t>&& edgelist_majors,
                               rmm::device_uvector<vertex_t>&& edgelist_minors,
                               std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                               rmm::cuda_stream_view stream_view)
{
  auto pair_first =
    thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));

  if (edgelist_weights) {
    thrust::sort_by_key(rmm::exec_policy(stream_view),
                        pair_first,
                        pair_first + edgelist_majors.size(),
                        (*edgelist_weights).begin());

    auto num_uniques =
      thrust::count_if(rmm::exec_policy(stream_view),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(edgelist_majors.size()),
                       detail::is_first_in_run_t<decltype(pair_first)>{pair_first});

    rmm::device_uvector<vertex_t> tmp_edgelist_majors(num_uniques, stream_view);
    rmm::device_uvector<vertex_t> tmp_edgelist_minors(tmp_edgelist_majors.size(), stream_view);
    rmm::device_uvector<weight_t> tmp_edgelist_weights(tmp_edgelist_majors.size(), stream_view);
    thrust::reduce_by_key(rmm::exec_policy(stream_view),
                          pair_first,
                          pair_first + edgelist_majors.size(),
                          (*edgelist_weights).begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(
                            tmp_edgelist_majors.begin(), tmp_edgelist_minors.begin())),
                          tmp_edgelist_weights.begin());

    edgelist_majors.resize(0, stream_view);
    edgelist_majors.shrink_to_fit(stream_view);
    edgelist_minors.resize(0, stream_view);
    edgelist_minors.shrink_to_fit(stream_view);
    (*edgelist_weights).resize(0, stream_view);
    (*edgelist_weights).shrink_to_fit(stream_view);

    return std::make_tuple(std::move(tmp_edgelist_majors),
                           std::move(tmp_edgelist_minors),
                           std::move(tmp_edgelist_weights));
  } else {
    thrust::sort(rmm::exec_policy(stream_view), pair_first, pair_first + edgelist_majors.size());
    auto num_uniques = static_cast<size_t>(thrust::distance(
      pair_first,
      thrust::unique(
        rmm::exec_policy(stream_view), pair_first, pair_first + edgelist_majors.size())));
    edgelist_majors.resize(num_uniques, stream_view);
    edgelist_majors.shrink_to_fit(stream_view);
    edgelist_minors.resize(num_uniques, stream_view);
    edgelist_minors.shrink_to_fit(stream_view);

    return std::make_tuple(std::move(edgelist_majors), std::move(edgelist_minors), std::nullopt);
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename EdgeMinorLabelInputWrapper>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
decompress_edge_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> const edge_partition,
  std::optional<detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>
    edge_partition_weight_view,
  vertex_t const* major_label_first,
  EdgeMinorLabelInputWrapper const minor_label_input,
  std::optional<std::vector<vertex_t>> const& segment_offsets,
  bool lower_triangular_only)
{
  static_assert(std::is_same_v<typename EdgeMinorLabelInputWrapper::value_type, vertex_t>);

  // FIXME: it might be possible to directly create relabled & coarsened edgelist from the
  // compressed sparse format to save memory

  rmm::device_uvector<vertex_t> edgelist_majors(edge_partition.number_of_edges(),
                                                handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
  auto edgelist_weights = edge_partition_weight_view
                            ? std::make_optional<rmm::device_uvector<weight_t>>(
                                edgelist_majors.size(), handle.get_stream())
                            : std::nullopt;
  detail::decompress_edge_partition_to_edgelist(
    handle,
    edge_partition,
    edge_partition_weight_view,
    edgelist_majors.data(),
    edgelist_minors.data(),
    edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data()} : std::nullopt,
    segment_offsets);

  auto pair_first =
    thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
  thrust::transform(handle.get_thrust_policy(),
                    pair_first,
                    pair_first + edgelist_majors.size(),
                    pair_first,
                    [major_label_first,
                     minor_label_input = detail::edge_partition_endpoint_property_device_view_t<
                       vertex_t,
                       decltype(minor_label_input.value_first())>(minor_label_input),
                     major_range_first = edge_partition.major_range_first(),
                     minor_range_first = edge_partition.minor_range_first()] __device__(auto val) {
                      return thrust::make_tuple(
                        *(major_label_first + (thrust::get<0>(val) - major_range_first)),
                        minor_label_input.get(thrust::get<1>(val) - minor_range_first));
                    });

  if (lower_triangular_only) {
    if (edgelist_weights) {
      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        edgelist_majors.begin(), edgelist_minors.begin(), (*edgelist_weights).begin()));
      edgelist_majors.resize(
        thrust::distance(
          edge_first,
          thrust::remove_if(
            handle.get_thrust_policy(),
            edge_first,
            edge_first + edgelist_majors.size(),
            is_not_lower_triangular_t<thrust::tuple<vertex_t, vertex_t, weight_t>>{})),
        handle.get_stream());
      edgelist_majors.shrink_to_fit(handle.get_stream());
      edgelist_minors.resize(edgelist_majors.size(), handle.get_stream());
      edgelist_minors.shrink_to_fit(handle.get_stream());
      (*edgelist_weights).resize(edgelist_majors.size(), handle.get_stream());
      (*edgelist_weights).shrink_to_fit(handle.get_stream());
    } else {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
      edgelist_majors.resize(
        thrust::distance(
          edge_first,
          thrust::remove_if(handle.get_thrust_policy(),
                            edge_first,
                            edge_first + edgelist_majors.size(),
                            is_not_lower_triangular_t<thrust::tuple<vertex_t, vertex_t>>{})),
        handle.get_stream());
      edgelist_majors.shrink_to_fit(handle.get_stream());
      edgelist_minors.resize(edgelist_majors.size(), handle.get_stream());
      edgelist_minors.shrink_to_fit(handle.get_stream());
    }
  }

  return groupby_e_and_coarsen_edgelist(std::move(edgelist_majors),
                                        std::move(edgelist_minors),
                                        std::move(edgelist_weights),
                                        handle.get_stream());
}

}  // namespace

namespace detail {

// FIXME: This function needs to be updated to support edge id/type
// multi-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
              std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
              vertex_t const* labels,
              bool renumber,
              bool do_expensive_check)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto const row_comm_rank = row_comm.get_rank();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();
  auto const col_comm_rank = col_comm.get_rank();

  CUGRAPH_EXPECTS(renumber,
                  "Invalid input arguments: renumber should be true if multi_gpu is true.");

  if (do_expensive_check) {
    // currently, nothing to do
  }

  // 1. construct coarsened edge lists from each local partition (if the input graph is symmetric,
  // start with only the lower triangular edges after relabeling, this is to prevent edge weights in
  // the coarsened graph becoming asymmmetric due to limited floatping point resolution)

  bool lower_triangular_only = graph_view.is_symmetric();

  std::conditional_t<
    store_transposed,
    edge_src_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, vertex_t>,
    edge_dst_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, vertex_t>>
    edge_minor_labels(handle, graph_view);
  if constexpr (store_transposed) {
    update_edge_src_property(handle, graph_view, labels, edge_minor_labels);
  } else {
    update_edge_dst_property(handle, graph_view, labels, edge_minor_labels);
  }

  std::vector<rmm::device_uvector<vertex_t>> coarsened_edgelist_majors{};
  std::vector<rmm::device_uvector<vertex_t>> coarsened_edgelist_minors{};
  auto coarsened_edgelist_weights =
    edge_weight_view ? std::make_optional<std::vector<rmm::device_uvector<weight_t>>>({})
                     : std::nullopt;
  coarsened_edgelist_majors.reserve(graph_view.number_of_local_edge_partitions());
  coarsened_edgelist_minors.reserve(coarsened_edgelist_majors.size());
  if (coarsened_edgelist_weights) {
    (*coarsened_edgelist_weights).reserve(coarsened_edgelist_majors.size());
  }
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    // 1-1. locally construct coarsened edge list

    vertex_t edge_partition_major_range_size{};
    if constexpr (store_transposed) {
      edge_partition_major_range_size = graph_view.local_edge_partition_dst_range_size(i);
    } else {
      edge_partition_major_range_size = graph_view.local_edge_partition_src_range_size(i);
    }
    rmm::device_uvector<vertex_t> major_labels(edge_partition_major_range_size,
                                               handle.get_stream());
    device_bcast(col_comm,
                 labels,
                 major_labels.data(),
                 major_labels.size(),
                 static_cast<int>(i),
                 handle.get_stream());

    auto [edgelist_majors, edgelist_minors, edgelist_weights] =
      decompress_edge_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
        handle,
        edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
          graph_view.local_edge_partition_view(i)),
        edge_weight_view
          ? std::make_optional<
              detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>(
              *edge_weight_view, i)
          : std::nullopt,
        major_labels.data(),
        edge_minor_labels.view(),
        graph_view.local_edge_partition_segment_offsets(i),
        lower_triangular_only);

    // 1-2. globally shuffle

    std::tie(edgelist_majors, edgelist_minors, edgelist_weights, std::ignore) =
      cugraph::detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                  edge_t,
                                                                                  weight_t,
                                                                                  int32_t>(
        handle,
        std::move(edgelist_majors),
        std::move(edgelist_minors),
        std::move(edgelist_weights),
        std::nullopt);

    // 1-3. groupby and coarsen again

    std::tie(edgelist_majors, edgelist_minors, edgelist_weights) =
      groupby_e_and_coarsen_edgelist(std::move(edgelist_majors),
                                     std::move(edgelist_minors),
                                     std::move(edgelist_weights),
                                     handle.get_stream());

    coarsened_edgelist_majors.push_back(std::move(edgelist_majors));
    coarsened_edgelist_minors.push_back(std::move(edgelist_minors));
    if (edgelist_weights) { (*coarsened_edgelist_weights).push_back(std::move(*edgelist_weights)); }
  }
  edge_minor_labels.clear(handle);

  // 2. concatenate and groupby and coarsen again (and if the input graph is symmetric, 1) create a
  // copy excluding self loops, 2) globally shuffle, and 3) concatenate again)

  edge_t tot_count{0};
  for (size_t i = 0; i < coarsened_edgelist_majors.size(); ++i) {
    tot_count += coarsened_edgelist_majors[i].size();
  }

  rmm::device_uvector<vertex_t> concatenated_edgelist_majors(tot_count, handle.get_stream());
  size_t major_offset{0};
  for (size_t i = 0; i < coarsened_edgelist_majors.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 coarsened_edgelist_majors[i].begin(),
                 coarsened_edgelist_majors[i].end(),
                 concatenated_edgelist_majors.begin() + major_offset);
    major_offset += coarsened_edgelist_majors[i].size();
    coarsened_edgelist_majors[i].resize(0, handle.get_stream());
    coarsened_edgelist_majors[i].shrink_to_fit(handle.get_stream());
  }

  rmm::device_uvector<vertex_t> concatenated_edgelist_minors(tot_count, handle.get_stream());
  size_t minor_offset{0};
  for (size_t i = 0; i < coarsened_edgelist_minors.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 coarsened_edgelist_minors[i].begin(),
                 coarsened_edgelist_minors[i].end(),
                 concatenated_edgelist_minors.begin() + minor_offset);
    minor_offset += coarsened_edgelist_minors[i].size();
    coarsened_edgelist_minors[i].resize(0, handle.get_stream());
    coarsened_edgelist_minors[i].shrink_to_fit(handle.get_stream());
  }

  std::optional<rmm::device_uvector<weight_t>> concatenated_edgelist_weights{std::nullopt};
  if (coarsened_edgelist_weights) {
    concatenated_edgelist_weights = rmm::device_uvector<weight_t>(tot_count, handle.get_stream());
    size_t weight_offset{0};
    for (size_t i = 0; i < (*coarsened_edgelist_weights).size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   (*coarsened_edgelist_weights)[i].begin(),
                   (*coarsened_edgelist_weights)[i].end(),
                   (*concatenated_edgelist_weights).begin() + weight_offset);
      weight_offset += (*coarsened_edgelist_weights)[i].size();
      (*coarsened_edgelist_weights)[i].resize(0, handle.get_stream());
      (*coarsened_edgelist_weights)[i].shrink_to_fit(handle.get_stream());
    }
  }

  std::tie(
    concatenated_edgelist_majors, concatenated_edgelist_minors, concatenated_edgelist_weights) =
    groupby_e_and_coarsen_edgelist(std::move(concatenated_edgelist_majors),
                                   std::move(concatenated_edgelist_minors),
                                   std::move(concatenated_edgelist_weights),
                                   handle.get_stream());

  if (lower_triangular_only) {
    rmm::device_uvector<vertex_t> reversed_edgelist_majors(0, handle.get_stream());
    rmm::device_uvector<vertex_t> reversed_edgelist_minors(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> reversed_edgelist_weights{std::nullopt};

    if (concatenated_edgelist_weights) {
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(concatenated_edgelist_majors.begin(),
                                                     concatenated_edgelist_minors.begin(),
                                                     (*concatenated_edgelist_weights).begin()));
      auto last =
        thrust::partition(handle.get_thrust_policy(),
                          edge_first,
                          edge_first + concatenated_edgelist_majors.size(),
                          is_not_self_loop_t<thrust::tuple<vertex_t, vertex_t, weight_t>>{});
      reversed_edgelist_majors.resize(thrust::distance(edge_first, last), handle.get_stream());
      reversed_edgelist_minors.resize(reversed_edgelist_majors.size(), handle.get_stream());
      reversed_edgelist_weights =
        rmm::device_uvector<weight_t>(reversed_edgelist_majors.size(), handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        edge_first,
        edge_first + reversed_edgelist_majors.size(),
        thrust::make_zip_iterator(thrust::make_tuple(reversed_edgelist_minors.begin(),
                                                     reversed_edgelist_majors.begin(),
                                                     (*reversed_edgelist_weights).begin())));
    } else {
      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        concatenated_edgelist_majors.begin(), concatenated_edgelist_minors.begin()));
      auto last       = thrust::partition(handle.get_thrust_policy(),
                                    edge_first,
                                    edge_first + concatenated_edgelist_majors.size(),
                                    is_not_self_loop_t<thrust::tuple<vertex_t, vertex_t>>{});
      reversed_edgelist_majors.resize(thrust::distance(edge_first, last), handle.get_stream());
      reversed_edgelist_minors.resize(reversed_edgelist_majors.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edge_first,
                   edge_first + reversed_edgelist_majors.size(),
                   thrust::make_zip_iterator(thrust::make_tuple(reversed_edgelist_minors.begin(),
                                                                reversed_edgelist_majors.begin())));
    }

    std::tie(
      reversed_edgelist_majors, reversed_edgelist_minors, reversed_edgelist_weights, std::ignore) =
      cugraph::detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                  edge_t,
                                                                                  weight_t,
                                                                                  int32_t>(
        handle,
        std::move(reversed_edgelist_majors),
        std::move(reversed_edgelist_minors),
        std::move(reversed_edgelist_weights),
        std::nullopt);

    auto output_offset = concatenated_edgelist_majors.size();

    concatenated_edgelist_majors.resize(
      concatenated_edgelist_majors.size() + reversed_edgelist_majors.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 reversed_edgelist_majors.begin(),
                 reversed_edgelist_majors.end(),
                 concatenated_edgelist_majors.begin() + output_offset);
    reversed_edgelist_majors.resize(0, handle.get_stream());
    reversed_edgelist_majors.shrink_to_fit(handle.get_stream());

    concatenated_edgelist_minors.resize(concatenated_edgelist_majors.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 reversed_edgelist_minors.begin(),
                 reversed_edgelist_minors.end(),
                 concatenated_edgelist_minors.begin() + output_offset);
    reversed_edgelist_minors.resize(0, handle.get_stream());
    reversed_edgelist_minors.shrink_to_fit(handle.get_stream());

    if (concatenated_edgelist_weights) {
      (*concatenated_edgelist_weights)
        .resize(concatenated_edgelist_majors.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*reversed_edgelist_weights).begin(),
                   (*reversed_edgelist_weights).end(),
                   (*concatenated_edgelist_weights).begin() + output_offset);
      (*reversed_edgelist_weights).resize(0, handle.get_stream());
      (*reversed_edgelist_weights).shrink_to_fit(handle.get_stream());
    }
  }

  // 3. find unique labels for this GPU

  rmm::device_uvector<vertex_t> unique_labels(graph_view.local_vertex_partition_range_size(),
                                              handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), labels, labels + unique_labels.size(), unique_labels.begin());
  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  unique_labels = cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
    handle, std::move(unique_labels));

  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  // 4. create a graph

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> coarsened_graph(handle);
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::tie(coarsened_graph, edge_weights, std::ignore, renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(unique_labels),
      store_transposed ? std::move(concatenated_edgelist_minors)
                       : std::move(concatenated_edgelist_majors),
      store_transposed ? std::move(concatenated_edgelist_majors)
                       : std::move(concatenated_edgelist_minors),
      std::move(concatenated_edgelist_weights),
      std::nullopt,
      graph_properties_t{graph_view.is_symmetric(), false},
      true,
      do_expensive_check);

  return std::make_tuple(std::move(coarsened_graph),
                         std::move(edge_weights),
                         std::optional<rmm::device_uvector<vertex_t>>{std::move(*renumber_map)});
}

// single-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<
    graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
              std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
              vertex_t const* labels,
              bool renumber,
              bool do_expensive_check)
{
  if (do_expensive_check) {
    if (!renumber) {
      auto num_invalids =
        thrust::count_if(handle.get_thrust_policy(),
                         labels,
                         labels + graph_view.number_of_vertices(),
                         check_out_of_range_t<vertex_t>{0, std::numeric_limits<vertex_t>::max()});
      CUGRAPH_EXPECTS(num_invalids == 0,
                      "Invalid input aguments: if renumber is false, labels should be non-negative "
                      "integers smaller than std::numeric_limits<vertex_t>::max().");
    }
  }

  bool lower_triangular_only = graph_view.is_symmetric();

  auto [coarsened_edgelist_majors, coarsened_edgelist_minors, coarsened_edgelist_weights] =
    decompress_edge_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
      handle,
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
        graph_view.local_edge_partition_view()),
      edge_weight_view
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>(
            *edge_weight_view, 0)
        : std::nullopt,
      labels,
      detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(labels, vertex_t{0}),
      graph_view.local_edge_partition_segment_offsets(0),
      lower_triangular_only);

  if (lower_triangular_only) {
    if (coarsened_edgelist_weights) {
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_majors.begin(),
                                                     coarsened_edgelist_minors.begin(),
                                                     (*coarsened_edgelist_weights).begin()));
      auto last =
        thrust::partition(handle.get_thrust_policy(),
                          edge_first,
                          edge_first + coarsened_edgelist_majors.size(),
                          is_not_self_loop_t<thrust::tuple<vertex_t, vertex_t, weight_t>>{});

      auto cur_size      = coarsened_edgelist_majors.size();
      auto reversed_size = static_cast<size_t>(thrust::distance(edge_first, last));

      coarsened_edgelist_majors.resize(cur_size + reversed_size, handle.get_stream());
      coarsened_edgelist_minors.resize(coarsened_edgelist_majors.size(), handle.get_stream());
      (*coarsened_edgelist_weights).resize(coarsened_edgelist_majors.size(), handle.get_stream());

      edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_majors.begin(),
                                                     coarsened_edgelist_minors.begin(),
                                                     (*coarsened_edgelist_weights).begin()));
      thrust::copy(
        handle.get_thrust_policy(),
        edge_first,
        edge_first + reversed_size,
        thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_minors.begin(),
                                                     coarsened_edgelist_majors.begin(),
                                                     (*coarsened_edgelist_weights).begin())) +
          cur_size);
    } else {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(coarsened_edgelist_majors.begin(), coarsened_edgelist_minors.begin()));
      auto last = thrust::partition(handle.get_thrust_policy(),
                                    edge_first,
                                    edge_first + coarsened_edgelist_majors.size(),
                                    is_not_self_loop_t<thrust::tuple<vertex_t, vertex_t>>{});

      auto cur_size      = coarsened_edgelist_majors.size();
      auto reversed_size = static_cast<size_t>(thrust::distance(edge_first, last));

      coarsened_edgelist_majors.resize(cur_size + reversed_size, handle.get_stream());
      coarsened_edgelist_minors.resize(coarsened_edgelist_majors.size(), handle.get_stream());

      edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(coarsened_edgelist_majors.begin(), coarsened_edgelist_minors.begin()));
      thrust::copy(handle.get_thrust_policy(),
                   edge_first,
                   edge_first + reversed_size,
                   thrust::make_zip_iterator(thrust::make_tuple(
                     coarsened_edgelist_minors.begin(), coarsened_edgelist_majors.begin())) +
                     cur_size);
    }
  }

  rmm::device_uvector<vertex_t> vertices(graph_view.number_of_vertices(), handle.get_stream());
  if (renumber) {
    thrust::copy(handle.get_thrust_policy(), labels, labels + vertices.size(), vertices.begin());
    thrust::sort(handle.get_thrust_policy(), vertices.begin(), vertices.end());
    vertices.resize(thrust::distance(
                      vertices.begin(),
                      thrust::unique(handle.get_thrust_policy(), vertices.begin(), vertices.end())),
                    handle.get_stream());
  } else {
    vertex_t number_of_vertices = thrust::reduce(handle.get_thrust_policy(),
                                                 labels,
                                                 labels + vertices.size(),
                                                 vertex_t{0},
                                                 thrust::maximum<vertex_t>{}) +
                                  1;
    vertices = rmm::device_uvector<vertex_t>(number_of_vertices, handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), vertices.begin(), vertices.end(), vertex_t{0});
  }

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> coarsened_graph(handle);
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::tie(coarsened_graph, edge_weights, std::ignore, renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::optional<rmm::device_uvector<vertex_t>>{std::move(vertices)},
      store_transposed ? std::move(coarsened_edgelist_minors)
                       : std::move(coarsened_edgelist_majors),
      store_transposed ? std::move(coarsened_edgelist_majors)
                       : std::move(coarsened_edgelist_minors),
      std::move(coarsened_edgelist_weights),
      std::nullopt,
      graph_properties_t{graph_view.is_symmetric(), false},
      renumber,
      do_expensive_check);

  return std::make_tuple(
    std::move(coarsened_graph), std::move(edge_weights), std::move(*renumber_map));
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
              std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
              vertex_t const* labels,
              bool renumber,
              bool do_expensive_check)
{
  return detail::coarsen_graph(
    handle, graph_view, edge_weight_view, labels, renumber, do_expensive_check);
}

}  // namespace cugraph
