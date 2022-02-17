/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/detail/decompress_matrix_partition.cuh>
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_to_adj_matrix_row_col.cuh>
#include <cugraph/prims/row_col_properties.cuh>
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

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

template <typename vertex_t, typename edge_t, typename weight_t>
edge_t groupby_e_and_coarsen_edgelist(vertex_t* edgelist_majors /* [INOUT] */,
                                      vertex_t* edgelist_minors /* [INOUT] */,
                                      std::optional<weight_t*> edgelist_weights /* [INOUT] */,
                                      edge_t number_of_edges,
                                      rmm::cuda_stream_view stream_view)
{
  auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors, edgelist_minors));

  if (edgelist_weights) {
    thrust::sort_by_key(
      rmm::exec_policy(stream_view), pair_first, pair_first + number_of_edges, *edgelist_weights);

    auto num_uniques =
      thrust::count_if(rmm::exec_policy(stream_view),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(static_cast<size_t>(number_of_edges)),
                       detail::is_first_in_run_pair_t<vertex_t>{edgelist_majors, edgelist_minors});

    rmm::device_uvector<vertex_t> tmp_edgelist_majors(num_uniques, stream_view);
    rmm::device_uvector<vertex_t> tmp_edgelist_minors(tmp_edgelist_majors.size(), stream_view);
    rmm::device_uvector<weight_t> tmp_edgelist_weights(tmp_edgelist_majors.size(), stream_view);
    thrust::reduce_by_key(rmm::exec_policy(stream_view),
                          pair_first,
                          pair_first + number_of_edges,
                          (*edgelist_weights),
                          thrust::make_zip_iterator(thrust::make_tuple(
                            tmp_edgelist_majors.begin(), tmp_edgelist_minors.begin())),
                          tmp_edgelist_weights.begin());

    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      tmp_edgelist_majors.begin(), tmp_edgelist_minors.begin(), tmp_edgelist_weights.begin()));
    thrust::copy(rmm::exec_policy(stream_view),
                 edge_first,
                 edge_first + num_uniques,
                 thrust::make_zip_iterator(
                   thrust::make_tuple(edgelist_majors, edgelist_minors, *edgelist_weights)));

    return num_uniques;
  } else {
    thrust::sort(rmm::exec_policy(stream_view), pair_first, pair_first + number_of_edges);
    return static_cast<edge_t>(thrust::distance(
      pair_first,
      thrust::unique(rmm::exec_policy(stream_view), pair_first, pair_first + number_of_edges)));
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename AdjMatrixMinorLabelInputWrapper>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
decompress_matrix_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
  raft::handle_t const& handle,
  matrix_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> const matrix_partition,
  vertex_t const* major_label_first,
  AdjMatrixMinorLabelInputWrapper const minor_label_input,
  std::optional<std::vector<vertex_t>> const& segment_offsets,
  bool lower_triangular_only)
{
  static_assert(std::is_same_v<typename AdjMatrixMinorLabelInputWrapper::value_type, vertex_t>);

  // FIXME: it might be possible to directly create relabled & coarsened edgelist from the
  // compressed sparse format to save memory

  rmm::device_uvector<vertex_t> edgelist_majors(matrix_partition.get_number_of_edges(),
                                                handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
  auto edgelist_weights = matrix_partition.get_weights()
                            ? std::make_optional<rmm::device_uvector<weight_t>>(
                                edgelist_majors.size(), handle.get_stream())
                            : std::nullopt;
  detail::decompress_matrix_partition_to_edgelist(
    handle,
    matrix_partition,
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
                     minor_label_input,
                     major_first = matrix_partition.get_major_first(),
                     minor_first = matrix_partition.get_minor_first()] __device__(auto val) {
                      return thrust::make_tuple(
                        *(major_label_first + (thrust::get<0>(val) - major_first)),
                        minor_label_input.get(thrust::get<1>(val) - minor_first));
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

  auto number_of_edges = groupby_e_and_coarsen_edgelist(
    edgelist_majors.data(),
    edgelist_minors.data(),
    edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data()} : std::nullopt,
    static_cast<edge_t>(edgelist_majors.size()),
    handle.get_stream());
  edgelist_majors.resize(number_of_edges, handle.get_stream());
  edgelist_majors.shrink_to_fit(handle.get_stream());
  edgelist_minors.resize(number_of_edges, handle.get_stream());
  edgelist_minors.shrink_to_fit(handle.get_stream());
  if (edgelist_weights) {
    (*edgelist_weights).resize(number_of_edges, handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(
    std::move(edgelist_majors), std::move(edgelist_minors), std::move(edgelist_weights));
}

}  // namespace

namespace detail {

// multi-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
             rmm::device_uvector<vertex_t>>>
coarsen_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels,
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

  if (do_expensive_check) {
    // currently, nothing to do
  }

  // 1. construct coarsened edge lists from each local partition (if the input graph is symmetric,
  // start with only the lower triangular edges after relabeling, this is to prevent edge weights in
  // the coarsened graph becoming asymmmetric due to limited floatping point resolution)

  bool lower_triangular_only = graph_view.is_symmetric();

  std::conditional_t<
    store_transposed,
    row_properties_t<graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
                     vertex_t>,
    col_properties_t<graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
                     vertex_t>>
    adj_matrix_minor_labels(handle, graph_view);
  if constexpr (store_transposed) {
    copy_to_adj_matrix_row(handle, graph_view, labels, adj_matrix_minor_labels);
  } else {
    copy_to_adj_matrix_col(handle, graph_view, labels, adj_matrix_minor_labels);
  }

  std::vector<rmm::device_uvector<vertex_t>> coarsened_edgelist_majors{};
  std::vector<rmm::device_uvector<vertex_t>> coarsened_edgelist_minors{};
  auto coarsened_edgelist_weights =
    graph_view.is_weighted() ? std::make_optional<std::vector<rmm::device_uvector<weight_t>>>({})
                             : std::nullopt;
  coarsened_edgelist_majors.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
  coarsened_edgelist_minors.reserve(coarsened_edgelist_majors.size());
  if (coarsened_edgelist_weights) {
    (*coarsened_edgelist_weights).reserve(coarsened_edgelist_majors.size());
  }
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    // 1-1. locally construct coarsened edge list

    rmm::device_uvector<vertex_t> major_labels(
      store_transposed ? graph_view.get_number_of_local_adj_matrix_partition_cols(i)
                       : graph_view.get_number_of_local_adj_matrix_partition_rows(i),
      handle.get_stream());
    device_bcast(col_comm,
                 labels,
                 major_labels.data(),
                 major_labels.size(),
                 static_cast<int>(i),
                 handle.get_stream());

    auto [edgelist_majors, edgelist_minors, edgelist_weights] =
      decompress_matrix_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
        handle,
        matrix_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu>(
          graph_view.get_matrix_partition_view(i)),
        major_labels.data(),
        adj_matrix_minor_labels.device_view(),
        graph_view.get_local_adj_matrix_partition_segment_offsets(i),
        lower_triangular_only);

    // 1-2. globally shuffle

    std::tie(edgelist_majors, edgelist_minors, edgelist_weights) =
      cugraph::detail::shuffle_edgelist_by_gpu_id(handle,
                                                  std::move(edgelist_majors),
                                                  std::move(edgelist_minors),
                                                  std::move(edgelist_weights));

    // 1-3. groupby and coarsen again

    auto coarsened_size = groupby_e_and_coarsen_edgelist(
      edgelist_majors.data(),
      edgelist_minors.data(),
      edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data()} : std::nullopt,
      edgelist_majors.size(),
      handle.get_stream());
    edgelist_majors.resize(coarsened_size, handle.get_stream());
    edgelist_majors.shrink_to_fit(handle.get_stream());
    edgelist_minors.resize(edgelist_majors.size(), handle.get_stream());
    edgelist_minors.shrink_to_fit(handle.get_stream());
    if (edgelist_weights) {
      (*edgelist_weights).resize(edgelist_majors.size(), handle.get_stream());
      (*edgelist_weights).shrink_to_fit(handle.get_stream());
    }

    coarsened_edgelist_majors.push_back(std::move(edgelist_majors));
    coarsened_edgelist_minors.push_back(std::move(edgelist_minors));
    if (edgelist_weights) { (*coarsened_edgelist_weights).push_back(std::move(*edgelist_weights)); }
  }

  // 2. concatenate and groupby and coarsen again (and if the input graph is symmetric, create a
  // copy excluding self loops and globally shuffle)

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

  auto concatenated_and_coarsened_size = groupby_e_and_coarsen_edgelist(
    concatenated_edgelist_majors.data(),
    concatenated_edgelist_minors.data(),
    concatenated_edgelist_weights
      ? std::optional<weight_t*>{(*concatenated_edgelist_weights).data()}
      : std::nullopt,
    concatenated_edgelist_majors.size(),
    handle.get_stream());
  concatenated_edgelist_majors.resize(concatenated_and_coarsened_size, handle.get_stream());
  concatenated_edgelist_majors.shrink_to_fit(handle.get_stream());
  concatenated_edgelist_minors.resize(concatenated_edgelist_majors.size(), handle.get_stream());
  concatenated_edgelist_minors.shrink_to_fit(handle.get_stream());
  if (concatenated_edgelist_weights) {
    (*concatenated_edgelist_weights)
      .resize(concatenated_edgelist_majors.size(), handle.get_stream());
    (*concatenated_edgelist_weights).shrink_to_fit(handle.get_stream());
  }

  std::optional<rmm::device_uvector<vertex_t>> reversed_edgelist_majors{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> reversed_edgelist_minors{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> reversed_edgelist_weights{std::nullopt};
  if (lower_triangular_only) {
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
      reversed_edgelist_majors =
        rmm::device_uvector<vertex_t>(thrust::distance(edge_first, last), handle.get_stream());
      reversed_edgelist_minors =
        rmm::device_uvector<vertex_t>((*reversed_edgelist_majors).size(), handle.get_stream());
      reversed_edgelist_weights =
        rmm::device_uvector<weight_t>((*reversed_edgelist_majors).size(), handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        edge_first,
        edge_first + (*reversed_edgelist_majors).size(),
        thrust::make_zip_iterator(thrust::make_tuple((*reversed_edgelist_minors).begin(),
                                                     (*reversed_edgelist_majors).begin(),
                                                     (*reversed_edgelist_weights).begin())));
    } else {
      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        concatenated_edgelist_majors.begin(), concatenated_edgelist_minors.begin()));
      auto last       = thrust::partition(handle.get_thrust_policy(),
                                    edge_first,
                                    edge_first + concatenated_edgelist_majors.size(),
                                    is_not_self_loop_t<thrust::tuple<vertex_t, vertex_t>>{});
      reversed_edgelist_majors =
        rmm::device_uvector<vertex_t>(thrust::distance(edge_first, last), handle.get_stream());
      reversed_edgelist_minors =
        rmm::device_uvector<vertex_t>((*reversed_edgelist_majors).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edge_first,
                   edge_first + (*reversed_edgelist_majors).size(),
                   thrust::make_zip_iterator(thrust::make_tuple(
                     (*reversed_edgelist_minors).begin(), (*reversed_edgelist_majors).begin())));
    }

    std::tie(*reversed_edgelist_majors, *reversed_edgelist_minors, reversed_edgelist_weights) =
      cugraph::detail::shuffle_edgelist_by_gpu_id(handle,
                                                  std::move(*reversed_edgelist_majors),
                                                  std::move(*reversed_edgelist_minors),
                                                  std::move(reversed_edgelist_weights));
  }

  // 3. split concatenated edge list to local partitions

  auto concatenated_counts =
    groupby_and_count_edgelist_by_local_partition_id(handle,
                                                     concatenated_edgelist_majors,
                                                     concatenated_edgelist_minors,
                                                     concatenated_edgelist_weights);

  std::vector<size_t> h_concatenated_counts(concatenated_counts.size());
  raft::update_host(h_concatenated_counts.data(),
                    concatenated_counts.data(),
                    concatenated_counts.size(),
                    handle.get_stream());

  std::optional<std::vector<size_t>> h_reversed_counts{std::nullopt};
  if (reversed_edgelist_majors) {
    auto reversed_counts = groupby_and_count_edgelist_by_local_partition_id(
      handle, *reversed_edgelist_majors, *reversed_edgelist_minors, reversed_edgelist_weights);

    h_reversed_counts = std::vector<size_t>(reversed_counts.size());
    raft::update_host((*h_reversed_counts).data(),
                      reversed_counts.data(),
                      reversed_counts.size(),
                      handle.get_stream());
  }

  handle.sync_stream();

  std::vector<size_t> h_concatenated_displacements(h_concatenated_counts.size(), size_t{0});
  std::partial_sum(h_concatenated_counts.begin(),
                   h_concatenated_counts.end() - 1,
                   h_concatenated_displacements.begin() + 1);

  std::optional<std::vector<size_t>> h_reversed_displacements{std::nullopt};
  if (h_reversed_counts) {
    h_reversed_displacements = std::vector<size_t>((*h_reversed_counts).size(), size_t{0});
    std::partial_sum((*h_reversed_counts).begin(),
                     (*h_reversed_counts).end() - 1,
                     (*h_reversed_displacements).begin() + 1);
  }

  for (size_t i = 0; i < coarsened_edgelist_majors.size(); ++i) {
    coarsened_edgelist_majors[i].resize(
      h_concatenated_counts[i] + (h_reversed_counts ? (*h_reversed_counts)[i] : size_t{0}),
      handle.get_stream());
    coarsened_edgelist_minors[i].resize(coarsened_edgelist_majors[i].size(), handle.get_stream());
    if (coarsened_edgelist_weights) {
      (*coarsened_edgelist_weights)[i].resize(coarsened_edgelist_majors[i].size(),
                                              handle.get_stream());
    }

    thrust::copy(handle.get_thrust_policy(),
                 concatenated_edgelist_majors.begin() + h_concatenated_displacements[i],
                 concatenated_edgelist_majors.begin() +
                   (h_concatenated_displacements[i] + h_concatenated_counts[i]),
                 coarsened_edgelist_majors[i].begin());
    thrust::copy(handle.get_thrust_policy(),
                 concatenated_edgelist_minors.begin() + h_concatenated_displacements[i],
                 concatenated_edgelist_minors.begin() +
                   (h_concatenated_displacements[i] + h_concatenated_counts[i]),
                 coarsened_edgelist_minors[i].begin());
    if (coarsened_edgelist_weights) {
      thrust::copy(handle.get_thrust_policy(),
                   (*concatenated_edgelist_weights).begin() + h_concatenated_displacements[i],
                   (*concatenated_edgelist_weights).begin() +
                     (h_concatenated_displacements[i] + h_concatenated_counts[i]),
                   (*coarsened_edgelist_weights)[i].begin());
    }

    if (reversed_edgelist_majors) {
      thrust::copy(handle.get_thrust_policy(),
                   (*reversed_edgelist_majors).begin() + (*h_reversed_displacements)[i],
                   (*reversed_edgelist_majors).begin() +
                     ((*h_reversed_displacements)[i] + (*h_reversed_counts)[i]),
                   coarsened_edgelist_majors[i].begin() + h_concatenated_counts[i]);
      thrust::copy(handle.get_thrust_policy(),
                   (*reversed_edgelist_minors).begin() + (*h_reversed_displacements)[i],
                   (*reversed_edgelist_minors).begin() +
                     ((*h_reversed_displacements)[i] + (*h_reversed_counts)[i]),
                   coarsened_edgelist_minors[i].begin() + h_concatenated_counts[i]);
      if (coarsened_edgelist_weights) {
        thrust::copy(handle.get_thrust_policy(),
                     (*reversed_edgelist_weights).begin() + (*h_reversed_displacements)[i],
                     (*reversed_edgelist_weights).begin() +
                       ((*h_reversed_displacements)[i] + (*h_reversed_counts)[i]),
                     (*coarsened_edgelist_weights)[i].begin() + h_concatenated_counts[i]);
      }
    }
  }

  // 4. find unique labels for this GPU

  rmm::device_uvector<vertex_t> unique_labels(graph_view.get_number_of_local_vertices(),
                                              handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), labels, labels + unique_labels.size(), unique_labels.begin());
  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  unique_labels = cugraph::detail::shuffle_vertices_by_gpu_id(handle, std::move(unique_labels));

  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  // 5. renumber

  rmm::device_uvector<vertex_t> renumber_map_labels(0, handle.get_stream());
  renumber_meta_t<vertex_t, edge_t, multi_gpu> meta{};
  {
    std::vector<vertex_t*> major_ptrs(coarsened_edgelist_majors.size());
    std::vector<vertex_t*> minor_ptrs(major_ptrs.size());
    std::vector<edge_t> counts(major_ptrs.size());
    for (size_t i = 0; i < coarsened_edgelist_majors.size(); ++i) {
      major_ptrs[i] = coarsened_edgelist_majors[i].data();
      minor_ptrs[i] = coarsened_edgelist_minors[i].data();
      counts[i]     = static_cast<edge_t>(coarsened_edgelist_majors[i].size());
    }
    std::tie(renumber_map_labels, meta) = renumber_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      std::optional<rmm::device_uvector<vertex_t>>{std::move(unique_labels)},
      major_ptrs,
      minor_ptrs,
      counts,
      std::nullopt,
      do_expensive_check);
  }

  // 6. build a graph

  std::vector<edgelist_t<vertex_t, edge_t, weight_t>> edgelists{};
  edgelists.resize(graph_view.get_number_of_local_adj_matrix_partitions());
  for (size_t i = 0; i < edgelists.size(); ++i) {
    edgelists[i].p_src_vertices =
      store_transposed ? coarsened_edgelist_minors[i].data() : coarsened_edgelist_majors[i].data();
    edgelists[i].p_dst_vertices =
      store_transposed ? coarsened_edgelist_majors[i].data() : coarsened_edgelist_minors[i].data();
    edgelists[i].p_edge_weights =
      coarsened_edgelist_weights
        ? std::optional<weight_t const*>{(*coarsened_edgelist_weights)[i].data()}
        : std::nullopt,
    edgelists[i].number_of_edges = static_cast<edge_t>(coarsened_edgelist_majors[i].size());
  }

  return std::make_tuple(
    std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
      handle,
      edgelists,
      graph_meta_t<vertex_t, edge_t, multi_gpu>{
        meta.number_of_vertices,
        meta.number_of_edges,
        graph_properties_t{graph_view.is_symmetric(), false},
        meta.partition,
        meta.segment_offsets}),
    std::move(renumber_map_labels));
}

// single-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
             rmm::device_uvector<vertex_t>>>
coarsen_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels,
  bool do_expensive_check)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  bool lower_triangular_only = graph_view.is_symmetric();

  auto [coarsened_edgelist_majors, coarsened_edgelist_minors, coarsened_edgelist_weights] =
    decompress_matrix_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
      handle,
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu>(
        graph_view.get_matrix_partition_view()),
      labels,
      detail::minor_properties_device_view_t<vertex_t, vertex_t const*>(labels),
      graph_view.get_local_adj_matrix_partition_segment_offsets(0),
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

  rmm::device_uvector<vertex_t> unique_labels(graph_view.get_number_of_vertices(),
                                              handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), labels, labels + unique_labels.size(), unique_labels.begin());
  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  auto [renumber_map_labels, meta] = renumber_edgelist<vertex_t, edge_t, multi_gpu>(
    handle,
    std::optional<rmm::device_uvector<vertex_t>>{std::move(unique_labels)},
    coarsened_edgelist_majors.data(),
    coarsened_edgelist_minors.data(),
    static_cast<edge_t>(coarsened_edgelist_majors.size()),
    do_expensive_check);

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{};
  edgelist.p_src_vertices =
    store_transposed ? coarsened_edgelist_minors.data() : coarsened_edgelist_majors.data();
  edgelist.p_dst_vertices =
    store_transposed ? coarsened_edgelist_majors.data() : coarsened_edgelist_minors.data();
  edgelist.p_edge_weights  = coarsened_edgelist_weights
                               ? std::optional<weight_t const*>{(*coarsened_edgelist_weights).data()}
                               : std::nullopt;
  edgelist.number_of_edges = static_cast<edge_t>(coarsened_edgelist_majors.size());

  return std::make_tuple(
    std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
      handle,
      edgelist,
      graph_meta_t<vertex_t, edge_t, multi_gpu>{
        static_cast<vertex_t>(renumber_map_labels.size()),
        graph_properties_t{graph_view.is_symmetric(), false},
        meta.segment_offsets}),
    std::move(renumber_map_labels));
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
           rmm::device_uvector<vertex_t>>
coarsen_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels,
  bool do_expensive_check)
{
  return detail::coarsen_graph(handle, graph_view, labels, do_expensive_check);
}

}  // namespace cugraph
